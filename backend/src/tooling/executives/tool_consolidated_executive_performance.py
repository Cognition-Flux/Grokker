# print(top_executives_report())
import json
from datetime import datetime, timedelta
from typing import List

import pandas as pd
from dateutil import parser
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.exc import ResourceClosedError

from tooling.db_instance import _engine
from tooling.utilities import add_docstring, remove_extra_spaces


def consolidated_executive_performance(
    executive_names: list[str],
    start_date: str = "2024/10/01",
    end_date: str = "2024/11/01",
):
    """
    Esta función toma una lista de nombres de ejecutivos y un rango de fechas,
    y retorna un informe consolidado del rendimiento diario de cada ejecutivo,
    incluyendo promedio de atenciones diarias, tiempo promedio de atención (minutos),
    series únicas atendidas, la oficina a la que pertenece cada ejecutivo,
    y el rendimiento calculado como PromedioAtencionesDiarias / TiempoPromedioAtencionMinutos.

    Parámetros:
    - executive_names: Lista de nombres completos de los ejecutivos.
    - start_date: Fecha de inicio en formato 'YYYY/MM/DD'. Por defecto es '2024/10/01'.
    - end_date: Fecha de fin en formato 'YYYY/MM/DD'. Por defecto es '2024/11/01'.
    """

    # Validar y parsear las fechas
    if not start_date:
        start_date = datetime.now().strftime("%Y/%m/%d")
    if not end_date:
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d")

    try:
        start_date_parsed = parser.parse(start_date).strftime("%Y/%m/%d")
        end_date_parsed = parser.parse(end_date).strftime("%Y/%m/%d")
    except ValueError:
        return "Error: Formato de fecha inválido. Por favor, use 'YYYY/MM/DD'."

    try:
        with _engine.connect() as conn:
            # Obtener IdEje y oficina de todos los ejecutivos
            # Construir una cadena de nombres de ejecutivos para la consulta SQL
            executive_names_str = "', '".join(executive_names)
            query_executives = f"""
            SELECT e.IdEje, e.Ejecutivo, o.Oficina
            FROM Ejecutivos e
            LEFT JOIN EjeEstado es ON e.IdEje = es.IdEje
            LEFT JOIN Oficinas o ON es.IdOficina = o.IdOficina
            WHERE e.Ejecutivo IN ('{executive_names_str}')
            """
            df_executives = pd.read_sql_query(query_executives, conn)
            if df_executives.empty:
                return "No se encontraron los ejecutivos en la base de datos."

            # Eliminar duplicados en caso de que un ejecutivo aparezca en múltiples oficinas
            df_executives = df_executives.drop_duplicates(subset=["IdEje"])

            # Crear un mapeo de IdEje a Ejecutivo y Oficina
            executive_info = df_executives.set_index("IdEje")[["Ejecutivo", "Oficina"]].to_dict(
                "index"
            )
            id_eje_list = df_executives["IdEje"].tolist()

            # Construir la consulta para obtener el rendimiento de los ejecutivos
            id_eje_str = ", ".join(map(str, id_eje_list))
            query_performance = f"""
            DECLARE @startDate DATE = '{start_date_parsed}';
            DECLARE @endDate DATE = '{end_date_parsed}';

            WITH ExecutivePerformance AS (
                SELECT
                    e.IdEje,
                    e.Ejecutivo,
                    o.Oficina,
                    COUNT(*) * 1.0 / COUNT(DISTINCT CAST(a.FH_Emi AS DATE)) AS PromedioAtencionesDiarias,
                    AVG(DATEDIFF(SECOND, a.FH_AteIni, a.FH_AteFin)) / 60.0 AS TiempoPromedioAtencionMinutos,
                    (COUNT(*) * 1.0 / COUNT(DISTINCT CAST(a.FH_Emi AS DATE))) / NULLIF(AVG(DATEDIFF(SECOND, a.FH_AteIni, a.FH_AteFin)) / 60.0, 0) AS Rendimiento,
                    STUFF((SELECT DISTINCT ', ' + s.Serie
                          FROM Atenciones a2
                          JOIN Series s ON s.IdSerie = a2.IdSerie
                          WHERE a2.IdEje = e.IdEje
                                AND a2.FH_Emi BETWEEN @startDate AND @endDate
                                AND a2.FH_AteIni IS NOT NULL AND a2.FH_AteFin IS NOT NULL
                          FOR XML PATH('')), 1, 2, '') AS Series,
                    @startDate AS StartDate,
                    @endDate AS EndDate
                FROM
                    Atenciones a
                    JOIN Oficinas o ON a.IdOficina = o.IdOficina
                    JOIN Ejecutivos e ON a.IdEje = e.IdEje
                WHERE
                    e.IdEje IN ({id_eje_str})
                    AND a.FH_AteIni IS NOT NULL
                    AND a.FH_AteFin IS NOT NULL
                    AND a.FH_Emi BETWEEN @startDate AND @endDate
                GROUP BY
                    e.IdEje, e.Ejecutivo, o.Oficina
            )
            SELECT
                IdEje,
                Ejecutivo,
                Oficina,
                Series,
                PromedioAtencionesDiarias,
                TiempoPromedioAtencionMinutos,
                Rendimiento,
                FORMAT(StartDate, 'dd/MM/yyyy') + ' - ' + FORMAT(EndDate, 'dd/MM/yyyy') AS RangoFechas
            FROM
                ExecutivePerformance
            ORDER BY
                Ejecutivo;
            """

            df_performance = pd.read_sql_query(query_performance, conn)
            if df_performance.empty:
                return "No se encontraron datos de rendimiento para los ejecutivos en el rango de fechas especificado."

            # Combinar la información de los ejecutivos con su rendimiento
            df_result = df_performance.copy()
            df_result["Oficina"] = df_result.apply(
                lambda row: executive_info.get(row["IdEje"], {}).get("Oficina", "Desconocida"),
                axis=1,
            )

            # Reordenar las columnas
            df_result = df_result[
                [
                    "Ejecutivo",
                    "Oficina",
                    "Series",
                    "PromedioAtencionesDiarias",
                    "TiempoPromedioAtencionMinutos",
                    "Rendimiento",
                    "RangoFechas",
                ]
            ]

            # Redondear las columnas numéricas
            df_result["PromedioAtencionesDiarias"] = df_result["PromedioAtencionesDiarias"].round(2)
            df_result["TiempoPromedioAtencionMinutos"] = df_result[
                "TiempoPromedioAtencionMinutos"
            ].round(2)
            df_result["Rendimiento"] = df_result["Rendimiento"].round(2)

            # Generar la tabla en formato Markdown
            markdown_table = df_result.to_markdown(index=False)

            # Construir el reporte
            report = f"""
Informe consolidado de rendimiento diario para los ejecutivos:
{', '.join(executive_names)}
Periodo: {start_date_parsed} - {end_date_parsed}

{remove_extra_spaces(markdown_table)}
"""

            return report

    except ResourceClosedError:
        return (
            "Error: La conexión a la base de datos está cerrada. Por favor, verifique la conexión."
        )
    except Exception as e:
        return f"Error al calcular el rendimiento: {e}"


class ConsolidatedExecutivePerformanceInput(BaseModel):
    executive_names: List[str] = Field(
        default=["Paulina Gonzalez Gallegos", "Leonardo Esteban Garces Soto"],
        description="List of full names of executives to analyze",
    )
    start_date: str = Field(default="2024/10/01", description="Start date in 'YYYY/MM/DD' format")
    end_date: str = Field(default="2024/11/01", description="End date in 'YYYY/MM/DD' format")

    @classmethod
    def get_documentation(cls) -> str:
        schema = cls.schema()
        docs = []

        # Extract title and description if available
        title = schema.get("title", cls.__name__)
        description = schema.get("description", "")
        docs.append(f"**{title}**")
        if description:
            docs.append(f"\n{description}\n")

        # Add field documentation
        for field_name, field_info in schema.get("properties", {}).items():
            field_type = field_info.get("type", "Unknown type")
            field_desc = field_info.get("description", "No description")
            default = field_info.get("default", "No default")
            constraints = ""

            # Include constraints like minimum or maximum values
            if "minimum" in field_info:
                constraints += f", minimum: {field_info['minimum']}"
            if "maximum" in field_info:
                constraints += f", maximum: {field_info['maximum']}"
            if "enum" in field_info:
                constraints += f", allowed values: {field_info['enum']}"

            field_doc = (
                f"- `{field_name}` ({field_type}{constraints}): {field_desc}\n"
                f" Default: `{default}`"
            )
            docs.append(field_doc)

        return "\n\n".join(docs)


@add_docstring(
    """
Get consolidated performance report for multiple executives.

This tool provides a consolidated report of daily performance for a list of executives,
including average daily attentions, average attention time (minutes), unique series attended,
the office they belong to, and performance calculated as AverageDailyAttentions / AverageAttentionTimeMinutes.

Parameters:
{params_doc}

Returns a table with:
| Ejecutivo | Oficina | Series | PromedioAtencionesDiarias | TiempoPromedioAtencionMinutos | Rendimiento | RangoFechas |
""".format(params_doc=ConsolidatedExecutivePerformanceInput.get_documentation())
)
def get_consolidated_executive_performance(input_string: str) -> str:
    try:
        params = json.loads(input_string) if input_string else {}
        input_data = ConsolidatedExecutivePerformanceInput(**params)
        return consolidated_executive_performance(**input_data.model_dump())
    except Exception as e:
        return f"Error: {str(e)}"


# Create the structured tool
consolidated_executive_performance_tool = StructuredTool.from_function(
    func=get_consolidated_executive_performance,
    name="ConsolidatedExecutivePerformanceTool",
    description=get_consolidated_executive_performance.__doc__,
    return_direct=True,
)
