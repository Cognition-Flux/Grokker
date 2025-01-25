import json
from typing import List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tooling.db_instance import _engine
from tooling.utilities import add_docstring


def calculate_daily_status_times(
    executive_names: list[str] = [
        "Abigail Betzabet Calabrano Avalos",
        "Maria Margarita Bahamondez Madrid",
    ],
    start_date: str = "2024-10-01",
    end_date: str = "2024-11-01",
):
    """
    Calcula el promedio diario en minutos que cada ejecutivo está en pausa ("P") y activo ("A")
    para las tablas EjeEstado y EjeEstado_D, en el rango de fechas especificado.

    Parámetros:
    - executive_names: Lista de nombres completos de los ejecutivos.
    - start_date: Fecha de inicio en formato 'YYYY-MM-DD'. Por defecto es '2024-10-01'.
    - end_date: Fecha de fin en formato 'YYYY-MM-DD'. Por defecto es '2024-11-01'.
    """
    from datetime import datetime, timedelta

    import pandas as pd
    from dateutil import parser
    from sqlalchemy.exc import ResourceClosedError

    # Validar y parsear las fechas
    if not start_date:
        start_date = datetime.now().strftime("%Y-%m-%d")
    if not end_date:
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        start_date_parsed = parser.parse(start_date).strftime("%Y-%m-%d")
        end_date_parsed = parser.parse(end_date).strftime("%Y-%m-%d")
    except ValueError:
        return "Error: Formato de fecha inválido. Por favor, use 'YYYY-MM-DD'."

    try:
        with _engine.connect() as conn:
            # Obtener IdEje de todos los ejecutivos
            executive_names_str = "', '".join(executive_names)
            query_id_eje = f"""
            SELECT IdEje, Ejecutivo
            FROM Ejecutivos
            WHERE Ejecutivo IN ('{executive_names_str}')
            """
            df_id_eje = pd.read_sql_query(query_id_eje, conn)
            if df_id_eje.empty:
                return "No se encontraron los ejecutivos en la base de datos."

            # Crear un mapeo de nombre de ejecutivo a IdEje
            executive_id_map = dict(zip(df_id_eje["Ejecutivo"], df_id_eje["IdEje"]))

            # Inicializar el reporte
            report = ""

            # Procesar cada ejecutivo
            for executive_name in executive_names:
                report += f"Promedio diario en minutos para el ejecutivo '{executive_name}' desde {start_date_parsed} hasta {end_date_parsed}:\n\n"
                if executive_name not in executive_id_map:
                    report += (
                        f"No se encontró el ejecutivo '{executive_name}' en la base de datos.\n\n"
                    )
                    continue
                IdEje = executive_id_map[executive_name]

                # Listas para almacenar resultados
                results = []

                # Tablas a procesar
                tables = ["EjeEstado", "EjeEstado_D"]

                for table in tables:
                    query = f"""
                    WITH EventDurations AS (
                        SELECT
                            e.IdEje,
                            CAST(e.FH_Eve AS DATE) AS Fecha,
                            e.Evento,
                            e.FH_Eve,
                            LEAD(e.FH_Eve) OVER (PARTITION BY e.IdEje ORDER BY e.FH_Eve) AS Next_FH_Eve,
                            LEAD(e.Evento) OVER (PARTITION BY e.IdEje ORDER BY e.FH_Eve) AS Next_Evento
                        FROM
                            {table} e
                        WHERE
                            e.IdEje = {IdEje}
                            AND e.FH_Eve BETWEEN '{start_date_parsed}' AND '{end_date_parsed}'
                    ),
                    Durations AS (
                        SELECT
                            IdEje,
                            Fecha,
                            Evento,
                            DATEDIFF(SECOND, FH_Eve, Next_FH_Eve) / 60.0 AS DurationMinutes
                        FROM
                            EventDurations
                        WHERE
                            Next_FH_Eve IS NOT NULL
                            AND Evento IN ('A', 'P')
                            AND DATEDIFF(SECOND, FH_Eve, Next_FH_Eve) > 0
                    ),
                    DailyTotals AS (
                        SELECT
                            Fecha,
                            Evento,
                            SUM(DurationMinutes) AS TotalMinutes
                        FROM
                            Durations
                        GROUP BY
                            Fecha, Evento
                    ),
                    DailyAverages AS (
                        SELECT
                            Evento,
                            AVG(TotalMinutes) AS AverageDailyMinutes
                        FROM
                            DailyTotals
                        GROUP BY
                            Evento
                    )
                    SELECT
                        Evento,
                        AverageDailyMinutes
                    FROM
                        DailyAverages
                    """

                    df_result = pd.read_sql_query(query, conn)
                    if df_result.empty:
                        message = f"No se encontraron datos en la tabla '{table}' para el ejecutivo y rango de fechas especificados."
                        results.append({"Tabla": table, "Mensaje": message})
                    else:
                        # Formatear los resultados
                        df_result["AverageDailyMinutes"] = df_result["AverageDailyMinutes"].round(2)
                        results.append({"Tabla": table, "Datos": df_result})

                # Añadir resultados al reporte
                for result in results:
                    report += f"Tabla: {result['Tabla']}\n"
                    if "Mensaje" in result:
                        report += result["Mensaje"] + "\n\n"
                    else:
                        df = result["Datos"]
                        report += df.to_markdown(index=False)
                        report += "\n\n"

                report += "-" * 80 + "\n\n"

            return report

    except ResourceClosedError:
        return (
            "Error: La conexión a la base de datos está cerrada. Por favor, verifique la conexión."
        )
    except Exception as e:
        return f"Error al calcular los tiempos: {e}"


class DailyStatusTimesInput(BaseModel):
    executive_names: List[str] = Field(
        default=[
            "Abigail Betzabet Calabrano Avalos",
            "Maria Margarita Bahamondez Madrid",
        ],
        description="List of full names of executives to analyze",
    )
    start_date: str = Field(default="2024-10-01", description="Start date in 'YYYY-MM-DD' format")
    end_date: str = Field(default="2024-11-01", description="End date in 'YYYY-MM-DD' format")

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
Calculate daily average time in minutes that executives spend in pause ("P") and active ("A") states.

This tool analyzes the EjeEstado and EjeEstado_D tables to calculate how much time each executive
spends in different states (Active/Pause) during their workday, averaged over the specified date range.

Parameters:
{params_doc}

Returns:
A detailed report containing for each executive:
- Average daily minutes in Active state (A)
- Average daily minutes in Pause state (P)
- Data from both EjeEstado and EjeEstado_D tables

Table format:
| Evento | AverageDailyMinutes |
Where:
- Evento: Status code (A=Active, P=Pause)
- AverageDailyMinutes: Average time spent in that status per day
""".format(params_doc=DailyStatusTimesInput.get_documentation())
)
def get_daily_status_times(input_string: str) -> str:
    try:
        params = json.loads(input_string) if input_string else {}
        input_data = DailyStatusTimesInput(**params)
        return calculate_daily_status_times(**input_data.model_dump())
    except Exception as e:
        return f"Error: {str(e)}"


# Create the structured tool
daily_status_times_tool = StructuredTool.from_function(
    func=get_daily_status_times,
    name="DailyStatusTimesTool",
    description=get_daily_status_times.__doc__,
    return_direct=True,
)
