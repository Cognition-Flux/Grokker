# print(top_executives_report())
import json
from datetime import datetime, timedelta
from typing import List

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tooling.db_instance import _engine
from tooling.utilities import add_docstring, remove_extra_spaces


def average_daily_attentions_by_executive(
    office_names: list[str] = ["159 - Providencia", "197 - Talagante"],
    start_date: str = "2024/10/01",
    end_date: str = "2024/11/01",
):
    """
    Calcula cuántas atenciones promedio diarias por serie realiza cada ejecutivo en las oficinas especificadas
    y en el rango de fechas dado. Es tolerante cuando se solicita un único día y maneja múltiples oficinas.
    """
    # Validar las fechas y asignar valores por defecto si es necesario
    start_date = datetime.now().strftime("%Y/%m/%d") if not start_date else start_date
    end_date = (
        (datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d") if not end_date else end_date
    )

    query = f"""
    DECLARE @OfficeList TABLE (OfficeName NVARCHAR(255));
    INSERT INTO @OfficeList (OfficeName)
    VALUES {', '.join([f"('{office}')" for office in office_names])};

    DECLARE @startDate DATE = '{start_date}';
    DECLARE @endDate DATE = '{end_date}';

    WITH ExecutiveSeriesStats AS (
        SELECT
            o.Oficina AS Oficina,
            e.Ejecutivo AS Ejecutivo,
            s.Serie AS Serie,
            COUNT(*) AS TotalAtenciones,
            COUNT(DISTINCT CAST(a.FH_Emi AS DATE)) AS DiasTrabajados
        FROM
            Atenciones a
            JOIN Ejecutivos e ON a.IdEje = e.IdEje
            JOIN Oficinas o ON a.IdOficina = o.IdOficina
            JOIN Series s ON a.IdSerie = s.IdSerie AND a.IdOficina = s.IdOficina
        WHERE
            o.Oficina IN (SELECT OfficeName FROM @OfficeList)
            AND a.FH_Emi BETWEEN @startDate AND @endDate
        GROUP BY
            o.Oficina, e.Ejecutivo, s.Serie
    )
    SELECT
        Oficina,
        Ejecutivo,
        Serie,
        CASE 
            WHEN DiasTrabajados > 0 THEN CAST(TotalAtenciones AS FLOAT) / DiasTrabajados 
            ELSE 0 
        END AS PromedioAtencionesDiarias,
        FORMAT(@startDate, 'dd/MM/yyyy') + ' - ' + FORMAT(@endDate, 'dd/MM/yyyy') AS RangoFechas
    FROM
        ExecutiveSeriesStats
    ORDER BY
        Oficina, Ejecutivo, Serie;
    """

    with _engine.connect() as conn:
        data: pd.DataFrame = pd.read_sql_query(query, conn)

    if data.empty:
        return (
            f"Sin data disponible en el rango ({start_date} - {end_date}) u oficinas seleccionadas"
        )

    # Obtener los nombres de las oficinas analizadas
    offices_analyzed = ", ".join(set(data["Oficina"]))

    markdown_table = data.to_markdown(index=False, floatfmt=".2f")

    return f"""
Atenciones promedio diarias por serie realizadas por cada ejecutivo en las oficinas: {offices_analyzed}
{remove_extra_spaces(markdown_table)}
"""


class AverageDailyAttentionsInput(BaseModel):
    office_names: List[str] = Field(
        default=["159 - Providencia", "197 - Talagante"],
        description="List of office names to analyze",
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
Calculate average daily attentions by series for each executive in specified offices.

This tool calculates how many average daily attentions per series each executive performs 
in the specified offices and date range. It handles single-day requests and multiple offices.

Parameters:
{params_doc}

Returns a table with:
| Oficina | Ejecutivo | Serie | PromedioAtencionesDiarias | RangoFechas |
""".format(params_doc=AverageDailyAttentionsInput.get_documentation())
)
def get_average_daily_attentions(input_string: str) -> str:
    try:
        params = json.loads(input_string) if input_string else {}
        input_data = AverageDailyAttentionsInput(**params)
        return average_daily_attentions_by_executive(**input_data.model_dump())
    except Exception as e:
        return f"Error: {str(e)}"


executive_specific_data_tool = StructuredTool.from_function(
    func=get_average_daily_attentions,
    name="AverageDailyAttentionsTool",
    description=get_average_daily_attentions.__doc__,
    return_direct=True,
)
