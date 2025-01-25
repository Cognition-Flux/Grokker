import json
from datetime import datetime, timedelta
from typing import List

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tooling.db_instance import _engine
from tooling.utilities import add_docstring, remove_extra_spaces


def average_executives_by_series(
    office_names: list[str] = [
        "159 - Providencia",
        "164 - Vitacura",
        "197 - Talagante",
    ],
    start_date: str = "2024/10/01",
    end_date: str = "2024/11/01",
):
    """
    Calcula el promedio diario de ejecutivos asignados por cada serie en las oficinas especificadas
    y en el rango de fechas dado. Es tolerante si se consulta por un único día.
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

    WITH DailyExecutives AS (
        SELECT
            o.Oficina AS Oficina,
            CAST(a.FH_Emi AS DATE) AS Fecha,
            s.Serie,
            COUNT(DISTINCT e.IdEje) AS NumEjecutivos
        FROM
            Atenciones a
            JOIN Ejecutivos e ON a.IdEje = e.IdEje
            JOIN Oficinas o ON a.IdOficina = o.IdOficina
            JOIN Series s ON a.IdSerie = s.IdSerie AND a.IdOficina = s.IdOficina
        WHERE
            o.Oficina IN (SELECT OfficeName FROM @OfficeList)
            AND a.FH_Emi BETWEEN @startDate AND @endDate
        GROUP BY
            o.Oficina, CAST(a.FH_Emi AS DATE), s.Serie
    ),
    AverageExecutives AS (
        SELECT
            Oficina,
            Serie,
            AVG(NumEjecutivos) AS PromedioEjecutivos
        FROM
            DailyExecutives
        GROUP BY
            Oficina, Serie
    )
    SELECT
        Oficina,
        Serie,
        PromedioEjecutivos,
        FORMAT(@startDate, 'dd/MM/yyyy') + ' - ' + FORMAT(@endDate, 'dd/MM/yyyy') AS RangoFechas
    FROM
        AverageExecutives
    ORDER BY
        Oficina, Serie;
    """

    with _engine.connect() as conn:
        data: pd.DataFrame = pd.read_sql_query(query, conn)

    if data.empty:
        return (
            f"Sin data disponible en el rango ({start_date} - {end_date}) u oficinas seleccionadas"
        )

    # Obtener los nombres de las oficinas analizadas
    offices_analyzed = ", ".join(set(data["Oficina"]))

    markdown_table = data.to_markdown(index=False)

    return f"""
Promedio diario de ejecutivos asignados por serie en las oficinas: {offices_analyzed}
{remove_extra_spaces(markdown_table)}
"""


class AverageExecutivesBySeriesInput(BaseModel):
    office_names: List[str] = Field(
        default=["159 - Providencia", "164 - Vitacura", "197 - Talagante"],
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
Calculate daily average number of executives assigned to each series by office.

This tool analyzes the number of executives handling each series in specified offices,
calculating daily averages over the given date range. It works for both single day
and date range queries.

Parameters:
{params_doc}

Returns:
A report containing:
- List of analyzed offices
- Table with columns:
  | Oficina | Serie | PromedioEjecutivos | RangoFechas |
  Where:
  - Oficina: Office name
  - Serie: Series type
  - PromedioEjecutivos: Average number of executives per day
  - RangoFechas: Date range of analysis
""".format(params_doc=AverageExecutivesBySeriesInput.get_documentation())
)
def get_average_executives_by_series(input_string: str) -> str:
    try:
        params = json.loads(input_string) if input_string else {}
        input_data = AverageExecutivesBySeriesInput(**params)
        return average_executives_by_series(**input_data.model_dump())
    except Exception as e:
        return f"Error: {str(e)}"


# Create the structured tool
average_executives_series_tool = StructuredTool.from_function(
    func=get_average_executives_by_series,
    name="AverageExecutivesSeriesTool",
    description=get_average_executives_by_series.__doc__,
    return_direct=True,
)
