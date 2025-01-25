import json

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tooling.db_instance import _engine
from tooling.utilities import add_docstring, remove_extra_spaces


def executive_daily_performance(
    executive_name: str = "Ana Maria Caceres Henriquez ",
    office_name: str = "159 - Providencia",
    start_date: str = "2024/10/01",
    end_date: str = "2024/11/01",
):
    """
    Use this function to get daily performance of an executive for the last specified number of weeks
    """
    query = f"""
    DECLARE @maxDate DATE = '{end_date}';
    DECLARE @startDate DATE = '{start_date}';
    SELECT
        FORMAT(a.FH_Emi, 'yyyy-MM-dd')                                                AS "Fecha",
        DATENAME(WEEKDAY, a.FH_Emi)                                                   AS "Dia",
        COUNT(*)                                                                      AS "Total Atenciones",
        AVG(DATEDIFF(MINUTE, a.FH_AteIni, a.FH_AteFin))                               AS "Tiempo Promedio de Atencion (minutos)",
        COUNT(*) * 60.0 / NULLIF(SUM(DATEDIFF(MINUTE, a.FH_AteIni, a.FH_AteFin)), 0)  AS "Rendimiento por hora"
    FROM
        Atenciones a
        JOIN Oficinas o ON o.IdOficina = a.IdOficina
        JOIN Ejecutivos e ON e.IdEje = a.IdEje
    WHERE
        o.Oficina = '{office_name}'
        AND e.Ejecutivo = '{executive_name}'
        AND a.FH_Emi BETWEEN @startDate AND @maxDate
        AND a.FH_AteIni IS NOT NULL
        AND a.FH_AteFin IS NOT NULL
    GROUP BY
        FORMAT(a.FH_Emi, 'yyyy-MM-dd'),
        DATENAME(WEEKDAY, a.FH_Emi)
    ORDER BY
    Fecha ASC;
    """

    with _engine.connect() as conn:
        data: pd.DataFrame = pd.read_sql_query(query, conn)

    if data.empty:
        return (
            f"Sin data disponible en el rango ({start_date} - {end_date}) u oficinas seleccioandas"
        )

    markdown_table = data.to_markdown(index=False)

    return f"""
# Reporte de rendimiento diario para {executive_name}

Oficina: {office_name}, Periodo: {start_date} - {end_date}

Desempeño diario de ejecutivo correctamente completado:
{remove_extra_spaces(markdown_table)}
"""


#########################################################################
# -------------------------Tools para agente EJECUTIVOS--------------------
#########################################################################
# ExecutiveRankingTool - Get executive performance rankings (top N best/worst performers)
# ExecutiveDailyPerformanceTool - Get detailed daily performance metrics for a specific executive
# ConsolidatedExecutivePerformanceTool - Get consolidated performance report for multiple executives
# AverageDailyAttentionsTool - Calculate average daily attentions by series for each executive
# ExecutiveSpecificDataTool - Retrieve specific data about executives from multiple database tables
# DailyStatusTimesTool - Analyze time spent in active/pause states for executives
# AverageExecutivesSeriesTool - Calculate average number of executives assigned to each series by office


class ExecutiveDailyPerformanceInput(BaseModel):
    executive_name: str = Field(
        default="Ana Maria Caceres Henriquez", description="Full name of the executive"
    )
    office_name: str = Field(default="159 - Providencia", description="Name of the office")
    start_date: str = Field(default="2024/09/01", description="Start date in 'YYYY/MM/DD' format")
    end_date: str = Field(default="2024/10/01", description="End date in 'YYYY/MM/DD' format")

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
Get daily performance details for a specific executive.

Note: If executive information is not known, use the ExecutiveRankingTool first to get a list of executives.

Parameters:
{params_doc}

Returns a table with:
| Fecha | Día | Total Atenciones | Tiempo Promedio de Atención (minutos) | Rendimiento por hora |
""".format(params_doc=ExecutiveDailyPerformanceInput.get_documentation())
)
def get_executive_daily_performance(input_string: str) -> str:
    try:
        params = json.loads(input_string) if input_string else {}
        input_data = ExecutiveDailyPerformanceInput(**params)
        return executive_daily_performance(**input_data.model_dump())
    except Exception as e:
        return f"Error: {str(e)}"


# Create the structured tool
executive_daily_performance_tool = StructuredTool.from_function(
    func=get_executive_daily_performance,
    name="ExecutiveDailyPerformanceTool",
    description=get_executive_daily_performance.__doc__,
    return_direct=True,
)
