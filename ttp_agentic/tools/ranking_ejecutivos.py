# %%
import os
from typing import List, Literal

import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ttp_agentic.db_instance import _engine
from ttp_agentic.tools.utilities import (
    add_docstring,
    get_documentation,
    parse_input,
    remove_extra_spaces,
)

load_dotenv(override=True)


def top_executives_report(
    office_names: list[str] = [
        "196 - Buin",
        "022 - Bombero Ossa ",
    ],
    start_date: str = "2024/10/01",
    end_date: str = "2024/11/01",
    top_ranking: int = 3,
    orden: str = "DESC",
):
    parsed_office_names = "', '".join(office_names)

    query = f"""
    DECLARE @startDate DATE = CONVERT(DATE, '{start_date}', 111);
    DECLARE @maxDate DATE = CONVERT(DATE, '{end_date}', 111);

    WITH
        ExecutivePerformance AS
        (
            SELECT
                o.Oficina,
                e.Ejecutivo,
                COUNT(*) AS TotalAtenciones,
                COUNT(*) * 1.0 / COUNT(DISTINCT CAST(a.FH_Emi AS DATE)) AS PromedioAtencionesdiarias,
                AVG(DATEDIFF(SECOND, a.FH_AteIni, a.FH_AteFin)) / 60.0 AS TiempoPromedioAtencionMinutos,
                ROW_NUMBER() OVER (PARTITION BY o.Oficina ORDER BY (COUNT(*) * 1.0 / COUNT(DISTINCT CAST(a.FH_Emi AS DATE))) {orden}) AS Ranking,
                STUFF((SELECT DISTINCT ', ' + s.Serie
                    FROM Atenciones a2
                    JOIN Series s ON s.IdSerie = a2.IdSerie
                    WHERE a2.IdEje = a.IdEje
                        AND CAST(a2.FH_Emi AS DATE) BETWEEN @startDate AND @maxDate
                        AND (a2.FH_AteIni IS NOT NULL) AND (a2.FH_AteFin IS NOT NULL)
                    FOR XML PATH('')), 1, 2, '') AS Series,
                MIN(CAST(a.FH_Emi AS DATE)) AS StartDate,
                MAX(CAST(a.FH_Emi AS DATE)) AS EndDate
            FROM
                Atenciones a
                JOIN Oficinas o ON a.IdOficina = o.IdOficina
                JOIN Ejecutivos e ON a.IdEje = e.IdEje
            WHERE
                o.Oficina IN ('{parsed_office_names}')
                AND a.FH_AteIni IS NOT NULL
                AND a.FH_AteFin IS NOT NULL
                AND CAST(a.FH_Emi AS DATE) BETWEEN @startDate AND @maxDate
            GROUP BY
                o.Oficina, e.Ejecutivo, a.IdEje
        )
    SELECT
        Ranking,
        Oficina,
        Ejecutivo,
        Series,
        TotalAtenciones AS "Total Atenciones",
        FORMAT(PromedioAtencionesdiarias, 'N2') AS "Atenciones Diarias promedio",
        FORMAT(TiempoPromedioAtencionMinutos, 'N2') AS "Tiempo de atencion promedio (min)",
        CASE
            WHEN StartDate = EndDate THEN FORMAT(StartDate, 'dd/MM/yyyy (dddd)', 'es-es')
            ELSE FORMAT(StartDate, 'dd/MM/yyyy (dddd)', 'es-es') + ' - ' + FORMAT(EndDate, 'dd/MM/yyyy (dddd)', 'es-es')
        END AS "Rango Registros"
    FROM
        ExecutivePerformance
    WHERE
        Ranking <= {top_ranking}
    ORDER BY
        Oficina,
        PromedioAtencionesdiarias {orden};
    """

    with _engine.connect() as conn:
        data: pd.DataFrame = pd.read_sql_query(query, conn)

    if data.empty:
        return f"Sin data disponible en el rango ({start_date} - {end_date}) u oficinas seleccionadas."

    markdown_table = data.to_markdown(index=False)

    return f"""Para la(s) oficina(s), el ranking se ordenó por Promedio Atenciones diarias
{"mostrando los peores ejecutivos primero (top 1 es el peor)" if orden == "ASC" else "mostrando los mejores ejecutivos primero (top 1 es el mejor)"}
(puede extraer el nombre completo de ejecutivos si se necesita para siguientes pasos de análisis), para cada Oficina:   
{remove_extra_spaces(markdown_table)}
(los resultados dependen de la disponibilidad de registros válidos)
"""


# top_executives_report()
## %%


class RankingEjecutivosInput(BaseModel):
    office_names: List[str] = Field(
        default=["196 - Buin", "022 - Bombero Ossa"],
        description="List of office names to consider in the ranking",
    )
    start_date: str = Field(
        default="2024/10/01", description="Start date in 'YYYY/MM/DD' format"
    )
    end_date: str = Field(
        default="2024/11/01", description="End date in 'YYYY/MM/DD' format"
    )
    top_ranking: int = Field(
        default=5,
        description="Number of executives to show, length of the ranking",
        ge=1,
    )
    orden: Literal["ASC", "DESC"] = Field(
        default="DESC",
        description="""Sorting order: usar 'DESC' para el mejor en top 1, usar 'ASC' para el peor en top 1.
                """,
    )
    # Bind the external implementations as classmethods
    parse_input_for_tool = classmethod(parse_input)
    get_documentation_for_tool = classmethod(get_documentation)


@add_docstring(
    """
Ejecutivos (funcionarios) que atendieron en una rango de tiempo ordenados por cantidad de atenciones  (ranking).
Entrega los peores (si orden=ASC) o mejores  (si orden=DESC) ejecutivos
Parameters:
{params_doc}
Returns a table with:
|Ranking|Oficina|Ejecutivo| Series|Total Atenciones|Atenciones Diarias promedio |Tiempo de atencion promedio (min)| Rango Registros|

""".format(
        params_doc=RankingEjecutivosInput.get_documentation_for_tool()
    )
)
def get_executive_ranking(input_string: str) -> str:
    try:
        input_data = RankingEjecutivosInput.parse_input_for_tool(input_string)
        return top_executives_report(**input_data.model_dump())
    except Exception as e:
        return f"Error: {str(e)}"


# Create the structured tool
executive_ranking_tool = StructuredTool.from_function(
    func=get_executive_ranking,
    name="executive_ranking_tool",
    description=get_executive_ranking.__doc__,
    return_direct=True,
)
if __name__ == "__main__":

    f"""{print(executive_ranking_tool.description)=},
      {print(executive_ranking_tool.invoke("{}"))=}"""
