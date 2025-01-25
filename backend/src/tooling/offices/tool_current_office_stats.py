# %% -- OFFICE STATS NOW
# @tool
from datetime import datetime

import pandas as pd
from langchain_core.tools import tool

from tooling.db_instance import _engine
from tooling.utilities import remove_extra_spaces


def get_current_office_stats(office_name: str) -> str:
    today: str = datetime.now().strftime("%Y/%m/%d")

    query_series = f"""
    SELECT  o.[Oficina]   AS "Oficina"
        , s.[Serie]       AS "Serie"
        , COUNT(*)        AS "Atenciones"
        , MAX(a.[FH_Emi]) AS "Ultima atencion"
        , FORMAT( AVG(a.[TpoAte]) / 60.0, 'N2')  AS "Tiempo Medio de Atencion (minutos)"
        , FORMAT( AVG(a.[TpoEsp]) / 60.0, 'N2')  AS "Tiempo Medio de Espera (minutos)"
        , SUM(CASE WHEN a.[Perdido] = 1 THEN 1 ELSE 0 END)  AS "Abandonos"
        , FORMAT(AVG(CASE WHEN a.[Perdido] = 1 THEN 100.0 ELSE 0 END), 'N2')  AS "Abandonos (%)"
    FROM [dbo].[Atenciones] a
        JOIN [dbo].[Series] s ON (s.[IdSerie] = a.[IdSerie]) AND (s.[IdOficina] = a.[IdOficina])
        JOIN [dbo].[Oficinas] o ON (o.[IdOficina] = a.[IdOficina])
    WHERE o.[Oficina] = '{office_name}' AND a.[FH_Emi] > '{today}'
    GROUP BY o.[Oficina], s.[Serie]
    ORDER BY COUNT(*) DESC
    """

    query_ejecutvos = f"""
    SELECT  o.[Oficina]   AS "Oficina"
        , e.[Ejecutivo]   AS "Ejecutivo"
        , COUNT(*)        AS "Atenciones"
        , MAX(a.[FH_Emi]) AS "Ultima atencion"
        , MIN(a.[FH_Emi]) AS "Primera atencion"
        , FORMAT( AVG(a.[TpoAte]) / 60.0, 'N2')  AS "Tiempo Medio de Atencion (minutos)"
    FROM [dbo].[Atenciones] a
        JOIN [dbo].[Ejecutivos] e ON (e.[IdEje] = a.[IdEje])
        JOIN [dbo].[Oficinas] o ON (o.[IdOficina] = a.[IdOficina])
    WHERE o.[Oficina] = '{office_name}' AND a.[FH_Emi] > '{today}'
    GROUP BY o.[Oficina], e.[Ejecutivo]
    ORDER BY COUNT(*) DESC
    """

    with _engine.connect() as conn:
        data_series: pd.DataFrame = pd.read_sql_query(query_series, conn)
        data_executives: pd.DataFrame = pd.read_sql_query(query_ejecutvos, conn)

    if data_series.empty:
        return "Sin data disponible en el rango u oficinas seleccioandas"

    markdown_table_series = data_series.to_markdown(index=False)
    markdown_table_executives = data_executives.to_markdown(index=False)

    return f"""
    # Reporte de rendimiento hoy para {office_name}. Fecha {today}
    
    Desempeño diario de oficina correctamente completado:
    
    ## Atenciones por serie
    {remove_extra_spaces(markdown_table_series)}

    ## Atenciones por ejecutivo
    {remove_extra_spaces(markdown_table_executives)}
    """


@tool
def tool_current_office_stats(office_name: str):
    """
    Use this tool when the user asks for the current stats of an office.
    This will return the current day. Do not use it for historical data.

    This will return two tables with the stats by series, and the stats by executive.

    Add a short analysis of the data and return the tables temselves.

    Parameters
    ----------
    office_name : str
        Name of the office to consider, for example "077 - Plaza Egaña"
    """
    return get_current_office_stats(office_name)
