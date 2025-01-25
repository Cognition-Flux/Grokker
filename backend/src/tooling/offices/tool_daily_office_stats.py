# %% -- DAILY OFFICE STATS (GLOBAL)
# @tool
from datetime import datetime, timedelta

import pandas as pd
from langchain_core.tools import BaseTool, tool
from typing_extensions import Literal

from tooling.db_instance import _engine
from tooling.utilities import remove_extra_spaces


def daily_office_stats(
    office_names: list[str] = [
        "AFC Miraflores II",
        "AFC Providencia",
    ],
    corte_espera: int = 600,
    start_date: str = datetime.now().strftime("%Y/%m/%d"),
    end_date: str = (datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d"),
    sort_by: Literal["Atenciones", "Perdidas", "Tasa_abandono", "Espera", "Oficina"] = "Oficina",
):
    """
    Use this function to get daily office statistics for the last two weeks
    """

    start_date = datetime.now().strftime("%Y/%m/%d") if start_date == "" else start_date
    end_date = (
        (datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d") if end_date == "" else end_date
    )

    match sort_by:
        case "Atenciones":
            order_by = "COUNT(*) DESC"
        case "Perdidas":
            order_by = "SUM(CASE WHEN a.Perdido = 1 THEN 1 ELSE 0 END) DESC"
        case "Tasa_abandono":
            order_by = "AVG(CASE WHEN a.Perdido = 1 THEN 100.0 ELSE 0.0 END) DESC"
        case "Espera":
            order_by = "AVG(DATEDIFF(SECOND, a.FH_Emi, a.FH_AteIni)) DESC"
        case "Oficina":
            order_by = "o.Oficina, CAST (a.FH_Emi AS DATE);"

    # office_names_str = ", ".join([f"'{name}'" for name in office_names])
    query = f"""
    DECLARE @OfficeList TABLE (OfficeName NVARCHAR(255));
    INSERT INTO @OfficeList (OfficeName)
    VALUES {', '.join([f"('{office}')" for office in office_names])};
    DECLARE @maxDate DATE = '{end_date}';
    DECLARE @startDate DATE = '{start_date}';
    DECLARE @corteEspera INT = {corte_espera};
    -- PROCEDURE ITSEFL FROM HERE
    SELECT
        o.Oficina                                                                                                   AS "Oficina",
        DATENAME(WEEKDAY, a.FH_Emi)                                                                                 AS "Dia",
        CAST(a.FH_Emi AS DATE)                                                                                      AS "Fecha",
        COUNT(*)                                                                                                    AS "Atenciones Totales",
        FORMAT(COUNT(*) * 1.0 / COUNT(DISTINCT a.IdEsc), 'N2')                                                      AS "Promedio Atenciones por Escritorio",
        COUNT(DISTINCT a.IdEsc)                                                                                     AS "Escritorios Utilizados",
        COUNT(DISTINCT a.IdEje)                                                                                     AS "Ejecutivos Atendieron",
        SUM(CASE WHEN a.Perdido = 1 THEN 1 ELSE 0 END)                                                              AS "Abandonos",
        FORMAT(AVG(CASE WHEN DATEDIFF(SECOND, a.FH_Emi, a.FH_AteIni) < @corteEspera THEN 100.0 ELSE 0.0 END),'N2')  AS "Nivel de Servicio (%)",
        FORMAT(AVG(DATEDIFF(SECOND, a.FH_Emi, a.FH_AteIni)) / 60.0, 'N2')                                           AS "Tiempo de Espera Promedio (minutos)",
        FORMAT(AVG(DATEDIFF(SECOND, a.FH_AteIni, a.FH_AteFin)) / 60.0, 'N2' )                                       AS "Tiempo de Atencion Promedio (minutos)",
        FORMAT(AVG(CASE WHEN a.Perdido = 1 THEN 100.0 ELSE 0.0 END), 'N2')                                          AS "Tasa de Abandono (%)"
    FROM
        Atenciones a
        JOIN
        Oficinas o ON o.IdOficina = a.IdOficina
    WHERE
                a.FH_Emi BETWEEN @startDate AND @maxDate
        AND o.Oficina IN
    (SELECT OfficeName
        FROM @OfficeList)
    GROUP BY
                o.Oficina, CAST
    (a.FH_Emi AS DATE), DATENAME
    (WEEKDAY, a.FH_Emi)
    ORDER BY {order_by};
    """

    with _engine.connect() as conn:
        data: pd.DataFrame = pd.read_sql_query(query, conn)

    if data.empty:
        return (
            f"Sin data disponible en el rango ({start_date} a {end_date}) u oficinas seleccioandas"
        )

    markdown_table = data.to_markdown(index=False)

    return f"""Desempeño diario de oficina(s) correctamente completado:
{remove_extra_spaces(markdown_table)}
"""


def tool_daily_office_stats(
    office_names: list[str],
    corte_espera: int = 600,
    start_date: str = datetime.now().strftime("%Y/%m/%d"),
    end_date: str = (datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d"),
    sort_by: Literal["Atenciones", "Perdidas", "Tasa_abandono", "Espera", "Oficina"] = "Oficina",
):
    """
    Use this when the user asks for daily stats for offices or the days with most demand, waiting times, lost calls, or proportion of lost calls.
    This tool returns the sum for all series.
    If no date has been specified, use today as the start date and tomorrow as the end date.
    Your answer should include the table that the tool responds with, and a short analysis of the data.

    Parameters
    ----------
    office_names : list[str]
        List of office names to consider, e.g., [ "077 - Plaza Egaña" ]
    corte_espera : int, optional
        Time in seconds for cutoff of service level, default is 600
    start_date : str, optional
        Begin date to consider, default is today's date
    end_date : str, optional
        End date to consider, default is tomorrow's date
    sort_by : Literal["Atenciones", "Perdidas", "Tasa_abandono", "Espera", "Oficina"], optional
        Use to sort the table by the selected column, default is "Oficina" wich is just the office name and date.
    """
    return daily_office_stats(
        office_names=office_names,
        corte_espera=corte_espera,
        start_date=start_date,
        end_date=end_date,
        sort_by=sort_by,
    )


tool_daily_office_stats: BaseTool = tool(tool_daily_office_stats)
