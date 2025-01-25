from datetime import datetime, timedelta

import pandas as pd
from langchain_core.tools import tool

from tooling.db_instance import _engine
from tooling.utilities import remove_extra_spaces


def daily_office_stats_by_series(
    office_names: list[str] = [
        "AFC Miraflores II",
        "AFC Providencia",
    ],
    corte_espera: int = 600,
    start_date: str = datetime.now().strftime("%Y/%m/%d"),
    end_date: str = (datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d"),
):
    """
    Use this function to get daily office statistics for the last two weeks
    """

    start_date = datetime.now().strftime("%Y/%m/%d") if start_date == "" else start_date
    end_date = (
        (datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d") if end_date == "" else end_date
    )

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
        o.Oficina                                                                                            AS "Oficina",
        DATENAME(WEEKDAY, a.FH_Emi)                                                                          AS "Dia",
        CAST(a.FH_Emi AS DATE)                                                                               AS "Fecha",
        s.[Serie]                                                                                            AS "Serie",
        COUNT(*)                                                                                             AS "Atenciones",
        COUNT(*) * 1.0 / COUNT(DISTINCT a.IdEsc)                                                             AS "Promedio Atenciones por Escritorio",
        COUNT(DISTINCT a.IdEsc)                                                                              AS "Escritorios Utilizados",
        COUNT(DISTINCT a.IdEje)                                                                              AS "Ejecutivos Atendieron",
        SUM(CASE WHEN a.Perdido = 1 THEN 1 ELSE 0 END)                                                       AS "Abandonos",
        100 * AVG(CASE WHEN DATEDIFF(SECOND, a.FH_Emi, a.FH_AteIni) < @corteEspera THEN 1.0 ELSE 0.0 END)    AS "Nivel de Servicio (%)",
        AVG(DATEDIFF(SECOND, a.FH_Emi, a.FH_AteIni)) / 60.0                                                  AS "Tiempo de Espera Promedio (minutos)",
        AVG(DATEDIFF(SECOND, a.FH_AteIni, a.FH_AteFin)) / 60.0                                               AS "Tiempo de Atencion Promedio (minutos)",
        100 * AVG(CASE WHEN a.Perdido = 1 THEN 1.0 ELSE 0.0 END)                                             AS "Tasa de Abandono (%)"
    FROM
        Atenciones a
        JOIN
        Oficinas o ON o.IdOficina = a.IdOficina
        JOIN
        Series s ON a.IdSerie = s.IdSerie AND a.IdOficina = s.IdOficina
    WHERE
            a.FH_Emi BETWEEN @startDate AND @maxDate
        AND o.Oficina IN (SELECT OfficeName
        FROM @OfficeList)
    GROUP BY
            o.Oficina, CAST(a.FH_Emi AS DATE), DATENAME(WEEKDAY, a.FH_Emi), s.[Serie]
    ORDER BY
            o.Oficina, CAST(a.FH_Emi AS DATE), s.Serie;
    """

    with _engine.connect() as conn:
        data: pd.DataFrame = pd.read_sql_query(query, conn)

    if data.empty:
        return "Sin data disponible en el rango u oficinas seleccioandas"

    markdown_table = data.to_markdown(index=False)

    return f"""Desempeño diario de oficina(s) correctamente completado:
    {remove_extra_spaces(markdown_table)}
    """


@tool
def tool_daily_office_stats_by_series(
    office_names: list[str],
    corte_espera: int = 600,
    start_date: str = datetime.now().strftime("%Y/%m/%d"),
    end_date: str = (datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d"),
):
    """
    Use this when the user asks for daily stats for offices by service series. Prefer tool_daily_office_stats if series are not needed.
    This tool provides daily statistics for offices based on service series.
    If no date has been specified, use today as the start date and tomorrow as the end date.
    The dates should be in the format "YYYY/MM/DD", and the cutoff time is at 00:00:00.
    The "Dia" column should be translated to Spanish. Return the table and a short analysis of the data.

    Parameters
    ----------
    office_names : list[str]
        List of office names to consider, e.g., [ "077 - Plaza Egaña", "199 - Melipilla" ]
    corte_espera : int, optional
        Time in seconds for cutoff of service level, default is 600
    start_date : str, optional
        Begin date to consider, default is today's date
    end_date : str, optional
        End date to consider, default is tomorrow's date
    """
    return daily_office_stats_by_series(
        office_names=office_names,
        corte_espera=corte_espera,
        start_date=start_date,
        end_date=end_date,
    )
