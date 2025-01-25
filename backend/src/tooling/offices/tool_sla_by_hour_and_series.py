# %%
import pandas as pd
from langchain_core.tools import BaseTool, tool

from tooling.db_instance import _engine
from tooling.utilities import remove_extra_spaces


def get_sla_by_hour_and_series(
    office_name: str, date: str = "2024/10/14", corte_espera: int = 300
) -> str:
    """
    Use this to get the SLA of a single office in a single day
    Returns it as a global SLA by hour, and a SLA by series by hour
    This tool is verbose, so prefer tool_daily_office_stats if you only need the global daily SLA.

    EXAMPLE INPUT: {{"office_name": "077 - Plaza Egaña", "date": "2024/10/01", "corte_espera": 600}}

    Parameters
    ----------
    office_name : str
        Office to consider, for example "077 - Plaza Egaña"
    date: str
        Date to consider, for example "2024/10/01"
    corte_espera : int, optional
        Time in seconds for cutoff of service level, by default 600
    """

    query_rango_data = f"""
    DECLARE @Office NVARCHAR(50) = '{office_name}';
    DECLARE @SetDate DATE = '{date}';
    DECLARE @CorteEspera INT = {corte_espera};

    SELECT
        MIN(a.FH_Emi) AS "Primer registro",
        MAX(a.FH_Emi) AS "Ultimo registro",
        COUNT(*) AS "Total Registros"
    FROM
        Atenciones a
    JOIN
        Oficinas o ON o.IdOficina = a.IdOficina
    WHERE
        o.Oficina = @Office
    """

    query_global = f"""
    DECLARE @Office NVARCHAR(50) = '{office_name}';
    DECLARE @SetDate DATE = '{date}';
    DECLARE @CorteEspera INT = {corte_espera};

    -- SLA GLOBAL POR HORA
    SELECT
        FORMAT(DATETRUNC(hour, a.FH_Emi)   , 'yyyy-MM-dd (dddd) HH:mm', 'es-es') AS "Fecha - Hora",
        COUNT(*)                                                                                                          AS "Atenciones",
        FORMAT( COUNT(*) * 1.0 / COUNT(DISTINCT a.IdEsc) ,                                                         'N2')  AS "Promedio Atenciones por Escritorio",
        COUNT(DISTINCT a.IdEsc)                                                                                           AS "Escritorios Utilizados",
        COUNT(DISTINCT a.IdEje)                                                                                           AS "Ejecutivos Atendieron",
        SUM(CASE WHEN a.Perdido = 1 THEN 1 ELSE 0 END)                                                                    AS "Abandonos",
        FORMAT( 100 * AVG(CASE WHEN DATEDIFF(SECOND, a.FH_Emi, a.FH_AteIni) < @corteEspera THEN 1.0 ELSE 0.0 END) ,'N2')  AS "Nivel de Servicio (%)",
        FORMAT( AVG(DATEDIFF(SECOND, a.FH_Emi, a.FH_AteIni)) / 60.0                                               ,'N2')  AS "Tiempo de Espera Promedio (minutos)",
        FORMAT( AVG(DATEDIFF(SECOND, a.FH_AteIni, a.FH_AteFin)) / 60.0                                            ,'N2')  AS "Tiempo de Atencion Promedio (minutos)",
        FORMAT( 100 * AVG(CASE WHEN a.Perdido = 1 THEN 1.0 ELSE 0.0 END)                                          ,'N2')  AS "Tasa de Abandono (%)"
    FROM
        Atenciones a
    JOIN
        Oficinas o ON o.IdOficina = a.IdOficina
    JOIN
        Series s ON a.IdSerie = s.IdSerie AND a.IdOficina = s.IdOficina
    WHERE
            a.FH_Emi BETWEEN @SetDate AND DATEADD(day, 1, @SetDate)
        AND o.Oficina = @Office
    GROUP BY
            DATETRUNC(hour, a.FH_Emi)
    ORDER BY
            DATETRUNC(hour, a.FH_Emi) ASC;
    """

    query_series = f"""
    DECLARE @Office NVARCHAR(50) = '{office_name}';
    DECLARE @SetDate DATE = '{date}';
    DECLARE @CorteEspera INT = {corte_espera};

    -- SLA POR SERIE
    SELECT
        s.Serie                                                                                                           AS "Serie",
        FORMAT(DATETRUNC(hour, a.FH_Emi)   , 'yyyy-MM-dd (dddd) HH:mm', 'es-es')                                          AS "Fecha - Hora",
        COUNT(*)                                                                                                          AS "Atenciones",
        FORMAT( COUNT(*) * 1.0 / COUNT(DISTINCT a.IdEsc) ,                                                         'N2')  AS "Promedio Atenciones por Escritorio",
        COUNT(DISTINCT a.IdEsc)                                                                                           AS "Escritorios Utilizados",
        COUNT(DISTINCT a.IdEje)                                                                                           AS "Ejecutivos Atendieron",
        SUM(CASE WHEN a.Perdido = 1 THEN 1 ELSE 0 END)                                                                    AS "Abandonos",
        FORMAT( 100 * AVG(CASE WHEN DATEDIFF(SECOND, a.FH_Emi, a.FH_AteIni) < @corteEspera THEN 1.0 ELSE 0.0 END) ,'N2')  AS "Nivel de Servicio (%)",
        FORMAT( AVG(DATEDIFF(SECOND, a.FH_Emi, a.FH_AteIni)) / 60.0                                               ,'N2')  AS "Tiempo de Espera Promedio (minutos)",
        FORMAT( AVG(DATEDIFF(SECOND, a.FH_AteIni, a.FH_AteFin)) / 60.0                                            ,'N2')  AS "Tiempo de Atencion Promedio (minutos)",
        FORMAT( 100 * AVG(CASE WHEN a.Perdido = 1 THEN 1.0 ELSE 0.0 END)                                          ,'N2')  AS "Tasa de Abandono (%)"
    FROM
        Atenciones a
    JOIN
        Oficinas o ON o.IdOficina = a.IdOficina
    JOIN
        Series s ON a.IdSerie = s.IdSerie AND a.IdOficina = s.IdOficina
    WHERE
            a.FH_Emi BETWEEN @SetDate AND DATEADD(day, 1, @SetDate)
        AND o.Oficina = @Office
    GROUP BY
            s.Serie, DATETRUNC(hour, a.FH_Emi)
    ORDER BY
            s.Serie ASC, DATETRUNC(hour, a.FH_Emi) ASC;
    """

    with _engine.connect() as conn:
        records_range: pd.DataFrame = pd.read_sql_query(query_rango_data, conn)
        data_global: pd.DataFrame = pd.read_sql_query(query_global, conn)
        data_series: pd.DataFrame = pd.read_sql_query(query_series, conn)

    if data_global.empty and not records_range.empty:
        return f"Sin data disponible en el rango u oficina seleccionada. Data disponible para oficina: {records_range.to_markdown()}"
    elif data_global.empty:
        return "Sin data disponible en el rango u oficina seleccionada"

    return f"""
    # SLA por hora y serie para {office_name}. Fecha {date}
    {remove_extra_spaces(data_series.to_markdown(index=False))}

    # SLA global por hora para {office_name}. Fecha {date}
    {remove_extra_spaces(data_global.to_markdown(index=False))}
    """


tool_get_sla_by_hour_and_series: BaseTool = tool(get_sla_by_hour_and_series)
