"""
Contains the functions per themselves, without the LLM wrappers
"""

# %%
import json
import os
from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd
import requests
from typing_extensions import Literal

from tooling.db_instance import _engine
from tooling.utilities import remove_extra_spaces


# %%
def top_executives_report(
    office_names: list[str] = [
        "AFC Miraflores II",
        "AFC Providencia",
    ],
    start_date: str = datetime.now().strftime("%Y/%m/%d"),
    end_date: str = (datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d"),
    top_ranking: int = 5,
):
    query = f"""
    DECLARE @maxDate DATE = '{end_date}';
    DECLARE @startDate DATE = '{start_date}';

    DECLARE @OfficeList TABLE (OfficeName NVARCHAR(255));
    INSERT INTO @OfficeList (OfficeName)
    VALUES {', '.join([f"('{office}')" for office in office_names])};

    WITH ExecutivePerformance AS (
        SELECT
            o.Oficina,
            e.Ejecutivo,
            COUNT(*) * 1.0 / COUNT(DISTINCT CAST(a.FH_Emi AS DATE)) AS PromedioAtencionesdiarias,
            AVG(DATEDIFF(SECOND, a.FH_AteIni, a.FH_AteFin)) / 60.0 AS TiempoPromedioAtencionMinutos,
            (COUNT(*) * 1.0 / COUNT(DISTINCT CAST(a.FH_Emi AS DATE))) /
            NULLIF(AVG(DATEDIFF(SECOND, a.FH_AteIni, a.FH_AteFin)) / 60.0, 0) AS Rendimiento,
            ROW_NUMBER() OVER (PARTITION BY o.Oficina ORDER BY (COUNT(*) * 1.0 / COUNT(DISTINCT CAST(a.FH_Emi AS DATE))) /
            NULLIF(AVG(DATEDIFF(MINUTE, a.FH_AteIni, a.FH_AteFin)), 0) DESC) AS Ranking,
            STUFF((SELECT DISTINCT ', ' + s.Serie
                FROM Atenciones a2
                JOIN Series s ON s.IdSerie = a2.IdSerie
                WHERE a2.IdEje = a.IdEje
                        AND a2.FH_Emi BETWEEN @startDate AND @maxDate
                        AND a2.FH_AteIni IS NOT NULL AND a2.FH_AteFin IS NOT NULL
                FOR XML PATH('')), 1, 2, '') AS Series,
            @startDate AS StartDate,
            @maxDate AS EndDate
        FROM
            Atenciones a
            JOIN Oficinas o ON a.IdOficina = o.IdOficina
            JOIN Ejecutivos e ON a.IdEje = e.IdEje
        WHERE
            o.Oficina IN (SELECT OfficeName FROM @OfficeList)
            AND a.FH_AteIni IS NOT NULL
            AND a.FH_AteFin IS NOT NULL
            AND a.FH_Emi BETWEEN @startDate AND @maxDate
        GROUP BY
            o.Oficina, e.Ejecutivo, a.IdEje
    )
    SELECT
        Oficina,
        Ejecutivo,
        Series,
        PromedioAtencionesdiarias,
        TiempoPromedioAtencionMinutos,
        Rendimiento,
        Ranking,
        FORMAT(StartDate, 'dd/MM/yyyy') + ' - ' + FORMAT(EndDate, 'dd/MM/yyyy') AS DateRange
    FROM
        ExecutivePerformance
    WHERE
        Ranking <= {top_ranking}
    ORDER BY
        Oficina,
        Ranking;
    """

    with _engine.connect() as conn:
        data: pd.DataFrame = pd.read_sql_query(query, conn)

    if data.empty:
        return "Sin data disponible en el rango u oficinas seleccioandas"

    markdown_table = data.to_markdown(index=False)

    return f"""
    Ranking de ejecutivos correctamente completado:
    {remove_extra_spaces(markdown_table)}"""


# %%
def executive_daily_performance(
    executive_name: str = "FABIOLA ANDREA GUERRERO GONZALEZ",
    office_name: str = "AFC Miraflores II",
    start_date: str = datetime.now().strftime("%Y/%m/%d"),
    end_date: str = (datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d"),
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
        return "Sin data disponible en el rango u oficinas seleccioandas"

    markdown_table = data.to_markdown(index=False)

    return f"""
        # Reporte de rendimiento diario para {executive_name}

        Oficina: {office_name}, Periodo: {start_date} - {end_date}

        Desempeño diario de ejecutivo correctamente completado:
        {markdown_table}
        """


def get_series_stats_by_granularity(
    office_name: str,
    series: list[str] | None,
    start_date: str = datetime.now().strftime("%Y/%m/%d"),
    end_date: str = (datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d"),
    granularity: Literal["1h", "2h", "4h", "8h", "1d", "1w"] = "2h",
) -> str:
    """
    Get the number of calls for a list of series, for a given time range, for an office in a given time granularity.
    """
    ...
    # FUCK THIS SQL GRANULARITY


@lru_cache(maxsize=10)
def get_office_names(start_date: str = "2024/01/01", end_date: str = "2025/01/01") -> str:
    """
    Get the list of available office names.
    Accepts a cutoff to select only the offices that have had activity after the given date.
    """
    query = f"""
    SELECT
          o.[Oficina]     AS "Oficina"
        , COUNT(*)        AS "Atenciones"
        , FORMAT( MIN(a.FH_Emi), 'yyyy-MM-dd HH:mm') AS "Primer Registro"
        , FORMAT( MAX(a.FH_Emi), 'yyyy-MM-dd HH:mm') AS "Ultimo Registro"
        , FORMAT( AVG(DATEDIFF(SECOND, a.FH_Emi, a.FH_AteIni)) / 60.0    ,'N2')  AS "Tiempo de Espera Promedio (minutos)"
        , FORMAT( AVG(DATEDIFF(SECOND, a.FH_AteIni, a.FH_AteFin)) / 60.0 ,'N2')  AS "Tiempo de Atencion Promedio (minutos)"
        , SUM(CASE WHEN a.Perdido = 1 THEN 1 ELSE 0 END)                         AS "Abandonos"
        , FORMAT( AVG(CASE WHEN a.Perdido = 1 THEN 100.0 ELSE 0.0 END) ,'N2')    AS "Tasa de Abandono (%)"
    FROM [dbo].[Atenciones] a
    JOIN [dbo].[Series] s ON (s.[IdSerie] = a.[IdSerie]) AND (s.[IdOficina] = a.[IdOficina])
    JOIN [dbo].[Oficinas] o ON (o.[IdOficina] = a.[IdOficina])
    JOIN [dbo].[Ejecutivos] e ON (e.[IdEje] = a.[IdEje])
    WHERE a.[FH_Emi] BETWEEN '{start_date}' AND '{end_date}' AND (o.[fDel] = 0)
    GROUP BY o.[Oficina]
    ORDER BY COUNT(*) DESC
    """

    with _engine.connect() as conn:
        data: pd.DataFrame = pd.read_sql_query(query, conn)

    if data.empty:
        return "<error>Sin data disponible en el rango u oficinas seleccioandas</error>"

    return f"""<desciption>Lista gobal de oficinas, atenciones totales desde {start_date} hasta {end_date}.
Primer y ultimo registro representan el rango de data disponible para el sistema.
Total de atenciones: {data['Atenciones'].sum()}. Total de oficinas: {data.shape[0]}.
</desciption>
<data>
{remove_extra_spaces(data.to_markdown(index=False))}
</data>"""


@lru_cache(maxsize=10)
def get_all_executives(
    start_date: str = "2024/01/01",
    end_date: str = "2025/01/01",
) -> str:
    """Generate a report with the name of all executives"""
    query = f"""
    SELECT DISTINCT
        o.Oficina, e.Ejecutivo -- HEADERS
        , FORMAT( MIN(a.FH_Emi), 'yyyy-MM-dd HH:mm') AS "Primer Registro"
        , FORMAT( MAX(a.FH_Emi), 'yyyy-MM-dd HH:mm') AS "Ultimo Registro"
        , COUNT(*) AS "Atenciones"
        -- , FORMAT( AVG(DATEDIFF(SECOND, a.FH_Emi, a.FH_AteIni)) / 60.0    ,'N2')  AS "Tiempo de Espera Promedio (minutos)"
        , FORMAT( AVG(DATEDIFF(SECOND, a.FH_AteIni, a.FH_AteFin)) / 60.0 ,'N2')  AS "Tiempo de Atencion Promedio (minutos)"
    FROM Ejecutivos e
    JOIN Atenciones a ON a.IdEje = e.IdEje
    JOIN Oficinas o ON o.IdOficina = a.IdOficina
    WHERE a.[FH_Emi] BETWEEN '{start_date}' AND '{end_date}'
      AND (o.[fDel] = 0) AND (e.[fDel] = 0)
    GROUP BY o.Oficina, e.Ejecutivo
    ORDER BY o.Oficina ASC, COUNT(*) DESC
    """

    with _engine.connect() as conn:
        data: pd.DataFrame = pd.read_sql_query(query, conn)

    if data.empty:
        return "<error>Sin data disponible en la rango u oficinas seleccioandas</error>"

    return f"""<desciption>Resumen de ejecutivos desde {start_date} hasta {end_date}, atenciones totales, y tiempos promedios de atencion. . Primer y ultimo registro representan el rango de data disponible para el sistema.</desciption>
<data>
{remove_extra_spaces(data.to_markdown(index=False))}
</data>"""


def get_just_office_names(cuttof_date: datetime = None, cuttof_number: int = 500) -> list[str]:
    """Not designed to be used with a chain per-se"""
    query = """
    SELECT
        o.[Oficina] AS "Oficina"
    FROM [dbo].[Oficinas] o
    """

    if cuttof_date:
        query = f"""
            SELECT
                o.[Oficina]
            FROM [dbo].[Atenciones] a
            JOIN [dbo].[Oficinas] o ON a.IdOficina = o.IdOficina
            WHERE (a.FH_Emi > '{cuttof_date}') AND (o.[fDel] = 0)
            GROUP BY o.[Oficina]
            HAVING (COUNT(*) > {cuttof_number})
            ORDER BY COUNT(*) DESC;
        """

    with _engine.connect() as conn:
        data: pd.DataFrame = pd.read_sql_query(query, conn)

    if data.empty:
        return []

    return data["Oficina"].to_list()


# reporte_historico()
# # %%
def get_response(url):
    payload = {}
    headers = {"APIKEY": os.environ["TTP_API_KEY"]}

    response = requests.request("GET", url, headers=headers, data=payload)

    return json.loads(response.text)


def ResumenDeOficinasOnline(target_oficinas: list[str]) -> list[dict]:
    domain = "https://adminv40.totalpack.cl/reporte/api/v1"
    servicio = "/Reportes/ResumenDeOficinas"

    Slug = "ttp_afc"
    idUsuario = "1"

    oficinas = get_response(f"{domain}{servicio}?Slug={Slug}&idUsuario={idUsuario}")

    global_series = [of for of in oficinas["data"] if of["serie"] == ""]
    filtered_data = [entry for entry in global_series if entry["oficina"] in target_oficinas]

    return filtered_data


# %%
