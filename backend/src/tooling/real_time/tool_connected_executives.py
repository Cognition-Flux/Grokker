from datetime import datetime

import pandas as pd
from langchain_core.tools import BaseTool, tool

from tooling.db_instance import _engine
from tooling.utilities import remove_extra_spaces


def get_connected_executives(
    office_names: list[str],
    include_series: bool = False,
):
    """
    Use this when the user asks for the connected executives, "ejecutivos conectados" in spanish.
    When the user does not specify any office, the function will return the connected executives for all offices.
    Use this when the user asks for "ejecutivos conectados"
    Your analysis should be the number of connected executives and count by state (Atencion, Iniciando, Pausa, Finalizado)

    Parameters
    ----------
    office_names : list[str]
        List of office names to consider
    include_series : bool, optional
        Include the series of each executive, default to False
    """

    if office_names:
        parsed_offices: str = ", ".join([f"'{office_name}'" for office_name in office_names])
        where_offices: str = f"WHERE o.Oficina IN ({parsed_offices})"
    else:
        where_offices: str = ""

    query_no_series = """
    WITH Diarios AS
        (
            SELECT TOP 1 WITH TIES
                IdOficina
                , IdEsc
                , (CASE WHEN Evento = 'F' THEN NULL ELSE IdEje END) AS IdEje
                , FH_Eve
                , (CASE Evento
                    WHEN 'F' THEN 'Finalizado'
                    WHEN 'A' THEN 'Atencion'
                    WHEN 'I' THEN 'Iniciando'
                    WHEN 'P' THEN 'Pausa'
                END) AS Evento
                , IdPausa
            FROM EjeEstado_D
            WHERE FH_Eve > {last_date_str}
            ORDER BY ROW_NUMBER() OVER (PARTITION BY IdOficina, IdEsc ORDER BY FH_Eve desc)
        )
    SELECT
        Oficina
        , Ejecutivo
        , IdEsc AS "Escritorio"
        , Evento AS "Estado"
    FROM Diarios d
        JOIN Ejecutivos e ON d.IdEje = e.IdEje
        JOIN Oficinas o ON d.IdOficina = o.IdOficina
    {where_offices}
    ORDER BY Oficina, Escritorio
    """

    query_with_series = """
    WITH Diarios AS
        (
            SELECT TOP 1 WITH TIES
                IdOficina
                , IdEsc
                , (CASE WHEN Evento = 'F' THEN NULL ELSE IdEje END) AS IdEje
                , FH_Eve
                , (CASE Evento
                    WHEN 'F' THEN 'Finalizado'
                    WHEN 'A' THEN 'Atencion'
                    WHEN 'I' THEN 'Iniciando'
                    WHEN 'P' THEN 'Pausa'
                END) AS Evento
                , IdPausa
            FROM EjeEstado
            WHERE FH_Eve > CAST((SELECT MAX(FH_Emi) FROM Atenciones) AS DATE)
            --CAST(getdate() as date)
            ORDER BY ROW_NUMBER() OVER (PARTITION BY IdOficina, IdEsc ORDER BY FH_Eve desc)
        )
    SELECT
        o.Oficina
        , e.Ejecutivo
        , d.IdEsc AS "Escritorio"
        , d.Evento AS "Estado"
        , STRING_AGG(s.Serie, ', ') AS "Series"
    FROM Diarios d
        JOIN Ejecutivos e ON d.IdEje = e.IdEje
        JOIN Oficinas o ON d.IdOficina = o.IdOficina
        JOIN EscritorioSerie es ON es.IdEsc = d.IdEsc AND es.IdOficina = d.IdOficina
        JOIN Series s ON s.IdSerie = es.IdSerie AND s.IdOficina = es.IdOficina
    {where_offices}
    GROUP BY o.Oficina, d.IdEsc, e.Ejecutivo, d.Evento
    ORDER BY Oficina, Escritorio
"""

    with _engine.connect() as conn:
        query = query_with_series if include_series else query_no_series

        data: pd.DataFrame = pd.read_sql_query(
            query.format(where_offices=where_offices, last_date_str="CAST(getdate() as date)"),
            conn,
        )

        data_last_available: pd.DataFrame = pd.read_sql_query(
            query.format(
                where_offices=where_offices,
                last_date_str="CAST((SELECT MAX(FH_Emi) FROM Atenciones) AS DATE)",
            ),
            conn,
        )

    if data_last_available.empty:
        return "Sin ejecutivos conectados en oficinas seleccioandas"

    if not data.empty:
        return f"""<description>Ejecutivos conectados en {'todas las oficinas' if not office_names else office_names}: {data.shape[0]}</description>
<data>
{remove_extra_spaces(data.to_markdown(index=False))}
</data>
"""

    else:
        return f"""<error>Data desactualizada. Sin data disponible al dia de hoy ({datetime.now():%Y/%m/%d}). Utilizando ultimo registro disponible</error>
<description>Data desactualizada. Ejecutivos conectados en {'todas las oficinas' if not office_names else office_names}: {data.shape[0]}</description>
<data>
{remove_extra_spaces(data_last_available.to_markdown(index=False))}
</data>
"""


tool_get_connected_executives: BaseTool = tool(get_connected_executives)
