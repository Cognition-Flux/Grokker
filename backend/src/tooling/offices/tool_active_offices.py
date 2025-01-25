from datetime import date

import pandas as pd
from langchain_core.tools import BaseTool, tool

from tooling.db_instance import _engine
from tooling.utilities import remove_extra_spaces


def get_active_offices(
    start_date: date,
    end_date: date,
):
    """
    Use this when the user asks for the active offices, "oficinas activas" in spanish.
    Do not ask the user for confirmation, just use the default values unless the user specifies them. In this case, refer to "los ultimos tres meses"
    Return the complete table including the total number of active offices and "DESACRUALIZADA" or "ONLINE" for each office.
    Note the ONLINE and DESACTUALIZADA mean the last update of the office is within 48 hours of the last update to the database;
    if the complete database is outdated you should alert

    Parameters
    ----------
    start_date : str, optional
        Begin date to consider, default to 3 months ago
    end_date : str, optional
        End date to consider, default to today's date
    """

    query = f"""
DECLARE @ReasonableUpdate DATETIME = DATEADD( HOUR, -48, (SELECT MAX([FH_Emi]) FROM Atenciones))

SELECT [Oficina]
      , FORMAT( MIN([FH_Emi]) , 'yyyy-MM-dd') + ' - ' + FORMAT( MAX([FH_Emi]) , 'yyyy-MM-dd')  AS "Rango de datos disponibles"
      , COUNT(*) AS "Atenciones totales"
      , SUM( CASE WHEN a.FH_Emi BETWEEN '{start_date}' AND '{end_date}' THEN 1 ELSE 0 END) AS 'Atenciones en rango seleccionado'
      , (CASE WHEN (MAX([FH_Emi]) > @ReasonableUpdate) THEN 'ONLINE' ELSE 'DESACTUALIZADA' END) AS 'Estado'
FROM [Atenciones] a
JOIN [Oficinas] o ON o.IdOficina = a.IdOficina
WHERE o.fDel = 0
GROUP BY o.[Oficina]
ORDER BY Estado, 'Atenciones en rango seleccionado' DESC
    """

    with _engine.connect() as conn:
        data: pd.DataFrame = pd.read_sql_query(query, conn)

    if data.empty:
        return "Sin data disponible en el rango u oficinas seleccioandas"

    response = f"""Oficinas activas en el rango de fechas seleccionado: {data.shape[0]}

{remove_extra_spaces(data.to_markdown())}
    """

    print(response)
    return response


tool_get_active_offices: BaseTool = tool(get_active_offices)
