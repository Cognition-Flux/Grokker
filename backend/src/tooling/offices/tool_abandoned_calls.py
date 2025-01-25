# %% -- ABANDONED CALLS
from datetime import datetime, timedelta

import pandas as pd
from langchain_core.tools import BaseTool, tool

from tooling.db_instance import _engine
from tooling.utilities import remove_extra_spaces


def get_abandoned_calls(
    office_names: list[str] = [
        "AFC Miraflores II",
        "AFC Providencia",
    ],
    start_date: str = datetime.now().strftime("%Y/%m/%d"),
    end_date: str = (datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d"),
):
    """
    Get the lost calls for a given time range.
    """
    parsed_office_names: str = ", ".join([f"'{office}'" for office in office_names])

    query_total = f"""
    SELECT
        o.[Oficina]                                                          AS "Oficina",
        COUNT(*)                                                             AS "Total Atenciones",
        SUM(CASE WHEN a.Perdido = 1 THEN 1 ELSE 0 END)                       AS "Abandonos",
        FORMAT( AVG(CASE WHEN a.Perdido = 1 THEN 100.0 ELSE 0.0 END) ,'N2')  AS "Tasa de Abandono (%)"
    FROM [dbo].[Atenciones] a
    JOIN [dbo].[Oficinas] o ON (o.[IdOficina] = a.[IdOficina])
    WHERE ([FH_Emi] BETWEEN '{start_date}' AND '{end_date}')
        AND o.[Oficina] IN ({parsed_office_names})
    GROUP BY o.[Oficina]
    ORDER BY  AVG(CASE WHEN a.Perdido = 1 THEN 100.0 ELSE 0.0 END) DESC
    """

    query = f"""
    SELECT
        o.[Oficina]                              AS "Oficina"
        ,[Serie]                                 AS "Serie"
        ,e.[Ejecutivo]                           AS "Ejecutivo"
        ,FORMAT([FH_Emi]   , 'yyyy-MM-dd HH:mm') AS "Fecha-Hora Emision"
        ,FORMAT([FH_Llama] , 'yyyy-MM-dd HH:mm') AS "Fecha-Hora Llamada"
        ,FORMAT([FH_AteIni], 'yyyy-MM-dd HH:mm') AS "Fecha-Hora Termino"
        ,FORMAT([TpoEsp] / 60.0, 'N2')           AS "Tiempo de Espera (minutos)"
    FROM [dbo].[Atenciones] a
    JOIN [dbo].[Series] s ON (s.[IdSerie] = a.[IdSerie]) AND (s.[IdOficina] = a.[IdOficina])
    JOIN [dbo].[Oficinas] o ON (o.[IdOficina] = a.[IdOficina])
    JOIN [dbo].[Ejecutivos] e ON (e.[IdEje] = a.[IdEje])
    WHERE ([Perdido] = 1 ) AND ([FH_Emi] BETWEEN '{start_date}' AND '{end_date}')
        AND o.[Oficina] IN ({parsed_office_names})
    ORDER BY [FH_Emi] DESC
    """

    # print(query)

    with _engine.connect() as conn:
        data_totales: pd.DataFrame = pd.read_sql_query(query_total, conn)
        data: pd.DataFrame = pd.read_sql_query(query, conn)

    if data_totales["Total Atenciones"].sum() == 0:
        return "Sin atenciones perdidas en el rango u oficinas seleccioandas"

    if data.empty:
        return "Sin data disponible en el rango u oficinas seleccioandas"

    # If the table is too long, return head 100 and a message indicating so
    if data.shape[0] > 50:
        markdown_table = data.head(15).to_markdown(index=False)
        aditional_text = "Registros truncados a 15 filas, la tabla es muy larga para mostrar. "
    else:
        markdown_table = data.to_markdown(index=False)
        aditional_text = ""

    if len(office_names) > 1:
        global_resume = f"""**Perdidas / Total**: { data_totales['Abandonos'].sum() } / { data_totales['Total Atenciones'].sum() } ({100 * data_totales['Abandonos'].sum() / data_totales['Total Atenciones'].sum()} %)
        """
    else:
        global_resume = ""

    response = f"""{global_resume}
Resumen total de atenciones perdidas:
{remove_extra_spaces(data_totales.to_markdown(index=False))}

{aditional_text}Detalle de atenciones perdidas:
{remove_extra_spaces(markdown_table)}
    """

    print(response)
    return response


def tool_get_abandoned_calls(
    office_names: list[str],
    start_date: str = datetime.now().strftime("%Y/%m/%d"),
    end_date: str = (datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d"),
) -> str:
    """
    Use this when the user asks for the details of abandoned or lost calls.
    This will return the abandoned calls for the selected time range.

    Parameters
    ----------
    office_names : list[str]
        List of office names to consider, for example [ "077 - Plaza Egaña", "199 - Melipilla" ]
    start_date : str, optional
        Begin date to consider, by default today's date
    end_date : str, optional
        End date to consider, by default tomorrows date
    """
    return get_abandoned_calls(
        office_names=office_names,
        start_date=start_date,
        end_date=end_date,
    )


tool_get_abandoned_calls: BaseTool = tool(tool_get_abandoned_calls)
