# %%
import json
import logging
from datetime import datetime, timedelta
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy import text

from ttp_agentic.db_instance import _engine
from ttp_agentic.tools.utilities import (
    add_docstring,
    get_documentation,
    parse_input,
    remove_extra_spaces,
    retry_decorator,
)

load_dotenv(override=True)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_data_consistency(
    global_stats: pd.DataFrame,
    data_series: pd.DataFrame,  # data_executives: pd.DataFrame
) -> Tuple[bool, List[str]]:
    """
    Validates data consistency across different tables.

    Args:
        global_stats (pd.DataFrame): DataFrame containing global statistics.
        data_series (pd.DataFrame): DataFrame containing statistics per series.
        data_executives (pd.DataFrame): DataFrame containing statistics per executive.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list of error messages)
    """
    errors = []

    # Validate total abandonos
    total_abandonos_global = global_stats["Total Abandonos"].iloc[0]
    total_abandonos_series = data_series["Abandonos"].sum()
    # total_abandonos_executives = data_executives["Abandonos"].sum()

    if total_abandonos_global != total_abandonos_series:
        errors.append(
            f"Inconsistencia en abandonos: Total global ({total_abandonos_global}) "
            f"≠ Suma por series ({total_abandonos_series})"
        )

    # if total_abandonos_global != total_abandonos_executives:
    #     errors.append(
    #         f"Inconsistencia en abandonos: Total global ({total_abandonos_global}) "
    #         f"≠ Suma por ejecutivos ({total_abandonos_executives})"
    #     )

    # Validate total atenciones
    total_atenciones_global = global_stats["Total Atenciones"].iloc[0]
    total_atenciones_series = data_series["Atenciones"].sum()
    # total_atenciones_executives = data_executives["Atenciones"].sum()

    if total_atenciones_global != total_atenciones_series:
        errors.append(
            f"Inconsistencia en atenciones: Total global ({total_atenciones_global}) "
            f"≠ Suma por series ({total_atenciones_series})"
        )

    # if total_atenciones_global != total_atenciones_executives:
    #     errors.append(
    #         f"Inconsistencia en atenciones: Total global ({total_atenciones_global}) "
    #         f"≠ Suma por ejecutivos ({total_atenciones_executives})"
    #     )

    is_valid = len(errors) == 0
    return is_valid, errors


def compute_global_statistics(
    office_data: pd.DataFrame, corte_espera: int
) -> pd.DataFrame:
    """
    Computes global statistics for the office.

    Args:
        office_data (pd.DataFrame): DataFrame containing office data.
        corte_espera (int): Threshold in seconds for service level calculation.

    Returns:
        pd.DataFrame: DataFrame containing global statistics.
    """
    total_atenciones = len(office_data)
    tiempo_medio_espera = (
        office_data["TpoEsp"].mean() / 60.0 if total_atenciones > 0 else 0
    )
    total_abandonos = office_data["Perdido"].sum()
    porcentaje_abandono = (
        (total_abandonos / total_atenciones) * 100 if total_atenciones > 0 else 0
    )
    dias_con_atenciones = office_data["FH_Emi"].dt.date.nunique()
    promedio_atenciones_diarias = (
        total_atenciones / dias_con_atenciones if dias_con_atenciones > 0 else 0
    )

    # Nivel de Servicio
    office_data["Tiempo_Espera"] = (
        office_data["FH_AteIni"] - office_data["FH_Emi"]
    ).dt.total_seconds()
    nivel_servicio = (
        (
            office_data[
                (office_data["Perdido"] == 0)
                & (office_data["Tiempo_Espera"] < corte_espera)
            ].shape[0]
            / total_atenciones
            * 100
        )
        if total_atenciones > 0
        else 0
    )

    total_series = office_data["Serie"].nunique()
    total_ejecutivos = office_data["Ejecutivo"].nunique()
    total_escritorios = office_data["IdEsc"].nunique()

    global_stats = pd.DataFrame(
        {
            "Total Atenciones": [total_atenciones],
            "Tiempo Medio de Espera Global (minutos)": [f"{tiempo_medio_espera:.2f}"],
            "Total Abandonos": [total_abandonos],
            "Porcentaje Abandono Global (%)": [f"{porcentaje_abandono:.2f}"],
            "Días con Atenciones": [dias_con_atenciones],
            "Promedio Atenciones Diarias": [f"{promedio_atenciones_diarias:.2f}"],
            "Nivel de Servicio (%)": [f"{nivel_servicio:.2f}"],
            "Total Series": [total_series],
            "Total Ejecutivos": [total_ejecutivos],
            "Total Escritorios": [total_escritorios],
        }
    )

    return global_stats


def compute_series_statistics(
    office_data: pd.DataFrame, total_atenciones: int
) -> pd.DataFrame:
    """
    Computes statistics per series.

    Args:
        office_data (pd.DataFrame): DataFrame containing office data.
        total_atenciones (int): Total number of attendances.

    Returns:
        pd.DataFrame: DataFrame containing series statistics.
    """
    data_series = (
        office_data.groupby("Serie")
        .agg(
            Atenciones=("Serie", "count"),
            Porcentaje_del_Total=("Serie", lambda x: (len(x) / total_atenciones) * 100),
            Ultima_atencion=("FH_Emi", "max"),
            Tiempo_Medio_de_Atencion_minutos=("TpoAte", lambda x: x.mean() / 60.0),
            Tiempo_Medio_de_Espera_minutos=("TpoEsp", lambda x: x.mean() / 60.0),
            Abandonos=("Perdido", "sum"),
        )
        .reset_index()
    )

    # Calculate percentages and format
    data_series["Porcentaje del Total (%)"] = data_series["Porcentaje_del_Total"].round(
        2
    )
    data_series["Abandonos (%)"] = (
        data_series["Abandonos"] / data_series["Atenciones"] * 100
    ).round(2)

    # Select columns to display
    data_series = data_series[
        [
            "Serie",
            "Atenciones",
            "Porcentaje del Total (%)",
            "Ultima_atencion",
            "Tiempo_Medio_de_Atencion_minutos",
            "Tiempo_Medio_de_Espera_minutos",
            "Abandonos",
            "Abandonos (%)",
        ]
    ]

    return data_series


def compute_executive_statistics(
    office_data: pd.DataFrame, total_atenciones: int
) -> pd.DataFrame:
    """
    Computes statistics per executive.

    Args:
        office_data (pd.DataFrame): DataFrame containing office data.
        total_atenciones (int): Total number of attendances.

    Returns:
        pd.DataFrame: DataFrame containing executive statistics.
    """
    data_executives = (
        office_data.groupby("Ejecutivo")
        .agg(
            Atenciones=("Ejecutivo", "count"),
            Porcentaje_del_Total=(
                "Ejecutivo",
                lambda x: (len(x) / total_atenciones) * 100,
            ),
            Ultima_atencion=("FH_Emi", "max"),
            Primera_atencion=("FH_Emi", "min"),
            Tiempo_Medio_de_Atencion_minutos=("TpoAte", lambda x: x.mean() / 60.0),
            Dias_con_Atenciones=("FH_Emi", lambda x: x.dt.date.nunique()),
            Abandonos=("Perdido", "sum"),
        )
        .reset_index()
    )

    # Calculate percentages and averages
    data_executives["Porcentaje del Total (%)"] = data_executives[
        "Porcentaje_del_Total"
    ].round(2)
    data_executives["Promedio Atenciones Diarias"] = (
        data_executives["Atenciones"] / data_executives["Dias_con_Atenciones"]
    ).round(2)

    # Select columns to display
    data_executives = data_executives[
        [
            "Ejecutivo",
            "Atenciones",
            "Porcentaje del Total (%)",
            "Ultima_atencion",
            "Primera_atencion",
            "Tiempo_Medio_de_Atencion_minutos",
            "Promedio Atenciones Diarias",
            "Abandonos",
        ]
    ]

    # Sort executives by Promedio Atenciones Diarias
    data_executives = data_executives.sort_values(
        by="Promedio Atenciones Diarias", ascending=False
    ).reset_index(drop=True)

    return data_executives


def compute_daily_statistics(
    office_data: pd.DataFrame, office_name: str, corte_espera: int
) -> pd.DataFrame:
    """
    Computes daily statistics for the office.

    Args:
        office_data (pd.DataFrame): DataFrame containing office data.
        office_name (str): Name of the office.
        corte_espera (int): Threshold in seconds for service level calculation.

    Returns:
        pd.DataFrame: DataFrame containing daily statistics.
    """
    # Mapping of weekday numbers to Spanish day names
    weekday_map = {
        0: "Lunes",
        1: "Martes",
        2: "Miércoles",
        3: "Jueves",
        4: "Viernes",
        5: "Sábado",
        6: "Domingo",
    }

    # Add 'Fecha' and 'Dia' columns
    office_data = office_data.copy()
    office_data["Fecha"] = office_data["FH_Emi"].dt.date
    office_data["Dia"] = office_data["FH_Emi"].dt.weekday.map(weekday_map)

    # Compute daily statistics
    daily_stats = (
        office_data.groupby(["Fecha", "Dia"])
        .agg(
            Atenciones_Totales=("FH_Emi", "count"),
            Escritorios_Utilizados=("IdEsc", pd.Series.nunique),
            Ejecutivos_Atendieron=("IdEje", pd.Series.nunique),
            Abandonos=("Perdido", "sum"),
            Tiempo_Espera_Promedio=("TpoEsp", lambda x: x.mean() / 60.0),
            Tiempo_Atencion_Promedio=("TpoAte", lambda x: x.mean() / 60.0),
        )
        .reset_index()
    )

    # Compute additional metrics
    daily_stats["Promedio_Atenciones_por_Escritorio"] = (
        daily_stats["Atenciones_Totales"] / daily_stats["Escritorios_Utilizados"]
    ).round(2)

    # Compute 'Nivel de Servicio (%)'
    office_data["EsNivelServicio"] = (
        (office_data["Perdido"] == 0) & (office_data["Tiempo_Espera"] < corte_espera)
    ).astype(int)
    nivel_servicio_series = (
        office_data.groupby(["Fecha", "Dia"])["EsNivelServicio"].mean().reset_index()
    )
    nivel_servicio_series["Nivel de Servicio (%)"] = (
        nivel_servicio_series["EsNivelServicio"] * 100
    )
    nivel_servicio_series.drop(columns=["EsNivelServicio"], inplace=True)

    # Merge with daily_stats
    daily_stats = pd.merge(daily_stats, nivel_servicio_series, on=["Fecha", "Dia"])

    # Compute 'Tasa de Abandono (%)'
    daily_stats["Tasa de Abandono (%)"] = (
        daily_stats["Abandonos"] / daily_stats["Atenciones_Totales"] * 100
    ).round(2)

    # Add 'Oficina' column
    daily_stats["Oficina"] = office_name

    # Reorder columns
    daily_stats = daily_stats[
        [
            "Oficina",
            "Dia",
            "Fecha",
            "Atenciones_Totales",
            "Promedio_Atenciones_por_Escritorio",
            "Escritorios_Utilizados",
            "Ejecutivos_Atendieron",
            "Abandonos",
            "Nivel de Servicio (%)",
            "Tiempo_Espera_Promedio",
            "Tiempo_Atencion_Promedio",
            "Tasa de Abandono (%)",
        ]
    ]

    # Rename columns for display
    daily_stats.columns = [
        "Oficina",
        "Día",
        "Fecha",
        "Atenciones Totales",
        "Promedio Atenciones por Escritorio",
        "Escritorios Utilizados",
        "Ejecutivos que Atendieron",
        "Abandonos",
        "Nivel de Servicio (%)",
        "Tiempo de Espera Promedio (minutos)",
        "Tiempo de Atención Promedio (minutos)",
        "Tasa de Abandono (%)",
    ]

    # Sort by Fecha
    daily_stats = daily_stats.sort_values(by="Fecha")

    # Format numeric columns
    numeric_cols = [
        "Atenciones Totales",
        "Promedio Atenciones por Escritorio",
        "Escritorios Utilizados",
        "Ejecutivos que Atendieron",
        "Abandonos",
        "Nivel de Servicio (%)",
        "Tiempo de Espera Promedio (minutos)",
        "Tiempo de Atención Promedio (minutos)",
        "Tasa de Abandono (%)",
    ]
    daily_stats[numeric_cols] = (
        daily_stats[numeric_cols].apply(pd.to_numeric, errors="coerce").round(2)
    )

    # Convert 'Fecha' column to string for better display
    daily_stats["Fecha"] = daily_stats["Fecha"].astype(str)

    return daily_stats


def get_office_stats(
    data: pd.DataFrame,
    office_name: str,
    corte_espera: int,
    start_date: datetime,
    end_date: datetime,
) -> str:
    """
    Get office statistics for a specified date range.

    Args:
        data (pd.DataFrame): The data for all offices.
        office_name (str): Name of the office to process.
        corte_espera (int): Threshold in seconds for service level calculation.
        start_date (datetime): Start date for the data.
        end_date (datetime): End date for the data.

    Returns:
        str: Formatted markdown report with office statistics.
    """
    # Filter data for the specific office and date range
    office_data = data[
        (data["Oficina"] == office_name)
        & (data["FH_Emi"] >= start_date)
        & (data["FH_Emi"] <= end_date)
    ]

    if office_data.empty:
        return f"Sin data disponible en el rango u oficina seleccionada: {office_name}"

    # Create a copy to avoid SettingWithCopyWarning
    office_data = office_data.copy()

    # Convert datetime columns to ensure correct dtype
    datetime_columns = ["FH_Emi", "FH_AteIni", "FH_AteFin"]
    for col in datetime_columns:
        office_data[col] = pd.to_datetime(office_data[col], errors="coerce")

    # Compute 'Tiempo_Espera'
    office_data["Tiempo_Espera"] = (
        office_data["FH_AteIni"] - office_data["FH_Emi"]
    ).dt.total_seconds()

    # Compute Global Statistics
    global_stats = compute_global_statistics(office_data, corte_espera)
    total_atenciones = int(global_stats["Total Atenciones"].iloc[0])

    # Compute Statistics per Series
    data_series = compute_series_statistics(office_data, total_atenciones)

    # # Compute Statistics per Executive
    # data_executives = compute_executive_statistics(office_data, total_atenciones)
    # executive_names = data_executives["Ejecutivo"].unique()

    # Validate Data Consistency
    is_valid, errors = validate_data_consistency(
        global_stats,
        data_series,  # data_executives
    )

    # Compute Daily Statistics
    daily_stats = compute_daily_statistics(office_data, office_name, corte_espera)

    # Convert DataFrames to Markdown
    global_stats_table = global_stats.to_markdown(index=False)
    markdown_table_series = data_series.to_markdown(index=False)
    # markdown_table_executives = data_executives.to_markdown(index=False)
    markdown_table_daily = daily_stats.to_markdown(index=False)

    # Build the report string
    corte_espera_min = corte_espera / 60.0
    start_date_str = start_date.strftime("%d/%m/%Y")
    end_date_str = end_date.strftime("%d/%m/%Y")

    report = f"""
### --------------------------------reporte para extraer información específica que requiere el usuario (solo extraer lo necesario, no mostrar estas tablas completas)----------------------------
# Reporte para la oficina: {office_name}
El período analizado es desde {start_date_str} hasta {end_date_str}
* Nivel de servicio (o SLA) se define como el porcentaje de clientes que esperaron menos de un máximo (umbral) de tiempo de espera.
* Tiempo máximo de espera para Nivel de Servicio (o SLA): {corte_espera_min:.1f} minutos.
"""

    if not is_valid:
        report += "\n## ⚠️ Advertencias de Validación\n"
        for error in errors:
            report += f"* {error}\n"

    # Add the list of executive names
    # report += "\n## Lista de Ejecutivos\n"
    # for name in executive_names:
    #     report += f"* {name}\n"

    report += f"""
## *Resumen de la sucursal/oficina* (usar la siguiente tabla cuando usuario solicita información general, resumen, o metricas/indicadores resumidos) solo extraer lo necesario.
{remove_extra_spaces(global_stats_table)}

## Indicadores por serie. solo extraer lo necesario
{remove_extra_spaces(markdown_table_series)}

## Desempeño diario de la sucursal/oficina. solo extraer lo necesario
{remove_extra_spaces(markdown_table_daily)}
### ---------------------------------------------------------------
"""

    return report


@retry_decorator(max_retries=5, delay=1.0)
def reporte_general_de_oficinas(
    office_names: List[str],
    days_back: Optional[int] = None,
    corte_espera: int = 600,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Generate reports for multiple offices.

    Args:
        office_names (List[str]): List of office names.
        days_back (Optional[int]): Number of days to look back. If None, custom date range is used.
        corte_espera (int): Threshold in seconds for service level calculation.
        start_date (Optional[str]): Start date in "DD/MM/YYYY" format (used when days_back is None).
        end_date (Optional[str]): End date in "DD/MM/YYYY" format (used when days_back is None).

    Returns:
        str: Combined reports for all offices.
    """
    try:
        if days_back is None:
            if start_date is None or end_date is None:
                return (
                    "When days_back is None, start_date and end_date must be provided."
                )
            # Parse start_date and end_date
            try:
                start_date_parsed = datetime.strptime(start_date, "%d/%m/%Y")
                end_date_parsed = datetime.strptime(end_date, "%d/%m/%Y") + timedelta(
                    days=1
                )
            except ValueError:
                return "Invalid date format. Please use DD/MM/YYYY."
            if end_date_parsed <= start_date_parsed:
                return "end_date must be greater than start_date."
            # Set the date range for data fetching
            earliest_start_date = start_date_parsed
            latest_end_date = end_date_parsed
        else:
            # Fetch the last valid register date per office
            query_last_valid_register_date = text(
                """
                SELECT o.[Oficina], MAX(a.[FH_Emi]) as last_valid_register_date
                FROM [dbo].[Atenciones] a
                JOIN [dbo].[Oficinas] o ON o.[IdOficina] = a.[IdOficina]
                WHERE o.[Oficina] IN :office_names
                GROUP BY o.[Oficina]
            """
            )
            params = {"office_names": tuple(office_names)}
            with _engine.connect() as conn:
                last_valid_dates_df = pd.read_sql_query(
                    query_last_valid_register_date, conn, params=params
                )
                if last_valid_dates_df.empty:
                    return "No data available."

                # Calculate earliest start date and latest end date across all offices
                last_valid_dates_df["start_date"] = last_valid_dates_df[
                    "last_valid_register_date"
                ] - pd.to_timedelta(days_back - 1, unit="d")
                earliest_start_date = last_valid_dates_df["start_date"].min()
                latest_end_date = last_valid_dates_df["last_valid_register_date"].max()

        # Fetch data for all offices between earliest_start_date and latest_end_date
        query_data = text(
            """
            SELECT
                a.*,
                s.[Serie],
                COALESCE(e.[Ejecutivo], 'No Asignado') AS [Ejecutivo],
                o.[Oficina]
            FROM [dbo].[Atenciones] a
            LEFT JOIN [dbo].[Series] s ON s.[IdSerie] = a.[IdSerie] AND s.[IdOficina] = a.[IdOficina]
            LEFT JOIN [dbo].[Ejecutivos] e ON e.[IdEje] = a.[IdEje]
            JOIN [dbo].[Oficinas] o ON o.[IdOficina] = a.[IdOficina]
            WHERE o.[Oficina] IN :office_names
            AND a.[FH_Emi] BETWEEN :start_date AND :end_date
        """
        )
        params_data = {
            "office_names": tuple(office_names),
            "start_date": earliest_start_date,
            "end_date": latest_end_date,
        }
        with _engine.connect() as conn:
            data = pd.read_sql_query(query_data, conn, params=params_data)

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return f"Error fetching data: {e}"

    if data.empty:
        return "Sin data disponible en el rango u oficinas seleccionadas."

    # Convert datetime columns to ensure correct dtype
    datetime_columns = ["FH_Emi", "FH_AteIni", "FH_AteFin"]
    for col in datetime_columns:
        data[col] = pd.to_datetime(data[col], errors="coerce")

    reports = []
    for office_name in office_names:
        logger.info(f"Generating report for office: {office_name}")
        if days_back is None:
            # Use custom date range
            report = get_office_stats(
                data=data,
                office_name=office_name,
                corte_espera=corte_espera,
                start_date=start_date_parsed,
                end_date=end_date_parsed,
            )
        else:
            # Use days_back to calculate date range for this office
            office_data = data[data["Oficina"] == office_name]
            if office_data.empty:
                reports.append(
                    f"Sin data disponible en el rango u oficina seleccionada: {office_name}"
                )
                continue
            last_valid_register_date = office_data["FH_Emi"].max()
            start_date_office = last_valid_register_date - timedelta(days=days_back - 1)
            report = get_office_stats(
                data=data,
                office_name=office_name,
                corte_espera=corte_espera,
                start_date=start_date_office,
                end_date=last_valid_register_date,
            )
        reports.append(report)

    return "\n".join(reports)


class ReporteDetalladoPorOficina(BaseModel):
    office_names: List[str] = Field(
        default=["356 - El Bosque", "362 - El Golf"],
        description="Lista de nombres completos de oficinas",
    )
    start_date: str = Field(
        default="07/10/2024", description="Start date in '%d/%m/%Y'"
    )
    end_date: str = Field(default="09/10/2024", description="End date in '%d/%m/%Y'")
    corte_espera: int = Field(
        default=900,
        description="Espera máximo para nivel de servicio SLA, en segundos",
    )
    parse_input_for_tool = classmethod(parse_input)
    get_documentation_for_tool = classmethod(get_documentation)


@add_docstring(
    """Reporte por Oficina
Utilizar para obtener información/indicadores sobre Oficinas.
Parameters:
{params_doc}
Returns
RESUMEN: de la sucursal/oficina: Total Atenciones Tiempo de Espera Abandonos Promedio Atenciones Diarias Nivel de Servicio (o SLA)  Ejecutivos  Escritorios
SERIES: Indicadores por serie.
DIARIO: Desempeño diario (Atenciones Totales por día).
""".format(
        params_doc=ReporteDetalladoPorOficina.get_documentation_for_tool()
    )
)
def get_reporte_general_de_oficinas(input_string: str) -> str:
    try:
        input_data = ReporteDetalladoPorOficina.parse_input_for_tool(input_string)
        reporte = reporte_general_de_oficinas(**input_data.model_dump())
        return reporte
    except Exception as e:
        return f"Error: {str(e)}"


# Create the structured tool
tool_reporte_extenso_de_oficinas = StructuredTool.from_function(
    func=get_reporte_general_de_oficinas,
    name="get_reporte_extenso_de_oficinas",
    description=get_reporte_general_de_oficinas.__doc__,
    return_direct=True,
)

if __name__ == "__main__":
    office_names = [
        "001 - Huerfanos 740 EDW",
    ]
    # Example with days_back
    print(reporte_general_de_oficinas(office_names, days_back=3, corte_espera=900))

    # # Example with custom date range
    # f"""{print(tool_reporte_general_de_oficinas.description)=},
    #     {print(tool_reporte_general_de_oficinas.invoke("{}"))=}"""
