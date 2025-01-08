# %%
import json
from datetime import datetime, timedelta
from typing import List, Literal

import numpy as np
import pandas as pd
from db_instance import _engine
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tools.utilities import (
    add_docstring,
    get_documentation,
    parse_input,
    remove_extra_spaces,
    retry_decorator,
)


def generar_reporte_especifico_de_estado(df, evento):
    """
    Genera un reporte específico de estado para un DataFrame y evento dados.

    Args:
        df (pd.DataFrame): DataFrame con las columnas FH_Eve y Evento
        evento (str): Tipo de evento a analizar ('P' para pausas, 'A' para atención)

    Returns:
        tuple: (reporte_str, tiempo_total, promedio_diario)
    """
    try:
        # Validar entrada
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Se requiere un DataFrame válido")

        if not {"FH_Eve", "Evento"}.issubset(df.columns):
            raise ValueError(
                "El DataFrame debe contener las columnas 'FH_Eve' y 'Evento'"
            )

        if evento not in ["P", "A"]:
            raise ValueError("El evento debe ser 'P' o 'A'")

        if len(df) == 0:
            return (
                f"No hay datos para analizar en el período especificado para el evento {evento}",
                0,
                0,
            )

        try:
            # Crear columna de fecha (sin hora)
            df = df.copy()  # Crear copia para evitar SettingWithCopyWarning
            df["Fecha"] = pd.to_datetime(df["FH_Eve"]).dt.date
        except Exception as e:
            raise ValueError(f"Error al procesar las fechas: {str(e)}")

        total_eventos_diarios = []
        cantidad_diaria = []
        reporte_e = ""

        try:
            # Agrupar por fecha
            for n_dia, (fecha, grupo) in enumerate(df.groupby("Fecha")):
                try:
                    # Calcular cantidad de eventos
                    cantidad_E = (grupo["Evento"] == evento).sum()
                    cantidad_diaria.append(cantidad_E)

                    # Obtener índices de eventos
                    eventos_idxs = np.where(grupo.Evento == evento)[0]
                    todas_los_eventos = []

                    # Calcular duración de eventos
                    for e in eventos_idxs:
                        try:
                            if e + 1 < len(grupo):
                                diff = (
                                    grupo.FH_Eve.iloc[e + 1] - grupo.FH_Eve.iloc[e]
                                ).total_seconds() / 60
                                todas_los_eventos.append(round(diff, 2))
                        except Exception as e:
                            continue  # Saltar este evento si hay error

                    # Calcular tiempo total de eventos para el día
                    tiempo_total_dia = (
                        np.array(todas_los_eventos).sum() if todas_los_eventos else 0
                    )

                    # Agregar al reporte
                    tipo_evento = "Pausas" if evento == "P" else "intervalos atendiendo"
                    reporte_e += f"\nDía: {fecha}, Cantidad de {tipo_evento}: {cantidad_E}, \
tiempo {'en Pausas' if evento == 'P' else 'atendiendo'}: {tiempo_total_dia:.2f} minutos"

                    total_eventos_diarios.append(tiempo_total_dia)

                except Exception as e:
                    reporte_e += f"\nError procesando día {fecha}: {str(e)}"
                    continue

            # Calcular estadísticas globales
            try:
                tiempo_total_en_el_evento = np.array(total_eventos_diarios).sum()
                n_dias = n_dia + 1 if n_dia >= 0 else 1
                promedio_diario_de_tiempo_en_el_evento = (
                    tiempo_total_en_el_evento / n_dias
                )
                promedio_cantidad = (
                    np.array(cantidad_diaria).mean() if cantidad_diaria else 0
                )

                fecha_min = min(df["Fecha"]) if len(df["Fecha"]) > 0 else "N/A"
                fecha_max = max(df["Fecha"]) if len(df["Fecha"]) > 0 else "N/A"

                # Generar reporte final
                tipo_evento = "Pausas" if evento == "P" else "Intervalos atendiendo"
                reporte_final = f"""
{tipo_evento} desde {fecha_min} hasta {fecha_max}
Global:
Tiempo total {"en Pausas" if evento == "P" else "atendiendo"}: {tiempo_total_en_el_evento/60:.2f} horas
Promedio diario de tiempo {"en Pausas" if evento == "P" else "atendiendo"}: {promedio_diario_de_tiempo_en_el_evento if evento == "P" else promedio_diario_de_tiempo_en_el_evento/60:.2f} {"minutos" if evento == "P" else "horas"}
Promedio diario de cantidad de {tipo_evento}: {int(promedio_cantidad)}
Detalles:
{reporte_e}
"""
                return (
                    reporte_final,
                    tiempo_total_en_el_evento,
                    promedio_diario_de_tiempo_en_el_evento,
                )

            except Exception as e:
                raise ValueError(f"Error al calcular estadísticas globales: {str(e)}")

        except Exception as e:
            raise ValueError(f"Error al procesar los eventos por día: {str(e)}")

    except Exception as e:
        return f"Error al generar reporte específico: {str(e)}", 0, 0


def reporte_general_estados(df, nombre):
    """
    Genera un reporte general de estados para un ejecutivo.

    Args:
        df (pd.DataFrame): DataFrame con los datos del ejecutivo
        nombre (str): Nombre del ejecutivo

    Returns:
        str: Reporte general
    """
    try:
        # Validar entrada
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Se requiere un DataFrame válido")

        if not nombre or not isinstance(nombre, str):
            raise ValueError("Se requiere un nombre válido")

        # Generar reportes específicos
        (
            reporteA,
            tiempo_total_en_el_eventoA,
            promedio_diario_de_tiempo_en_el_eventoA,
        ) = generar_reporte_especifico_de_estado(df, "A")

        (
            reporteP,
            tiempo_total_en_el_eventoP,
            promedio_diario_de_tiempo_en_el_eventoP,
        ) = generar_reporte_especifico_de_estado(df, "P")

        reporte_ejecutivo = ""

        # Validar que se generaron los reportes correctamente
        if isinstance(reporteA, str) and isinstance(reporteP, str):
            reporte_ejecutivo += reporteA
            reporte_ejecutivo += reporteP

            # Calcular porcentaje de tiempo en pausa
            try:
                tiempo_total = tiempo_total_en_el_eventoA + tiempo_total_en_el_eventoP
                if tiempo_total > 0:
                    p = 100 * tiempo_total_en_el_eventoP / tiempo_total
                    reporte_ejecutivo += f"\n {nombre} tiene un porcentaje de tiempo total en pausa de {p:.2f}%"
                else:
                    reporte_ejecutivo += (
                        f"\n {nombre} no tiene tiempo registrado en el período"
                    )
            except Exception as e:
                reporte_ejecutivo += (
                    f"\nError al calcular porcentaje de tiempo en pausa: {str(e)}"
                )

            reporte_ejecutivo += "\n\nFin del reporte"
            return reporte_ejecutivo
        else:
            raise ValueError("Error al generar los reportes específicos")

    except Exception as e:
        return f"Error al generar reporte general: {str(e)}"


DATE_FORMAT = "%d/%m/%Y"


def format_date(date):
    """Formats a datetime object to a string."""
    return date.strftime(DATE_FORMAT)


@retry_decorator(max_retries=5, delay=1.0)
def reporte_detallado_por_ejecutivo(
    executive_names: list[str] = [
        "Abigail Betzabet Calabrano Avalos",
        "Maria Margarita Bahamondez Madrid",
    ],
    start_date: str = "01/10/2024",
    end_date: str = "15/10/2024",
):
    try:
        query_id_eje = f"""
        SELECT IdEje, Ejecutivo
        FROM Ejecutivos
        WHERE Ejecutivo IN ('{"', '".join(executive_names)}')
        """
        with _engine.connect() as conn:
            df_id_eje = pd.read_sql_query(query_id_eje, conn)

        if df_id_eje.empty or len(df_id_eje.index) == 0:
            raise ValueError(
                f"No se encontraron ejecutivos con los nombres proporcionados: {executive_names}"
            )

        executive_id_map = dict(zip(df_id_eje["Ejecutivo"], df_id_eje["IdEje"]))
        start_date_parsed = datetime.strptime(start_date, "%d/%m/%Y").strftime(
            "%Y-%m-%d"
        )
        end_date_parsed = (
            datetime.strptime(end_date, "%d/%m/%Y") + timedelta(days=1)
        ).strftime("%Y-%m-%d")
        tabla: str = "EjeEstado"
        reporte_final = ""

        for nombre, id in executive_id_map.items():
            reporte_final += f"\n\nInicio del reporte para ejecutivo {nombre}\n"

            # Primera consulta - Series atendidas
            query_1 = f"""
                        DECLARE @startDate DATE = '{start_date_parsed}';
                        DECLARE @endDate DATE = '{end_date_parsed}';
                        SELECT
                            o.Oficina,
                            STUFF((SELECT DISTINCT ', ' + s.Serie
                                FROM Atenciones a2
                                JOIN Series s ON s.IdSerie = a2.IdSerie
                                WHERE a2.IdEje = e.IdEje
                                        AND a2.FH_Emi BETWEEN @startDate AND @endDate
                                        AND a2.FH_AteIni IS NOT NULL AND a2.FH_AteFin IS NOT NULL
                                FOR XML PATH('')), 1, 2, '') AS 'Series que atiende'
                        FROM
                            Atenciones a
                            JOIN Oficinas o ON a.IdOficina = o.IdOficina
                            JOIN Ejecutivos e ON a.IdEje = e.IdEje
                        WHERE
                            e.IdEje IN ({id})
                            AND a.FH_AteIni IS NOT NULL
                            AND a.FH_AteFin IS NOT NULL
                            AND a.FH_Emi BETWEEN @startDate AND @endDate
                        GROUP BY
                            e.IdEje, e.Ejecutivo, o.Oficina;
                """
            try:
                with _engine.connect() as conn:
                    df = pd.read_sql_query(query_1, conn)
                if df.empty or len(df.index) == 0:
                    reporte_final += f"\nNo se encontraron series atendidas para el ejecutivo {nombre} en el período {start_date} - {end_date}."
                else:
                    reporte_final += f"\n\nSeries atendidas ({len(df.index)} registros encontrados):\n{df.to_markdown(index=False)}"
            except Exception as e:
                reporte_final += f"\nError al obtener las series atendidas: {str(e)}"

            # Segunda consulta - Resumen de atenciones
            q = f"""
                        DECLARE @startDate DATE = '{start_date_parsed}';
                        DECLARE @endDate DATE = '{end_date_parsed}';
                        SELECT
                            FORMAT(a.FH_Emi, 'yyyy-MM-dd')     AS "Fecha",
                            DATENAME(WEEKDAY, a.FH_Emi)        AS "Dia",
                            COUNT(*)                           AS "Atenciones",
                            AVG(DATEDIFF(MINUTE, a.FH_AteIni, a.FH_AteFin)) AS "Tiempo Promedio por Atencion (minutos)"
                                                                        
                        FROM
                            Atenciones a
                            JOIN Oficinas o ON a.IdOficina = o.IdOficina
                            JOIN Ejecutivos e ON a.IdEje = e.IdEje
                        WHERE
                            e.IdEje IN ({id})
                            AND a.FH_AteIni IS NOT NULL
                            AND a.FH_AteFin IS NOT NULL
                            AND a.FH_Emi BETWEEN @startDate AND @endDate
                        GROUP BY
                            FORMAT(a.FH_Emi, 'yyyy-MM-dd'),
                            DATENAME(WEEKDAY, a.FH_Emi)
                        ORDER BY
                        Fecha ASC;
                """

            try:
                with _engine.connect() as conn:
                    df = pd.read_sql_query(q, conn)
                if df.empty or len(df.index) == 0:
                    reporte_final += f"\nNo se encontraron atenciones para el ejecutivo {nombre} en el período {start_date} - {end_date}."
                else:
                    reporte_final += f"\n\nResumen de atenciones diarias ({len(df.index)} días con atenciones):\n{df.to_markdown(index=False)}\n\n"
                    reporte_final += f"\n\nConsolidado para el período:\n"
                    reporte_final += f"Total de atenciones: {df.Atenciones.sum()}\n"
                    reporte_final += f"Promedio diario de tiempo por atención: {df['Tiempo Promedio por Atencion (minutos)'].mean():.2f} minutos\n\n"
            except Exception as e:
                reporte_final += (
                    f"\nError al obtener el resumen de atenciones: {str(e)}"
                )

            # Tercera consulta - Eventos
            query_2 = f"""
                        SELECT
                            e.IdEje,
                            e.FH_Eve,
                            e.Evento
                        FROM
                            {tabla} e
                        WHERE
                            e.IdEje = {id}
                            AND e.FH_Eve BETWEEN '{start_date_parsed}' AND '{end_date_parsed}'
                        ORDER BY
                            e.FH_Eve
                        """
            try:
                with _engine.connect() as conn:
                    df = pd.read_sql_query(query_2, conn)
                if df.empty or len(df.index) == 0:
                    reporte_final += f"\nNo se encontraron eventos para el ejecutivo {nombre} en el período {start_date} - {end_date}."
                else:
                    # Verificar que el DataFrame tiene las columnas necesarias y al menos una fila
                    required_columns = ["IdEje", "FH_Eve", "Evento"]
                    if all(col in df.columns for col in required_columns):
                        try:
                            reporte_final += (
                                f"\n\nEventos encontrados ({len(df.index)} eventos):\n"
                            )
                            # print(nombre, df, nombre)
                            reporte_eventos = reporte_general_estados(df, nombre)
                            if reporte_eventos and len(reporte_eventos.strip()) > 0:
                                reporte_final += reporte_eventos
                            else:
                                reporte_final += "\nNo se pudo generar el reporte de eventos aunque se encontraron registros."
                        except Exception as e:
                            reporte_final += (
                                f"\nError al procesar el reporte de eventos: {str(e)}"
                            )
                    else:
                        missing_cols = [
                            col for col in required_columns if col not in df.columns
                        ]
                        reporte_final += f"\nFaltan columnas necesarias en los datos de eventos: {', '.join(missing_cols)}"
            except Exception as e:
                reporte_final += f"\nError al obtener los eventos: {str(e)}"

        return reporte_final

    except Exception as e:
        return f"Error general en la generación del reporte: {str(e)}"


# print(
#     reporte_detallado_por_ejecutivo(
#         start_date="01/01/2024",
#         end_date="01/10/2024",
#     )
# )


# input = {
#     **{
#         "executive_names": [
#             "Luis Hernan Labarca Montecino",
#             "Natalia Belen Troncoso Silva",
#             "Ricardo Andres Cataldo Veloso",
#             "Ivonne Alejandra Munoz Diaz",
#         ],
#         "start_date": "01/08/2024",
#         "end_date": "31/08/2024",
#     }
# }

# print(reporte_detallado_por_ejecutivo(**input))


class ReporteDetalladoPorEjecutivo(BaseModel):
    executive_names: List[str] = Field(
        default=[
            "Abigail Betzabet Calabrano Avalos",
            "Maria Margarita Bahamondez Madrid",
        ],
        description="Lista de nombres completos de ejecutivos",
    )
    start_date: str = Field(
        default="01/10/2024", description="Start date in '%d/%m/%Y' format"
    )
    end_date: str = Field(
        default="15/10/2024", description="End date in '%d/%m/%Y'  format"
    )
    parse_input_for_tool = classmethod(parse_input)
    get_documentation_for_tool = classmethod(get_documentation)


@add_docstring(
    """
Detalles de ejecutivos
Utilizar sólo cuando el usuario/humano explícitamente requiera información detallada o solicite profundizar sobre ejecutivos.
Parameters:
{params_doc}
Para obtener un único día start_date y end_date deben ser iguales a la fecha requerida.
Returns: 
Reporte detallado con información sobre uno o más ejecutivos para un período de tiempo.
El reporte entrega Oficina, Series que atiende, Resumen de atenciones diarias y tiempo en Pausas. 
Importante: Extraer y sintetizar únicamente la información específica que el usuario/humano requiere, nunca muestre todo el reporte, nunca.
""".format(
        params_doc=ReporteDetalladoPorEjecutivo.get_documentation_for_tool()
    )
)
def get_reporte_detallado_por_ejecutivo(input_string: str) -> str:
    try:
        input_data = ReporteDetalladoPorEjecutivo.parse_input_for_tool(input_string)
        reporte = reporte_detallado_por_ejecutivo(**input_data.model_dump())
        return reporte
    except Exception as e:
        return f"Error: {str(e)}"


# Create the structured tool
tool_reporte_detallado_por_ejecutivo = StructuredTool.from_function(
    func=get_reporte_detallado_por_ejecutivo,
    name="get_reporte_detallado_por_ejecutivo",
    description=get_reporte_detallado_por_ejecutivo.__doc__,
    return_direct=True,
)
if __name__ == "__main__":
    input_string = '{"executive_names":["Luis Hernan Labarca Montecino", "Natalia Belen Troncoso Silva", "Ricardo Andres Cataldo Veloso", "Ivonne Alejandra Munoz Diaz"], "start_date":"01/08/2024", "end_date":"31/08/2024"}'

    f"""{print(tool_reporte_detallado_por_ejecutivo.description)=},
        {print(tool_reporte_detallado_por_ejecutivo.invoke(
        input_string))=}"""


# %%
