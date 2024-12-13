# %%
# Importar las bibliotecas necesarias
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

#
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory

# from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.messages import AIMessage, HumanMessage
from typing import List, Union
from langchain_core.messages import BaseMessage
import plotly.graph_objects as go
import plotly.express as px  # Para paletas de colores y temas
from langchain_openai import AzureChatOpenAI
import os
from typing import Any, List, Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import logging
import pandas as pd
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()


class Trace(BaseModel):
    type: str = Field(
        ...,
        description="Tipo de traza (por ejemplo, 'scatter', 'bar', 'pie', 'histogram', etc.)",
    )
    x: Optional[List[Any]] = Field(
        None,
        description="Puntos de datos del eje X (pueden ser fechas (dates) para series temporales)",
    )
    y: Optional[List[Any]] = Field(None, description="Puntos de datos del eje Y")
    mode: Optional[str] = Field(
        None,
        description="Modo de dibujo para trazas scatter (por ejemplo, 'lines', 'markers', 'lines+markers')",
    )
    name: Optional[str] = Field(None, description="Nombre de la traza para la leyenda")
    text: Optional[List[Any]] = Field(
        None, description="Elementos de texto asociados a cada punto de datos"
    )
    hovertext: Optional[List[str]] = Field(
        None, description="Elementos de texto al pasar el ratón"
    )
    marker: Optional[dict] = Field(
        None,
        description="Opciones de estilo del marcador (por ejemplo, 'size', 'color', 'symbol', 'line').",
    )
    line: Optional[dict] = Field(
        None,
        description="Opciones de estilo de línea (por ejemplo, 'width', 'color', 'dash').",
    )
    fill: Optional[str] = Field(
        None,
        description="Opciones de relleno de área (por ejemplo, 'tozeroy', 'tonexty')",
    )
    opacity: Optional[float] = Field(
        None, description="Opacidad de la traza entre 0 y 1"
    )
    orientation: Optional[str] = Field(
        None, description="Orientación para gráficos de barras ('v' o 'h')"
    )
    labels: Optional[List[Any]] = Field(
        None, description="Etiquetas para sectores del gráfico de pastel"
    )
    values: Optional[List[Any]] = Field(
        None, description="Valores para sectores del gráfico de pastel"
    )
    textposition: Optional[str] = Field(
        None,
        description="Posición de las etiquetas de texto (por ejemplo, 'inside', 'outside')",
    )
    hole: Optional[float] = Field(
        None, description="Tamaño del agujero en un gráfico de dona (0 a 1)"
    )
    direction: Optional[str] = Field(
        None,
        description="Dirección de los sectores del gráfico de pastel ('clockwise', 'counterclockwise')",
    )
    sort: Optional[bool] = Field(
        None, description="Si ordenar los sectores del gráfico de pastel"
    )
    customdata: Optional[List[Any]] = Field(
        None, description="Datos adicionales para eventos de pasar el ratón y clic"
    )
    error_x: Optional[dict] = Field(None, description="Barras de error para el eje x")
    error_y: Optional[dict] = Field(None, description="Barras de error para el eje y")
    visible: Optional[bool] = Field(None, description="Visibilidad de la traza")
    showlegend: Optional[bool] = Field(
        None, description="Si mostrar la traza en la leyenda"
    )
    legendgroup: Optional[str] = Field(
        None, description="Nombre del grupo para la leyenda"
    )
    offsetgroup: Optional[str] = Field(
        None, description="Nombre del grupo para barras agrupadas"
    )
    hoverinfo: Optional[str] = Field(
        None, description="Información que se muestra al pasar el ratón"
    )
    texttemplate: Optional[str] = Field(
        None, description="Plantilla para el texto mostrado en las etiquetas"
    )
    width: Optional[float] = Field(
        None, description="Ancho de las barras en gráficos de barras"
    )
    # Añade más campos según sea necesario para diferentes tipos de traza

    @field_validator("text")
    def convert_text_to_string(cls, v):
        if v is not None:
            return [str(item) for item in v]
        return v

    class Config:
        extra = "allow"  # Permitir campos extra


class Layout(BaseModel):
    title: Optional[str] = Field(None, description="Título del gráfico")
    xaxis: Optional[dict] = Field(None, description="Configuración del eje X")
    yaxis: Optional[dict] = Field(None, description="Configuración del eje Y")
    legend: Optional[dict] = Field(None, description="Configuración de la leyenda")
    template: Optional[str] = Field(None, description="Plantilla de tema del gráfico")
    margin: Optional[dict] = Field(None, description="Márgenes del gráfico")
    annotations: Optional[List[dict]] = Field(
        None, description="Lista de anotaciones para agregar al gráfico"
    )
    shapes: Optional[List[dict]] = Field(
        None, description="Lista de formas para agregar al gráfico"
    )
    bargap: Optional[float] = Field(
        None, description="Espacio entre barras en gráficos de barras"
    )
    bargroupgap: Optional[float] = Field(
        None, description="Espacio entre grupos de barras en gráficos de barras"
    )
    barmode: Optional[str] = Field(
        None,
        description="Modo de gráfico de barras ('stack', 'group', 'overlay', 'relative')",
    )
    hovermode: Optional[str] = Field(
        None, description="Modo de interacción al pasar el ratón"
    )
    polar: Optional[dict] = Field(
        None, description="Configuración para gráficos polares"
    )
    radialaxis: Optional[dict] = Field(
        None, description="Configuración del eje radial en gráficos polares"
    )
    angularaxis: Optional[dict] = Field(
        None, description="Configuración del eje angular en gráficos polares"
    )
    showlegend: Optional[bool] = Field(None, description="Si mostrar la leyenda")
    font: Optional[dict] = Field(
        None, description="Configuración de fuente para el texto del gráfico"
    )
    paper_bgcolor: Optional[str] = Field(None, description="Color de fondo del papel")
    plot_bgcolor: Optional[str] = Field(
        None, description="Color de fondo del área del gráfico"
    )
    width: Optional[int] = Field(None, description="Ancho del gráfico en píxeles")
    height: Optional[int] = Field(None, description="Altura del gráfico en píxeles")
    autosize: Optional[bool] = Field(
        None,
        description="Si el gráfico debe ajustarse automáticamente al tamaño del contenedor",
    )
    title_font: Optional[dict] = Field(
        None, description="Configuración de fuente para el título"
    )
    xaxis_title_font: Optional[dict] = Field(
        None, description="Configuración de fuente para el título del eje X"
    )
    yaxis_title_font: Optional[dict] = Field(
        None, description="Configuración de fuente para el título del eje Y"
    )
    xaxis_tickfont: Optional[dict] = Field(
        None, description="Configuración de fuente para las etiquetas del eje X"
    )
    yaxis_tickfont: Optional[dict] = Field(
        None, description="Configuración de fuente para las etiquetas del eje Y"
    )
    legend_font: Optional[dict] = Field(
        None, description="Configuración de fuente para la leyenda"
    )
    # Añade más campos según sea necesario

    class Config:
        extra = "allow"  # Permitir campos extra


class PlotlySchema(BaseModel):
    data: List[Trace] = Field(..., description="Lista de trazas de datos")
    layout: Optional[Layout] = Field(None, description="Configuración del diseño")


# Obtener las variables de entorno necesarias
# AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
# AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT")
# AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION")


def initialize_model():
    """Inicializa el modelo AzureChatOpenAI."""
    # model_instance = AzureChatOpenAI(
    #     azure_deployment="gpt-4o",
    #     api_version=AZURE_API_VERSION,
    #     temperature=0,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=5,
    #     api_key=AZURE_API_KEY,
    #     azure_endpoint=AZURE_ENDPOINT,
    #     streaming=True,
    # )
    model_instance = ChatOpenAI(
        model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
    )
    return model_instance


system_prompt = """
    Recuerda siempre poner nombres(labels) de las variables, leyendas (legends) para cada variable y título.
    Los datos pueden ser no-temporales o temporales.
        A continuación, solo y exclusivamente si los datos son NO-TEMPORALES:
            - Si es una sola/unica oficina, haga una barra para cada variable con diferentes colores y no muestre (esconda) eje Y (axis-y).
            - Cuando son varias oficinas, utilizar barras agrupadas (barmode='group') y no muestre (esconda) eje Y (axis-y).
            - Siempre: Mostrar los valores de las variables.
            - Siempre: Las etiquetas (valores de las variables) posicionarlas afuera, 'outside'.

        A continuación, solo y exclusivamente si los datos son series temporales (time series):
            Utiliza los datos proporcionados para generar un gráfico de líneas que muestre las atenciones diarias a lo largo del tiempo para cada oficina. Asegúrate de:

            - Usar la fecha como eje X (asegúrate de que las fechas estén en el formato adecuado).
            - Diferenciar cada oficina con colores distintos.
            - Si hay más de una oficina, incluir una leyenda para identificar cada línea.
            - Asegurarte de que el eje X esté ordenado cronológicamente.
            
    Sobre las solicitudes de modificaciones/actualizaciones del usuario a un gráfico existente:
        - Debes respetar el gráfico previamente existente e ir aplicando cuidadosamemte las modificaciones sin hacer alteraciones que el usuario no haya solicitado.
        - Tu actualización debe ser consistente con la cronología del historial de mensajes; siempre verifica el hilo de la conversación.
    """
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)


def make_plot(structured_output):
    # Aplicar el tema especificado por el usuario o por defecto 'plotly_dark'
    theme = (
        structured_output.layout.template
        if structured_output.layout and structured_output.layout.template
        else "plotly_dark"
    )

    fig = go.Figure()

    # Definir colores personalizados si se proporcionan
    custom_colors = []
    for trace in structured_output.data:
        if trace.marker and "colors" in trace.marker:
            custom_colors.extend(trace.marker["colors"])

    # Usar colores personalizados o paleta predeterminada
    if custom_colors:
        colors = custom_colors
    else:
        color_palette = px.colors.qualitative.Plotly
        num_traces = len(structured_output.data)
        if num_traces > len(color_palette):
            colors = color_palette * ((num_traces // len(color_palette)) + 1)
        else:
            colors = color_palette

    # Mapeo de tipo de traza a objetos de gráfico de Plotly
    trace_type_mapping = {
        "scatter": go.Scatter,
        "bar": go.Bar,
        "pie": go.Pie,
        "histogram": go.Histogram,
        "box": go.Box,
        "scatter3d": go.Scatter3d,
        "surface": go.Surface,
        "heatmap": go.Heatmap,
        "violin": go.Violin,
        "area": go.Scatter,  # Los gráficos de área son scatter plots con 'fill' establecido
        # Añade más mapeos según sea necesario
    }

    # Añadir trazas con colores de la paleta o colores definidos por el usuario
    for idx, trace in enumerate(structured_output.data):
        trace_dict = trace.model_dump(exclude_unset=True)
        trace_type = trace_dict.pop("type", None)  # Eliminar la clave 'type'

        # Intentar convertir 'x' a datetime si es apropiado
        if "x" in trace_dict:
            try:
                converted_x = pd.to_datetime(trace_dict["x"])
                # Comprobar si la conversión tuvo éxito sin generar valores NaT
                if not converted_x.isnull().all():
                    trace_dict["x"] = converted_x
            except (ValueError, TypeError):
                # Si falla la conversión, dejar 'x' como está
                pass

        if trace_type in trace_type_mapping:
            trace_class = trace_type_mapping[trace_type]

            # Asignar color a la traza apropiadamente
            if trace_type in ["scatter", "scatter3d", "line", "area"]:
                trace_dict["line"] = trace_dict.get("line", {})
                if "color" not in trace_dict["line"]:
                    trace_dict["line"]["color"] = colors[idx]
            elif trace_type in ["bar", "histogram", "box", "violin", "heatmap"]:
                trace_dict["marker"] = trace_dict.get("marker", {})
                if "color" not in trace_dict["marker"]:
                    trace_dict["marker"]["color"] = colors[idx]
            elif trace_type == "pie":
                trace_dict["marker"] = trace_dict.get("marker", {})
                if "colors" not in trace_dict["marker"]:
                    trace_dict["marker"]["colors"] = colors[
                        : len(trace_dict.get("values", []))
                    ]
                if "values" not in trace_dict:
                    trace_dict["values"] = trace_dict.get("y", [])
                if "labels" not in trace_dict:
                    trace_dict["labels"] = trace_dict.get("x", [])
            else:
                pass

            fig.add_trace(trace_class(**trace_dict))
        else:
            trace_dict["line"] = trace_dict.get("line", {})
            if "color" not in trace_dict["line"]:
                trace_dict["line"]["color"] = colors[idx]
            fig.add_trace(go.Scatter(**trace_dict))

    # Ajustar el rango del eje Y si es necesario
    all_y_values = []
    for trace in structured_output.data:
        if trace.y:
            all_y_values.extend(trace.y)

    if all_y_values:
        max_y = max(all_y_values)
        adjusted_max_y = max_y * 1.15
        fig.update_yaxes(range=[0, adjusted_max_y])

    # Actualizar el diseño con cualquier opción de personalización adicional
    if structured_output.layout:
        fig.update_layout(**structured_output.layout.model_dump(exclude_unset=True))

    # Asegurar que las líneas de cuadrícula se muestren por defecto y aplicar el tema
    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        template=theme,
    )
    return fig


def instantiate_model_with_prompt_and_PlotlySchema():
    modelo = initialize_model()
    model_with_structure_and_prompt = prompt | modelo.with_structured_output(
        PlotlySchema
    )
    return model_with_structure_and_prompt


def llm_json_to_plot_from_text(
    input_instructions, model_with_structure_and_prompt, chat_history, max_retries=3
):
    retries = 0
    while retries < max_retries:
        # Invocar al modelo con el input del usuario y el historial de chat
        respuesta_estructurada = model_with_structure_and_prompt.invoke(
            {
                "input": input_instructions,
                "chat_history": chat_history.messages,
            }
        )

        try:
            fig = make_plot(respuesta_estructurada)
            # Si la validación es exitosa, actualizar el historial y retornar la respuesta
            chat_history.append(HumanMessage(content=input_instructions))
            chat_history.append(AIMessage(content=str(respuesta_estructurada)))
            return fig
        except Exception as e:
            # Capturar el mensaje de error
            error_message = str(e)
            # Actualizar el historial con el mensaje del usuario y la respuesta fallida
            chat_history.append(HumanMessage(content=input_instructions))
            chat_history.append(AIMessage(content=str(respuesta_estructurada)))
            # Informar  sobre el error y solicitar que vuelva a intentarlo
            input_instructions = f"{input_instructions=}\n\nHa ocurrido un error: {error_message}\nPor favor, intenta corregirlo y vuelve a intentarlo."
            retries += 1
            continue
    # Si se excede el número máximo de reintentos, lanzar una excepción
    raise Exception(f"Se ha excedido el número máximo de reintentos ({max_retries=}).")


class ChatHistory(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)

    def append(self, message: BaseMessage):
        self.messages.append(message)


if __name__ == "__main__":
    chat_history = ChatHistory()
    model_with_structure_and_prompt = instantiate_model_with_prompt_and_PlotlySchema()
    human_input = "tres barras de color: hueso, ciruela, oliva"
    fig = llm_json_to_plot_from_text(
        human_input, model_with_structure_and_prompt, chat_history
    )
    fig.show()
    human_input = "cambia el gráfico a torta"
    fig = llm_json_to_plot_from_text(
        human_input, model_with_structure_and_prompt, chat_history
    )
    fig.show()
    human_input = "ahora asigna 15%, 55%, y 30% a las variables. Y cambia la leyenda a  hueso = 'hola', ciruela='q tal' , oliva='chao'"
    fig = llm_json_to_plot_from_text(
        human_input, model_with_structure_and_prompt, chat_history
    )
    fig.show()
