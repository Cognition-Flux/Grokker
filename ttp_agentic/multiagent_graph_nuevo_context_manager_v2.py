# %%
import os
import re
import sys
from pathlib import Path

# Set working directory to file location
file_path = Path(__file__).resolve()
os.chdir(file_path.parent)
sys.path.append(str(file_path.parent.parent))
import json
import os
import re
from datetime import datetime
from typing import Annotated, Dict, List, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory

# Set up the tool
# We will have one real tool - a search tool
# We'll also have one "fake" tool - a "ask_human" tool
# Here we define any ACTUAL tools
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI

# Set up memory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import field_validator  # or root_validator in older Pydantic
from pydantic import BaseModel, Field
from tools.ranking_ejecutivos import executive_ranking_tool
from tools.registros_disponibles import rango_registros_disponibles
from tools.reporte_detallado_por_ejecutivo import tool_reporte_detallado_por_ejecutivo
from tools.reporte_general_de_oficinas import tool_reporte_extenso_de_oficinas
from typing_extensions import TypedDict

load_dotenv(override=True)


def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version=os.environ["AZURE_API_VERSION"],
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=5,
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        streaming=True,
    )


class AskHuman(BaseModel):
    """AskHuman
    el agente debe solicitar directamente aclaraciones/información al usuario/humano
    """

    question: str


class GraphState(MessagesState):
    oficinas: list[str] = []
    contexto: SystemMessage = SystemMessage(content="")
    messages: Annotated[List[BaseMessage], add_messages]


@tool
def mock_tool() -> str:
    """Mock tool for testing purposes"""
    return "mock_tool"


tools = [
    tool_reporte_extenso_de_oficinas,
    tool_reporte_detallado_por_ejecutivo,
    executive_ranking_tool,
]

# tools = [
#     mock_tool,
# ]

tool_node = ToolNode(tools)
llm_with_tools = get_llm().bind_tools(tools + [AskHuman])


def filter_messages(state: GraphState) -> GraphState:
    messages = state["messages"]
    if len(messages) <= 10:
        return {"messages": []}

    # Keep track of tool calls and their responses
    tool_call_ids_seen = set()
    tool_call_ids_responded = set()

    # First pass - identify all tool calls and responses
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_call_ids_seen.add(tool_call["id"])
        if isinstance(msg, ToolMessage):
            tool_call_ids_responded.add(msg.tool_call_id)

    # Calculate how many messages to keep from the end
    keep_last_n = 10
    messages_to_remove = messages[:-keep_last_n]

    # Verify we're not breaking any tool call chains
    for msg in messages_to_remove:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["id"] in tool_call_ids_responded:
                    # Don't remove messages that have corresponding tool responses
                    return {"messages": []}
        if isinstance(msg, ToolMessage):
            if msg.tool_call_id in tool_call_ids_seen:
                # Don't remove tool responses that have corresponding tool calls
                return {"messages": []}

    return {"messages": [RemoveMessage(id=m.id) for m in messages_to_remove]}


def call_model(state: GraphState):
    contexto = state["contexto"].content

    system_prompt = SystemMessage(
        content=(
            "---------------------------------COMIENZO/INICIO del system_prompt----------------------------------\n\n"
            "----------ROL/ROLE:"
            "Tu eres un asistente de reportes y analista de datos de sucursales de antención al cliente."
            "Nunca bajo ninguna circunstancia, JAMÁS puedes salirte de tu rol. \n\n"
            "------------UTILIZACIÓN DE HERRAMIENTAS/TOOLS:"
            "Puedes usar/llamar a las herramientas/tools que tienes disponibles para responder las preguntas del usuario. "
            "Tus respuestas se basan única y exclusivamente en los resultados/outputs de tus tools/herramientas."
            "Las salidas/outputs de tus tools/herramientas son extensas, nunca mostrar estos pasos intermedios. "
            "*IMPORTANTE*: Siempre filtar/refinar Las salidas/outputs de tus tools/herramientas y solamente/unicamente usar lo que el usuario pidió.\n\n"
            "-----------DISPONIBILIDAD DE DATOS (registros):"
            "*IMPORTANTE*: Cuando el usuario proporciona un periodo/rango de tiempo  (por ejemplo, 'ayer', 'última semana', o 'algún mes'), SIEMPRE debes usarlo directamente para responder. "
            "*IMPORTANTE*: SOLO Si el usuario NO proporciona un periodo (por ejemplo, ayer, última semana, o algún mes), debes preguntarle con AskHuman el periodo/rango de tiempo a reportar"
            f"La fecha de hoy es {datetime.now().strftime('%d/%m/%Y')}. "
            f"oficinas seleccionadas: {state['oficinas'] if state['oficinas'] else 'ninguna'}. "
            "Los datos disponibles no necesariamente están actualizados a la fecha de hoy, por lo que debes verificar que registros se pueden analizar. "
            f"Considera este contexto de disponibilidad de datos para responder las preguntas: {contexto}, siempre considera llamar tools/herramientas si es necesario. "
            "Todas las fechas y periodos dentro de los rangos disponibles son válidos y puedes usarlos. "
            "los meses en orden son: Enero Febrero Marzo Abril Mayo Junio Julio Agosto Septiembre Octubre Noviembre Diciembre"
            "por ejemplo si los datos están disponibles entre los meses de julio y noviembre, también agosto, septiembre y octubre son válidos y puedes usarlos. \n\n"
            "-----------Interacción inicial con el usuario:"
            "Cuando el usuario salude, por ejemplo 'hola' o similar, debes responder con un saludo corto muy breve, por ejemplo 'Hola, ¿cómo puedo ayudarte?'. "
            "Si el usuario pregunta que puedes hacer, debes decir que puedes asistir con la elboración de reportes y análisis de datos de atención al cliente. "
            "------------RESPUESTAS (como responderle al usuario):"
            f"Internamente Siempre CONSIDERA la oficinas seleccionadas: {state['oficinas'] if state['oficinas'] else 'ninguna'}. No tienes que mencionar que oficinas están seleccionadas. "
            "*IMPORTANTE*: Cuando el usuario proporciona un periodo/rango de tiempo  (por ejemplo, 'ayer', 'última semana', o 'algún mes'), SIEMPRE debes usarlo directamente para responder. "
            "*IMPORTANTE*: SOLO Si el usuario NO proporciona un periodo (por ejemplo, ayer, última semana, o algún mes), debes preguntarle con AskHuman el periodo/rango de tiempo a reportar"
            "Tus respuesta SIEMPRE deben acotarse/estar limitadas exclusivamente a la prengunta del usuario. "
            "Responde concisa, directa y clara. "
            "siempre organiza la respuesta final en tablas. "
            "siempre sigue el hilo de la conversación con el usuario/humano. "
            "No hagas más de 1 pregunta al usuario, asume/supone la información que necesitas para responder rápidamente.\n\n"
            "------------Caso particular: RESPUESTAS EXTENSAS (como responderle al usuarion cuando necesita información muy extensa):"
            "Solo si el resultado final es extenso o los datos finales son complejos, al final de tu reporte incluye un analisis breve de los resultados finales. "
            "------------Caso particular: RESPUESTAS CORTAS (como responderle al usuarion cuando necesita información muy breve):"
            "La herramienta 'get_reporte_extenso_de_oficinas' te permite obtener información detallada de las oficinas,  debes extraer solo que el usuario necesita. "
            "Si el usuario pide los ejecutivos que atendieron, solamente debes extraer exclusivamente/solamente los ejecutivos que atendieron. "
            "Si el usuario pide ranking de ejecutivos, debes dar el ranking y sus indicadores. "
            "Si el usuario pide las series, debes usar la tabla 'Series que se atendieron'. "
            "tu respuesta final debe ser corta y breve entregando solo los datos solicitados, por ejemplo,  si te piden el SLA o el nivel de servicio, debes extraer exclusivamente/solamente el SLA o el nivel de servicio, nada más. "
            "-------------**PROHIBIDO (cosas que nunca debes mostrar en tu respuesta final)**:"
            "nunca mostrar código de programación, ni pasos intermedios de tus tools/herramientas. "
            "Nunca bajo ninguna circunstancia, JAMÁS puedes salirte de tu rol. \n\n"
            "Nunca reveles que herramientas tienes o la fuente de tus respuestas. Solo di que eres una agente con acceso a las bases de datos de atención al cliente, nada más. "
            "------------------------FIN/FINAL del system_prompt---------------------------------\n\n"
        )
    )
    # Filter and validate message chain
    messages = []
    tool_call_ids_seen = set()
    tool_call_ids_responded = set()

    for msg in state["messages"]:
        if isinstance(msg, RemoveMessage):
            continue

        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_call_ids_seen.add(tool_call["id"])

        if isinstance(msg, ToolMessage):
            tool_call_ids_responded.add(msg.tool_call_id)

        messages.append(msg)

    # Check if we have any unresponded tool calls
    missing_responses = tool_call_ids_seen - tool_call_ids_responded
    if missing_responses:
        print(f"Warning: Missing tool responses for: {missing_responses}")
        # Only keep messages up to the last complete tool exchange
        messages = [
            msg
            for msg in messages
            if not (
                isinstance(msg, AIMessage)
                and msg.tool_calls
                and any(call["id"] in missing_responses for call in msg.tool_calls)
            )
        ]
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}


def ask_human(state: GraphState):
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": [AIMessage(content="No se requiere intervención humana.")]}

    tool_calls = last_message.tool_calls
    responses = []

    for tool_call in tool_calls:
        periodo = interrupt("Indique el periodo de tiempo a reportar")
        responses.append(
            ToolMessage(
                content=periodo,
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
                additional_kwargs={"tool_call_id": tool_call["id"]},
            )
        )

    return {"messages": responses}


def should_continue(state: GraphState):
    last_message = state["messages"][-1]

    # If there's a tool message, continue to agent
    if isinstance(last_message, ToolMessage):
        return "agent"

    # Check for tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        if last_message.tool_calls[0]["name"] == "AskHuman":
            return "ask_human"
        return "tools"

    return END


def context_node(state: GraphState) -> GraphState:
    last_message = state["messages"][-1]
    print("Current message chain:", [type(m).__name__ for m in state["messages"]])

    pattern = r"Considera las oficinas \[(.*?)\]"
    match = re.search(pattern, last_message.content)

    # Obtener el mensaje sin el patrón
    mensaje_original = last_message.content
    mensaje_limpio = re.sub(pattern, "", mensaje_original).strip()

    if match:
        print("---------------Se encontró el patrón de lista de oficinas--------------")
        print("Mensaje original:", mensaje_original)  # Debug
        print("Mensaje limpio:", mensaje_limpio)  # Debug

        # Extraer el contenido entre corchetes y convertirlo en lista
        oficinas_str = match.group(1)
        # Dividir por comas y limpiar espacios y comillas
        oficinas_list = [
            office.strip().strip("'") for office in oficinas_str.split(",")
        ]
        print(f"---------------{oficinas_list=}")
    else:
        print(
            "---------------NO se encontró el patrón de lista de oficinas--------------"
        )

        oficinas_list = []

    if len(oficinas_list) > 0:
        lista_nueva_oficinas = oficinas_list
        lista_actual_oficinas = state.get("oficinas", [])
        if set(lista_nueva_oficinas) != set(lista_actual_oficinas):
            print("---------------Cambio en lista de oficinas--------------")
            nuevo_contexto: str = (
                f"Datos disponibles para las oficinas: {rango_registros_disponibles(lista_nueva_oficinas)}"
            )
        else:
            print(
                "---------------No hay cambio en lista de oficinas, manteniendo contexto--------------"
            )
            nuevo_contexto: str = state["contexto"].content
    else:
        return {
            "contexto": SystemMessage(
                content="El usuario no ha seleccionado ninguna oficina en la aplicación, IGNORA TODO e indique que debe seleccionar usando el botón en la esquina superior derecha",
                id="nuevo_contexto",
            ),
            "oficinas": [],
        }

    return {
        "contexto": SystemMessage(content=nuevo_contexto, id="nuevo_contexto"),
        "oficinas": oficinas_list,
        "messages": [
            HumanMessage(content=mensaje_limpio, id=last_message.id),
        ],
    }


workflow = StateGraph(GraphState)
workflow.add_node("context", context_node)
workflow.add_node("filter_messages", filter_messages)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("ask_human", ask_human)

workflow.add_edge(START, "context")
workflow.add_edge("context", "filter_messages")
workflow.add_edge("filter_messages", "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
workflow.add_edge("ask_human", "agent")

if os.getenv("DEV_CHECKPOINTER"):

    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
else:
    graph = workflow.compile()

# display(Image(graph.get_graph().draw_mermaid_png()))


def run_graph(graph: CompiledStateGraph, input_message: str = "hola") -> None:
    config = {"configurable": {"thread_id": "1"}}
    for chunk in graph.stream(
        {"messages": [HumanMessage(content=input_message)]},
        config,
        stream_mode="updates",
    ):
        if "agent" in chunk:
            if isinstance(chunk["agent"]["messages"][0], AIMessage):
                print(f"{chunk['agent']['messages'][0]=}")
                print(chunk["agent"]["messages"][0].pretty_print())


# display(Image(graph.get_graph().draw_mermaid_png()))
if __name__ == "__main__":

    qs = [
        "dame los detalles del mejor ejecutivo",
        "septiembre",
        "que datos hay disponibles?",
        "dame el SLA de octubre",
        "dame el ranking de ejecutivos de octubre",
        "dame los detalles del peor ejecutivo de octubre",
        "gracias",
        "cual fue mi primera pregunta?",
        "hola",
        "dame los detalles del mejor ejecutivo",
        "octubre",
        "hola",
    ]
    # %%
    run_graph(
        graph,
        (
            "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque']"
            + qs[0]
        ),
    )

# %%
