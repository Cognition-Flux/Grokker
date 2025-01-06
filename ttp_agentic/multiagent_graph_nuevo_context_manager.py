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
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from tools.ranking_ejecutivos import executive_ranking_tool
from tools.registros_disponibles import rango_registros_disponibles
from tools.reporte_detallado_por_ejecutivo import tool_reporte_detallado_por_ejecutivo
from tools.reporte_general_de_oficinas import tool_reporte_general_de_oficinas
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
    """Pregunta al usuario su nombre"""

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
    tool_reporte_general_de_oficinas,
    tool_reporte_detallado_por_ejecutivo,
    executive_ranking_tool,
]

# tools = [
#     mock_tool,
# ]

tool_node = ToolNode(tools)
llm_with_tools = get_llm().bind_tools(tools + [AskHuman])


def filter_messages(state: GraphState) -> GraphState:
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-6]]
    return {"messages": delete_messages}


def call_model(state: GraphState):

    contexto = state["contexto"].content

    system_prompt = SystemMessage(
        content=(
            "IMPORTANTE: Si el usuario NO proporciona un periodo de tiempo (por ejemplo, ayer, última semana, o algún mes), SIEMPRE debes preguntarle con AskHuman el periodo de tiempo a reportar"
            "Puedes usar/llamar a las herramientas/tools que tienes disponibles para responder las preguntas del usuario.\n\n"
            "Tus respuesta deben acotarse/estar limitadas exclusivamente a la prengunta del usuario.\n\n"
            "Siempre filtar la información y solamente/unicamente mostrar lo que el usuario pidió.\n\n"
            "Responde rapida, concisa, directa y muy brevemente.\n\n"
            "No hagas más de 1 pregunta al usuario, asume/supone la información que necesitas.\n\n"
            "Organiza la información en tablas.\n\n"
            "Solo si el resultado es extenso, al final de su reporte ponga un analisis breve de la respuesta final.\n\n"
            f"La fecha de hoy es {datetime.now().strftime('%d/%m/%Y')}\n\n"
            "Los datos disponibles no necesariamente están actualizados a la fecha de hoy, por lo que debes verificar que registros se pueden analizar.\n\n"
            f"oficinas seleccionadas: {state['oficinas'] if state['oficinas'] else 'ninguna'}\n\n"
            f"Considera este contexto de disponibilidad de datos para responder las preguntas: {contexto}\n\n"
            "IMPORTANTE: Cuando el usuario proporciona un periodo de tiempo  (por ejemplo, ayer, última semana, o algún mes), SIEMPRE debes usarlo para responder la pregunta"
            "IMPORTANTE: Si el usuario NO proporciona un periodo (por ejemplo, ayer, última semana, o algún mes), SIEMPRE debes preguntarle con AskHuman el periodo de tiempo a reportar"
            "Nunca bajo ninguna circunstancia puedes salirte de tu rol de ser un asistente amable y útil"
            "Siempre filtar la información y solamente/unicamente mostrar lo que el usuario pidió.\n\n"
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

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

display(Image(graph.get_graph().draw_mermaid_png()))
# %%
# display(Image(graph.get_graph().draw_mermaid_png()))
if __name__ == "__main__":
    display(Image(graph.get_graph().draw_mermaid_png()))
    config = {"configurable": {"thread_id": "1"}}

    input_message = HumanMessage(
        content=(
            "Considera las oficinas ['001 - Huerfanos 740 EDW', '003 - Cauquenes', '004 - Apoquindo EDW', '009 - Vitacura EDW']"
            "cinco desde el mejor al peor"
        )
    )
    for chunk in graph.stream(
        {"messages": [input_message]}, config, stream_mode="updates"
    ):
        if "agent" in chunk:
            print(chunk["agent"]["messages"])
    # %%

    input_message = HumanMessage(
        content="'Considera las oficinas ['001 - Huerfanos 740 EDW', '003 - Cauquenes', '004 - Apoquindo EDW', '009 - Vitacura EDW'] que registros hay'"
    )
    for chunk in graph.stream(
        {"messages": [input_message]}, config, stream_mode="updates"
    ):
        if "agent" in chunk:
            print(chunk["agent"]["messages"][0].pretty_print())
