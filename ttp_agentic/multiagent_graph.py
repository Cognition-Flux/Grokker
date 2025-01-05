# %%
import os
import sys
from pathlib import Path

# Set working directory to file location
file_path = Path(__file__).resolve()
os.chdir(file_path.parent)
sys.path.append(str(file_path.parent.parent))
import json
import os
from datetime import datetime
from typing import Annotated, Dict, List, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
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


# office_names = [
#     "001 - Huerfanos 740 EDW",
#     "003 - Cauquenes",
#     "004 - Apoquindo EDW",
#     "009 - Vitacura EDW",
# ]
# # Example with days_back
# print(rango_registros_disponibles(office_names))
# # %%
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
    oficinas_seleccionadas: set = set()
    contexto: str = ""


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


def call_model(state: GraphState):
    messages = state["messages"]

    contexto = state.get(
        "contexto",
        "no hay contexto, IGNORA TODO LO DEMÁS Y SOLO PIDELE/SOLICITALE AL USUARIO QUE SELECCIONE OFICINAS!",
    )

    system_prompt = SystemMessage(
        content=(
            f"La fecha de hoy es {datetime.now().strftime('%d/%m/%Y')}\n\n"
            "Los datos disponibles no necesariamente están actualizados a la fecha de hoy, por lo que debes verificar que registros se pueden analizar.\n\n"
            f"Contexto de registros disponibles: {contexto}\n\n"
            "Si el usuario NO proporciona un periodo de tiempo (por ejemplo, ayer, última semana, o algún mes), SIEMPRE debes preguntarle con AskHuman el periodo de tiempo a reportar\n\n"
            "solamente si el usuario lo requiere explicitamente Puedes ajustar los otros parámetros como el corte de espera, \n\n"
            "Si no hay datos disponibles, debes indicar los registros disponibles y preguntar si desea continuar con los registros disponibles\n\n"
            "Si no se especifica por el ususario, SIEMPRE debes preguntar  AskHuman el periodo de tiempo a reportar\n\n"
            "Resulve rápida y directamente las solicitudes de los usuarios\n\n"
            "Utilice tablas para oganizar la información\n\n"
            "Al final de su reporte ponga un analisis breve de los resultados relevantes\n\n"
            "Cuando las oficinas son seleccionadas en el frontend, no es necesario que las vuelvas a mencionar \n\n"
        )
    )
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}


def ask_human(state: GraphState):
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    periodo = interrupt("Indique el periodo de tiempo a reportar")
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": periodo}]
    return {"messages": tool_message}


def should_continue(state: GraphState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    # Otherwise if there is, we continue
    else:
        return "tools"


def generar_contexto(state: GraphState) -> dict:
    try:
        # Obtener el último mensaje
        last_message = state["messages"][-1]

        # Usar regex para extraer la lista de oficinas
        import re

        pattern = r"Considera las oficinas \[(.*?)\]"
        match = re.search(pattern, last_message.content)

        if match:
            # Extraer el contenido entre corchetes y convertirlo en lista
            oficinas_str = match.group(1)
            # Dividir por comas y limpiar espacios y comillas
            oficinas_list = [
                office.strip().strip("'") for office in oficinas_str.split(",")
            ]
            content = {
                "oficinas_seleccionadas": oficinas_list,
                "mensaje": last_message.content,
            }
        else:
            content = {
                "oficinas_seleccionadas": [],
                # "mensaje": last_message.content,
            }
        # Obtener el nuevo set de oficinas
        new_oficinas = set(content.get("oficinas_seleccionadas", []))
        # Obtener el set actual de oficinas
        current_oficinas = state.get("oficinas_seleccionadas", set())

        # Crear nuevo mensaje solo con el contenido del mensaje
        new_message = HumanMessage(content=content.get("mensaje", ""))

        # Crear el estado actualizado
        updated_state = {
            "messages": [RemoveMessage(id=last_message.id), new_message],
        }

        print("\n=== Verificación de cambios en oficinas ===")
        if new_oficinas != current_oficinas:
            # Oficinas agregadas
            agregadas = new_oficinas - current_oficinas
            if agregadas:
                print(f"Oficinas agregadas: {', '.join(sorted(agregadas))}")

            # Oficinas eliminadas
            eliminadas = current_oficinas - new_oficinas
            if eliminadas:
                print(f"Oficinas eliminadas: {', '.join(sorted(eliminadas))}")

            updated_state["oficinas_seleccionadas"] = new_oficinas
            updated_state["contexto"] = format_oficinas_context(
                {"oficinas_seleccionadas": new_oficinas}
            )
        else:
            print("No hay cambios en las oficinas seleccionadas")
            print(f"Oficinas actuales: {', '.join(sorted(current_oficinas))}")

        print("=================================\n")
        return updated_state

    except Exception as e:
        print(f"Error parsing input: {e}")
        return {}


def format_oficinas_context(state_values: dict) -> str:
    oficinas = state_values.get("oficinas_seleccionadas")

    if not oficinas:
        return "No hay oficinas en contexto"
    # Convertir set a lista ordenada para consistencia
    oficinas_list = sorted(oficinas)

    if len(oficinas_list) == 1:
        return f"Oficina en contexto: {oficinas_list[0]}"

    # Unir todas las oficinas con comas y "y" para la última
    # oficinas_str = ", ".join(oficinas_list[:-1]) + " y " + oficinas_list[-1]
    return f"Contexto: {rango_registros_disponibles(oficinas_list)}"


workflow = StateGraph(GraphState)
workflow.add_node("context", generar_contexto)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("ask_human", ask_human)

workflow.add_edge(START, "context")

workflow.add_edge("context", "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
)

workflow.add_edge("tools", "agent")

# After we get back the human response, we go back to the agent
workflow.add_edge("ask_human", "agent")


memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# display(Image(graph.get_graph().draw_mermaid_png()))
# %%
# display(Image(graph.get_graph().draw_mermaid_png()))
if __name__ == "__main__":
    display(Image(graph.get_graph().draw_mermaid_png()))
    config = {"configurable": {"thread_id": "1"}}

    input_message = HumanMessage(
        content="'Considera las oficinas ['001 - Huerfanos 740 EDW', '003 - Cauquenes', '004 - Apoquindo EDW', '009 - Vitacura EDW'] que año es'"
    )
    for chunk in graph.stream(
        {"messages": [input_message]}, config, stream_mode="updates"
    ):
        if "agent" in chunk:
            print(chunk["agent"]["messages"][0].pretty_print())
    # %%

    input_message = HumanMessage(
        content="'Considera las oficinas ['001 - Huerfanos 740 EDW', '003 - Cauquenes', '004 - Apoquindo EDW', '009 - Vitacura EDW'] que registros hay'"
    )
    for chunk in graph.stream(
        {"messages": [input_message]}, config, stream_mode="updates"
    ):
        if "agent" in chunk:
            print(chunk["agent"]["messages"][0].pretty_print())
