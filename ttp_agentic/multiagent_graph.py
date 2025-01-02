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
from tools.reporte_detallado_por_ejecutivo import tool_reporte_detallado_por_ejecutivo
from tools.reporte_general_de_oficinas import tool_reporte_general_de_oficinas
from typing_extensions import TypedDict

load_dotenv(override=True)
oficinas_seleccionadas = [
    "160 - Ñuñoa",
    "162 - Torre Las Condes",
    "001 - Huerfanos 740 EDW",
    "003 - Cauquenes",
    "004 - Apoquindo EDW",
    "005 - Los Leones",
    "007 - Viña del Mar EDW",
    "223 - Talcahuano",
    "225 - Concepcion",
    "229 - Curico OHiggins",
    "230 - Los Angeles",
    "232 - Angol",
    "234 - Pirque",
    "235 - Victoria",
]


# %%
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


# tools = [
#     tool_reporte_general_de_oficinas,
#     tool_reporte_detallado_por_ejecutivo,
#     executive_ranking_tool,
# ]

tools = [
    mock_tool,
]

tool_node = ToolNode(tools)
llm_with_tools = get_llm().bind_tools(tools + [AskHuman])


def call_model(state):
    messages = state["messages"]

    # contexto = state.get("contexto", oficinas_seleccionadas)

    system_prompt = SystemMessage(
        content=(
            # f"Oficinas seleccionadas: {contexto}\n\n"
            "El año actual es 2024."
            # "Al inicio del chat, pregunta al usuario su nombre (AskHuman)."
            "si el usuario dice que quiere el reporte, debes preguntarle con AskHuman el periodo de tiempo a reportar"
            "cuando el usuaio da su periodo de tiempo, solamento debes llamar mock_tool, decir que se llamó y finalizar."
        )
    )
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}


def ask_human(state):
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    periodo = interrupt("Indique el periodo de tiempo a reportar")
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": periodo}]
    return {"messages": tool_message}


def should_continue(state):
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


workflow = StateGraph(GraphState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("ask_human", ask_human)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
)

workflow.add_edge("tools", "agent")

# After we get back the human response, we go back to the agent
workflow.add_edge("ask_human", "agent")


memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
