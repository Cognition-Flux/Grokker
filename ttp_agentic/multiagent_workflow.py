# %%
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
from typing_extensions import TypedDict

from ttp_agentic.tools.ranking_ejecutivos import executive_ranking_tool
from ttp_agentic.tools.reporte_detallado_por_ejecutivo import (
    tool_reporte_detallado_por_ejecutivo,
)
from ttp_agentic.tools.reporte_general_de_oficinas import (
    tool_reporte_general_de_oficinas,
)

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
    """Ask the human a question"""

    question: str


class GraphState(MessagesState):
    oficinas_seleccionadas: set = set()
    contexto: str = ""


tools = [
    tool_reporte_general_de_oficinas,
    tool_reporte_detallado_por_ejecutivo,
    executive_ranking_tool,
]
tool_node = ToolNode(tools)
llm_with_tools = get_llm().bind_tools(tools + [AskHuman])


def call_model(state):
    messages = state["messages"]

    contexto = state.get("contexto", oficinas_seleccionadas)

    system_prompt = SystemMessage(
        content=(
            f"Oficinas seleccionadas: {contexto}\n\n"
            "El año actual es 2024."
            "Usa las herramientas disponibles para responder la pregunta del usuario."
            "Si el usuario no entrega un periodo de tiempo, usa ask_human para pedirlo."
            "Si es que Sin data disponible, pidele al usuario otro periodo de tiempo."
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
    # If tool call is asking Human, we return that node
    # You could also add logic here to let some system know that there's something that requires Human input
    # For example, send a slack message, etc
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    # Otherwise if there is, we continue
    else:
        return "action"


workflow = StateGraph(GraphState)

# Define the three nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node("ask_human", ask_human)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# After we get back the human response, we go back to the agent
workflow.add_edge("ask_human", "agent")


memory = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
# We add a breakpoint BEFORE the `ask_human` node so it never executes
app = workflow.compile(checkpointer=memory)

# display(Image(app.get_graph().draw_mermaid_png()))

config = {"configurable": {"thread_id": "2"}}
for event in app.stream(
    {
        "messages": [
            (
                "user",
                "dame el reporte general de oficinas",
            )
        ]
    },
    config,
    stream_mode="values",
):
    event["messages"][-1].pretty_print()

# util para condicional input tipo command/resume. if ('ask_human',)
print(app.get_state(config).next)
# %%
for event in app.stream(Command(resume="ultimo mes"), config, stream_mode="values"):
    event["messages"][-1].pretty_print()
# %%
config = {"configurable": {"thread_id": "2"}}
for event in app.stream(
    {
        "messages": [
            (
                "user",
                "dame el ranking de ejecutivos",
            )
        ]
    },
    config,
    stream_mode="values",
):
    event["messages"][-1].pretty_print()

# util para condicional input tipo command/resume. if ('ask_human',)
print(app.get_state(config).next)
# %%
for event in app.stream(Command(resume="ultimo mes"), config, stream_mode="values"):
    event["messages"][-1].pretty_print()
