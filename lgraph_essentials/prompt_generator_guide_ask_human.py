# %%
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Sequence, TypedDict

import yaml
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import TavilySearchResults
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
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

from lgraph_essentials.groker_dev_utils import get_llm  # CustomGraphState,
from lgraph_essentials.groker_dev_utils import (
    safely_remove_messages,
    validate_message_chain,
)
from ttp_agentic.tools.ranking_ejecutivos import executive_ranking_tool
from ttp_agentic.tools.registros_disponibles import rango_registros_disponibles
from ttp_agentic.tools.reporte_detallado_por_ejecutivo import (
    tool_reporte_detallado_por_ejecutivo,
)
from ttp_agentic.tools.reporte_general_de_oficinas import (
    tool_reporte_extenso_de_oficinas,
)

load_dotenv(override=True)

search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    # include_domains=[...],
    # exclude_domains=[...],
    # name="...",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)


@tool
def make_prompt(sub_prompt: str):
    """Elabora un prompt que será usado por otro agente"""
    return f"{sub_prompt}"


tool_node_prompt = ToolNode([make_prompt])
tool_node_internet = ToolNode([search_tool])

# Set up the model


# We are going "bind" all tools to the model
# We have the ACTUAL tools from above, but we also need a mock tool to ask a human
# Since `bind_tools` takes in tools but also just tool definitions,
# We can define a tool definition for `ask_human`
class AskHuman(BaseModel):
    """Ask the human a question"""

    question: str


# Define nodes and conditional edges


# Define the function that determines whether to continue or not
def should_continue(state) -> Literal[END, "ask_human", "tool_node_prompt"]:
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
        return "tool_node_prompt"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]

    system_promtp = SystemMessage(
        content=(
            "debes ver si el usuario ha proporcionado algun año"
            "si no ha proporcionado un año, debes preguntarle usando AskHuman"
            "y luego debes hacer un prompt (tool make_prompt) que solicite una busqueda en internet de hechos científicos relevantes sobre la fecha, este prompt será usado otro agente para buscar en internet"
        )
    )
    model = ChatAnthropic(model="claude-3-5-sonnet-latest")
    model = model.bind_tools([make_prompt] + [AskHuman])
    response = model.invoke([system_promtp] + messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def search_internet(state) -> Command[Literal["tool_node_internet", END]]:
    messages = state["messages"]
    # extraer el prompt hecho por el agente anterior
    prompt = messages[-1].content
    print(f"---------SUB-PROMPT: {prompt=}")
    model = ChatAnthropic(model="claude-3-5-sonnet-latest")
    model = model.bind_tools([search_tool])

    system_promtp = SystemMessage(
        content=(
            "Puedes hacer una busqueda en internet, y con los resultados que obtengas, debes hacer un resumen de los resultados"
        )
    )

    response = model.invoke([system_promtp] + [prompt])
    print(f"---------RESPONSE search_internet: {response=}")

    if len(response.tool_calls) > 0:
        next_node = "tool_node_internet"
    else:
        next_node = END

    return Command(goto=next_node, update={"messages": [response]})


# We define a fake node to ask the human
def ask_human(state):
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    location = interrupt("Por favor, proporciona información")
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": location}]
    return {"messages": tool_message}


# Build the graph

# Define a new graph
workflow = StateGraph(MessagesState)

# Define the three nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tool_node_prompt", tool_node_prompt)
workflow.add_node("ask_human", ask_human)
workflow.add_node("internet_agent", search_internet)
workflow.add_node("tool_node_internet", tool_node_internet)
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
# workflow.add_edge("action", "agent")

# After we get back the human response, we go back to the agent
workflow.add_edge("ask_human", "agent")
workflow.add_edge("tool_node_prompt", "internet_agent")
workflow.add_edge("tool_node_internet", "internet_agent")
workflow.add_edge("internet_agent", END)
# Set up memory

memory = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
# We add a breakpoint BEFORE the `ask_human` node so it never executes
app = workflow.compile(checkpointer=memory)

display(Image(app.get_graph().draw_mermaid_png()))


def run_graph(graph: CompiledStateGraph, input_message: str = "hola") -> None:
    config = {"configurable": {"thread_id": "1"}}
    print(f"## INICIO: Próximo paso del grafo: {graph.get_state(config).next}")
    for event in graph.stream(
        {"messages": [HumanMessage(content=input_message)]},
        config,
        stream_mode="updates",
    ):
        mss = next(iter(event.values()))
        print(f"mensages internos: type: {type(mss)}, mensajes: {mss}")
        if isinstance(mss, dict):
            if mss.get("messages"):
                if isinstance(mss.get("messages")[0], BaseMessage):
                    mss.get("messages")[0].pretty_print()
        else:
            pass
    print(f"## FINAL: Próximo paso del grafo: {graph.get_state(config).next}")


def resume_graph(graph: CompiledStateGraph, input_message: str = "1980") -> None:
    config = {"configurable": {"thread_id": "1"}}
    print(f"## INICIO: Próximo paso del grafo: {app.get_state(config).next}")
    for event in graph.stream(
        Command(resume=input_message),
        config,
        stream_mode="updates",
    ):
        mss = next(iter(event.values()))
        print(f"mensages internos: type: {type(mss)}, mensajes: {mss}")
        if mss.get("messages"):
            if isinstance(mss.get("messages")[0], BaseMessage):
                mss.get("messages")[0].pretty_print()

    print(f"## FINAL: Próximo paso del grafo: {graph.get_state(config).next}")


if __name__ == "__main__":

    run_graph(
        app,
        "hola",
    )
    # %%
    resume_graph(app, "1980")
