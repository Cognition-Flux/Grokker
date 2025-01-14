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


class CustomGraphState(MessagesState):
    oficinas: list[str] = []
    contexto: SystemMessage = SystemMessage(content="")
    guidance: str = ""  # SystemMessage = SystemMessage(content="")
    messages: Annotated[List[BaseMessage], add_messages]


class AskHuman(BaseModel):
    """AskHuman
    el agente debe solicitar directamente aclaraciones/información al usuario/humano
    """

    question_for_human: str


@tool
def make_prompt(internal_prompt: str) -> str:
    """Elabora un prompt que será pasado a otro agente"""
    return f"{internal_prompt}"


tool_node_prompt = ToolNode([make_prompt])

llm_guide_agent = get_llm().bind_tools([make_prompt] + [AskHuman])


def clean_messages(state: CustomGraphState) -> CustomGraphState:
    # messages_to_remove = safely_remove_messages(state)
    # return {"messages": [RemoveMessage(id=m.id) for m in messages_to_remove]}
    return state


def guidance_agent(
    state: CustomGraphState,
) -> Command[Literal["ask_human", "tool_node_prompt"]]:

    prompt_for_guidance = SystemMessage(
        content=(
            "Debes ver si el usuario ha proporcionado algun año o periodo de tiempo. "
            "si no ha proporcionado un año o periodo de tiempo, debes preguntarle usando AskHuman"
            "Despues usa la tool make_prompt para crear un prompt que explique lo que el usuario está pidiendo basado en el historial de la conversación"
        )
    )

    mensajes = state["messages"] + [prompt_for_guidance]

    response = llm_guide_agent.invoke(mensajes)

    if len(response.tool_calls) > 0:
        if response.tool_calls[0]["name"] == "AskHuman":
            next_node = "ask_human"
        else:
            next_node = "tool_node_prompt"
    # else:
    #     next_node = END

    return Command(goto=next_node, update={"messages": [response]})


def ask_human(state: CustomGraphState):
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    intervencion_humana = interrupt("Por favor, proporciona información")
    tool_message = [
        {"tool_call_id": tool_call_id, "type": "tool", "content": intervencion_humana}
    ]
    return {"messages": tool_message}


workflow = StateGraph(CustomGraphState)
workflow.add_node("clean_messages", clean_messages)
workflow.add_node("guidance_agent", guidance_agent)
workflow.add_node("tool_node_prompt", tool_node_prompt)
workflow.add_node("ask_human", ask_human)


workflow.add_edge(START, "clean_messages")
workflow.add_edge("clean_messages", "guidance_agent")
workflow.add_edge("ask_human", "guidance_agent")

if os.getenv("DEV_CHECKPOINTER"):

    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
else:
    graph = workflow.compile()

display(Image(graph.get_graph().draw_mermaid_png()))


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
                    if hasattr(mss.get("messages")[0], "tool_calls"):
                        if mss.get("messages")[0].tool_calls:
                            print(f"tool_calls: {mss.get('messages')[0].tool_calls=}")
                            print(
                                f"tool_calls: {mss.get('messages')[0].tool_calls[0]['name']=}"
                            )
                        else:
                            print(
                                f"No hay tool_calls {mss.get('messages')[0].tool_calls=}"
                            )

                    mss.get("messages")[0].pretty_print()
            pass

    print(f"## FINAL: Próximo paso del grafo: {graph.get_state(config).next}")


def resume_graph(graph: CompiledStateGraph, input_message: str = "1980") -> None:
    config = {"configurable": {"thread_id": "1"}}
    print(f"## INICIO: Próximo paso del grafo: {graph.get_state(config).next}")
    for event in graph.stream(
        Command(resume=input_message),
        config,
        stream_mode="updates",
    ):
        mss = next(iter(event.values()))
        print(f"mensages internos: type: {type(mss)}, mensajes: {mss}")
        if isinstance(mss, dict):
            if mss.get("messages"):
                if isinstance(mss.get("messages")[0], BaseMessage):
                    if hasattr(mss.get("messages")[0], "tool_calls"):
                        if mss.get("messages")[0].tool_calls:
                            print(f"tool_calls: {mss.get('messages')[0].tool_calls=}")
                            print(
                                f"tool_calls: {mss.get('messages')[0].tool_calls[0]['name']=}"
                            )
                        else:
                            print(
                                f"No hay tool_calls {mss.get('messages')[0].tool_calls=}"
                            )

                    mss.get("messages")[0].pretty_print()
            pass

    print(f"## FINAL: Próximo paso del grafo: {graph.get_state(config).next}")


run_graph(
    graph,
    "hola dame los datos de los experimentos",
)
# %%
resume_graph(graph, "los hechos con ratones en 1980")
