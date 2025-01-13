# %%
####
# Set working directory to file location
# file_path = Path(__file__).resolve()
# os.chdir(file_path.parent)
# sys.path.append(str(file_path.parent.parent))
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
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


class CustomGraphState(MessagesState):
    oficinas: list[str] = []
    contexto: SystemMessage = SystemMessage(content="")
    messages: Annotated[List[BaseMessage], add_messages]


def safely_remove_messages(
    state: CustomGraphState, keep_last_n: int = 14
) -> List[BaseMessage]:
    """
    Revisa los mensajes del estado y determina cuáles pueden ser removidos de forma segura.

    Args:
        state: Estado actual del grafo

    Returns:
        Lista de mensajes que pueden ser removidos
    """
    messages = state["messages"]
    if len(messages) <= keep_last_n:
        return []

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
    messages_to_remove = messages[:-keep_last_n]

    # Verify we're not breaking any tool call chains
    for msg in messages_to_remove:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["id"] in tool_call_ids_responded:
                    # Don't remove messages that have corresponding tool responses
                    return []
        if isinstance(msg, ToolMessage):
            if msg.tool_call_id in tool_call_ids_seen:
                # Don't remove tool responses that have corresponding tool calls
                return []

    return messages_to_remove


def validate_message_chain(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Valida y filtra la cadena de mensajes para asegurar que todas las llamadas a herramientas
    tengan sus respectivas respuestas.

    Args:
        messages: Lista de mensajes a validar

    Returns:
        Lista filtrada de mensajes
    """
    filtered_messages = []
    tool_call_ids_seen = set()
    tool_call_ids_responded = set()

    for msg in messages:
        if isinstance(msg, RemoveMessage):
            continue

        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_call_ids_seen.add(tool_call["id"])

        if isinstance(msg, ToolMessage):
            tool_call_ids_responded.add(msg.tool_call_id)

        filtered_messages.append(msg)

    # Check if we have any unresponded tool calls
    missing_responses = tool_call_ids_seen - tool_call_ids_responded
    if missing_responses:
        print(f"Warning: Missing tool responses for: {missing_responses}")
        # Only keep messages up to the last complete tool exchange
        filtered_messages = [
            msg
            for msg in filtered_messages
            if not (
                isinstance(msg, AIMessage)
                and msg.tool_calls
                and any(call["id"] in missing_responses for call in msg.tool_calls)
            )
        ]

    return filtered_messages
