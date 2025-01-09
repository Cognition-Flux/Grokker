# %%
import os
import re
import sys
from pathlib import Path

####
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


def safely_remove_messages(state: CustomGraphState) -> List[BaseMessage]:

    messages = state["messages"]
    if len(messages) <= 10:
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
    return messages_to_remove
