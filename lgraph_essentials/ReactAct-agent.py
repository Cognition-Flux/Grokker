# %%
import json
import os
from typing import Annotated, Dict, List, Sequence, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

load_dotenv(override=True)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def multiply(a: float, b: float) -> float:
    """Multiplica dos números enteros.

    Args:
        a (float): Primer número a multiplicar
        b (float): Segundo número a multiplicar

    Returns:
        int: El producto de a y b
    """
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Suma dos números enteros.

    Args:
        a (float): Primer número a sumar
        b (float): Segundo número a sumar

    Returns:
        int: La suma de a y b
    """
    return a + b


@tool
def divide(a: float, b: float) -> float:
    """Divide dos números enteros.

    Args:
        a (float): Numerador
        b (float): Denominador

    Returns:
        float: El resultado de dividir a entre b
    """
    return a / b


def tool_node(state: AgentState) -> dict[str, list[ToolMessage]]:
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


tools = [multiply, add, divide]
tools_by_name = {tool.name: tool for tool in tools}

llm = AzureChatOpenAI(
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

llm_with_tools = llm.bind_tools(tools)


def call_model(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str, List[BaseMessage]]:
    system_prompt = SystemMessage(
        content=(
            "Planifica en detalle y ejecuta todas las acciones necesarias"
            "para responder a la pregunta del usuario. "
            "Tu respuesta Final debe ser breve y coherente con la pregunta del usuario."
        )
    )
    response = llm_with_tools.invoke([system_prompt] + state["messages"], config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

workflow.add_edge("tools", "agent")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception as e:
#     print(f"Could not display graph visualization: {e}")


def print_stream(stream: list) -> None:
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


thread_id = "0"
config = {"configurable": {"thread_id": thread_id}}
inputs = {
    "messages": [
        (
            "user",
            "multiplica 7789679 por 9939, el resultado lo divides por 1089711"
            " y lo restas con 67823. Este resultado debes elevarlo al cuadrado (multiplicarlo por si mismo) "
            "y luego dividir por el número inicial. Mostrar el resultado con dos decimales",
        )
    ]
}
print_stream(graph.stream(inputs, config, stream_mode="values"))
# %%
inputs = {
    "messages": [
        (
            "user",
            "cual fue el resultado de la operacion anterior?",
        )
    ]
}
print_stream(graph.stream(inputs, config, stream_mode="values"))
