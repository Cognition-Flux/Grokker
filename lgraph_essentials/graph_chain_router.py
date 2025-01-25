# %%
from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from lgraph_essentials.llm import llm
from typing import TypedDict, List, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from typing import TypedDict
import random
from typing import Literal
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver

# %%

messages = [AIMessage(content="como te puedo ayudar?", name="Model")]
messages.extend([HumanMessage(content="a investigar sobre bacterias", name="Ale")])
for m in messages:
    m.pretty_print()

result = llm.invoke(messages)
type(result)
result.response_metadata


# %%
def multiplicar(a, b):
    """multiplicar a x b
    a: int
    b: int
    """
    return a * b


llm_with_tools = llm.bind_tools([multiplicar])
tool_call = llm_with_tools.invoke(
    [HumanMessage(content="multiplicar 11 por 77", name="Ale")]
)
tool_call.content


# %%
# class MessagesState(TypedDict):
#     messages: Annotated[List[AnyMessage], add_messages]
class MessagesState(MessagesState):
    # Add any keys needed beyond messages
    pass


def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
# %%
graph.invoke({"messages": HumanMessage(content="hola")})
graph.invoke({"messages": HumanMessage(content="multiplica 4 por 100")})
# %%
# Grafo con tool node
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiplicar]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # si el último mensaje es una tool call -> tools_condition routes to tools
    # si el último mensaje NO es una tool call -> tools_condition routes to END
    tools_condition,
)

builder.add_edge("tools", END)
graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
# %%
messages = HumanMessage(content="mutiplique 345 por 777")
messages = graph.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()


# %%
def sumar(a: float, b: float) -> float:
    """sumar a + b
    a: float
    b: float
    """
    return a + b


def dividir(a: float, b: float) -> float:
    """dividir a / b
    a: float
    b: float
    """
    return a / b


def multiplicar(a: float, b: float) -> float:
    """multiplicar a x b
    a: float
    b: float
    """
    return a * b


tools = [multiplicar, dividir, sumar]
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="tu tienes tools aritmeticas")


def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # si el último mensaje es una tool call -> tools_condition routes to tools
    # si el último mensaje NO es una tool call -> tools_condition routes to END
    tools_condition,
)

builder.add_edge("tools", "assistant")
memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}
graph = builder.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))

messages = HumanMessage(
    content="mutiplique 345 por 777, luego divida el resultado por 7 y finalmente reste 10"
)
messages = graph.invoke({"messages": messages}, config)
for m in messages["messages"]:
    m.pretty_print()
# %%
messages = HumanMessage(content="divida eso por 10000")
messages = graph.invoke({"messages": messages}, config)
for m in messages["messages"]:
    m.pretty_print()
