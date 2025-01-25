# %%
from operator import add
from typing import Annotated, TypedDict

from IPython.display import Image, display
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

init_messages = [
    SystemMessage(content="Hola, ¿cómo estás?", id="1"),
    SystemMessage(content="Estoy bien, gracias", id="2"),
]
new_messages = [SystemMessage(content="chao", id="1")]

add_messages(init_messages, new_messages)


# %%
class State(TypedDict):
    oficinas_seleccionadas: list[str]
    contexto: str
    mensajes: Annotated[list[str], add]


def node_1(state: State) -> TypedDict:
    print("---Node 1---")
    return {"mensajes": [state["mensajes"][-1] + "ffds"]}


builder = StateGraph(State)
builder.add_node("node_1", node_1)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

graph = builder.compile()
# display(Image(graph.get_graph().draw_mermaid_png()))
# %%
for event in graph.stream({"mensajes": ["hola"]}, stream_mode="values"):
    print(event)


# %%
# Custom reducer
def custom_reducer(left: list[str] | None, right: list[str] | None) -> list[str]:
    if left is None:
        left = []
    if right is None:
        right = []

    if left == right:
        return left
    return left + right


class CustomReducerState(TypedDict):
    mensajes: Annotated[list[str], custom_reducer]


def node_1(state: CustomReducerState) -> TypedDict:
    print("---Node 1---")
    return state  # {"mensajes": [state["mensajes"][-1] + "ffds"]}


builder = StateGraph(CustomReducerState)
builder.add_node("node_1", node_1)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

graph = builder.compile()
# display(Image(graph.get_graph().draw_mermaid_png()))
# %%
for event in graph.stream({"mensajes": None}, stream_mode="values"):
    print(event)
# %%
