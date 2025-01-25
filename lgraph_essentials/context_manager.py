# %%
import os
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.message import add_messages

load_dotenv(override=True)
print(
    f"""
LANGCHAIN_API_KEY: {os.environ["LANGCHAIN_API_KEY"]}
LANGCHAIN_TRACING_V2: {os.environ["LANGCHAIN_TRACING_V2"]}
LANGCHAIN_PROJECT: {os.environ["LANGCHAIN_PROJECT"]}
"""
)


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


llm = get_llm()


class GraphState(MessagesState):
    contexto: Annotated[list[str], add_messages]
    oficinas: list[str]


def filter_messages(state: GraphState) -> GraphState:
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages}


def context_node(state: GraphState) -> GraphState:
    last_message = state["messages"][-1]
    pattern = r"Considera las oficinas \[(.*?)\]"
    match = re.search(pattern, last_message.content)

    # Obtener el mensaje sin el patrón
    mensaje_original = last_message.content
    mensaje_limpio = re.sub(pattern, "", mensaje_original).strip()

    if match:
        print("---------------Se encontró el patrón de lista de oficinas--------------")
        print("Mensaje original:", mensaje_original)  # Debug
        print("Mensaje limpio:", mensaje_limpio)  # Debug

        # Extraer el contenido entre corchetes y convertirlo en lista
        oficinas_str = match.group(1)
        # Dividir por comas y limpiar espacios y comillas
        oficinas_list = [
            office.strip().strip("'") for office in oficinas_str.split(",")
        ]
        print(f"---------------{oficinas_list=}")
    else:
        print(
            "---------------NO se encontró el patrón de lista de oficinas--------------"
        )

        oficinas_list = []

    if len(oficinas_list) > 0:
        lista_nueva_oficinas = oficinas_list
        lista_actual_oficinas = state.get("oficinas", [])
        if set(lista_nueva_oficinas) != set(lista_actual_oficinas):
            nuevo_contexto: str = "lista de oficinas actualizada"
        else:
            nuevo_contexto: str = "lista de oficinas no ha cambiado"
    else:
        return {
            "contexto": SystemMessage(content="No hay oficinas", id="nuevo_contexto"),
            "oficinas": oficinas_list,
            "messages": [RemoveMessage(id=m.id) for m in state["messages"]],
        }

    return {
        "contexto": SystemMessage(content=nuevo_contexto, id="nuevo_contexto"),
        "oficinas": oficinas_list,
        "messages": [
            RemoveMessage(id=last_message.id),
            HumanMessage(content=mensaje_limpio),
        ],
    }


def chat_model_node(state: GraphState) -> GraphState:
    return {
        "messages": llm.invoke(
            state["messages"]
            + [
                SystemMessage(
                    content="solo responde informando esto:"
                    + state["contexto"][-1].content
                )
            ]
        )
    }


builder = StateGraph(GraphState)
builder.add_node("filter_messages", filter_messages)
builder.add_node("chat_model_node", chat_model_node)
builder.add_node("context_node", context_node)
builder.add_edge(START, "context_node")  #
builder.add_edge("context_node", "filter_messages")  #
builder.add_edge("filter_messages", "chat_model_node")
builder.add_edge("chat_model_node", END)  #
memory = MemorySaver()

graph = builder.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))

config = {"configurable": {"thread_id": "2"}}
# %%
input_message = HumanMessage(
    content=("Considera las oficinas ['009 - Vitacura EDW'] ..responde en inglés")
)

for event in graph.stream(
    {"messages": [input_message]},
    config,
    stream_mode="values",
):
    for m in event["messages"]:
        print(m.pretty_print())

# %%
