# %%
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.message import add_messages

load_dotenv(override=True)
messages = [AIMessage(content="Hola", id="1", name="ia")]
messages.append(HumanMessage(content="que eres", id="2", name="user"))
for m in messages:
    print(m.pretty_print())


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
llm.invoke(messages)


# %%
def chat_model_node(state: MessagesState) -> MessagesState:
    return {"messages": llm.invoke(state["messages"])}


builder = StateGraph(MessagesState)
builder.add_node("chat_model_node", chat_model_node)
builder.add_edge(START, "chat_model_node")
builder.add_edge("chat_model_node", END)

graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
# %%
output = graph.invoke({"messages": messages})
for m in output["messages"]:
    print(m.pretty_print())

# %%
from langchain_core.messages import RemoveMessage


def filter_messages(state: MessagesState) -> MessagesState:
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages}


def chat_model_node(state: MessagesState) -> MessagesState:
    return {"messages": llm.invoke(state["messages"])}


builder = StateGraph(MessagesState)
builder.add_node("filter_messages", filter_messages)
builder.add_node("chat_model_node", chat_model_node)
builder.add_edge(START, "filter_messages")
builder.add_edge("filter_messages", "chat_model_node")
builder.add_edge("chat_model_node", END)

graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
# %%
messages = [
    HumanMessage(content="Soy alejandro", id="1", name="user"),
    AIMessage(content="como estas alejandro", id="2", name="ia"),
    HumanMessage(content="que eres", id="3", name="user"),
    AIMessage(content="una IA", id="4", name="ia"),
]
output = graph.invoke({"messages": messages})
for m in output["messages"]:
    print(m.pretty_print())

# %%
