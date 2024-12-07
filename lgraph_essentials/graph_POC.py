# %%
from langchain_community.tools import TavilySearchResults
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

tavily_search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=False,
)
# tavily_search.invoke("what is optknock")
tools = [tavily_search]
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(
    content="""
Eres un asistente que puede buscar en internet. 
Antes de buscar has un plan de cada búsqueda individual para responder la pregunta del usuario"""
)


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
# %%
messages = HumanMessage(content="que es optknock? y que es una levadura?")
messages = graph.invoke({"messages": messages}, config)
for m in messages["messages"]:
    m.pretty_print()
