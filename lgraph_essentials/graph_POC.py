# %%
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from lgraph_essentials.llm import llm

from langgraph.graph import MessagesState

from typing import Literal
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, RemoveMessage

tavily_search = TavilySearchResults(
    max_results=5,
    search_depth="basic",  # "advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    include_domains=["https://pubmed.ncbi.nlm.nih.gov/", "https://scholar.google.com/"],
    verbose=True,
)
# tavily_search.invoke("what is optknock")
tools = [tavily_search]
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(
    content="""
Eres un asistente que puede buscar en internet. 
Antes de buscar has un plan de cada búsqueda individual para responder las preguntas del usuario.
Utilice lenguaje técnico especializado.
"""
)


class State(MessagesState):
    summary: str


def assistant(state: State):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# def call_model(state: State):
#     summary = state.get("summary", "")
#     if summary:
#         system_message = f"Summary of conversation earlier:{summary}"
#         messages = [SystemMessage(content=system_message) + state["messages"]]
#     else:
#         messages = state["messages"]

#     response = llm.invoke(messages)
#     return {"messages": response}


builder = StateGraph(State)
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
# %%
##################################################################################
####----------------Summarize conversation-----------------------------------
################################################################################


# Define the logic to call the model
def call_model(state: State):
    # If a summary exists, we add this in as a system message
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = llm.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# We now define the logic for determining whether to end or summarize the conversation
def should_continue(state: State) -> Literal["summarize_conversation", "assistant"]:
    """Return the next node to execute."""
    messages = state["messages"]
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 2:
        return "summarize_conversation"
    # Otherwise we can just end
    return "assistant"


def summarize_conversation(state: State):
    # First, we summarize the conversation
    summary = state.get("summary", "")
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    # We now need to delete messages that we no longer want to show up
    # I will delete all but the last two messages, but you can change this
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


def assistant(state: State):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Define a new graph
workflow = StateGraph(State)

# Define the conversation node and the summarize node
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)
workflow.add_node("assistant", assistant)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `conversation`.
    # This means these are the edges taken after the `conversation` node is called.
    "conversation",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `summarize_conversation` to END.
# This means that after `summarize_conversation` is called, we end.
workflow.add_edge("summarize_conversation", "assistant")
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge("tools", "assistant")

workflow.add_conditional_edges(
    "assistant",
    # si el último mensaje es una tool call -> tools_condition routes to tools
    # si el último mensaje NO es una tool call -> tools_condition routes to END
    tools_condition,
)

# Finally, we compile it!
app = workflow.compile(checkpointer=memory)
display(Image(app.get_graph().draw_mermaid_png()))

# %%

messages = HumanMessage(content="hola")
messages = app.invoke({"messages": messages}, config)
for m in messages["messages"]:
    m.pretty_print()


# %%
def print_update(update):
    for k, v in update.items():
        for m in v["messages"]:
            m.pretty_print()
        if "summary" in v:
            print(v["summary"])


config = {"configurable": {"thread_id": "4"}}
input_message = HumanMessage(content="hola, soy Ale")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)
# %%
input_message = HumanMessage(content="what's my name?")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)
# %%
app.get_state(config).values.get("summary", "")
