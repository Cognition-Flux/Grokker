# %%
"""Módulo para gestionar la memoria de la aplicación."""  # Añadir docstring al módulo

import inspect
import os
import sqlite3

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0)

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "memory.db")


# Initialize database connection in a function instead of global
def get_db_connection() -> sqlite3.Connection:
    """Get a new database connection."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


conn = get_db_connection()
memory = SqliteSaver(conn)

print(inspect.getsource(SqliteSaver))


# %%
class State(MessagesState):
    summary: str


def call_model(state: State) -> State:

    # Get summary if it exists
    summary = state.get("summary", "")
    if summary:
        system_message = f"The summary of the conversation: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    response = model.invoke(messages)
    return {"messages": response}


def summarize_conversation(state: State) -> State:

    summary = state.get("summary", "")
    if summary:

        summary_message = (
            f"The summary of the conversation to date: {summary} \n\n"
            "Extend the summary by taking into account the new messages above"
        )
    else:
        summary_message = "Create a summary of the conversation above"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    # Delete all but the 2 most recent messages
    two_last_messages = [RemoveMessage(id=m.i) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": two_last_messages}


def should_continue(state: State) -> str:
    messages = state["messages"]
    if len(messages) > 6:
        return "summarize_conversation"
    return END


workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)


graph = workflow.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))
# %%
config = {"configurable": {"thread_id": "1"}}
input_message = HumanMessage(content="hola, soy Ale")
output = graph.invoke({"messages": [input_message]}, config)
for m in output["messages"][-1:]:
    m.pretty_print()
# %%

config = {"configurable": {"thread_id": "1"}}
print(graph.get_state(config))
input_message = HumanMessage(content="cuales mi nombre?")
output = graph.invoke({"messages": [input_message]}, config)
for m in output["messages"][-1:]:
    m.pretty_print()
