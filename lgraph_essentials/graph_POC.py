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
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from lgraph_essentials.llm import llm
import numpy as np
from datetime import datetime, timedelta
import json


def create_random_dataframe(num_rows=100, seed=None):
    """
    Create a random DataFrame with various data types.

    Parameters:
    num_rows (int): Number of rows to generate (default: 100)
    seed (int): Random seed for reproducibility (default: None)

    Returns:
    pd.DataFrame: DataFrame with random data
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(num_rows)]

    # Generate random categorical data
    categories = ["A", "B", "C", "D"]
    departments = ["Sales", "Marketing", "Engineering", "HR"]

    # Create dictionary of data
    data = {
        "date": dates,
        "category": np.random.choice(categories, num_rows),
        "department": np.random.choice(departments, num_rows),
        "value": np.random.normal(100, 25, num_rows),  # Normal distribution
        "quantity": np.random.randint(1, 100, num_rows),
        "is_active": np.random.choice([True, False], num_rows),
        "score": np.random.uniform(0, 1, num_rows),  # Uniform distribution
        "priority": np.random.choice(["Low", "Medium", "High"], num_rows),
        "cost": np.random.exponential(500, num_rows),  # Exponential distribution
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add some formatting
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df["value"].round(2)
    df["score"] = df["score"].round(3)
    df["cost"] = df["cost"].round(2)

    return df


# Example usage
df = create_random_dataframe(num_rows=5, seed=42)

pandas_dataframe_agent = create_pandas_dataframe_agent(
    llm, df, agent_type="tool-calling", verbose=False, allow_dangerous_code=True
)


def llamar_agente_pandas(question: str) -> str:
    """llamar_agente_pandas: Agente analista
    Usar para analizar los datos del usuario.
    solo hay q pasar la consulta para obtener una respuesta.
    question: str, consulta.

    return: respuesta
    """
    response = pandas_dataframe_agent.invoke(question)
    return str(response["output"][0]["text"])


tavily_search = TavilySearchResults(
    max_results=5,
    search_depth="basic",  # "advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    include_domains=["https://pubmed.ncbi.nlm.nih.gov/"],
    verbose=True,
)
tools = [tavily_search, llamar_agente_pandas]
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(
    content="""
                        Puedes buscar en internet sólo si es necesario.
                        También puedes consultar los datos del usuario, 

                        sea detallado cuando el usuario pide analizar los datos (llamar_agente_pandas)
                        Nunca hay auq cargar datos, sólo directamente usar llamar_agente_pandas
                        Siempre sigue el hilo de la conversación.
                        """
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
# display(Image(graph.get_graph().draw_mermaid_png()))

# %%
node_to_stream = "assistant"

config = {"configurable": {"thread_id": "5"}}
input_message = HumanMessage(content="dame un resumen de mis datos")
async for event in graph.astream_events(
    {"messages": [input_message]}, config, version="v2"
):
    # Get chat model tokens from a particular node
    if (
        event["event"] == "on_chat_model_stream"
        and event["metadata"].get("langgraph_node", "") == node_to_stream
    ):
        data = event["data"]
        if data["chunk"].content:
            print(data["chunk"].content[0].get("text", ""), end="|**")

            # print(
            #     json.loads(data["chunk"].content[0].replace("'", '"'))["text"], end="|"
            # )


# %%
messages = HumanMessage(content="dame un resumen de mis datos")
messages = graph.invoke({"messages": messages}, config)
for m in messages["messages"]:
    m.pretty_print()
# %%
input_message = HumanMessage(content="dame un resumen de mis datos")
async for event in graph.astream_events(
    {"messages": [input_message]}, config, version="v2"
):
    print(
        f"Node: {event['metadata'].get('langgraph_node','')}. Type: {event['event']}. Name: {event['name']}"
    )
# %%
node_to_stream = "assistant"
config = {"configurable": {"thread_id": "4"}}
input_message = HumanMessage(content="dame un resumen de mis datos")
async for event in graph.astream_events(
    {"messages": [input_message]}, config, version="v2"
):
    # Get chat model tokens from a particular node
    if (
        event["event"] == "on_chat_model_stream"
        and event["metadata"].get("langgraph_node", "") == node_to_stream
    ):
        print(event["data"])
# %%
