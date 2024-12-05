# %%
from typing import TypedDict
import random
from typing import Literal
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    graph_state: str


def node_1(state):
    print("--Node 1--")
    return {"graph_state": state["graph_state"] + "I am"}


def node_2(state):
    print("--Node 2--")
    return {"graph_state": state["graph_state"] + "happy"}


def node_3(state):
    print("--Node 3--")
    return {"graph_state": state["graph_state"] + "happy"}


def decide_mood(state) -> Literal["node_1", "node_2"]:
    user_input = state["graph_state"]

    if random.random() < 0.5:
        return "node_2"
    return "node_3"
