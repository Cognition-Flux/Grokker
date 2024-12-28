# %%
import operator
import os
from typing import Annotated, List, Literal, Tuple, Union

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain import hub
from langchain_core.messages import HumanMessage  # , AIMessage, ToolMessage
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate  # , MessagesPlaceholder
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    """Multiplica dos números enteros.

    Args:
        a (int): Primer número a multiplicar
        b (int): Segundo número a multiplicar

    Returns:
        int: El producto de a y b
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Suma dos números enteros.

    Args:
        a (int): Primer número a sumar
        b (int): Segundo número a sumar

    Returns:
        int: La suma de a y b
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide dos números enteros.

    Args:
        a (int): Numerador
        b (int): Denominador

    Returns:
        float: El resultado de dividir a entre b
    """
    return a / b


tools = [multiply, add, divide]
tools_by_name = {tool.name: tool for tool in tools}

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_retries=5,
    streaming=False,
)
ReAct_system_prompt = """
Debes ejecutar por completo las N-acciones.
El imput de cada tool es un serialized json-formatted string.
    """
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ReAct_system_prompt),
        MessagesPlaceholder("messages"),
    ]
)
agent_executor = create_react_agent(llm, tools, state_modifier=prompt)


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective/request/question,
Siempre debes hacer una planificación lógica sistemática muy detallada paso-a-paso o N-acciones (Chain-of-Thought) sobre como responder/resolver la consulta/pregunta.
Usar estas herramientas para definir las N-acciones:
{tools}
This plan should involve individual tasks, that if executed correctly will yield the Response.
""".format(
                tools="\n".join(
                    [f"{tool.name=}: {tool.description=}" for tool in tools]
                ),
            ),
        ),
        ("placeholder", "{messages}"),
    ]
)
planner_prompt.pretty_print()
planner = planner_prompt | llm.with_structured_output(Plan)


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """
puedes usar estas herramientas:
{tools}
Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Observando los steps pasados update your plan accordingly. If no more steps are needed and you can to respond to user (use Response). Otherwise, just fill out the plan. 
Only add/execute ALL THE steps to the plan that ARE still NEEDed/REQUIRED to be done - do not skip steps. do not repeat previously done steps OF THE PLAN.
"""
)

replanner_prompt = replanner_prompt.partial(
    tools="\n".join([f"{tool.name = }: {tool.description = }" for tool in tools])
)


replanner = replanner_prompt | llm.with_structured_output(Act)


async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"


workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges(
    "replan",
    should_end,
    ["agent", END],
)
app = workflow.compile()
# display(Image(app.get_graph(xray=True).draw_mermaid_png()))

query = (
    "multiplica 7789679 por 9939, el resultado lo divides por 1089711"
    " y lo restas con 67823. Este resultado debes elevarlo al cuadrado (multiplicarlo por si mismo) "
    "y luego dividir por el número inicial. Mostrar el resultado con dos decimales"
)
config = {"recursion_limit": 20}
inputs = {"input": query}
gathered = inputs
async for event in app.astream(inputs, config=config, stream_mode="updates"):
    gathered = gathered | next(iter(event.values()))
    print(event)
