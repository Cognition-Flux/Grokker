# %%
import json
import os
from typing import Annotated, Dict, List, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv(override=True)


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


class State(MessagesState):
    oficinas_seleccionadas: set = set()
    contexto: str = ""


def transferir_a_react() -> Command:
    """Llamar esto para transferir el control al agente ReAct"""
    return Command(goto="ReAct")


tools = [transferir_a_react]

llm = get_llm()
llm_with_tools = llm.bind_tools(tools)


def llamar_llm(
    state: State,
    config: RunnableConfig,
):
    # Obtener el contexto procesado
    contexto = state.get("contexto", "No hay elementos seleccionados en contexto")

    system_prompt = SystemMessage(
        content=(
            f"{contexto}\n\n"
            "Si el usuario pregunta por las oficinas seleccionadas, responde "
            "directamente al usuario con las oficinas seleccionadas. Si NO, pasale el "
            "requimiento al agente ReAct."
        )
    )
    response = llm_with_tools.invoke([system_prompt] + state["messages"], config)

    if len(response.tool_calls) > 0:
        return Command(goto="ReAct")
    else:
        return Command(goto=END, update={"messages": [response]})


def parse_input_and_update_state(state: State) -> dict:
    try:
        # Obtener el último mensaje
        last_message = state["messages"][-1]
        # Parsear el contenido como diccionario
        content = eval(last_message.content)

        # Obtener el nuevo set de oficinas
        new_oficinas = set(content.get("oficinas_seleccionadas", []))
        # Obtener el set actual de oficinas
        current_oficinas = state.get("oficinas_seleccionadas", set())

        # Crear nuevo mensaje solo con el contenido del mensaje
        new_message = HumanMessage(content=content.get("mensaje", ""))

        # Crear el estado actualizado
        updated_state = {
            "messages": [RemoveMessage(id=last_message.id), new_message],
        }

        print("\n=== Verificación de cambios en oficinas ===")
        if new_oficinas != current_oficinas:
            # Oficinas agregadas
            agregadas = new_oficinas - current_oficinas
            if agregadas:
                print(f"Oficinas agregadas: {', '.join(sorted(agregadas))}")

            # Oficinas eliminadas
            eliminadas = current_oficinas - new_oficinas
            if eliminadas:
                print(f"Oficinas eliminadas: {', '.join(sorted(eliminadas))}")

            updated_state["oficinas_seleccionadas"] = new_oficinas
            updated_state["contexto"] = format_oficinas_context(
                {"oficinas_seleccionadas": new_oficinas}
            )
        else:
            print("No hay cambios en las oficinas seleccionadas")
            print(f"Oficinas actuales: {', '.join(sorted(current_oficinas))}")

        print("=================================\n")
        return updated_state

    except Exception as e:
        print(f"Error parsing input: {e}")
        return {}


def format_oficinas_context(state_values: dict) -> str:
    oficinas = state_values.get("oficinas_seleccionadas")

    if not oficinas:
        return "No hay oficinas en contexto"
    # Convertir set a lista ordenada para consistencia
    oficinas_list = sorted(oficinas)

    if len(oficinas_list) == 1:
        return f"Oficina en contexto: {oficinas_list[0]}"

    # Unir todas las oficinas con comas y "y" para la última
    oficinas_str = ", ".join(oficinas_list[:-1]) + " y " + oficinas_list[-1]
    return f"Elemento seleccionados en contexto: {oficinas_str}"


def react_agent(state: State):
    """
    Mock function for ReAct agent node that processes the request and returns a response.
    """
    # Obtener el último mensaje
    last_message = state["messages"][-1]

    # Obtener el contexto actual
    contexto = state.get("contexto", "No hay elementos seleccionados en contexto")

    # Crear una respuesta que incluya el contexto
    response = HumanMessage(
        content=f"ReAct Agent procesó el mensaje: '{last_message.content}' con el contexto: {contexto}"
    )

    return {
        "messages": [response]
    }  # Command(goto=END, update={"messages": [response]})


def prompt_generator(state: State):
    """
    Mock function for prompt generator node that enhances the request with additional context.
    """
    # Obtener el último mensaje
    last_message = state["messages"][-1]

    # Obtener el contexto actual
    contexto = state.get("contexto", "No hay elementos seleccionados en contexto")

    # Crear un prompt enriquecido que combina el mensaje original y el contexto
    enhanced_prompt = HumanMessage(
        content=f"[CONTEXTO]: {contexto}\n[CONSULTA ORIGINAL]: {last_message.content}\n"
        "Por favor, procesa esta consulta considerando el contexto proporcionado."
    )

    return {"messages": [enhanced_prompt]}


# Actualizar la definición del grafo
workflow = StateGraph(State)
workflow.add_node("parse_input", parse_input_and_update_state)
workflow.add_node("supervisor", llamar_llm)
workflow.add_node("prompt_generator", prompt_generator)  # Agregar el nuevo nodo
workflow.add_node("ReAct", react_agent)

# Actualizar las conexiones del grafo
workflow.add_edge(START, "parse_input")
workflow.add_edge("parse_input", "supervisor")
workflow.add_edge("supervisor", "prompt_generator")  # Supervisor -> Prompt Generator
workflow.add_edge("prompt_generator", "ReAct")  # Prompt Generator -> ReAct
workflow.add_edge("supervisor", END)
workflow.add_edge("ReAct", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))


# %%
config = {"configurable": {"thread_id": "1"}}

input_message = HumanMessage(
    content="{'oficinas_seleccionadas': ['oficina_1', 'oficina_2'], 'mensaje': 'hola'}"
)
output = graph.invoke({"messages": [input_message]}, config)

for m in output["messages"][-1:]:
    m.pretty_print()

# %%

input_message = HumanMessage(
    content="{'oficinas_seleccionadas': ['oficina_1', 'oficina_2'], 'mensaje': 'que está seleccionado??'}"
)
output = graph.invoke({"messages": [input_message]}, config)

for m in output["messages"][-1:]:
    m.pretty_print()
# %%


# %%
######------Agente ReAct con herramientas------######


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def obtener_contexto_de_la_atencion() -> str:
    """
    Obtiene el contexto de la atención del cliente.
    """
    return "El cliente requiere que la respuesta use jerga de biología"


@tool
def multiply(a: float, b: float) -> float:
    """Multiplica dos números enteros.

    Args:
        a (float): Primer número a multiplicar
        b (float): Segundo número a multiplicar

    Returns:
        int: El producto de a y b
    """
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Suma dos números enteros.

    Args:
        a (float): Primer número a sumar
        b (float): Segundo número a sumar

    Returns:
        int: La suma de a y b
    """
    return a + b


@tool
def divide(a: float, b: float) -> float:
    """Divide dos números enteros.

    Args:
        a (float): Numerador
        b (float): Denominador

    Returns:
        float: El resultado de dividir a entre b
    """
    return a / b


def tool_node(state: AgentState) -> dict[str, list[ToolMessage]]:
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


tools = [multiply, add, divide]
tools_by_name = {tool.name: tool for tool in tools}

llm = AzureChatOpenAI(
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


llm_with_tools = get_llm().bind_tools(tools)


def call_model(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str, List[BaseMessage]]:
    system_prompt = SystemMessage(
        content=(
            "Planifica en detalle y ejecuta todas las acciones necesarias"
            "para responder a la pregunta del usuario. "
            "Tu respuesta Final debe ser breve y coherente con la pregunta del usuario."
        )
    )
    response = llm_with_tools.invoke([system_prompt] + state["messages"], config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


ReAct = StateGraph(AgentState)

ReAct.add_node("agent", call_model)
ReAct.add_node("tools", tool_node)

ReAct.set_entry_point("agent")

ReAct.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

ReAct.add_edge("tools", "agent")

memory = MemorySaver()
graph = ReAct.compile(checkpointer=memory)

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Could not display graph visualization: {e}")


# %%


# %%
def print_stream(stream: list) -> None:
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


thread_id = "0"
config = {"configurable": {"thread_id": thread_id}}
inputs = {
    "messages": [
        (
            "user",
            "multiplica 7789679 por 9939, el resultado lo divides por 1089711"
            " y lo restas con 67823. Este resultado debes elevarlo al cuadrado (multiplicarlo por si mismo) "
            "y luego dividir por el número inicial. Mostrar el resultado con dos decimales",
        )
    ]
}
print_stream(graph.stream(inputs, config, stream_mode="values"))
# %%
inputs = {
    "messages": [
        (
            "user",
            "cual fue el resultado de la operacion anterior?",
        )
    ]
}
print_stream(graph.stream(inputs, config, stream_mode="values"))
