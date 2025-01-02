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
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from ttp_agentic.tools.registros_disponibles import rango_registros_disponibles

# office_names = [
#     "001 - Huerfanos 740 EDW",
#     "003 - Cauquenes",
#     "004 - Apoquindo EDW",
#     "009 - Vitacura EDW",
# ]
# # Example with days_back
# print(rango_registros_disponibles(office_names))
# # %%

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


def transferir_a_prompt_generator() -> Command:
    """Llamar esto para transferir el control al agente prompt_generator"""
    return Command(goto="prompt_generator")


tools = [transferir_a_prompt_generator]

llm = get_llm()
llm_with_tools = llm.bind_tools(tools)


def llamar_llm(
    state: State,
    config: RunnableConfig,
) -> Command[Literal["prompt_generator", END]]:
    # Obtener el contexto procesado
    contexto = state.get("contexto", "No hay elementos seleccionados en contexto")

    system_prompt = SystemMessage(
        content=(
            "Estos son los registros/datos disponibles para las oficinas seleccionadas:\n"
            f"{contexto}\n\n"
            "Si el usuario pregunta por las oficinas seleccionadas, responde "
            "directamente al usuario y finaliza."
        )
    )
    response = llm_with_tools.invoke([system_prompt] + state["messages"], config)

    if len(response.tool_calls) > 0:
        return Command(goto="prompt_generator")
    else:
        return Command(goto=END, update={"messages": [response]})


def generar_contexto(state: State) -> dict:
    try:
        # Obtener el último mensaje
        last_message = state["messages"][-1]

        # Usar regex para extraer la lista de oficinas
        import re

        pattern = r"Considera las oficinas \[(.*?)\]"
        match = re.search(pattern, last_message.content)

        if match:
            # Extraer el contenido entre corchetes y convertirlo en lista
            oficinas_str = match.group(1)
            # Dividir por comas y limpiar espacios y comillas
            oficinas_list = [
                office.strip().strip("'") for office in oficinas_str.split(",")
            ]
            content = {
                "oficinas_seleccionadas": oficinas_list,
                "mensaje": last_message.content,
            }
        else:
            # Si no encuentra el patrón, intentar el eval original
            content = {}

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
    # oficinas_str = ", ".join(oficinas_list[:-1]) + " y " + oficinas_list[-1]
    return f"Contexto: {rango_registros_disponibles(oficinas_list)}"


def prompt_generator(state: State):
    """
    Prompt-generator agent node that enhances the request with additional context.
    """
    # Obtener el último mensaje
    last_message = state["messages"][-1]

    # Obtener el contexto actual
    contexto = state.get("contexto", "No hay elementos seleccionados en contexto")

    # Crear un prompt enriquecido que combina el mensaje original y el contexto
    enhanced_prompt = get_llm().invoke(
        [
            SystemMessage(
                content=f"[CONTEXTO]: {contexto}\n[CONSULTA ORIGINAL]: {last_message.content}\n"
                "Por favor, PREGUNTA AL USUARIO QUE FECHAS NECESITA."
                "CUANDO LA RESPUESTA la entrege el usuario, inmediatamente directamente entrega un prompt"
            )
        ]
    )

    return {"messages": [enhanced_prompt]}


workflow = StateGraph(State)
workflow.add_node("generar_contexto", generar_contexto)
workflow.add_node("supervisor", llamar_llm)
workflow.add_node("prompt_generator", prompt_generator)

workflow.add_edge(START, "generar_contexto")
workflow.add_edge("generar_contexto", "supervisor")
workflow.add_edge("prompt_generator", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))


config = {"configurable": {"thread_id": "1"}}


input_message = HumanMessage(
    content="'Considera las oficinas ['001 - Huerfanos 740 EDW', '003 - Cauquenes', '004 - Apoquindo EDW', '009 - Vitacura EDW'] que registros hay'"
)
output = graph.invoke({"messages": [input_message]}, config)

for m in output["messages"][-1:]:
    m.pretty_print()

# %%

input_message = HumanMessage(
    content="'Considera las oficinas ['001 - Huerfanos 740 EDW', '003 - Cauquenes', '004 - Apoquindo EDW', '009 - Vitacura EDW'] que periodo es ese SLA?"
)
output = graph.invoke({"messages": [input_message]}, config)

for m in output["messages"][-1:]:
    m.pretty_print()
# %%
input_message = HumanMessage(
    content="{'oficinas_seleccionadas': ['oficina_1', 'oficina_2'], 'mensaje': 'mes desde el 01 al 31'}"
)
output = graph.invoke({"messages": [input_message]}, config)

for m in output["messages"][-1:]:
    m.pretty_print()

# %%
