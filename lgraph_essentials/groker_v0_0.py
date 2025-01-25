# %%
# Set working directory to file location
# file_path = Path(__file__).resolve()
# os.chdir(file_path.parent)
# sys.path.append(str(file_path.parent.parent))
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Sequence, TypedDict

import yaml
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
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
from langgraph.constants import END, START
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

from lgraph_essentials.groker_dev_utils import get_llm  # CustomGraphState,
from lgraph_essentials.groker_dev_utils import (
    safely_remove_messages,
    validate_message_chain,
)
from ttp_agentic.tools.ranking_ejecutivos import executive_ranking_tool
from ttp_agentic.tools.registros_disponibles import rango_registros_disponibles
from ttp_agentic.tools.reporte_detallado_por_ejecutivo import (
    tool_reporte_detallado_por_ejecutivo,
)
from ttp_agentic.tools.reporte_general_de_oficinas import (
    tool_reporte_extenso_de_oficinas,
)

load_dotenv(override=True)


class CustomGraphState(MessagesState):
    oficinas: list[str] = []
    contexto: SystemMessage = SystemMessage(content="")
    guidance: SystemMessage = SystemMessage(content="")
    messages: Annotated[List[BaseMessage], add_messages]


class AskHuman(BaseModel):
    """AskHuman
    el agente debe solicitar directamente aclaraciones/información al usuario/humano
    """

    question: str


tools = [
    tool_reporte_extenso_de_oficinas,
    tool_reporte_detallado_por_ejecutivo,
    executive_ranking_tool,
]


tool_node = ToolNode(tools)
llm_with_tools = get_llm().bind_tools(tools + [AskHuman])


def filter_messages(state: CustomGraphState) -> dict:

    messages_to_remove = safely_remove_messages(state)
    return {"messages": [RemoveMessage(id=m.id) for m in messages_to_remove]}


def context_node(
    state: CustomGraphState,
) -> Command[Literal["agent", "request_context"]]:
    last_message = state["messages"][-1]
    print("Current message chain:", [type(m).__name__ for m in state["messages"]])

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
            print("---------------Cambio en lista de oficinas--------------")
            nuevo_contexto: str = (
                f"Datos disponibles para las oficinas: {rango_registros_disponibles(lista_nueva_oficinas)}"
            )
        else:
            print(
                "---------------No hay cambio en lista de oficinas, manteniendo contexto--------------"
            )
            nuevo_contexto: str = state["contexto"].content
    else:
        return Command(
            goto="request_context",
        )

    return Command(
        goto="agent",
        update={
            "contexto": SystemMessage(content=nuevo_contexto, id="nuevo_contexto"),
            "oficinas": oficinas_list,
            "messages": [
                HumanMessage(content=mensaje_limpio, id=last_message.id),
            ],
        },
    )


def call_model(state: CustomGraphState) -> dict:
    contexto = state["contexto"].content
    guidance = state["guidance"].content
    # Leer el system prompt desde el archivo YAML
    yaml_path = Path(__file__).parent / "system_prompts.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    # Obtener el template del prompt y hacer el formato
    prompt_template = prompts["groker_v0_0"]["system_prompt"]
    formatted_prompt = prompt_template.format(
        date=datetime.now().strftime("%d/%m/%Y"),
        oficinas=state["oficinas"] if state["oficinas"] else "ninguna",
        contexto=contexto,
        guidance=guidance,
    )

    # Guardar el prompt formateado en un nuevo archivo YAML
    runtime_yaml_path = Path(__file__).parent / "system_prompt_runtime.yaml"
    with open(runtime_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({"system_prompt": formatted_prompt}, f, allow_unicode=True)

    system_prompt = SystemMessage(content=formatted_prompt)
    messages = validate_message_chain(state["messages"])
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}


def ask_human(state: CustomGraphState) -> dict:
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    user_intervention = interrupt(
        "Por favor, indique el período de tiempo que desea consultar"
    )
    tool_message = [
        {"tool_call_id": tool_call_id, "type": "tool", "content": user_intervention}
    ]
    return {"messages": tool_message}


def should_continue(state: CustomGraphState) -> Literal["tools", "ask_human", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END
    # If tool call is asking Human, we return that node
    # You could also add logic here to let some system know that there's something that requires Human input
    # For example, send a slack message, etc
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    # Otherwise if there is, we continue
    else:
        return "tools"


def request_office_selection(state: CustomGraphState) -> dict:
    """
    Nodo que entrega un mensaje al usuario solicitando la selección de oficinas.
    """
    return {
        "messages": [
            AIMessage(
                content="Por favor, selecciona las oficinas que deseas consultar usando el botón en la esquina superior derecha."
            )
        ]
    }


def update_guidance(state: CustomGraphState) -> dict:
    """
    Genera una guía de razonamiento basada en el historial de la conversación usando el LLM.
    """
    messages = validate_message_chain(state["messages"])

    # Retornar guidance vacío si hay 4 o menos mensajes
    if len(messages) <= 4:
        print("---------------------------------------------")
        print(f"---------------NO guidance_update {len(messages)=}---------------")
        print("---------------------------------------------")
        return {"guidance": SystemMessage(content="", id="guidance_update")}
    # Usamos una instancia separada del LLM sin herramientas
    print("---------------------------------------------")
    print(f"---------------guidance_update {len(messages)=}---------------")
    print("---------------------------------------------")

    llm = get_llm()

    # Creamos un prompt para solicitar el resumen
    system_message = SystemMessage(
        content="Eres un agente que entiende lo que el usuario está pidiendo. "
        "Tienes que hacer un resumen que servirá como guía  breve para que otro agente entienda la necesidad actual del usuario. "
        "Sé muy breve, conciso y enfócate en lo que el usuario está pidiendo."
    )

    # Convertimos los mensajes a formato de resumen
    conversation_summary = "\n".join(
        [
            f"{'Usuario' if isinstance(m, HumanMessage) else 'Asistente'}: {m.content}"
            for m in messages
        ]
    )

    human_message = HumanMessage(
        content=f"Debes considerar el historial de mensajes:\n{conversation_summary}"
    )

    # Obtenemos el resumen
    response = llm.invoke([system_message, human_message])
    print(
        f"---------------{SystemMessage(content=response.content, id='guidance_update').pretty_print()}---------------"
    )
    return {"guidance": SystemMessage(content=response.content, id="guidance_update")}


# Actualización del grafo
workflow = StateGraph(CustomGraphState)
workflow.add_node("filter_messages", filter_messages)
workflow.add_node("guide", update_guidance)  # Nuevo nodo
workflow.add_node("context", context_node)
workflow.add_node("request_context", request_office_selection)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("ask_human", ask_human)

# Edges
workflow.add_edge(START, "filter_messages")
workflow.add_edge("filter_messages", "guide")  # Nueva conexión
workflow.add_edge("filter_messages", "context")  # Nueva conexión

workflow.add_edge("guide", "agent")  # Nueva conexión
workflow.add_edge("request_context", END)
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
workflow.add_edge("ask_human", "agent")
if os.getenv("DEV_CHECKPOINTER"):

    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
else:
    graph = workflow.compile()


def run_graph(graph: CompiledStateGraph, input_message: str = "hola") -> None:
    config = {"configurable": {"thread_id": "1"}}
    for chunk in graph.stream(
        {"messages": [HumanMessage(content=input_message)]},
        config,
        stream_mode="updates",
    ):
        if "agent" in chunk:
            # if isinstance(chunk["agent"]["messages"][0], AIMessage):
            print(f"{chunk['agent']['messages'][0]=}")
            print(chunk["agent"]["messages"][0].pretty_print())
        if "request_context" in chunk:
            print(chunk["request_context"]["messages"][0].pretty_print())


display(Image(graph.get_graph().draw_mermaid_png()))
if __name__ == "__main__":

    qs = [
        "dame los detalles del mejor ejecutivo",
        "septiembre",
        "que datos hay disponibles?",
        "dame el SLA de octubre",
        "dame el ranking de ejecutivos de octubre",
        "dame los detalles del peor ejecutivo de octubre",
        "gracias",
        "cual fue mi primera pregunta?",
        "hola",
        "dame los detalles del mejor ejecutivo",
        "octubre",
        "hola",
    ]
    # %%
    run_graph(
        graph,
        (
            "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque']"
            # ""
            + qs[0]
        ),
    )
