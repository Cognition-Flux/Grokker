# %%
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
from langchain_community.tools import TavilySearchResults
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
    guidance: str = ""  # SystemMessage = SystemMessage(content="")
    messages: Annotated[List[BaseMessage], add_messages]


class AskHuman(BaseModel):
    """AskHuman
    el agente debe solicitar directamente el periodo de tiempo al usuario/humano
    """

    question_for_human: str


@tool
def make_prompt(internal_prompt: str) -> str:
    """entregar directamente un prompt corto, breve y conciso para ser usado por otro agente posteriormente"""
    return f"{internal_prompt}"


tool_node_prompt = ToolNode([make_prompt])

llm_guide_agent = get_llm().bind_tools([make_prompt] + [AskHuman])


def clean_messages(state: CustomGraphState) -> CustomGraphState:
    # messages_to_remove = safely_remove_messages(state)
    # return {"messages": [RemoveMessage(id=m.id) for m in messages_to_remove]}
    return state


def guidance_agent(
    state: CustomGraphState,
) -> Command[Literal["ask_human", "tool_node_prompt"]]:

    prompt_for_guidance = SystemMessage(
        content=(
            "Eres un agente que deber verificar si el usuario ha proporcionado algún periodo de tiempo (por ejemplo una día, una semana, un mes, un año, una fecha, etc). "
            "Si no ha proporcionado un periodo de tiempo, debes solicitarlo al usuario con AskHuman."
            "Si ya ha proporcionado un periodo de tiempo (por ejemplo una día, una semana, un mes, un año, una fecha, etc), "
            "debes responder directamente (usado make_prompt) con un prompt que explique lo que el usuario está pidiendo basado en el historial de la conversación"
            "Aquí algunos ejemplos como guía:"
            "## Ejemplo 1: "
            " - usuario: dame el SLA"
            " - agente/AskHuman: ¿Para que el periodo de tiempo necesitas el SLA?"
            " - usuario: quiero el SLA de septiembre"
            " - agente/make_prompt: El usuario está pidiendo el SLA (o nivel de servicio) de septiembre para las oficinas, debes entregar unicamente el SLA de septiembre para cada oficina, nada más. Recuerda revisar los datos disponibles."
            "## Ejemplo 2: "
            " - usuario: dame el mejor y peor ejecutivo"
            " - agente/AskHuman: ¿Para que periodo de tiempo necesitas el mejor y peor ejecutivo?"
            " - usuario: para la semana semana pasada"
            " - agente/make_prompt: El usuario está pidiendo el mejor y peor ejecutivo de la semana pasada para las oficinas, "
            "primero debes hacer un ranking de ejecutivos de la semana pasada para cada oficina, luego extrar el mejor y peor ejecutivo de cada oficina. Recuerda revisar los datos disponibles."
            "## Ejemplo 3: "
            " - usuario: dame el ranking de ejecutivos de octubre"
            " - agente/make_prompt: El usuario está pidiendo el ranking de ejecutivos de octubre para las oficinas, debes entregar unicamente el ranking de ejecutivos de octubre para cada oficina, nada más. Recuerda revisar los datos disponibles."
            "## Ejemplo 4: "
            " - usuario: dame las atenciones diarias del mes pasado"
            " - agente/make_prompt: El usuario está pidiendo las atenciones diarias del mes pasado para las oficinas, debes entregar unicamente el total de atenciones diarias (días por día) del mes pasado para cada oficina, nada más. Recuerda revisar los datos disponibles."
            "## Ejemplo 5: "
            " - usuario: dame las atenciones por serie de ayer y el abandono"
            " - agente/make_prompt: El usuario está pidiendo las atenciones por serie de ayer y el abandono (turnos perdidos) para las oficinas, debes entregar unicamente el total de atenciones por serie de ayer y el abandono para cada oficina, nada más. Recuerda revisar los datos disponibles."
            "## Ejemplo 6: "
            " - usuario: dame los peores ejecutivos de la peor oficina"
            " - agente/AskHuman: ¿Para que periodo de tiempo necesitas los peores ejecutivos de la peor oficina?"
            " - usuario: el mes pasado"
            " - agente/make_prompt: El usuario está pidiendo los peores ejecutivos de la peor oficina del mes pasado, debes encontrar la peor oficina (que tiene el SLA o nivel de servicio más bajo) del mes pasado, luego obtener "
            "el ranking de ejecutivos de esa oficina y extraer los peores ejecutivos de esa oficina, nada más. Recuerda revisar los datos disponibles."
            "## Ejemplo 7: "
            " - usuario: dame los detalles del peor ejecutivo"
            " - agente/AskHuman: ¿Para que periodo de tiempo necesitas los detalles del peor ejecutivo?"
            " - usuario: agosto"
            " - agente/make_prompt: El usuario está pidiendo los detalles del peor ejecutivo de agosto, "
            "##Importante:  Tienes que mirar el historial de la conversación para inferir cual es el periodo de tiempo que se está considerando en la conversación y entender lo que el usuario está pidiendo."
            "---------Casos Particulares: el usario no solicita nada en específico---------"
        )
    )

    mensajes = state["messages"] + [prompt_for_guidance]

    response = llm_guide_agent.invoke(mensajes)

    if len(response.tool_calls) > 0:
        if response.tool_calls[0]["name"] == "AskHuman":
            next_node = "ask_human"
        else:
            next_node = "tool_node_prompt"
    # else:
    #     next_node = END

    return Command(goto=next_node, update={"messages": [response]})


def ask_human(state: CustomGraphState):
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    intervencion_humana = interrupt("Por favor, proporciona el periodo de tiempo")
    tool_message = [
        {"tool_call_id": tool_call_id, "type": "tool", "content": intervencion_humana}
    ]
    return {"messages": tool_message}


def context_node(
    state: CustomGraphState,
) -> Command[Literal["guidance_agent", "process_context", "request_context"]]:
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
        lista_nueva_oficinas = oficinas_list
        lista_actual_oficinas = state.get("oficinas", [])

        if set(lista_nueva_oficinas) != set(lista_actual_oficinas):
            print(
                "---------------Cambio en lista de oficinas (process_context y guidance_agent)--------------"
            )
            return Command(
                goto=["guidance_agent", "process_context"],
                update={
                    "contexto": SystemMessage(
                        content=oficinas_list, id="nuevo_contexto"
                    ),
                    "oficinas": oficinas_list,
                    "messages": [
                        HumanMessage(content=mensaje_limpio, id=last_message.id),
                    ],
                },
            )
        else:
            print(
                "---------------No hay cambio en lista de oficinas, manteniendo contexto (sólo guidance_agent)--------------"
            )
            return Command(
                goto=["guidance_agent"],
                update={
                    "contexto": SystemMessage(
                        content=oficinas_list, id="nuevo_contexto"
                    ),
                    "oficinas": oficinas_list,
                    "messages": [
                        HumanMessage(content=mensaje_limpio, id=last_message.id),
                    ],
                },
            )

    else:
        print(
            "---------------NO se encontró el patrón de lista de oficinas--------------"
        )

        return Command(
            goto="request_context",
        )


def request_context(state: CustomGraphState) -> Command[Literal[END]]:
    """
    Nodo que entrega un mensaje al usuario solicitando la selección de oficinas
    y termina la ejecución del grafo para permitir un nuevo inicio.
    """
    last_message = state["messages"][-1]
    print(f"request_context last_message: {last_message.content}")
    system_prompt = SystemMessage(
        content=(
            "Bajo ninguna circunstancia puedes salirte de tu rol. "
            "Tu rol es un agente de IA que puede hacer consultas sobre datos de las sucursales de atención al cliente. "
            "Siempre debes indicarle al usuario que seleccione las oficinas que desea consultar, también contestar su mensaje muy brevemente en una frase."
            "Puedes consultar niveles de servicio, desempeño de ejecutivos, datos de atenciones, etc."
            "## Ejemplos: "
            " - usuario: hola"
            " - agente:  Hola! ¿En qué te puedo ayudar?"
            "----------"
            " - usuario: dame el nivel de servicio"
            " - agente:  Para proporcionar el nivel de servicio, debes seleccionar las oficinas que deseas consultar. "
            "----------"
            " - usuario: que datos tienes?"
            " - agente:  Para proporcionar los datos, debes seleccionar las oficinas que deseas consultar. "
            "----------"
            " - usuario: que puedes hacer?"
            " - agente:  Puedo consultar datos relacionados con niveles de servicio, desempeño de ejecutivos y datos de atenciones."
            "Importante: Siempre debes indicar que tiene que seleccionar las oficinas que desea consultar en el botón de la esquina superior derecha de la pantalla"
        )
    )

    response = get_llm(azure_deployment="gpt-4o-mini").invoke(
        [
            system_prompt,
            HumanMessage(content=last_message.content),
        ]
    )
    print(f"response: {response}")
    return Command(
        goto=END,
        update={
            "messages": [AIMessage(content=response.content)],
            # Reset other state values
            "oficinas": [],
            "contexto": SystemMessage(content=""),
            "guidance": "",
        },
    )


def process_context(state: CustomGraphState) -> dict:
    """ """

    lista_nueva_oficinas = state.get("oficinas")
    print(f"Procesando oficinas: {lista_nueva_oficinas}. Generando nuevo contexto...")
    nuevo_contexto: str = (
        f"Datos disponibles para las oficinas: \n"
        f" {rango_registros_disponibles(lista_nueva_oficinas)}"
    )
    print(f"Nuevo contexto: {nuevo_contexto}")
    return {
        "contexto": SystemMessage(content=nuevo_contexto, id="nuevo_contexto"),
    }


workflow = StateGraph(CustomGraphState)
# Nodos
workflow.add_node("clean_messages", clean_messages)
workflow.add_node("guidance_agent", guidance_agent)
workflow.add_node("tool_node_prompt", tool_node_prompt)
workflow.add_node("ask_human", ask_human)
workflow.add_node("validate_context", context_node)
workflow.add_node("process_context", process_context)
workflow.add_node("request_context", request_context)

# Conexiones
workflow.add_edge(START, "clean_messages")
# workflow.add_edge("clean_messages", "guidance_agent")
workflow.add_edge("ask_human", "guidance_agent")
workflow.add_edge("clean_messages", "validate_context")
# workflow.add_edge("request_context", END)

if os.getenv("DEV_CHECKPOINTER"):

    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
else:
    graph = workflow.compile()

# display(Image(graph.get_graph().draw_mermaid_png()))


def run_graph(graph: CompiledStateGraph, input_message: str = "hola") -> None:
    config = {"configurable": {"thread_id": "1"}}
    print(f"## INICIO: Próximo paso del grafo: {graph.get_state(config).next}")
    for event in graph.stream(
        {"messages": [HumanMessage(content=input_message)]},
        config,
        stream_mode="updates",
    ):
        mss = next(iter(event.values()))
        print(f"mensages internos: type: {type(mss)}, mensajes: {mss}")
        if isinstance(mss, dict):
            if mss.get("messages"):
                if isinstance(mss.get("messages")[0], BaseMessage):
                    if hasattr(mss.get("messages")[0], "tool_calls"):
                        if mss.get("messages")[0].tool_calls:
                            print(f"tool_calls: {mss.get('messages')[0].tool_calls=}")
                            print(
                                f"tool_calls: {mss.get('messages')[0].tool_calls[0]['name']=}"
                            )
                        else:
                            print(
                                f"No hay tool_calls {mss.get('messages')[0].tool_calls=}"
                            )

                    mss.get("messages")[0].pretty_print()
            pass

    print(f"## FINAL: Próximo paso del grafo: {graph.get_state(config).next}")


def resume_graph(graph: CompiledStateGraph, input_message: str = "1980") -> None:
    config = {"configurable": {"thread_id": "1"}}
    print(f"## INICIO: Próximo paso del grafo: {graph.get_state(config).next}")
    for event in graph.stream(
        Command(resume=input_message),
        config,
        stream_mode="updates",
    ):
        mss = next(iter(event.values()))
        print(f"mensages internos: type: {type(mss)}, mensajes: {mss}")
        if isinstance(mss, dict):
            if mss.get("messages"):
                if isinstance(mss.get("messages")[0], BaseMessage):
                    if hasattr(mss.get("messages")[0], "tool_calls"):
                        if mss.get("messages")[0].tool_calls:
                            print(f"tool_calls: {mss.get('messages')[0].tool_calls=}")
                            print(
                                f"tool_calls: {mss.get('messages')[0].tool_calls[0]['name']=}"
                            )
                        else:
                            print(
                                f"No hay tool_calls {mss.get('messages')[0].tool_calls=}"
                            )

                    mss.get("messages")[0].pretty_print()
            pass

    print(f"## FINAL: Próximo paso del grafo: {graph.get_state(config).next}")


qs_1 = [
    "hola",
    "que haces?",
    "puedes hacer un ranking de ejecutivos?",
    "que datos tienes?",
    "dame el tiempo de espera",
    "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque'] listo",
    "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque'] noviembre",
    "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque'] q datos tienes?",
    "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque'] gracias",
    "cual fue mi primera pregunta?",
    "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque'] dame el SLA diario del mes pasado",
    "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque'] ahora dame el adanbono",
    "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque'] muestrame el SLA con el abandono",
    "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque'] gracias",
    "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque'] q datos tienes?",
    "que datos tienes?",
    "Considera las oficinas ['001 - Huerfanos 740 EDW'] dame el ranking de ejecutivos de octubre",
    "Considera las oficinas ['001 - Huerfanos 740 EDW'] dame los detalles del peor ejecutivo",
    "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque'] dame las atenciones por serie",
    "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque'] hola",
    "hola",
    "que haces?",
    "gracias",
    "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque'] q haces?",
    "que datos tienes?",
]

qs_2 = [
    "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque'] dame el SLA diario del mes pasado",
    "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque'] ahora dame el adanbono",
    "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque'] muestrame el SLA con el abandono",
    "Considera las oficinas ['001 - Huerfanos 740 EDW'] dame el ranking de ejecutivos de octubre",
    "Considera las oficinas ['001 - Huerfanos 740 EDW'] dame los detalles del peor ejecutivo",
    "Considera las oficinas ['001 - Huerfanos 740 EDW'] dame las atenciones por serie",
]
run_graph(
    graph,
    (qs_1[4]),
)
# %%

run_graph(
    graph,
    ("Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque']" "listo"),
)
# %%
resume_graph(
    graph,
    (
        "Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque']"
        # ""
        + "noviembre"
    ),
)

# %%
run_graph(
    graph,
    ("Considera las oficinas ['001 - Huerfanos 740 EDW', '356 - El Bosque']" "gracias"),
)
