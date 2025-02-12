"""
Módulo para la ejecución de agentes y la generación de prompts para
la extracción, análisis y reporte de información.
"""

# %%
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Literal

import yaml
from agents.grokker.tools.ranking_ejecutivos import executive_ranking_tool
from agents.grokker.tools.registros_disponibles import rango_registros_disponibles
from agents.grokker.tools.reporte_detallado_por_ejecutivo import (
    tool_reporte_detallado_por_ejecutivo,
)
from agents.grokker.tools.reporte_general_de_oficinas import (
    tool_reporte_extenso_de_oficinas,
)
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt
from pydantic import BaseModel

# Configuración del registro (logging)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
)

# Establecer directorio de trabajo (ajustar según sea necesario)
# os.chdir("/home/alejandro/Desktop/repos/groker/backend/src")

# Cargar prompts desde YAML
PROMPTS_PATH = Path("agents/grokker/system_prompts/agents_prompts.yaml")
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

with open(
    "agents/grokker/system_prompts/tests_user_prompts.yaml", "r", encoding="utf-8"
) as file:
    user_prompts = yaml.safe_load(file)

system_prompt_prohibited_actions = SystemMessage(
    content=prompts["prohibited_actions"]["prompt"]
)

load_dotenv(override=True)


def get_llm(deployment: Literal["openai", "azure"] = "openai") -> ChatOpenAI:
    """Retorna una instancia de ChatOpenAI configurada."""
    if deployment == "openai":
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_retries=5,
            streaming=True,
            api_key=os.environ["OPENAI_API_KEY"],
        )
    elif deployment == "azure":
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


# get_llm(deployment="azure").invoke("hola")


# %%
class CustomGraphState(MessagesState):
    """Estado del grafo con información adicional."""

    oficinas: list[str] = []
    contexto: SystemMessage = SystemMessage(content="")
    guidance: str = ""
    messages: Annotated[List[BaseMessage], add_messages]


class GuidanceAgentAskHuman(BaseModel):
    """
    GuidanceAgentAskHuman: solicita directamente al usuario el periodo de tiempo.
    """

    question_for_human: str


class AnalystAgentAskHuman(BaseModel):
    """
    AnalystAgentAskHuman: solicita directamente aclaraciones o guía al usuario.
    """

    question_for_human: str


@tool
def make_prompt(internal_prompt: str) -> str:
    """
    Retorna un prompt breve y conciso para uso interno por otros agentes.
    """
    return internal_prompt


# Nodo de herramienta para make_prompt
tool_node_prompt = ToolNode([make_prompt])

llm_guide_agent = get_llm().bind_tools([make_prompt, GuidanceAgentAskHuman])

tools_analyst = [
    tool_reporte_extenso_de_oficinas,
    tool_reporte_detallado_por_ejecutivo,
    executive_ranking_tool,
]
tools_analyst_description = "\n".join(
    [f"name: {t.name} - {t.description}" for t in tools_analyst]
)

analyst_llm = get_llm().bind_tools(tools_analyst)
tools_node_analyst = ToolNode(tools_analyst)
context_request_llm = get_llm()


def clean_messages(state: CustomGraphState) -> CustomGraphState:
    """
    Función para limpiar mensajes.
    Actualmente no elimina mensajes, solo retorna el estado tal cual.
    """
    return state


def guidance_agent(
    state: CustomGraphState,
) -> Command[Literal["guidance_agent_ask_human", "tool_node_prompt", END]]:
    """
    Agente de guidance que implementa la lógica para solicitar o procesar información
    según el prompt configurado.
    """
    logger.debug("##################### --- guidance_agent --- #####################")
    prompt_template = prompts["guidance_agent"]["prompt"]
    formatted_prompt = prompt_template.format(
        tools_analyst_description=tools_analyst_description,
        hoy=datetime.now().strftime("%d/%m/%Y"),
    )
    prompt_for_guidance = SystemMessage(content=formatted_prompt)
    prompt_for_guidance.pretty_print()

    # Se visualiza el último mensaje recibido
    state["messages"][-1].pretty_print()

    response = llm_guide_agent.invoke(
        [prompt_for_guidance, system_prompt_prohibited_actions] + state["messages"]
    )
    response.pretty_print()
    logger.debug("Número de tool_calls: %d", len(response.tool_calls))
    logger.debug("response.tool_calls: %s", response.tool_calls)

    if response.tool_calls:
        if response.tool_calls[0]["name"] == "GuidanceAgentAskHuman":
            next_node = "guidance_agent_ask_human"
        else:
            next_node = "tool_node_prompt"
    else:
        next_node = END

    return Command(goto=next_node, update={"messages": [response]})


def guidance_agent_ask_human(state: CustomGraphState) -> dict:
    """
    Solicita al usuario el periodo de tiempo.
    """
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    logger.debug("Solicitando intervención humana para el periodo de tiempo.")
    intervencion_humana = interrupt("Por favor, proporciona el periodo de tiempo")
    tool_message = ToolMessage(
        content=intervencion_humana, tool_call_id=tool_call_id, type="tool"
    )
    return {"messages": [tool_message]}


def context_node(
    state: CustomGraphState,
) -> Command[Literal["guidance_agent", "process_context", "context_request_agent"]]:
    """
    Verifica si en el mensaje se especifica una lista de oficinas y actualiza el estado.
    """
    last_message = state["messages"][-1]
    pattern = r"Considera las oficinas \[(.*?)\]"
    mensaje_original = last_message.content
    # Actualmente no se modifica el mensaje
    mensaje_limpio = mensaje_original

    match = re.search(pattern, last_message.content)
    if match:
        logger.debug("Se encontró el patrón de lista de oficinas.")
        logger.debug("Mensaje original: %s", mensaje_original)
        logger.debug("Mensaje limpio: %s", mensaje_limpio)

        oficinas_str = match.group(1)
        oficinas_list = [
            office.strip().strip("'") for office in oficinas_str.split(",")
        ]
        logger.debug("Oficinas extraídas: %s", oficinas_list)

        lista_actual_oficinas = state.get("oficinas", [])

        if set(oficinas_list) != set(lista_actual_oficinas):
            logger.debug("Cambio en la lista de oficinas, se actualizará el contexto.")
            return Command(
                goto=["guidance_agent", "process_context"],
                update={
                    "oficinas": oficinas_list,
                    "messages": [
                        HumanMessage(content=mensaje_limpio, id=last_message.id)
                    ],
                },
            )
        else:
            logger.debug("No hay cambio en la lista de oficinas.")
            return Command(
                goto=["guidance_agent"],
                update={
                    "messages": [
                        HumanMessage(content=mensaje_limpio, id=last_message.id)
                    ],
                },
            )
    else:
        logger.debug("NO se encontró el patrón de lista de oficinas.")
        return Command(goto="context_request_agent")


def context_request_agent(
    state: CustomGraphState,
) -> Command[Literal[END]]:
    """
    Solicita al usuario mayor contexto si no se encontró el patrón.
    """
    logger.debug(
        "##################### --- context_request_agent --- #####################"
    )
    last_message = state["messages"][-1]
    formatted_prompt = prompts["context_request_agent"]["prompt"].format(
        hoy=datetime.now().strftime("%d/%m/%Y")
    )
    system_prompt = SystemMessage(content=formatted_prompt)
    system_prompt.pretty_print()
    input_message = HumanMessage(content=last_message.content)
    input_message.pretty_print()

    logger.debug("Enviando solicitud de mayor contexto...")
    response = context_request_llm.invoke(
        [system_prompt, system_prompt_prohibited_actions, input_message]
    )
    response.pretty_print()

    return Command(
        goto=END,
        update={
            "messages": [response],
            "oficinas": [],
            "contexto": SystemMessage(content=""),
            "guidance": "",
        },
    )


def process_context(state: CustomGraphState) -> dict:
    """
    Procesa la lista de oficinas y genera un nuevo contexto.
    """
    lista_nueva_oficinas = state.get("oficinas")
    logger.debug(
        "Procesando oficinas: %s. Generando nuevo contexto...",
        lista_nueva_oficinas,
    )
    nuevo_contexto = (
        f"Datos disponibles para las oficinas:\n"
        f"{rango_registros_disponibles(lista_nueva_oficinas)}"
    )
    logger.debug("Nuevo contexto generado: %s", nuevo_contexto)
    return {
        "contexto": SystemMessage(content=nuevo_contexto, id="nuevo_contexto"),
    }


def validate_state(state: CustomGraphState) -> Command[Literal["analyst_agent"]]:
    """
    Valida que el contexto y la lista de oficinas estén completos para pasar al siguiente paso.
    Si no se cumple la validación, se finaliza el flujo.
    """
    last_message = state["messages"][-1]
    contexto = state.get("contexto")
    oficinas = state.get("oficinas")
    if contexto and oficinas:
        if isinstance(last_message, ToolMessage) and last_message.content != "":
            logger.debug("Estado validado, pasando a analyst_agent.")
            return Command(
                goto="analyst_agent",
                update={"guidance": last_message.content, "messages": [last_message]},
            )

    logger.error("validate_state: datos incompletos, finalizando flujo.")
    return Command(goto=END, update={"messages": state["messages"]})


def analyst_agent(
    state: CustomGraphState,
) -> Command[Literal["tools_node_analyst", END]]:
    """
    Agente analista que genera el prompt definitivo para el análisis.
    """
    last_message = state["messages"][-1]
    contexto = state.get("contexto")
    oficinas = state.get("oficinas")
    guidance = state.get("guidance")

    logger.debug(
        "##################### --- analyst_agent --- #####################\n"
        "Último mensaje: %s\n"
        "Guidance: %s\n"
        "Contexto: %s\n"
        "Oficinas: %s",
        last_message.content,
        guidance,
        contexto.content,
        oficinas,
    )

    formatted_prompt = prompts["analyst_agent"]["prompt"].format(
        oficinas=oficinas,
        contexto=contexto.content,
        hoy=datetime.now().strftime("%d/%m/%Y"),
    )
    system_prompt = SystemMessage(content=formatted_prompt)
    system_prompt.pretty_print()
    response = analyst_llm.invoke(
        [system_prompt, system_prompt_prohibited_actions] + state["messages"]
    )
    logger.debug("Respuesta de analyst_agent: %s", response)
    logger.debug("Número de tool_calls: %d", len(response.tool_calls))

    if response.tool_calls:
        logger.debug("Se encontraron tool_calls en la respuesta.")
    else:
        response.pretty_print()

    next_node = "tools_node_analyst" if response.tool_calls else END
    return Command(goto=next_node, update={"messages": [response]})


# Configuración y compilación del grafo de estados
workflow = StateGraph(CustomGraphState)
workflow.add_node("clean_messages", clean_messages)
workflow.add_node("guidance_agent", guidance_agent)
workflow.add_node("tool_node_prompt", tool_node_prompt)
workflow.add_node("guidance_agent_ask_human", guidance_agent_ask_human)
workflow.add_node("validate_context", context_node)
workflow.add_node("process_context", process_context)
workflow.add_node("context_request_agent", context_request_agent)
workflow.add_node("analyst_agent", analyst_agent)
workflow.add_node("tools_node_analyst", tools_node_analyst)
workflow.add_node("validate_state", validate_state)

workflow.add_edge(START, "clean_messages")
workflow.add_edge("guidance_agent_ask_human", "guidance_agent")
workflow.add_edge("clean_messages", "validate_context")
workflow.add_edge("process_context", "validate_state")
workflow.add_edge("tool_node_prompt", "validate_state")
workflow.add_edge("tools_node_analyst", "analyst_agent")

# Configuración de memorias
across_thread_memory = InMemoryStore()
within_thread_memory = MemorySaver()

graph = workflow.compile(checkpointer=within_thread_memory, store=across_thread_memory)
