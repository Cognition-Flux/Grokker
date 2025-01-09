# %%
import os
import re
import sys
from pathlib import Path

# Set working directory to file location
file_path = Path(__file__).resolve()
os.chdir(file_path.parent)
sys.path.append(str(file_path.parent.parent))
import json
import os
import re
from datetime import datetime
from typing import Annotated, Dict, List, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from groker_dev_utils import CustomGraphState, safely_remove_messages
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

load_dotenv(override=True)


class AskHuman(BaseModel):
    """AskHuman
    el agente debe solicitar directamente aclaraciones/información al usuario/humano
    """

    question: str


def context_node(state: CustomGraphState) -> CustomGraphState:
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
        return {
            "contexto": SystemMessage(
                content="El usuario no ha seleccionado ninguna oficina en la aplicación, IGNORA TODO e indique que debe seleccionar usando el botón en la esquina superior derecha",
                id="nuevo_contexto",
            ),
            "oficinas": [],
        }

    return {
        "contexto": SystemMessage(content=nuevo_contexto, id="nuevo_contexto"),
        "oficinas": oficinas_list,
        "messages": [
            HumanMessage(content=mensaje_limpio, id=last_message.id),
        ],
    }
