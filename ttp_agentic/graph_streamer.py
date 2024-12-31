# %%
import os
import sys
from pathlib import Path

# Set working directory to file location
file_path = Path(__file__).resolve()
os.chdir(file_path.parent)
sys.path.append(str(file_path.parent.parent))
import asyncio
import typing
from collections.abc import AsyncGenerator
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from IPython.display import Image, display
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from multiagent_graph import graph
from pydantic import BaseModel, Field

# display(Image(graph.get_graph().draw_mermaid_png()))
app = FastAPI()


class UserInput(BaseModel):
    """Basic user input for the agent."""

    message: str = Field(
        description="User input to the agent.",
        examples=["hola, que puedes hacer?"],
    )
    model: str = Field(
        description="LLM Model to use for the agent.",
        default="gpt-4o",
        examples=["gpt-4o", "gpt-4o-mini", "llama-3.1-70b"],
    )
    thread_id: str | None = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )


class StreamInput(UserInput):
    """User input for streaming the agent's response."""

    stream_tokens: bool = Field(
        description="Whether to stream LLM tokens to the client.",
        default=True,
    )


def _parse_input(user_input: UserInput) -> tuple[dict[str, Any], str]:
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    kwargs = {
        "input": {"messages": [HumanMessage(content=user_input.message)]},
        "config": RunnableConfig(
            configurable={"thread_id": thread_id, "model": user_input.model},
            run_id=run_id,
        ),
    }
    return kwargs, run_id


# %%
async def call_graph(graph: CompiledStateGraph, user_input: StreamInput) -> None:
    kwargs, run_id = _parse_input(user_input)
    config = {
        "configurable": {"thread_id": kwargs["config"]["configurable"]["thread_id"]}
    }
    mensaje_del_usuario = kwargs["input"]["messages"][-1]
    node_to_stream: str = "agent"
    if graph.get_state(config).next == () or not graph.get_state(config).next:
        async for event in graph.astream_events(
            {"messages": [mensaje_del_usuario]}, config, version="v2"
        ):
            if not event:
                continue
            if (
                event["event"] == "on_chat_model_stream"
                and event["metadata"].get("langgraph_node", "") == node_to_stream
            ):
                data = event["data"]
                if data["chunk"].content:
                    yield data["chunk"].content
    else:
        async for event in graph.astream_events(
            Command(resume=mensaje_del_usuario), config, version="v2"
        ):
            if not event:
                continue
            if (
                event["event"] == "on_chat_model_stream"
                and event["metadata"].get("langgraph_node", "") == node_to_stream
            ):
                data = event["data"]
                if data["chunk"].content:
                    yield data["chunk"].content


# %%

# user_input = UserInput(message="para la ultima semana", model="gpt-4o", thread_id=None)
# async_generator = call_graph(graph, user_input)

# async for content in async_generator:
#     print(content, end="|")
# %%


def _sse_response_example() -> dict[int, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@app.post(
    "/stream", response_class=StreamingResponse, responses=_sse_response_example()
)
async def stream_agent(user_input: StreamInput) -> StreamingResponse:

    async def generate() -> typing.AsyncIterator[str]:
        accumulated_content = []
        async for content in call_graph(graph, user_input):
            accumulated_content.append(content)
            yield f'data: {{"type": "token", "content": "{content}"}}\n\n'
        full_message = "".join(accumulated_content)
        yield f'data: {{"type": "message", "content": "{full_message}"}}\n\n'
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


"""
curl -X 'POST' \
  'http://0.0.0.0:8080/stream' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "message": "hola",
  "model": "gpt-4o",
  "thread_id": "847c6285-8fc9-4560-a83f-4e6285809254",
  "stream_tokens": true
}'


data: {"type": "token", "content": "\u00a1"}

data: {"type": "token", "content": "Hola"}

data: {"type": "token", "content": "!"}

data: {"type": "token", "content": " \u00bf"}

data: {"type": "token", "content": "En"}

data: {"type": "token", "content": " qu\u00e9"}

data: {"type": "token", "content": " puedo"}

data: {"type": "token", "content": " ayudarte"}

data: {"type": "token", "content": " hoy"}

data: {"type": "token", "content": "?"}

data: {"type": "token", "content": " Estoy"}

data: {"type": "token", "content": " aqu\u00ed"}

data: {"type": "token", "content": " para"}

data: {"type": "token", "content": " proporcion"}

data: {"type": "token", "content": "arte"}

data: {"type": "token", "content": " informaci\u00f3n"}

data: {"type": "token", "content": " y"}

data: {"type": "token", "content": " an\u00e1lisis"}

data: {"type": "token", "content": " sobre"}

data: {"type": "token", "content": " el"}

data: {"type": "token", "content": " desempe\u00f1o"}

data: {"type": "token", "content": " de"}

data: {"type": "token", "content": " suc"}

data: {"type": "token", "content": "urs"}

data: {"type": "token", "content": "ales"}

data: {"type": "token", "content": ","}

data: {"type": "token", "content": " tiempos"}

data: {"type": "token", "content": " de"}

data: {"type": "token", "content": " atenci\u00f3n"}

data: {"type": "token", "content": ","}

data: {"type": "token", "content": " niveles"}

data: {"type": "token", "content": " de"}

data: {"type": "token", "content": " servicio"}

data: {"type": "token", "content": ","}

data: {"type": "token", "content": " y"}

data: {"type": "token", "content": " m\u00e1s"}

data: {"type": "token", "content": "."}

data: {"type": "message", "content": {"type": "ai", "content": "\u00a1Hola! \u00bfEn qu\u00e9 puedo ayudarte hoy? Estoy aqu\u00ed para proporcionarte informaci\u00f3n y an\u00e1lisis sobre el desempe\u00f1o de sucursales, tiempos de atenci\u00f3n, niveles de servicio, y m\u00e1s.", "tool_calls": [], "tool_call_id": null, "run_id": "6a0170fb-0908-4a9e-941e-7cc9bbfdb97e", "original": {"type": "ai", "data": {"content": "\u00a1Hola! \u00bfEn qu\u00e9 puedo ayudarte hoy? Estoy aqu\u00ed para proporcionarte informaci\u00f3n y an\u00e1lisis sobre el desempe\u00f1o de sucursales, tiempos de atenci\u00f3n, niveles de servicio, y m\u00e1s.", "additional_kwargs": {}, "response_metadata": {"finish_reason": "stop", "model_name": "gpt-4o-2024-11-20", "system_fingerprint": "fp_82ce25c0d4"}, "type": "ai", "name": null, "id": "run-92880019-a2d0-4bc9-af2d-a29ec83c14bb", "example": false, "tool_calls": [], "invalid_tool_calls": [], "usage_metadata": null}}}}

data: [DONE]

"""

# config = {"configurable": {"thread_id": "0"}}
# user_input = HumanMessage(content="dame el reporte general de oficinas")
# async_generator = call_graph(graph, user_input, config)
# # %%
# async for content in async_generator:
#     print(content, end="|")

# # %%

# user_input = HumanMessage(content="para el ultimo mes")
# async_generator = call_graph(graph, user_input, config)

# async for content in async_generator:
#     print(content, end="|")
# # %%
# user_input = HumanMessage(content="dame el ranking de ejecutivos")
# async_generator = call_graph(graph, user_input, config)

# async for content in async_generator:
#     print(content, end="|")
# # %%
# user_input = HumanMessage(content="para el ultimo mes")
# async_generator = call_graph(graph, user_input, config)

# async for content in async_generator:
#     print(content, end="|")

# # %%
# config = {"configurable": {"thread_id": "3"}}
# for event in graph.stream(
#     {"messages": [HumanMessage(content="hola")]},
#     config,
#     stream_mode="updates",
# ):
#     print(event["agent"]["messages"][-1].pretty_print())
#     print(graph.get_state(config).next == ())
# # %%
# config = {"configurable": {"thread_id": "4"}}
# user_input = HumanMessage(content="hola")
# node_to_stream = "agent"
# async for event in graph.astream_events(
#     {"messages": [user_input]}, config, version="v2"
# ):
#     # Get chat model tokens from node_to_stream
#     if (
#         event["event"] == "on_chat_model_stream"
#         and event["metadata"].get("langgraph_node", "") == node_to_stream
#     ):
#         data = event["data"]
#         print(data["chunk"].content, end="1")
# # %%
