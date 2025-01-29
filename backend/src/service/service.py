import json
import os
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from langsmith import Client as LangsmithClient
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
from agents.grokker.multiagent_graph_v2 import graph
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    StreamInput,
    UserInput,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)

PymongoInstrumentor().instrument()

warnings.filterwarnings("ignore", category=LangChainBetaWarning)


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.")),
    ],
) -> None:
    if http_auth.credentials != os.getenv("AUTH_SECRET"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


bearer_depend = [Depends(verify_bearer)] if os.getenv("AUTH_SECRET") else None
app = FastAPI()


router = APIRouter(dependencies=bearer_depend)


def _parse_input(user_input: UserInput) -> tuple[dict[str, Any], str]:
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    input_message = ChatMessage(type="human", content=user_input.message)
    kwargs = {
        "input": {"messages": [input_message.to_langchain()]},
        "config": RunnableConfig(
            configurable={"thread_id": thread_id, "model": user_input.model},
            run_id=run_id,
        ),
    }
    return kwargs, run_id


async def message_generator_v2(user_input: StreamInput) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    kwargs, run_id = _parse_input(user_input)
    mensaje_del_usuario = kwargs["input"]["messages"][-1]

    # if os.getenv("DEV_CHECKPOINTER"):
    #     config = {"configurable": {"thread_id": "0"}}
    # else:
    #     config = {"configurable": {"thread_id": kwargs["config"]["configurable"]["thread_id"]}}

    config = {
        "configurable": {"thread_id": kwargs["config"]["configurable"]["thread_id"]}
    }

    # node_to_stream: str = "agent"

    # FIXME: This should be completly async, but instead the get_state is blocking the thread

    latest_checkpoint = await graph.aget_state(config)

    # if graph.get_state(config).next == () or not graph.get_state(config).next:
    # if latest_checkpoint.next == () or not latest_checkpoint.next:
    print(
        f"#--------------------------------{mensaje_del_usuario=}-------------------------------------------------------#"
    )
    # Process streamed events from the graph and yield messages over the SSE stream.
    print(f"## INICIO: Próximo paso del grafo: {latest_checkpoint.next}")
    message = None
    # while not message:
    async for event in graph.astream_events(
        (
            {"messages": [mensaje_del_usuario]}
            if not latest_checkpoint.next
            else Command(resume=mensaje_del_usuario)
        ),
        config,
        version="v2",
    ):
        # print("-----------------------------------")
        # print(f"***event***: {event}")
        # print("-----------------------------------")
        if (
            "on_chat_model_end" in event["event"]
            and event["metadata"].get("langgraph_node", "") == "context_request_agent"
        ):
            print(f"#-------------context_request_agent: {event['data']['output']=}")
            message = event["data"]["output"]
            print(f"#-------------context_request_agent: {message=}")
            if message:
                print(f"Procesando mensaje: {message=}")
                message.pretty_print()
                try:
                    chat_message = ChatMessage.from_langchain(message)
                    print(f"1111 {chat_message=}")
                    chat_message.run_id = str(run_id)
                    print(f"1111 {chat_message.run_id=}")
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                    continue

                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

        if (
            "on_chat_model_end" in event["event"]
            and event["metadata"].get("langgraph_node", "") == "guidance_agent"
        ):
            print(f"#-------------guidance_agent: {event['data']['output']=}")
            if len(event["data"]["output"].tool_calls) > 0:
                if (
                    event["data"]["output"].tool_calls[0]["name"]
                    == "GuidanceAgentAskHuman"
                ):
                    message = event["data"]["output"].tool_calls[0]["args"][
                        "question_for_human"
                    ]
                    message = AIMessage(content=message)
                    print(f"#-------------guidance_agent: {message=}")
                    if message:
                        print(f"Procesando mensaje: {message=}")
                        message.pretty_print()
                        try:
                            chat_message = ChatMessage.from_langchain(message)
                            print(f"1111 {chat_message=}")
                            chat_message.run_id = str(run_id)
                            print(f"1111 {chat_message.run_id=}")
                        except Exception as e:
                            yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                            continue

                        yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            else:
                message = event["data"]["output"]
                print(f"#-------------guidance_agent: {message=}")
                if message:
                    print(f"Procesando mensaje: {message=}")
                    message.pretty_print()
                    try:
                        chat_message = ChatMessage.from_langchain(message)
                        print(f"1111 {chat_message=}")
                        chat_message.run_id = str(run_id)
                        print(f"1111 {chat_message.run_id=}")
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                        continue

                    yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

        if (
            "on_chat_model_end" in event["event"]
            and event["metadata"].get("langgraph_node", "") == "analyst_agent"
        ):
            print(f"#-------------analyst_agent: {event['data']['output']=}")
            message = event["data"]["output"]
            print(f"#-------------analyst_agent: {message=}")

            if message.content != "":
                print(f"Procesando mensaje: {message=}")
                message.pretty_print()
                try:
                    chat_message = ChatMessage.from_langchain(message)
                    print(f"1111 {chat_message=}")
                    chat_message.run_id = str(run_id)
                    print(f"1111 {chat_message.run_id=}")
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                    continue

                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

    yield "data: [DONE]\n\n"
    print(
        f"## FINAL: Próximo paso del grafo: {latest_checkpoint.next}--------------------------------------------------------------------#"
    )


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


# TODO: add cache here!
@router.post(
    "/stream", response_class=StreamingResponse, responses=_sse_response_example()
)
async def stream_agent(user_input: StreamInput) -> StreamingResponse:
    """
    Stream the agent's response to a user input, including intermediate messages and tokens.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    """
    return StreamingResponse(
        message_generator_v2(user_input), media_type="text/event-stream"
    )


app.include_router(router)

# region Utilities

utilities_router = APIRouter()
"""Utilites router for endpoints not directly related to the agent."""


@utilities_router.get("/model-graph", response_model=str)
def get_model_graph():
    return office_assistant.get_graph().draw_mermaid()


@utilities_router.get("/health")
def healt() -> dict[str, str]:
    return {"status": "ok"}


@utilities_router.get("/offices")
def get_offices():
    from tooling.db_instance import get_offices

    return get_offices()


@utilities_router.get("/last-db-update")
def get_last_db_update():
    from tooling.db_instance import get_last_database_update

    return {"last_update": get_last_database_update()}


app.include_router(utilities_router, include_in_schema=True)
# endregion
app.add_middleware(
    CORSMiddleware,
    # allow_origins=cors_origins,
    allow_origin_regex=r"http://localhost(:\d+)?",  # Permite http://localhost con cualquier puerto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add instrumentation
FastAPIInstrumentor.instrument_app(app, excluded_urls="offices,health")
