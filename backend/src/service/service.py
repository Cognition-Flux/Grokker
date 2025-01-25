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

import service.app_insights

# from agent import research_assistant as office_assistant
from agent import graph as office_assistant

# from agent.ttp_agentic.multiagent_graph import graph
# from agent.ttp_agentic.multiagent_graph_v2 import system_prompt_prohibited_actions
from agent.ttp_agentic.multiagent_graph_v2 import graph
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

service.app_insights.run_empty()


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
            if (
                event["event"] == "on_chat_model_stream"
                and event["metadata"].get("langgraph_node", "") == node_to_stream
            ):
                data = event["data"]
                if data["chunk"].content:
                    yield data["chunk"].content


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.")),
    ],
) -> None:
    if http_auth.credentials != os.getenv("AUTH_SECRET"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


bearer_depend = [Depends(verify_bearer)] if os.getenv("AUTH_SECRET") else None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Construct agent with Sqlite checkpointer
    async with AsyncMongoDBSaver.from_conn_string(
        os.getenv("MONGO_CONNECTION_STRING")
    ) as saver:
        graph.checkpointer = saver
        app.state.agent = graph
        yield
    # context manager will clean up the AsyncSqliteSaver on exit


app = FastAPI(lifespan=lifespan)
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


def _remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content."""
    if isinstance(content, str):
        return content
    # Currently only Anthropic models stream tool calls, using content item type tool_use.
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str)
        or (content_item["type"] != "tool_use" and content_item["type"] != "tool")
    ]


@router.post("/invoke")
async def invoke(user_input: UserInput) -> ChatMessage:
    """
    Invoke the agent with user input to retrieve a final response.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    agent: CompiledStateGraph = graph  # app.state.agent
    kwargs, run_id = _parse_input(user_input)
    try:
        response = await agent.ainvoke(**kwargs)
        output = ChatMessage.from_langchain(response["messages"][-1])
        output.run_id = str(run_id)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def message_generator(user_input: StreamInput) -> AsyncGenerator[str, None]:
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

    node_to_stream: str = "agent"

    # FIXME: This should be completly async, but instead the get_state is blocking the thread

    latest_checkpoint = await graph.aget_state(config)

    # if graph.get_state(config).next == () or not graph.get_state(config).next:
    if latest_checkpoint.next == () or not latest_checkpoint.next:
        # Process streamed events from the graph and yield messages over the SSE stream.
        async for event in graph.astream_events(
            {"messages": [mensaje_del_usuario]}, config, version="v2"
        ):

            # Yield messages written to the graph state after node execution finishes.
            if (
                event["event"] == "on_chain_end"
                # on_chain_end gets called a bunch of times in a graph execution
                # This filters out everything except for "graph node finished"
                and any(t.startswith("graph:step:") for t in event.get("tags", []))
                and "messages" in event["data"]["output"]
                and event["metadata"].get("langgraph_node", "") == node_to_stream
            ):
                new_messages = event["data"]["output"]["messages"]

                print(f"#### 11111 {new_messages=}")
                for message in new_messages:
                    if not isinstance(message, AIMessage):
                        continue
                    print(f"1111 {message=}")
                    message.pretty_print()
                    try:
                        chat_message = ChatMessage.from_langchain(message)
                        print(f"1111 {chat_message=}")
                        chat_message.run_id = str(run_id)
                        print(f"1111 {chat_message.run_id=}")
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                        continue

                    try:
                        if message.tool_calls:
                            for tool_call in message.tool_calls:
                                print(
                                    f"111111 Tool Call: {tool_call} {type(tool_call)}"
                                )
                                if (
                                    tool_call["type"] == "tool_call"
                                    and tool_call["name"] == "AskHuman"
                                ):
                                    question = tool_call["args"]["question"]
                                    print(f"Question: {question}")
                                    chat_message.content = question

                        yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"
                    except Exception as e:
                        print(f"Error: {e}")
                        continue
        yield "data: [DONE]\n\n"

    else:
        async for event in graph.astream_events(
            Command(resume=mensaje_del_usuario), config, version="v2"
        ):
            if (
                event["event"] == "on_chain_end"
                # on_chain_end gets called a bunch of times in a graph execution
                # This filters out everything except for "graph node finished"
                and any(t.startswith("graph:step:") for t in event.get("tags", []))
                and "messages" in event["data"]["output"]
                and event["metadata"].get("langgraph_node", "") == node_to_stream
            ):
                new_messages = event["data"]["output"]["messages"]
                print(f"####  2222 {new_messages=}")
                for message in new_messages:
                    if not isinstance(message, AIMessage):
                        continue
                    print(f"2222 {message=}")
                    message.pretty_print()
                    try:
                        chat_message = ChatMessage.from_langchain(message)
                        print(f"2222 {chat_message=}")
                        chat_message.run_id = str(run_id)
                        print(f"2222 {chat_message.run_id=}")
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                        continue
                    try:
                        if message.tool_calls:
                            for tool_call in message.tool_calls:
                                print(
                                    f"111111 Tool Call: {tool_call} {type(tool_call)}"
                                )
                                if (
                                    tool_call["type"] == "tool_call"
                                    and tool_call["name"] == "AskHuman"
                                ):
                                    question = tool_call["args"]["question"]
                                    print(f"Question: {question}")
                                    chat_message.content = question
                        yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"
                    except Exception as e:
                        print(f"Error: {e}")
                        continue
        yield "data: [DONE]\n\n"


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


# @app.post(
#     "/stream", response_class=StreamingResponse, responses=_sse_response_example()
# )
# async def stream_agent(user_input: StreamInput) -> StreamingResponse:

#     async def generate() -> typing.AsyncIterator[str]:
#         accumulated_content = []
#         async for content in call_graph(graph, user_input):
#             accumulated_content.append(content)
#             yield f'data: {{"type": "token", "content": "{content}"}}\n\n'
#         full_message = "".join(accumulated_content)
#         yield f'data: {{"type": "message", "content": "{full_message}"}}\n\n'
#         yield "data: [DONE]\n\n"

#     return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


@router.post("/history")
def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    agent: CompiledStateGraph = app.state.agent
    try:
        state_snapshot = agent.get_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = []
        for message in messages:
            chat_messages.append(ChatMessage.from_langchain(message))
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
