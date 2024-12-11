# %%
# SDK

from langgraph_sdk import get_client
from langgraph.pregel.remote import RemoteGraph
from langchain_core.messages import convert_to_messages
from langchain_core.messages import HumanMessage, SystemMessage

url_for_cli_deployment = "http://localhost:8123"
client = get_client(url=url_for_cli_deployment)

# %%
# Remote Graph

url = "http://localhost:8123"
graph_name = "task_maistro"
remote_graph = RemoteGraph(graph_name, url=url)
# %%
# create thread
thread = await client.threads.create()  # noqa: F704
# %%
# check runs on a thread

runs = await client.runs.list(thread["thread_id"])
print(runs)
# %%

# Ensure we've created some ToDos and saved them to my user_id
user_input = "Add a ToDo to finish booking travel to Hong Kong by end of next week. Also, add a ToDo to call parents back about Thanksgiving plans."
config = {"configurable": {"user_id": "Test"}}
graph_name = "task_maistro"
run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input)]},
    config=config,
)
# %%
# Kick off a new thread and a new run
thread = await client.threads.create()
user_input = "Give me a summary of all ToDos."
config = {"configurable": {"user_id": "Test"}}
graph_name = "task_maistro"
run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input)]},
    config=config,
)
# %%
# Wait until the run completes
await client.runs.join(thread["thread_id"], run["run_id"])
print(await client.runs.get(thread["thread_id"], run["run_id"]))
# %%
user_input = "What ToDo should I focus on first."
print(f"{thread['thread_id']=}")
print(f"{graph_name=}")
async for chunk in client.runs.stream(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input)]},
    config=config,
    stream_mode="messages-tuple",
):
    if chunk.event == "messages":
        print(
            "".join(
                data_item["content"]
                for data_item in chunk.data
                if "content" in data_item
            ),
            end="",
            flush=True,
        )
# %%
"""


curl http://localhost:8123/threads/90ad2fef-e6fc-4600-92b3-d34b268a33ac/runs/stream\
  --request POST \
  --header 'Content-Type: application/json' \
  --data '{
  "assistant_id": "task_maistro",
  "input": {
    "messages": [
      {
        "role": "user",
        "content": "What ToDo should I focus on first"
      }
    ]},
"config": {
    "recursion_limit": 100,
     "configurable": {"user_id": "Test"}
  },
    "stream_mode": [
    "messages-tuple"
  ]
}'


"""
