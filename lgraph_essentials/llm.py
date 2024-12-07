# %%
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatAnthropic(
    model="claude-3-5-haiku-latest",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_retries=5,
    # model_kwargs={
    #     "streaming": False,
    # },
)


# llm.invoke("hola")
