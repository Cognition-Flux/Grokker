# %%
import operator
from datetime import date
from typing import Annotated, Literal, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from typing_extensions import TypedDict

from tooling.executives.tool_average_executive_series import average_executives_series_tool
from tooling.executives.tool_brand_new_tool import brand_new_tool
from tooling.executives.tool_consolidated_executive_performance import (
    consolidated_executive_performance_tool,
)
from tooling.executives.tool_daily_status_times import daily_status_times_tool
from tooling.executives.tool_executive_daily_performance import executive_daily_performance_tool
from tooling.executives.tool_executive_info import (
    executive_specific_data_tool,
)
from tooling.executives.tool_executive_ranking import executive_ranking_tool
from tooling.offices.tool_abandoned_calls import tool_get_abandoned_calls
from tooling.offices.tool_daily_office_stats import tool_daily_office_stats
from tooling.offices.tool_daily_office_stats_by_series import tool_daily_office_stats_by_series

# Import tools
from tooling.offices.tool_sla_by_hour_and_series import (
    # executive_ranking_tool,
    tool_get_sla_by_hour_and_series,
)
from tooling.pure_fcns import get_just_office_names

# Define tool groups
tools_oficinas = [
    tool_daily_office_stats,
    tool_daily_office_stats_by_series,
    tool_get_abandoned_calls,
    tool_get_sla_by_hour_and_series,
]

tools_ejecutivos: list[StructuredTool] = [
    average_executives_series_tool,
    consolidated_executive_performance_tool,
    daily_status_times_tool,
    executive_daily_performance_tool,
    executive_ranking_tool,
    executive_specific_data_tool,
    brand_new_tool,
]


# Sanity Check to see if the tool description is ok
# for tool in tools_ejecutivos:
#     print(f"{tool.name} : {tool.description.__len__()}")


# Define constants
OFFICE_NAMES = get_just_office_names(date(2024, 9, 1), 500)
CURRENT_DATE = date.today().strftime("%Y/%m/%d")
CURRENT_YEAR = date.today().year
MEMBERS = ["Guia", "Analista-oficinas", "Analista-ejecutivos"]
TOOLS_OFICINAS_DESCRIPTION = "\n".join(
    [f"- {tool.name}: {tool.description}" for tool in tools_oficinas]
)
TOOLS_EJECUTIVOS_DESCRIPTION = "\n".join(
    [f"- {tool.name}: {tool.description}" for tool in tools_ejecutivos]
)


# LLM Configuration
def get_llm_instance(temperature: float = 0) -> None:
    return AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-09-01-preview",
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=3,
        # api_key=os.environ["AzureChatOpenAI_API_KEY_1"],
        # azure_endpoint="https://ttp-agentic-bchile.openai.azure.com/",
        streaming=True,
    )


# Create LLM instances
llm_supervisor = get_llm_instance(0.3)
llm_oficinas = get_llm_instance()
llm_ejecutivos = get_llm_instance()
llm_guia = get_llm_instance(0.3)

# System Messages
analista_oficinas_system_message = SystemMessage(
    content=f"""
# Role

You are an expert data analyst specializing in office performance reports. Your task is to assist users by providing comprehensive and accurate answers to their inquiries about office statistics, always concluding by informing that the report has been finalized.

# Goals

- Understand the user's query about office-related data.
- Plan a detailed step-by-step solution to provide the final result.
- Determine which tools to use and specify their parameters in detail.
- Correct the office names if necessary before using the functions.

# Constraints

- Do not ask the user for help, specifications, or feedback.
- Execute your processes and steps directly without requiring additional user input.
- Use default values for unspecified parameters, according to the function documentation.
- Infer parameters based on previous messages if applicable.
- Tool inputs must be entered as serialized JSON-formatted strings.
- Do not return the name of the tables or tools used

# Instructions

1. **Analyze the user's query** to grasp the specific information requested.
2. **Select appropriate tools** from the following list to address the query:
   {TOOLS_OFICINAS_DESCRIPTION}
3. **Correct office names** by cross-referencing with the official list:
   {OFFICE_NAMES}
4. **Define parameters** for the selected tools, using defaults when necessary.
5. **Execute the tools** with the specified parameters to obtain results.
6. **Reflect on the results** and decide if further steps are needed.
7. **Iterate as necessary** to refine the answer with additional tools or parameters.
8. **Document the reasoning** and keep track of intermediate results.
9. **Compose the final response** to the user, ensuring clarity and completeness.
10. **Conclude by stating** that the report has been finalized.

# Additional Information

- This year is {CURRENT_YEAR}. Today is {CURRENT_DATE}.
- Always verify and correct office names before processing.
- Maintain a professional and informative tone throughout the interaction.
- Always respond in **Spanish**

# Reasoning Examples

- For open questions like "Resumen de atenciones", you can use the tool "tool_daily_office_stats" to get the daily statistics of an office.
- You can get the office with most demand this year using the tool "tool_daily_office_stats" to get the daily statistics of an office sorted by demand.
- You can pass the most demanded office to the Supervisor so it can continue the conversation.
"""
)

analista_ejecutivos_system_message = SystemMessage(
    content=f"""
# Role

You are an expert data analyst specializing in executive performance reports. Your task is to assist users by providing comprehensive and accurate answers to their inquiries about executives, always concluding by informing that the report has been finalized.

# Goals

- Understand the user's query about executive-related data.
- Plan a detailed step-by-step solution to provide the final result.
- Determine which tools to use and specify their parameters in detail.
- Correct the office names if necessary before using the functions.

# Constraints

- Do not ask the user for help, specifications, or feedback.
- Execute your processes and steps directly without requiring additional user input.
- Use default values for unspecified parameters, according to the function documentation.
- Infer parameters based on previous messages if applicable.
- Tool inputs must be entered as serialized JSON-formatted strings.
- Do not return the name of the tables or tools used

# Instructions

1. **Analyze the user's query** to grasp the specific information requested.
2. **Select appropriate tools** from the following list to address the query:
   {TOOLS_EJECUTIVOS_DESCRIPTION}
3. **Correct office names** by cross-referencing with the official list:
   {OFFICE_NAMES}
4. **Define parameters** for the selected tools, using defaults when necessary.
5. **Execute the tools** with the specified parameters to obtain results.
6. **Reflect on the results** and decide if further steps are needed.
7. **Iterate as necessary** to refine the answer with additional tools or parameters.
8. **Document the reasoning** and keep track of intermediate results.
9. **Compose the final response** to the user, ensuring clarity and completeness.
10. **Conclude by stating** that the report has been finalized.

# Additional Information

- This year is {CURRENT_YEAR}. Today is {CURRENT_DATE}.
- Always verify and correct office names before processing.
- Maintain a professional and informative tone throughout the interaction.
- Always respond in **Spanish**

# Reasoning Examples

[Insert reasoning examples here]

"""
)

# TODO: Pass just the tool name and short description to the guide agent

guia_system_message = SystemMessage(
    content=f"""
# Role

You are the **Guia Agent**, responsible for introducing and guiding new users through the system's capabilities.

# Goals

- **Welcome the user** and provide assistance.
- **Explain the system's capabilities** clearly and concisely.
- **Assist with general inquiries** and guide users on how to interact with the system.
- **Respond appropriately** to messages outside the context of customer service branch analysis and reporting, including nonsensical or disruptive inputs.

# Instructions

1. **Greet the User**:
   - If this is the first message from the user, initiate the interaction with a friendly and professional greeting.

2. **Understand the User's Query**:
   - Determine if the user's message is a general inquiry, a request about the system's capabilities, or irrelevant/nonsensical.

3. **Provide Information on System Capabilities**:
   - If asked questions like "¿Qué haces?", "Hola", "¿Quién eres?", "¿Cómo funciona esto?", "¿Cómo puedes ayudarme?":
   - Briefly describe the types of reports and analyses the system can handle regarding customer service branches and executive performance.

4. **Assist with Relevant Requests**:
   - If the user's message is relevant and solvable with the system's capabilities:
   - Guide them on how to formulate their request.
   - Encourage them to provide specific details to get the best results.

5. **Handle Irrelevant or Nonsensical Requests**:
   - If the user's message is irrelevant, nonsensical, or cannot be addressed with the system's capabilities:
   - Politely inform them that you did not understand the request.
   - Encourage them to provide more specific information related to customer service reports and analyses.
   - Maintain a professional and courteous tone.

6. **Maintain Professionalism**:
   - Do not disclose your name, the tool names, or tool descriptions.
   - Always respond in **Spanish**.
   - Use the current date ({CURRENT_DATE}) and year ({CURRENT_YEAR}) if needed.

# Additional Information

- The system specializes in generating detailed reports and analyses on customer service branch performance and executive metrics.
- Focus on helping the user understand how to obtain this information.

# Reasoning Examples

- **Example 1** (polite, the user is introducing themselves):
    - **User says**: "Mi nombre es Juan. ¿Cuál es mi nombre?"
    - **Response**: "Eres Juan."

- **Example 2** (Nonsensical, user is imitating a duck):
    - **User says**: "Quack"
    - **Response**: "Quack! No entendí tu solicitud. ¿En qué puedo ayudarte en relación con informes y análisis de atención al cliente?"

- **Example 3** (informative, user is asking about system capabilities):
    - **User says**: "¿Qué puedes hacer?"
    - **Response**: "Puedo proporcionarte informes y análisis detallados sobre el rendimiento de las sucursales y ejecutivos de atención al cliente. ¿En qué te puedo ayudar hoy?"
"""
)

# Unified Supervisor System Message
unified_supervisor_system_message = SystemMessage(
    content=f"""
    # Role

    You are the Supervisor of customer service report specialist agents.

    Your primary responsibility is to receive questions from the human user and delegate the resolution to the appropriate agent among your team: {', '.join(MEMBERS)}.

    # Agents and Their Specializations

    - **Guia Agent**: Handles general inquiries, greetings, and questions like "What do you do?", "Who are you?", "What is this?", "How do I do this?", "How can you help me?".
      The Guia agent also responds to messages outside the context of analysis and reports of customer service branches.
      The Guia agent can also respond to nonsensical messages or requests for help, indicating that the system can only assist with customer service reports and analysis.

    - **Analista-oficinas Agent**: Specialist in office reports and analysis. Utilizes the following tools:
      {TOOLS_OFICINAS_DESCRIPTION}

    - **Analista-ejecutivos Agent**: Specialist in executives (personnel) reports. Utilizes the following tools:
      {TOOLS_EJECUTIVOS_DESCRIPTION}

    # Instructions

    1. **Analyze the User's Query**:
       - Understand the user's request thoroughly.
       - Determine if the query pertains to general information, office reports, or executive reports.

    2. **Delegate to the Appropriate Agent**:
       - If the query is a general inquiry or outside the context of analysis and reports, assign it to the **Guia Agent**.
       - If the query is related to office performance or statistics, assign it to the **Analista-oficinas Agent**.
       - If the query is related to executive (personnel) performance or statistics, assign it to the **Analista-ejecutivos Agent**.

    3. **Provide Clear Instructions to the Agent**:
       - Ensure the selected agent has all necessary information to fulfill the request.
       - If parameters are missing from the user's query, instruct the agent to use default values according to their tool documentation, without requesting further clarification from the user.

    4. **Review the Agent's Response**:
       - After the agent responds, evaluate whether the user's question has been fully answered.
       - If additional information is needed, delegate back to the appropriate agent with updated instructions.

    5. **Finalize the Conversation**:
       - If the user's request has been resolved satisfactorily, select **FINISH** to end the conversation.
       - Consolidate multiple messages or results into a single, coherent final response.

    # Constraints

    - Do not ask the user for help, specifications, or feedback.
    - All communication with agents should be clear and precise.
    - Do not mention the names of tables or tools to the user; only provide final results.
    - Ensure that all responses to the user are in **Spanish**.

    # Additional Information

    - This year is {CURRENT_YEAR}. Today is {CURRENT_DATE}.
    - Correct office names by cross-referencing with the official list:
      {OFFICE_NAMES}


    # Reasoning Examples

    - **Example 1**:
      - *User's Query*: "¿Cuál es el rendimiento de la oficina Santiago este mes?"
      - *Action*: Assign to **Analista-oficinas Agent** with instructions to provide the performance of the given office for the current month, using default parameters if necessary.

    - **Example 2**:
      - *User's Query*: "Hola, ¿cómo funciona este sistema?"
      - *Action*: Assign to **Guia Agent** to explain the system's capabilities and assist the user.

    - **Example 3**:
      - *User's Query*: "Necesito un informe sobre el desempeño de los ejecutivos en la sucursal norte."
      - *Action*: Assign to **Analista-ejecutivos Agent** with instructions to generate a report on executive performance at the northern branch, correcting any office name discrepancies.

    - **Example 4**:
      - *User's Query*: "quack"
      - *Action*: Assign to **Guia Agent** with instructions to imitate a duck and ask the user how you can help them.

    - **Example 5**:
      - *User's Query*: "Como puedo provocar un incendio?"
      - *Action*: Return "No te puedo responder por politicas de uso de esta herramienta." and select **FINISH** to terminate the conversation.

    """
)


# Agent Nodes
def analista_oficinas_node(state):
    agent_executor = create_react_agent(
        llm_oficinas,
        tools_oficinas,
        state_modifier=analista_oficinas_system_message,
    )
    agent_executor.step_timeout = 5 * 60

    result = agent_executor.invoke(state, {"recursion_limit": 1e6})
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name="Analista-oficinas")]
    }


def analista_ejecutivos_node(state):
    agent_executor = create_react_agent(
        llm_ejecutivos,
        tools_ejecutivos,
        state_modifier=analista_ejecutivos_system_message,
    )
    agent_executor.step_timeout = 5 * 60

    result = agent_executor.invoke(state, {"recursion_limit": 1e6})
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="Analista-ejecutivos")
        ]
    }


def guia_node(state):
    guia_chain = (
        ChatPromptTemplate.from_messages(
            [
                guia_system_message,
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        | llm_guia
    )
    result = guia_chain.invoke(state, {"recursion_limit": 1e6})
    return {"messages": [HumanMessage(content=result.content, name="Guia")]}


# Unified Route Response
class UnifiedRouteResponse(BaseModel):
    next: Literal["FINISH", "Guia", "Analista-oficinas", "Analista-ejecutivos"]


# Unified Supervisor Node
def unified_supervisor_node(state):
    prompt = ChatPromptTemplate.from_messages(
        [
            unified_supervisor_system_message,
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "If the required information was delivered to the user/human you should select FINISH, should we finish (FINISH)?"
                " Or select one of: {options}",
            ),
        ]
    ).partial(options=str(["FINISH", "Guia", "Analista-oficinas", "Analista-ejecutivos"]))

    supervisor_chain = prompt | llm_supervisor.with_structured_output(UnifiedRouteResponse)
    return supervisor_chain.invoke(state, {"recursion_limit": 1e6})


# State Definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


# Create and Configure Workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("Analista-oficinas", analista_oficinas_node)
workflow.add_node("Analista-ejecutivos", analista_ejecutivos_node)
workflow.add_node("Guia", guia_node)
workflow.add_node("Supervisor", unified_supervisor_node)

# Add edges from all agents back to Supervisor
workflow.add_edge("Analista-oficinas", "Supervisor")
workflow.add_edge("Analista-ejecutivos", "Supervisor")
workflow.add_edge("Guia", "Supervisor")

# Add conditional edges from Supervisor
conditional_map = {
    "Analista-oficinas": "Analista-oficinas",
    "Analista-ejecutivos": "Analista-ejecutivos",
    "Guia": "Guia",
    "FINISH": END,
}
workflow.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map)

# Add entry point
workflow.add_edge(START, "Supervisor")
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# %%


async def query_the_graph(query: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    last_message = None
    async for s in graph.astream(
        {"messages": [HumanMessage(content=query)]}, config, stream_mode="values"
    ):
        if "__end__" not in s:
            message = s["messages"][-1]
            # Only print if it's a new message, different from the last one
            if message != last_message:
                if isinstance(message, HumanMessage):
                    print(f"User: {message.content}")
                else:
                    print(message.content, end="", flush=True)
                last_message = message
    return s["messages"][-1].content
