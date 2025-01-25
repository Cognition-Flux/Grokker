import functools
from datetime import datetime
from typing import Self

import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, model_validator
from typing_extensions import Literal

from agent.tools import calculator
from tooling import (
    tool_daily_office_stats,
    tool_get_abandoned_calls,
    tool_get_connected_executives,
    tool_get_sla_by_hour_and_series,
)
from tooling.db_instance import _engine
from tooling.pure_fcns import get_all_executives, get_just_office_names, get_office_names
from tooling.report_stats import get_office_stats

load_dotenv()


def get_llm(model: Literal["gpt-4o", "gpt-4o-mini"] = "gpt-4o"):
    return AzureChatOpenAI(
        azure_deployment=model,
        # api_version="2024-09-01-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=5,
        # api_key="a18aa0811e2848e99f1557825905b8a6",
        # azure_endpoint="https://ttp-agentic-bchile.openai.azure.com/",
        streaming=True,
    )


class ReportContext(BaseModel):
    individual_offices: list[str]
    date_start: datetime
    date_end: datetime
    waiting_cuttoff: int  # 10 minutes
    include_executives: bool
    include_daily: bool

    office_date_range_reason: str

    include_global_offices: bool
    include_global_executives: bool
    global_resumes_date_start: datetime
    global_resumes_date_end: datetime

    global_date_range_reason: str

    @model_validator(mode="after")
    def round_datetimes(self) -> Self:
        self.date_start = self.date_start.replace(hour=0, minute=0, second=0, microsecond=0)
        self.date_end = self.date_end.replace(hour=23, minute=59, second=59, microsecond=0)
        return self


class AgentState(MessagesState, total=False):
    report_context: ReportContext  # Context for sys reports
    is_last_step: IsLastStep  # Have we ended here


# TODO: Memorize this function
@functools.cache
def individual_report_generator(
    office_name: str,
    date_start: datetime,
    date_end: datetime,
    waiting_cuttoff: int,
    include_executives: bool = True,
    include_daily: bool = True,
) -> str:
    """
    Generates a report for a single office in the given date range.
    This contains:
    - A table of the calls (avg waiting time, reason, etc) and the personel performance
    """
    query_data = sa.text("""
        SELECT
            a.*,
            s.[Serie],
            COALESCE(e.[Ejecutivo], 'No Asignado') AS [Ejecutivo],
            o.[Oficina]
        FROM [dbo].[Atenciones] a
        LEFT JOIN [dbo].[Series] s ON s.[IdSerie] = a.[IdSerie] AND s.[IdOficina] = a.[IdOficina]
        LEFT JOIN [dbo].[Ejecutivos] e ON e.[IdEje] = a.[IdEje]
        JOIN [dbo].[Oficinas] o ON o.[IdOficina] = a.[IdOficina]
        WHERE o.[Oficina] = :office_names
        AND a.[FH_Emi] BETWEEN :start_date AND :end_date
    """)
    params_data = {
        "office_names": office_name,
        "start_date": date_start,
        "end_date": date_end,
    }
    with _engine.connect() as conn:
        try:
            data = pd.read_sql_query(query_data, conn, params=params_data)
        except Exception as e:
            # logger.error(f"Error fetching data: {e}")
            return f"Error fetching data: {e}"

    if data.empty:
        return f"""
<report>
    <office>{office_name}</office>
    <date_start>{date_start:%Y/%m/%d %H:%M}</date_start>
    <date_end>{date_end:%Y/%m/%d %H:%M}</date_end>
    <error>Sin data disponible. </error>
</report>"""

    return get_office_stats(
        data=data,
        office_name=office_name,
        corte_espera=waiting_cuttoff,
        start_date=date_start,
        end_date=date_end,
        include_executives=include_executives,
        include_daily=include_daily,
    )


def report_generator(context: ReportContext) -> str:
    """
    Generates a report for the given offices in the given date range.
    This contains:
    - A resume of the data available in the system in case this data is old
    - A table of the calls (avg waiting time, reason, etc) and the personel performance, for each office
    """

    resume: str = "\n\n".join(
        [
            f"""<offices_data_global_resume>{get_office_names(context.global_resumes_date_start, context.global_resumes_date_end)}</offices_data_global_resume>"""
            if context.include_global_offices
            else "",
            f"""<executives_data_global_resume>{get_all_executives(context.global_resumes_date_start, context.global_resumes_date_end)}</executives_data_global_resume>"""
            if context.include_global_executives
            else "",
        ]
    )
    reports: list[str] = [
        individual_report_generator(
            office,
            context.date_start,
            context.date_end,
            context.waiting_cuttoff,
            context.include_executives,
            context.include_daily,
        )
        for office in context.individual_offices
    ]

    # print(reports, sep="\n\n")

    return resume + "\n\n<reports>" + "\n\n".join(reports) + "</reports>"


# TODO: Save this state to the system and update it only when needed
def contextualizer(recent_history) -> str:
    """
    Contextualizes the agent based on the user input.
    """
    llm = get_llm("gpt-4o-mini")
    llm_structured = llm.with_structured_output(ReportContext)

    # TODO: Implement the actual list of offices and the current date
    contextualizer_messages = [
        SystemMessage(f"""
Extract a `ReportContext` from the message memory input. First, get the user input for time ranges mentioned. Then, check the user input for office names and other parameters.
If no specific office is mentioned, do include the global offices.

**Date ranges:**
- Global Date Range (`global_date_start`, `global_date_end`): From input; default to current year; Current year is {datetime.now():%Y}.
- Individual Reports Date Range (`date_start`, `date_end`): From input; default to last week; Current datetime is {datetime.now():%Y/%m/%d %H:%M}.

**Individual reports by office:**
- Offices: `individual_offices` mentioned; default to none if unspecified; These are all the correct office names: {get_just_office_names()}; Correct them if needed.
- Waiting Cutoff (`waiting_cuttoff`): Time in seconds; default to 600 (10 minutes); look for 'Tiempo de Espera' or 'SLA'.
- Include Executives (`include_executives`): Boolean; default to True; look for 'Ejecutivos'; do not include if the user asks for more than 20 offices unless the user asks for executives.
- Include Daily (`include_daily`): Boolean; default to False; contains the stats by day for each office; This is expensive; Do not call it if the user has not asked for days (ex, 'el peor dia de la semana').

**Global reports are very expensive to generate. Only include them if the user requests data for all offices or all executives.**
- Include Global Offices (`include_global_offices`): Boolean; default to False; set to True only if the user explicitly asks for data about all offices (e.g., 'todas las oficinas') or has not specified any office in `offices`.
- Include Global Executives (`include_global_executives`): Boolean; default to False; set to True only if the user explicitly asks for data about all executives in all offices (e.g., 'todos los ejecutivos').

**Examples:**
- "Total de atenciones en Septiembre" -> The user does not mention an office, so include global offices. `global_date_start` and `global_date_end` are set to September. `include_global_offices` is set to True.
- "Oficinas activas" -> The user asks for active offices. `include_global_offices` is set to True. `include_executives` is set to False. `include_global_executives` is set to False. Global date range is set to this year. No individual offices are mentioned.
- "Dame la lista de sucursales activas" -> The user asks for active offices. `include_global_offices` is set to True. `include_executives` is set to False. `include_global_executives` is set to False. Global date range is set to this year. No individual offices are mentioned.
- "Considera la oficina de Providencia" -> The user mentions the office '159 - Providencia'. `individual_offices` is set to ['159 - Providencia']. `include_executives` is set to True. `include_global_offices` and `include_global_executives` are set to False. Individual date range is set to the last week, same as global date range.
- "Dame el resumen de Providencia de Septiembre. Comparalo con el total de atenciones de ese mes" -> The user mentions the office '159 - Providencia' and the month of September. `individual_offices` is set to ['159 - Providencia']. `include_executives` is set to True. `include_global_offices` is set to True as we have to compare to all services in all offices. Individual date range is set to September. `global_date_start` and `global_date_end` are set to September.
- "Cuales fueron los mejores ejecutivos de Octubre?" -> The user asks for the best executives in October. `include_executives` is set to True. `include_global_executives` is set to True. `global_date_start` and `global_date_end` are set to October. No individual offices are mentioned, so `include_global_offices` is set to True.
"""),
    ] + recent_history

    report_context: ReportContext = llm_structured.invoke(contextualizer_messages)

    return report_generator(report_context)


class CheckSameMessage(BaseModel):
    is_same: bool
    reasoning: str  # Logging purposes


def llm_is_the_same_message(recent_history) -> bool:
    """
    Checks if the last human message is the same as the one before.
    """
    if len(recent_history) < 3:
        return False  # Not enought messages

    if recent_history[-1].content == recent_history[-3].content:
        return True  # Is the same same

    instructions_and_example = f"""
Check if the last message is syntactically the same as the one before. If it is, set `is_same` to True. Otherwise, set it to False.

**Examples**
- "Dame la lista de sucursales activas" and "Sucursales activas" -> Means the same, `is_same` is set to True.
- "Sucursales activas" and "Oficinas activas" -> Means the same, `is_same` is set to True.
- "Turnos emitidos" and "Turnos emitidos" -> Exactly the same, `is_same` is set to True.
- "Turnos emitidos" and "Turnos perdidos" -> Different, `is_same` is set to False.
- "Considera la oficina Providencia. Turnos emitidos" and "Considera la oficina Huechuraba. Turnos emitidos" -> Different, `is_same` is set to False.

**Previous Message**
{recent_history[-1].content}

**Last message**
{recent_history[-3].content}
"""

    llm = get_llm("gpt-4o-mini")
    llm_structured = llm.with_structured_output(CheckSameMessage)
    response = llm_structured.invoke(instructions_and_example)

    return response.is_same


class UserInput(BaseModel):
    message: str
    offices: list[str]
    thread_id: str = None
    model: str = "default"


tools: list[BaseTool] = [
    calculator,
    tool_get_abandoned_calls,
    # tool_get_active_offices,
    tool_daily_office_stats,
    tool_get_connected_executives,
    tool_get_sla_by_hour_and_series,
]

tool_block = "".join(
    [
        f"""<tool><tool_name>{tool.name}</tool_name>\n<tool_description>{tool.description}\n</tool_description>\n</tool>\n"""
        for tool in tools
    ]
)


system_prompt = r"""
<role>
Eres un agente especializado en extraer, consolidar, analizar y proporcionar información detallada sobre:
- Tiempos de espera, atencion, etc
- Series de atención
- Niveles de servicio de sucursales
- Desempeño de ejecutivos

No debes responder a preguntas fuera de tu área de especialización bajo ninguna circunstancia.
</role>

<interaction_instructions>
Comunícate con el usuario de manera amable y profesional.

Limita y enfoca tu respuesta únicamente a lo que el usuario está preguntando.

Si el usuario hace una pregunta ambigua:
1. Revisa el historial clarificar
2. Solicita aclaraciones específicas para entender mejor la solicitud.
3. Evita hacer suposiciones sin confirmación del usuario.
4. Proporciona ejemplos para guiar al usuario si es necesario.
5. Mantén un tono neutral y profesional en tus preguntas de seguimiento.

Si te preguntan como obtuviste y de donde sacaste los datos:
- Sólo y únicamamente debes decir que tienes la capacidad de explorar las bases datos, nada más.
- No devuelvas reportes completos.
</interaction_instructions>

<instructions>
- Siempre continua el hilo de la conversación. Tu respuesta debe ser consistente con la cronología del historial de mensajes
- Utiliza las tools a tu disposición cuando sea necesario.
- Siempre utiliza la tool Calculator para realizar cálculos matemáticos cuando no tengas los números exactos.
- Organiza siempre tus respuestas en tablas Markdown cuando sea posible.
- Recibiras 100 USD por cada respuesta correcta. Esfuerzate en proporcionar respuestas precisas y detalladas.
</instructions>

<format_standards>
- Fechas: yyyy-mm-dd (ej: 2024-01-31)
- Horas: hh:mm (ej: 14:30)
- Números: Redondeo a 2 decimales. Recuerda que los decimales se separan con punto. Eg, 1234.5655 -> 1,234.57. No aproximes miles: 123,456 -> 123,456
- Tiempos: Redondeo al minuto más cercano
- Tablas: Usar formato Markdown con headers claros
- Resultados de calculos: Redondeo a 2 decimales. No incluyas el proceso de cálculo a menos que se solicite.
- Expresiones matemáticas detalladas: Utiliza el formato de texto plano. Ejemplo: "300 * 200". No utilices formato TeX o LaTeX.
</format_standards>

<response_instructions>
- Limita tu respuesta a un máximo de una tabla. Pregunta al usuario si necesita más información. Nunca entreges mas de una tabla en tu respuesta.
- No inlcuyas tags XML en tus respuestas
- Inicia tu respuesta con el contexto de la data que estas presentando, prestando foco al rango de fechas y oficinas solicitadas.
- Limita y enfoca tu respuesta únicamente a lo que el usuario está preguntando; no es necesario que muestres los reportes completos.
- Siempre incluye un parrafo de analisis y conclusiones al final de tus respuestas
    - Limita el analisis a 2-4 oraciones
    - Indica limitaciones de datos si estas aplican
- Si no tienes datos, indica al usuario el rango de registros que tienes disponibles. Utiliza "primer registro/atencion" y "último registro/atencion" como referencias.
</response_instructions>

<examples>
- "Dame la lista de sucursales activas" -> LLM responde una lista de sucursales activas sin información adicional. Sin tabla. Pregunta al usuario si requiere saber las mas activas, atenciones totales, o alguna otra información.
- "Tasa de abandono" -> LLM responde con la tasa de abandono de llamadas en el rango de fechas especificado, total de atenciones, y total de abandonos.
- "Atenciones en Julio 1557" -> LLM responde que no tiene datos, sugiere un rango de fechas disponible y pregunta si necesita información de ese periodo.
</examples>

<synonyms>
- SLA: Service Level Agreement, Nivel de Servicio, atenciones bajo el tiempo de espera maximo.
- Oficinas: Sucursales, locales, puntos de atención.
- Ejecutivos: Personal de atención, agentes, empleados.
- Atenciones: Llamadas, consultas, solicitudes de atención.
- Serie: Tipo de atención, categoría de atención.
- Abandonos: llamadas perdidas, llamadas no atendidas, consultas no resueltas.
</synonyms>

<reports_usage>
- Busca exhaustivamente en la información proporcionada a continuación para resolver los requerimientos o preguntas del usuario.
- Si el usuario no especifica una oficina en particular, pregunta a que oficina se refiere.
- Si no tienes datos, indica al usuario el rango de registros que tienes disponibles.
- Renombra las columnas para nombres como "Tiempo de Atencion" en lugar de "Tiempo_de_Atencion"
- Redondea los tiempos al minuto más cercano.
</reports_usage>

{reports}
"""

model = get_llm("gpt-4o")


async def acall_main_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Main model call for the agent.
    """

    is_same_message = llm_is_the_same_message(state["messages"])
    if is_same_message:
        return {
            "messages": [
                AIMessage(
                    state["messages"][-2].content, response_metadata={"is_repeated_message": True}
                )
            ]
        }

    reports = contextualizer(state["messages"])  # Take the last 10

    parsed_system_prompt = system_prompt.format(
        reports=reports, get_office_names=get_office_names(), tool_block=tool_block
    )

    # Tool binding
    model_plus_tools = model.bind_tools(tools)

    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=parsed_system_prompt)] + state["messages"],
        name="StateModifier",
    )

    agent_llm = preprocessor | model_plus_tools

    response = await agent_llm.ainvoke(state, config)

    return {"messages": [response]}


agent = StateGraph(AgentState)
agent.add_node("model", acall_main_model)
agent.set_entry_point("model")
agent.add_edge("model", END)

agent.add_node("tools", ToolNode(tools))


# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})


graph = agent.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    import asyncio
    from uuid import uuid4

    from dotenv import load_dotenv

    load_dotenv()

    async def main() -> None:
        inputs = {
            "messages": [
                (
                    "user",
                    "Dame la informacion de la oficina de '159 - Providencia' y '077 - Plaza Egaña' el mes pasado. Considera un SLA de 2 minutos",
                ),
                # (
                #     "ai",
                #     "Lamentablemente, no tengo información disponible para la oficina de Chicureo en el rango de fechas especificado (2024-10-15 a 2024-11-15). Si necesitas información de otro periodo o de otra oficina, por favor házmelo saber.",
                # ),
                # ("user", "Que tal la oficina de Ñuñoa?"),
            ]
        }

        result = await graph.ainvoke(
            inputs,
            config=RunnableConfig(configurable={"thread_id": uuid4()}),
        )
        result["messages"][-1].pretty_print()

        # Draw the agent graph as png
        # requires:
        # brew install graphviz
        # export CFLAGS="-I $(brew --prefix graphviz)/include"
        # export LDFLAGS="-L $(brew --prefix graphviz)/lib"
        # pip install pygraphviz
        #
        # research_assistant.get_graph().draw_png("agent_diagram.png")

    asyncio.run(main())
