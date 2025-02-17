# %%
import os
from typing import Literal

from bs4 import BeautifulSoup as Soup
from dotenv import load_dotenv
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool, tool
from langchain_experimental.utilities import PythonREPL
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field
import pandas as pd
from typing import List
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableLambda


def describe_dataframe(data_df):
    """
    Elabora una descripción estadística de las variables numéricas y categóricas presentes en data_df,
    mostrando además los valores únicos específicos de cada variable categórica.

    Args:
        data_df (pd.DataFrame): DataFrame con los datos.

    Returns:
        str: Cadena de texto con la descripción que incluye:
             - Variables numéricas: count, mean, std, min, 25%, 50%, 75%, max.
             - Variables categóricas: count, unique, top, freq (si existen).
             - Los unique values de cada variable categórica.
             - Conteo de valores faltantes en cada variable.
    """
    # Descripción de variables numéricas
    numeric_desc = data_df.describe().to_string()

    # Filtrar variables categóricas (tipo object o category)
    categorical_cols = data_df.select_dtypes(include=["object", "category"])
    if categorical_cols.empty:
        categorical_desc = "No hay variables categóricas en el DataFrame."
        unique_categorical_str = (
            "No hay variables categóricas para mostrar sus unique values."
        )
    else:
        categorical_desc = categorical_cols.describe().to_string()

        # Obtener y formatear los unique values para cada variable categórica
        unique_lines = []
        for col in categorical_cols.columns:
            # Se omiten los NaN para que no los aparezcan como unique values
            unique_values = categorical_cols[col].dropna().unique()
            unique_values_str = ", ".join(map(str, unique_values))
            unique_lines.append(f"{col}: {unique_values_str}")
        unique_categorical_str = "\n".join(unique_lines)

    # Conteo de valores faltantes por variable
    missing_values = data_df.isnull().sum().to_string()

    # Construir la cadena final con todos los componentes
    description = (
        "=== Descripción de variables numéricas ===\n"
        + numeric_desc
        + "\n\n=== Descripción de variables categóricas ===\n"
        + categorical_desc
        + "\n\n=== Unique values de variables categóricas ===\n"
        + unique_categorical_str
        + "\n\n=== Conteo de valores faltantes en cada variable ===\n"
        + missing_values
    )
    return description


load_dotenv(override=True)

print(f"{os.environ['GROQ_API_KEY']=}")


data_df = pd.read_csv("backend/src/tooling/social_media_entertainment_data.csv").dropna(
    subset=["Country"], inplace=False
)
df_description = describe_dataframe(data_df)
primary_libs = " ".join([lib for lib in ["scipy", "numpy", "pandas"]])


def get_llm(model: str = "deepseek-r1-distill-llama-70b"):
    return ChatGroq(
        model=model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.environ["GROQ_API_KEY"],
    )


code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a python programmer and data analyst who primarily uses these libraries: {primary_libs} to programmatically analyze the dataframe named 'data_df', which is already loaded in your environment. This is the description of the dataframe: {df_description}

When constructing your answer, you must strictly invoke the provided tool 'code' with the following JSON schema:
{{
    "prefix": <string>,  // A brief explanation of your solution.
    "imports": <string>, // All necessary import statements.
    "code": <string>     // All remaining code excluding imports.
}}

Always include a print(...) command in your code to display the final result.
Do not include any additional text, markdown formatting, or code outside of this JSON object.
Answer in Spanish.
Now, continue with the conversation:
            """,
        ),
        ("placeholder", "{messages}"),
    ]
)


# Data model
class code(BaseModel):
    """Schema for code solutions to questions."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


llm = get_llm()


# Optional: Check for errors in case tool use is flaky
def check_claude_output(tool_output):
    """Check for parse error or failure to call the tool"""

    # Error with parsing
    if tool_output["parsing_error"]:
        # Report back output and parsing errors
        print("Parsing error!")
        raw_output = str(tool_output["raw"].content)
        error = tool_output["parsing_error"]
        raise ValueError(
            f"Error parsing your output! Be sure to invoke the tool. Output: {raw_output}. \n Parse error: {error}"
        )

    # Tool was not invoked
    elif not tool_output["parsed"]:
        print("Failed to invoke tool!")
        raise ValueError(
            "You did not use the provided tool! Be sure to invoke the tool to structure the output."
        )
    return tool_output


# Chain with output check
code_chain_claude_raw = (
    code_gen_prompt
    | llm.with_structured_output(code, include_raw=True)
    | check_claude_output
)


def insert_errors(inputs):
    """Insert errors for tool parsing in the messages"""

    # Get errors
    error = inputs["error"]
    messages = inputs["messages"]
    messages += [
        (
            "assistant",
            f"Retry. You are required to fix the parsing errors: {error} \n\n You must invoke the provided tool.",
        )
    ]
    return {
        "messages": messages,
        "primary_libs": inputs["primary_libs"],
        "df_description": inputs["df_description"],
    }


# This will be run as a fallback chain
fallback_chain = insert_errors | code_chain_claude_raw
N = 3  # Max re-tries
code_gen_chain_re_try = code_chain_claude_raw.with_fallbacks(
    fallbacks=[fallback_chain] * N, exception_key="error"
)


def parse_output(solution):
    """When we add 'include_raw=True' to structured output,
    it will return a dict w 'raw', 'parsed', 'parsing_error'."""

    return solution["parsed"]


# Optional: With re-try to correct for failure to invoke tool
code_gen_chain = code_gen_chain_re_try | parse_output


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """

    error: str
    messages: List
    generation: str
    iterations: int
    output: str = None


### Parameter

# Max tries
max_iterations = 3
# Reflect
flag = "reflect"
# flag = "do not reflect"


### Nodes
def fallback_handler(inputs):
    """
    Fallback handler para errores en la invocación del tool.

    En caso de error (por ejemplo, al detectar formato Markdown no válido en la salida),
    se añade un mensaje al historial solicitando explícitamente que se responda utilizando un JSON
    sin ningún tipo de formateo Markdown.
    """
    error_message = inputs.get("error", "Error desconocido")
    modified_messages = inputs["messages"] + [
        (
            "assistant",
            f"Se detectó el siguiente error en la generación del código: {error_message}. "
            "Por favor, responde únicamente con un objeto JSON sin ningún formato Markdown, "
            'es decir, sin incluir signos de triple backticks o etiquetas de código. Utiliza el siguiente formato EXACTO: {"prefix": <string>, "imports": <string>, "code": <string>}.',
        )
    ]
    return code_chain_claude_raw.invoke(
        {
            "primary_libs": inputs["primary_libs"],
            "df_description": inputs["df_description"],
            "messages": modified_messages,
        }
    )


# Número máximo de reintentos
N = 3
# Envolver el fallback_handler en un RunnableLambda para cumplir con el tipo requerido
fallback_runnable = RunnableLambda(fallback_handler)
code_gen_chain_tolerant = code_chain_claude_raw.with_fallbacks(
    fallbacks=[fallback_runnable] * N, exception_key="error"
)


def generate(state: GraphState):
    """
    Genera una solución en código de forma tolerante a errores en la invocación del tool.
    Si se produce un error, se utiliza el fallback que reenvía un mensaje adicional solicitando el formato
    correcto (JSON sin ningún Markdown) y reintenta la generación.
    """
    print("---GENERANDO SOLUCIÓN CON FALLBACK TOLERANTE---")
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    # Si hubo error en el último intento, se añade un recordatorio al historial.
    if error == "yes":
        messages += [
            (
                "user",
                "Se produjo un error en la generación del código. "
                "Intenta nuevamente invocar el tool siguiendo el formato JSON EXACTO (sin formato Markdown).",
            )
        ]

    try:
        code_solution = code_gen_chain_tolerant.invoke(
            {
                "primary_libs": primary_libs,
                "df_description": df_description,
                "messages": messages,
            }
        )
    except Exception as e:
        raise ValueError(f"Fallos repetidos en la generación del código: {e}")

    messages += [
        (
            "assistant",
            f"{code_solution.prefix}\nImports: {code_solution.imports}\nCode: {code_solution.code}",
        )
    ]
    iterations += 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


def code_check(state: GraphState) -> Command[Literal["reflect", END]]:
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state: error and output containing the execution result.
    """

    print("---CHECKING CODE---")

    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code

    # Creamos un entorno de ejecución que ya cuenta con data_df
    env = {"data_df": data_df}

    # Verificar que las importaciones sean correctas
    try:
        exec(imports, env)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the import test: {e}")]
        messages += error_message
        return Command(
            goto="reflect",
            update={
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
            },
        )

    # Ejecución del código y captura de la salida, agregando data_df.describe()
    try:
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()  # Redirigir la salida
        # Se construye el código final a ejecutar que incluye:
        # - Las importaciones
        # - El código generado
        # - La ejecución de data_df.describe() y su impresión
        full_code = imports + "\n" + code
        exec(full_code, env)
        output = sys.stdout.getvalue()  # Capturar la salida generada
        sys.stdout = old_stdout  # Restaurar la salida estándar original
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the code execution test: {e}")]
        messages += error_message
        return Command(
            goto="reflect",
            update={
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
            },
        )

    print("---NO CODE TEST FAILURES---")
    # Se almacena la salida correcta (incluyendo el resultado de data_df.describe())
    return Command(
        goto=END,
        update={
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "no",
            "output": output,
        },
    )


def reflect(state: GraphState):
    """
    Reflect on errors

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("---GENERATING CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]

    # Prompt reflection

    # Add reflection
    reflections = code_gen_chain.invoke(
        {
            "primary_libs": primary_libs,
            "df_description": df_description,
            "messages": messages,
        }
    )
    messages += [("assistant", f"Here are reflections on the error: {reflections}")]
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


# def decide_to_finish(state: GraphState):
#     """
#     Determines whether to finish.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Next node to call
#     """
#     error = state["error"]
#     iterations = state["iterations"]

#     if error == "no" or iterations == max_iterations:
#         print("---DECISION: FINISH---")
#         return "end"
#     else:
#         print("---DECISION: RE-TRY SOLUTION---")
#         if flag == "reflect":
#             return "reflect"
#         else:
#             return "generate"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("generate", generate)  # generation solution
workflow.add_node("check_code", code_check)  # check code
workflow.add_node("reflect", reflect)  # reflect

# Build graph
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "check_code")
# workflow.add_conditional_edges(
#     "check_code",
#     decide_to_finish,
#     {
#         "end": END,
#         "reflect": "reflect",
#         "generate": "generate",
#     },
# )
workflow.add_edge("reflect", "generate")
app = workflow.compile()
from IPython.display import Image, display

display(Image(app.get_graph().draw_mermaid_png()))


# %%
question = "Dame la red social con usuarios con más actividad física y la red social con menos, y dame el delta entre ellas"
solution = app.invoke(
    {"messages": [HumanMessage(content=question)], "iterations": 0, "error": ""}
)

# print(solution["generation"].code)
print(solution["output"])

# %%
# %%
