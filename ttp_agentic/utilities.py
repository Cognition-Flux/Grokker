# %%
import ast
import json
import time
from functools import wraps
from typing import Any, Callable, List, Literal

from pydantic import BaseModel, Field


def retry_decorator(max_retries: int = 5, delay: float = 1.0):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            last_exception = None

            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    retries += 1
                    if retries < max_retries:
                        time.sleep(delay)  # Espera antes de reintentar
                    continue

            # Si llegamos aquí, todos los intentos fallaron
            return f"Error después de {max_retries} intentos: {str(last_exception)}"

        return wrapper

    return decorator


def parse_input(cls, input_data):
    # Handle empty dict or empty dict string - return instance with default values
    if (isinstance(input_data, dict) and not input_data) or (
        isinstance(input_data, str) and input_data.strip() in ["{}", ""]
    ):
        return cls()  # Pydantic will use the default values defined in the model

    # Handle other dict cases
    if isinstance(input_data, dict):
        return cls(**input_data)
    elif isinstance(input_data, str):
        data = None

        # Try to parse as JSON
        try:
            data = json.loads(input_data)
            # Check if parsed data is empty dict
            if not data:
                return cls()  # Use default values
        except json.JSONDecodeError:
            # Try to replace single quotes with double quotes
            try:
                data = json.loads(input_data.replace("'", '"'))
                # Check if parsed data is empty dict
                if not data:
                    return cls()  # Use default values
            except json.JSONDecodeError:
                # Try to parse using ast.literal_eval
                try:
                    data = ast.literal_eval(input_data)
                    # Check if parsed data is empty dict
                    if not data:
                        return cls()  # Use default values
                except Exception as e:
                    raise ValueError(f"Invalid input data: {e}")

        # Handle nested 'input_string'
        if isinstance(data, dict) and "input_string" in data:
            input_string = data["input_string"]
            if isinstance(input_string, str):
                try:
                    data = json.loads(input_string)
                    # Check if parsed nested data is empty dict
                    if not data:
                        return cls()  # Use default values
                except json.JSONDecodeError:
                    try:
                        data = json.loads(input_string.replace("'", '"'))
                        # Check if parsed nested data is empty dict
                        if not data:
                            return cls()  # Use default values
                    except json.JSONDecodeError:
                        try:
                            data = ast.literal_eval(input_string)
                            # Check if parsed nested data is empty dict
                            if not data:
                                return cls()  # Use default values
                        except Exception as e:
                            raise ValueError(f"Invalid 'input_string' data: {e}")
            elif isinstance(input_string, dict):
                # Check if nested dict is empty
                if not input_string:
                    return cls()  # Use default values
                data = input_string
            else:
                raise ValueError("Invalid 'input_string' data type")

        return cls(**data)
    else:
        raise ValueError("Invalid input data type")


def get_documentation(cls) -> str:
    schema = cls.schema()
    docs = []
    description = schema.get("description", "")
    if description:
        docs.append(f"\n{description}\n")
    docs.append(
        " The parameters should be passed into serialized JSON formatted string"
    )
    # Add field documentation
    for field_name, field_info in schema.get("properties", {}).items():
        field_type = field_info.get("type", "Unknown type")
        field_desc = field_info.get("description", "No description")
        default = field_info.get("default", "No default")
        constraints = ""

        # Include constraints like minimum or maximum values
        if "minimum" in field_info:
            constraints += f", minimum: {field_info['minimum']}"
        if "maximum" in field_info:
            constraints += f", maximum: {field_info['maximum']}"
        if "enum" in field_info:
            constraints += f", allowed values: {field_info['enum']}"

        field_doc = (
            f"- `{field_name}` ({field_type}{constraints}): {field_desc}\n"
            f"  Default: `{default}`"
        )
        docs.append(field_doc)

    return "\n\n".join(docs)


def remove_extra_spaces(s: str) -> str:
    """Replaces extra spaces (3 or more) with a double space"""
    import re

    despaced = re.sub(r"-{5,}", "-----", s, count=0)  # Replace "------"
    return re.sub(r"\s{3,}", "  ", despaced, count=0)  # Replace "      "


# Define the decorator once after imports
def add_docstring(doc):
    def decorator(func):
        func.__doc__ = doc
        return func

    return decorator


class ReporteDetalladoPorEjecutivo(BaseModel):
    executive_names: List[str] = Field(
        default=[
            "Abigail Betzabet Calabrano Avalos",
            "Maria Margarita Bahamondez Madrid",
        ],
        description="Lista de nombres completos de ejecutivos",
    )
    start_date: str = Field(
        default="01/10/2024", description="Start date in '%d/%m/%Y' format"
    )
    end_date: str = Field(
        default="15/10/2024", description="End date in '%d/%m/%Y'  format"
    )
    parse_input_for_tool = classmethod(parse_input)
    get_documentation_for_tool = classmethod(get_documentation)


# input_string = '{"executive_names":["Luis Hernan Labarca Montecino", "Natalia Belen Troncoso Silva", "Ricardo Andres Cataldo Veloso", "Ivonne Alejandra Munoz Diaz"], "start_date":"01/08/2024", "end_date":"31/08/2024"}'

# input_data = ReporteDetalladoPorEjecutivo.parse_input_for_tool(input_string)
# input_data.model_dump()
