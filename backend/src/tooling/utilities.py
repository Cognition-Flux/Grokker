import ast
import json
import time
from functools import wraps
from typing import Any, Callable

"""
Contains common use functions to parse the tools i/o and docs gen
"""


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
    """
    More fault-tolerant parser that tries:
      1. If input_data is empty or only '{}' -> returns a default model instance (cls()).
      2. If input_data is a dict, serialize it to JSON first.
      3. Attempts to parse (a) direct JSON, (b) quotes replaced, (c) ast.literal_eval
      4. Recursively parse "input_string" if it exists in the parsed data.
         - If 'input_string' is a str, parse again with the same approach.
         - If 'input_string' is a dict, convert it to JSON and parse.
      5. Finally, tries to instantiate the Pydantic model:
         - If that fails, it attempts a partial parse ignoring any unknown fields.
         - If that also fails, returns default instance of the model.
    """

    import ast
    import json

    from pydantic import ValidationError

    # 1. If input_data is an empty dict or minimal JSON-like string, return defaults
    if (isinstance(input_data, dict) and not input_data) or (
        isinstance(input_data, str) and input_data.strip() in ["{}", ""]
    ):
        return cls()

    # 2. Convert dict to string for consistent handling
    if isinstance(input_data, dict):
        try:
            input_data = json.dumps(input_data)
        except Exception as e:
            print(f"[WARN] Could not serialize input_data as JSON: {e}")
            return cls()

    # input_data should now be a string
    if not isinstance(input_data, str):
        print(
            "[WARN] input_data is neither a dict nor a string; returning default model."
        )
        return cls()

    # Attempt multi-step parsing
    raw_data = None
    try:
        # Step 3a. Direct JSON
        raw_data = json.loads(input_data)
    except json.JSONDecodeError:
        try:
            # Step 3b. Replace single quotes -> double quotes
            raw_data = json.loads(input_data.replace("'", '"'))
        except json.JSONDecodeError:
            try:
                # Step 3c. ast.literal_eval
                raw_data = ast.literal_eval(input_data)
            except Exception as e:
                print(f"[WARN] Could not parse input_data: {e}")
                print("[WARN] Returning default model.")
                return cls()

    # If empty or None, fall back to defaults
    if not raw_data:
        return cls()

    # Nested input_string handling
    def parse_nested_input_string(data):
        """
        If data is a dict and has an 'input_string' key:
          - If that key is a str, re-parse it.
          - If that key is a dict, convert to JSON -> parse.
        Replace data with the result of that parse.
        """
        if not isinstance(data, dict):
            return data
        if "input_string" not in data:
            return data

        nested = data["input_string"]

        if isinstance(nested, dict):
            # Convert dict to JSON, then parse
            try:
                nested_str = json.dumps(nested)
                nested_data = json.loads(nested_str)
                if nested_data:
                    data = nested_data
                else:
                    # empty => defaults
                    data = {}
            except json.JSONDecodeError as e:
                print(f"[WARN] Could not parse nested dict: {e}")
                data = {}
        elif isinstance(nested, str):
            # Attempt the same multi-step parse
            converted = None
            try:
                converted = json.loads(nested)
            except json.JSONDecodeError:
                try:
                    converted = json.loads(nested.replace("'", '"'))
                except json.JSONDecodeError:
                    try:
                        converted = ast.literal_eval(nested)
                    except Exception as e:
                        print(f"[WARN] Could not parse 'input_string' str: {e}")
                        converted = {}
            if converted:
                data = converted
            else:
                data = {}
        else:
            print("[WARN] 'input_string' is neither a str nor a dict; ignoring it.")
            data = {}

        return data

    raw_data = parse_nested_input_string(raw_data)

    # Step 5: Instantiate the Pydantic model
    try:
        return cls(**raw_data)
    except ValidationError as e:
        # Attempt partial parse: keep only recognized fields
        print(f"[WARN] Full parse into the model failed: {e}")
        recognized_fields = {k: v for k, v in raw_data.items() if k in cls.__fields__}
        try:
            return cls(**recognized_fields)
        except Exception as inner_e:
            print(f"[WARN] Partial parse also failed: {inner_e}")
            return cls()
    except Exception as e:
        print(f"[WARN] Unknown error while instantiating model: {e}")
        return cls()


def get_documentation(cls) -> str:
    schema = cls.schema()
    docs = []
    description = schema.get("description", "")
    if description:
        docs.append(f"\n{description}\n")
    docs.append(
        "The parameters should be serialized JSON formatted string like '{'input_string': '{ ... }'}'."
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
