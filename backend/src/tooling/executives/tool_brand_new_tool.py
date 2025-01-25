# %%import json

from langchain_core.tools import StructuredTool

# from tooling.db_instance import _engine
# from tooling.utilities import add_docstring, remove_extra_spaces


def brand_new(input_string: str) -> str:
    """
    la nueva herramieta(brand new tool) de ejecutivos:

    usar cuando el usuario llama a la herramienta nueva y entregar el código 23
    """
    return "esta es la herramienta nueva, informar que está funcionando y entregar el código 23"


# Create the structured tool
brand_new_tool = StructuredTool.from_function(
    func=brand_new,
    name="brand_new_tool",
    description=brand_new.__doc__,
    return_direct=True,
)

# %%
