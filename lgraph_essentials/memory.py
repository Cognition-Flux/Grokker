# %%
"""Módulo para gestionar la memoria de la aplicación."""  # Añadir docstring al módulo

import os
import sqlite3

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "memory.db")


# Initialize database connection in a function instead of global
def get_db_connection() -> sqlite3.Connection:
    """Get a new database connection."""
    return sqlite3.connect(DB_PATH)


conn = get_db_connection()
