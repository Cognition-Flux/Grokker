#!/usr/bin/env python
# %%
import logging
import os
from datetime import datetime

import sqlalchemy
from dotenv import load_dotenv
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from pydantic import BaseModel

load_dotenv(override=True)

logging.basicConfig()
logger: logging.Logger = logging.getLogger("sqlalchemy.engine")
# logger.setLevel(logging.DEBUG)

# Variables de entorno
DB_USERNAME = os.getenv("DB_USERNAME")  #
DB_PASSWORD = os.getenv("DB_PASSWORD")  #
DB_SERVER = os.getenv("DB_SERVER")  #
DB_PORT = os.getenv("DB_PORT")  #
DB_DATABASE = os.getenv("DB_DATABASE")  #
# Construcción de la cadena de conexión
connection_url = (
    f"mssql+pymssql://{DB_USERNAME}:{DB_PASSWORD}"
    f"@{DB_SERVER}:{DB_PORT}/{DB_DATABASE}"
)
logger.info(
    f"{DB_USERNAME = }\n{DB_PASSWORD = }\n{DB_SERVER = }\n{DB_PORT = }\n{DB_DATABASE = }"
)

_engine = sqlalchemy.create_engine(
    connection_url,
    pool_recycle=3600,
    pool_pre_ping=True,
)

assert isinstance(
    _engine, sqlalchemy.engine.base.Engine
), "SQLAlchemy Engine was not properly instantiated"


SQLAlchemyInstrumentor().instrument(
    engine=_engine,
)

# These are the ones used by the LLM model
_relevant_tables: list[str] = ["Atenciones", "EjeEstado", "Series", "Oficinas"]

# db = SQLDatabase(
#     engine=_engine, include_tables=_relevant_tables, sample_rows_in_table_info=5
# )

# assert isinstance(
#     db, langchain_community.utilities.sql_database.SQLDatabase
# ), "langchain_community's SQLAlchemy wrapper was not properly instantiated"


# region Other methods


class GetOfficesResponseOffices(BaseModel):
    name: str  #  "Oficina 1"
    ref: str  #  "ofc_01"
    region: str = "Oficinas"  #  "Norte"


class GetOfficesResponse(BaseModel):
    offices: list[GetOfficesResponseOffices]


def get_offices(group_by_zone=False) -> GetOfficesResponse:
    """Get the list of offices from the database.

    Current implementation just returns a list of all offices as a string.
    """
    with _engine.connect() as conn:
        if group_by_zone:
            ...
        else:
            offices = conn.execute(
                sqlalchemy.text(
                    """
                    SELECT o.[Oficina], o.[IdOficina]
                    FROM [Oficinas] o
                    JOIN [Atenciones] a ON a.IdOficina = o.IdOficina
                    WHERE o.fDel = 0
                    GROUP BY o.[Oficina], o.[IdOficina]
                    HAVING COUNT(*) > 0
                    ORDER BY [Oficina] ASC
                    """
                )
            )
            rows = offices.all()

            return GetOfficesResponse(
                offices=[
                    GetOfficesResponseOffices(name=row[0], ref=str(row[1]))
                    for row in rows
                ]
            )


def get_just_office_names() -> list[str]:
    """Not designed to be used with a chain per-se"""
    import pandas as pd

    query = """
    SELECT
          o.[Oficina] AS "Oficina"
    FROM [dbo].[Oficinas] o
    """

    with _engine.connect() as conn:
        data: pd.DataFrame = pd.read_sql_query(query, conn)

    return data["Oficina"].to_list()


def get_last_database_update() -> datetime:
    """Get the last time the database was updated."""
    with _engine.connect() as conn:
        query = 'SELECT MAX(a.[FH_Emi]) AS "Ultima atencion" FROM [dbo].[Atenciones] a'
        last_update = conn.execute(sqlalchemy.text(query)).scalar()

        return last_update


if __name__ == "__main__":
    print(get_just_office_names())

# %%
