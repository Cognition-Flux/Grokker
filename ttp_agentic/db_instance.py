#!/usr/bin/env python
# %%
import logging
import os
from datetime import datetime

import sqlalchemy
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=True)

logging.basicConfig()
logger: logging.Logger = logging.getLogger("sqlalchemy.engine")

_DB_USERNAME: str = os.getenv("DB_USERNAME")  # , "v40_afc_r")
_DB_PASSWORD: str = os.getenv("DB_PASSWORD")  # , "ttp_turnos1")
_DB_SERVER: str = os.getenv("DB_SERVER")  # , "10.0.1.21")
_DB_DATABASE: str = os.getenv("DB_DATABASE")  # , "TTP_AFC")

# mssql: adaptador para Microsoft SQL Server database
# pymssql: Python driver.
_engine = sqlalchemy.create_engine(
    url=f"mssql+pymssql://{_DB_USERNAME}:{_DB_PASSWORD}@{_DB_SERVER}/{_DB_DATABASE}"
)

if not isinstance(_engine, sqlalchemy.engine.base.Engine):
    raise TypeError("SQLAlchemy Engine was not properly instantiated")


# class LastUpdateByOffice(BaseModel):
#     office_name: str
#     office_id: str
#     last_update: datetime


# def get_last_database_update(
#     by_office: bool = False,
# ) -> datetime | list[LastUpdateByOffice]:
#     """Get the last time the database was updated.

#     Args:
#         by_office: If True, returns a list of last updates per office. If False, returns
#                   the overall last update.
#     """
#     with _engine.connect() as conn:
#         if by_office:
#             # Optimized query using window function instead of GROUP BY
#             query = """
#                 WITH LastUpdates AS (
#                     SELECT
#                         o.Oficina,
#                         o.IdOficina,
#                         a.[FH_Emi],
#                         ROW_NUMBER() OVER (PARTITION BY o.IdOficina ORDER BY a.[FH_Emi] DESC) as rn
#                     FROM [dbo].[Oficinas] o
#                     INNER JOIN [dbo].[Atenciones] a WITH (NOLOCK)
#                         ON o.IdOficina = a.IdOficina
#                     WHERE o.fDel = 0
#                 )
#                 SELECT
#                     Oficina,
#                     IdOficina,
#                     [FH_Emi] as Ultima_atencion
#                 FROM LastUpdates
#                 WHERE rn = 1
#                 ORDER BY Oficina;
#             """
#             # Use server-side cursor for memory efficiency
#             results = conn.execution_options(stream_results=True).execute(
#                 sqlalchemy.text(query)
#             )
#             return [
#                 LastUpdateByOffice(
#                     office_name=row[0], office_id=str(row[1]), last_update=row[2]
#                 )
#                 for row in results
#             ]
#         else:
#             # Optimized single update query with NOLOCK hint
#             query = """
#                 SELECT MAX(a.[FH_Emi])
#                 FROM [dbo].[Atenciones] a WITH (NOLOCK)
#             """
#             last_update = conn.execute(sqlalchemy.text(query)).scalar()
#             return last_update


# def format_last_updates(updates: list[LastUpdateByOffice]) -> str:
#     """Format the last updates list into a human readable string.

#     Args:
#         updates: List of LastUpdateByOffice objects

#     Returns:
#         Formatted string with the last updates information
#     """
#     now = datetime.now()

#     # Find the longest office name (including quotes) for padding
#     max_name_length = (
#         max(len(update.office_name) for update in updates) + 2
#     )  # +2 for quotes

#     formatted_updates = []
#     for update in updates:
#         # Calculate time difference
#         time_diff = now - update.last_update

#         # Format the time difference
#         if time_diff.days > 0:
#             time_ago = f"{time_diff.days} días atrás"
#         else:
#             hours = time_diff.seconds // 3600
#             minutes = (time_diff.seconds % 3600) // 60
#             if hours > 0:
#                 time_ago = f"{hours} horas y {minutes} minutos atrás"
#             else:
#                 time_ago = f"{minutes} minutos atrás"

#         # Format each part separately and then join them
#         office_part = f"nombre: '{update.office_name}'".ljust(max_name_length + 8)
#         id_part = f"office_id: {update.office_id} - "
#         update_part = (
#             f"Última actualización: {update.last_update.strftime('%Y-%m-%d %H:%M:%S')} "
#         )
#         time_part = f"({time_ago})"

#         # Join all parts
#         line = f"{office_part}{id_part}{update_part}{time_part}"
#         formatted_updates.append(line)

#     return "\n".join(formatted_updates)


# updates_by_office = get_last_database_update(by_office=True)
# ultima_actualizacion = format_last_updates(updates_by_office)
