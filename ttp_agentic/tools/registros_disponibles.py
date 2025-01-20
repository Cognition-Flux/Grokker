# %%
import json
import logging
from datetime import datetime, timedelta
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy import text

from ttp_agentic.db_instance import _engine
from ttp_agentic.tools.utilities import (
    add_docstring,
    get_documentation,
    parse_input,
    remove_extra_spaces,
    retry_decorator,
)

load_dotenv(override=True)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@retry_decorator(max_retries=5, delay=1.0)
def rango_registros_disponibles(office_names: List[str]) -> pd.DataFrame:
    """
    Genera una tabla con información de registros disponibles por mes para cada
    oficina dada en la lista de oficinas, incluyendo días disponibles, total
    de atenciones y además muestra la primera y última fecha (first_valid_date,
    last_valid_date) donde existan registros en cada mes de la consulta.

    Retorna un DataFrame con las columnas:
      - oficina
      - mes (en formato AAAA-MM)
      - first_valid_date (fecha mínima del mes)
      - last_valid_date (fecha máxima del mes)
      - total_dias_registrados (cuántos días del mes hay registros)
      - total_atenciones (número de atenciones totales en ese mes)

    Args:
        office_names (List[str]): Lista con los nombres de las oficinas
            a consultar en la base de datos.

    Returns:
        pd.DataFrame: DataFrame que muestra, para cada oficina y mes:
            - Fecha mínima y fecha máxima con registros.
            - Número de días con registros.
            - Cantidad de atenciones totales.
    """
    monthly_query = text(
        """
        SELECT 
            o.[Oficina] AS oficina,
            FORMAT(a.[FH_Emi], 'yyyy-MM') AS mes,
            MIN(CAST(a.[FH_Emi] AS DATE)) AS first_valid_date,
            MAX(CAST(a.[FH_Emi] AS DATE)) AS last_valid_date,
            COUNT(DISTINCT CAST(a.[FH_Emi] AS DATE)) AS total_dias_registrados,
            COUNT(*) AS total_atenciones
        FROM [dbo].[Atenciones] a
        JOIN [dbo].[Oficinas] o ON o.[IdOficina] = a.[IdOficina]
        WHERE o.[Oficina] IN :office_names
            AND a.[FH_Emi] IS NOT NULL
        GROUP BY o.[Oficina], FORMAT(a.[FH_Emi], 'yyyy-MM')
        ORDER BY o.[Oficina], mes
        """
    )

    try:
        with _engine.connect() as conn:
            monthly_data = pd.read_sql_query(
                monthly_query,
                conn,
                params={"office_names": tuple(office_names)},
            )

        if monthly_data.empty:
            logger.warning("No se encontraron datos para las oficinas especificadas.")
            return pd.DataFrame()

        return monthly_data

    except Exception as e:
        logger.error(f"Error al consultar los registros disponibles por mes: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    office_names = ["356 - El Bosque", "362 - El Golf"]
    date_ranges = rango_registros_disponibles(office_names)
    print(date_ranges)
