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
    Get the most recent and oldest valid records for specified offices.

    Args:
        office_names (List[str]): List of office names to query.

    Returns:
        pd.DataFrame: DataFrame containing office names and their date ranges.
    """
    query = text(
        """
        SELECT 
            o.[Oficina],
            MIN(a.[FH_Emi]) as first_valid_date,
            MAX(a.[FH_Emi]) as last_valid_date,
            COUNT(DISTINCT CAST(a.[FH_Emi] AS DATE)) as total_days_with_data
        FROM [dbo].[Atenciones] a
        JOIN [dbo].[Oficinas] o ON o.[IdOficina] = a.[IdOficina]
        WHERE o.[Oficina] IN :office_names
        AND a.[FH_Emi] IS NOT NULL
        GROUP BY o.[Oficina]
        ORDER BY o.[Oficina]
    """
    )

    try:
        with _engine.connect() as conn:
            date_ranges = pd.read_sql_query(
                query, conn, params={"office_names": tuple(office_names)}
            )

        if date_ranges.empty:
            logger.warning("No data found for the specified offices")
            return pd.DataFrame()

        # Primero calculamos total_weeks con las fechas originales
        # date_ranges["total_weeks"] = (
        #     (date_ranges["last_valid_date"] - date_ranges["first_valid_date"]).dt.days
        #     / 7
        # ).round(1)

        # Luego formateamos las fechas para mostrar
        date_ranges["first_valid_date"] = pd.to_datetime(
            date_ranges["first_valid_date"]
        ).dt.strftime("%d-%m-%Y")
        date_ranges["last_valid_date"] = pd.to_datetime(
            date_ranges["last_valid_date"]
        ).dt.strftime("%d-%m-%Y")

        return date_ranges

    except Exception as e:
        logger.error(f"Error fetching date ranges: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    office_names = ["356 - El Bosque", "362 - El Golf"]
    date_ranges = rango_registros_disponibles(office_names)
    print(date_ranges)
