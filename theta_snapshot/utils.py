from dataclasses import dataclass, field
import pandas as pd
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from loguru import logger as logging

# --------------------------------------------------------------
# Data Classes
# --------------------------------------------------------------


@dataclass
class CalendarSnapData:
    symbol: str
    rdatedt: datetime
    weeks: int
    right: str
    rdate: int = field(init=False)  # Automatically computed
    _fexp: Optional[int] = field(default=None, init=False)
    _bexp: Optional[int] = field(default=None, init=False)
    fexpdt: Optional[pd.Timestamp] = field(default=None, init=False)
    bexpdt: Optional[pd.Timestamp] = field(default=None, init=False)
    greeks_front: Optional[pd.DataFrame] = field(default=None, init=False)
    greeks_back: Optional[pd.DataFrame] = field(default=None, init=False)
    greeks: Optional[pd.DataFrame] = field(default=None, init=False)
    quotes_front: Optional[pd.DataFrame] = field(default=None, init=False)
    quotes_back: Optional[pd.DataFrame] = field(default=None, init=False)
    quotes: Optional[pd.DataFrame] = field(default=None, init=False)
    oi_front: Optional[pd.DataFrame] = field(default=None, init=False)
    oi_back: Optional[pd.DataFrame] = field(default=None, init=False)
    oi: Optional[pd.DataFrame] = field(default=None, init=False)

    def __post_init__(self):
        """Convert rdatedt to rdate as an integer in YYYYMMDD format."""
        self.rdate = int(self.rdatedt.strftime("%Y%m%d"))

    @property
    def fexp(self) -> Optional[int]:
        """Getter for fexp."""
        return self._fexp

    @fexp.setter
    def fexp(self, value: Optional[int]):
        """Setter for fexp with automatic conversion to fexpdt."""
        self._fexp = value
        self.fexpdt = (
            pd.to_datetime(str(value), format="%Y%m%d") if value is not None else None
        )

    @property
    def bexp(self) -> Optional[int]:
        """Getter for bexp."""
        return self._bexp

    @bexp.setter
    def bexp(self, value: Optional[int]):
        """Setter for bexp with automatic conversion to bexpdt."""
        self._bexp = value
        self.bexpdt = (
            pd.to_datetime(str(value), format="%Y%m%d") if value is not None else None
        )

    @property
    def rdate_dt(self) -> pd.Timestamp:
        """Getter for rdatedt."""
        return pd.Timestamp(self.rdatedt)


# --------------------------------------------------------------
# Filter Functions
# --------------------------------------------------------------


def underlying_price_filter(df: pd.DataFrame, max_rows: int = 2):
    assert "undPricePctDiff" in df.columns, "undPricePctDiff column not found"
    df = df[(df["undPricePctDiff"] >= -0.003) & (df["undPricePctDiff"] <= 0.05)]
    if len(df) > max_rows:
        for i in range(20):
            lower_bound = round(i * -0.00025, 6)
            upper_bound = round(i * 0.003, 6)
            if upper_bound > 0.05:
                break
            filtered_df = df[
                (df["undPricePctDiff"] >= lower_bound)
                & (df["undPricePctDiff"] <= upper_bound)
            ]
            if len(filtered_df) >= max_rows:
                df = filtered_df
                break

    if len(df) > max_rows:
        df = df.head(max_rows)

    return df


def oi_filter(df: pd.DataFrame, min_oi: int = 20):
    assert "open_interest_front" in df.columns, "open_interest_front column not found"
    assert "open_interest_back" in df.columns, "open_interest_back column not found"
    df = df[(df["open_interest_front"] > min_oi) & (df["open_interest_back"] > min_oi)]
    return df


def snapshot_filter(df: pd.DataFrame, min_oi: int = 20, max_rows: int = 2):
    df = oi_filter(df, min_oi)
    df = underlying_price_filter(df, max_rows)
    return df


# --------------------------------------------------------------
# Database Functions
# --------------------------------------------------------------


def read_from_db(table: str = None, query: str = None) -> pd.DataFrame:
    load_dotenv(".env")
    assert table or query, "Table or query must be provided"
    assert not (table and query), "Only one of table or query must be provided"
    if query:
        try:
            return pd.read_sql(query, create_engine(os.getenv("POSTGRESSSQL_URL")))
        except Exception as err:
            logging.error(f"FAILED to read DataFrame from database. ERROR: {err}")
            return pd.DataFrame()

    elif table:
        try:
            return pd.read_sql_table(
                table, create_engine(os.getenv("POSTGRESSSQL_URL"))
            )
        except Exception as err:
            logging.error(f"FAILED to read DataFrame from database. ERROR: {err}")
            return pd.DataFrame()


def write_to_db(
    df: pd.DataFrame, table_name: str, conn_string: str = None, if_exists="replace"
) -> None:
    load_dotenv(".env")
    if conn_string is None:
        conn_string = "POSTGRESSSQL_URL"
    try:
        df.to_sql(
            table_name,
            create_engine(os.getenv(conn_string)),
            if_exists=if_exists,
            index=False,
        )
        logging.info(f"DataFrame written to database successfully. Table: {table_name}")
    except Exception as err:
        logging.error(f"FAILED to write DataFrame to database. ERROR: {err}")
