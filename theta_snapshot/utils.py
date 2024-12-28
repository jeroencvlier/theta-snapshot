from dataclasses import dataclass, field
import pandas as pd
import pytz
from datetime import datetime as dt
from typing import Optional
from sqlalchemy import create_engine
import os
import sys

# import logging
import httpx
import option_emporium as oe

from loguru import logger as log

# --------------------------------------------------------------
# Data Classes
# --------------------------------------------------------------


@dataclass
class CalendarSnapData:
    symbol: str
    rdatedt: dt
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
        self.fexpdt = pd.to_datetime(str(value), format="%Y%m%d") if value is not None else None

    @property
    def bexp(self) -> Optional[int]:
        """Getter for bexp."""
        return self._bexp

    @bexp.setter
    def bexp(self, value: Optional[int]):
        """Setter for bexp with automatic conversion to bexpdt."""
        self._bexp = value
        self.bexpdt = pd.to_datetime(str(value), format="%Y%m%d") if value is not None else None

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
                (df["undPricePctDiff"] >= lower_bound) & (df["undPricePctDiff"] <= upper_bound)
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
    assert table or query, "Table or query must be provided"
    assert not (table and query), "Only one of table or query must be provided"
    if query:
        try:
            return pd.read_sql(query, create_engine(os.getenv("POSTGRESSSQL_URL")))
        except Exception as err:
            log.error(f"FAILED to read DataFrame from database. ERROR: {err}")
            return pd.DataFrame()

    elif table:
        try:
            return pd.read_sql_table(table, create_engine(os.getenv("POSTGRESSSQL_URL")))
        except Exception as err:
            log.error(f"FAILED to read DataFrame from database. ERROR: {err}")
            return pd.DataFrame()


def write_to_db(
    df: pd.DataFrame, table_name: str, conn_db: str = None, if_exists="replace"
) -> None:
    if conn_db is None:
        conn_db = "POSTGRESSSQL_URL"
    try:
        df.to_sql(
            table_name,
            create_engine(os.getenv(conn_db)),
            if_exists=if_exists,
            index=False,
        )
        log.info(f"DataFrame written to database successfully. Table: {table_name}")
    except Exception as err:
        log.error(f"FAILED to write DataFrame to database. ERROR: {err}")


# --------------------------------------------------------------
# HTTPX Functions
# --------------------------------------------------------------


def request_pagination(url, params, max_retries=3, timeout=20.0):
    """
    Fetch paginated responses from an API.

    Args:
        url (str): The initial URL to request.
        params (dict): Query parameters for the request.
        max_retries (int): Maximum number of retries for transient errors.
        timeout (float): Timeout for the HTTP request in seconds.

    Returns:
        tuple: A list of combined responses and the format header.
    """
    responses = []
    retries = 0

    while url is not None:
        try:
            # Make the HTTP request
            response = httpx.get(url, params=params, timeout=timeout)
            response.raise_for_status()  # Raise for HTTP errors

            # Parse and append the response
            data = response.json()
            responses.extend(data.get("response", []))
            # log.debug(f"Fetched {len(data.get('response', []))} items from {url}")

            # Handle pagination
            next_page = response.headers.get("Next-Page")
            url = next_page if next_page and next_page != "null" else None

        except httpx.RequestError as e:
            log.error(f"Request error: {e}")
            if retries < max_retries:
                retries += 1
                log.warning(f"Retrying... attempt {retries}")
            else:
                log.error("Max retries exceeded. Exiting pagination.")
                break
        except KeyError as e:
            log.error(f"Key error: {e}. Response format may have changed.")
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            break

    # Return responses and format header (if available)
    format_header = data.get("header", {}).get("format", None) if "data" in locals() else None
    return responses, format_header


def get_expiry_dates(symbol):
    params = {"root": symbol}
    url = os.getenv("BASE_URL") + "/list/expirations"
    responses, _ = request_pagination(url, params)
    return responses


def get_greeks(symbol: str, exp: int, right: str):
    url = os.getenv("BASE_URL") + "/bulk_snapshot/option/greeks"
    params = {"root": symbol, "exp": exp}
    response, columns = request_pagination(url, params)
    df = response_to_df(response, columns)
    if right in ["C", "P"]:
        df = df[df["right"] == right]
    df.drop(columns=["ms_of_day2", "bid", "ask", "ms_of_day"], inplace=True)
    return df


def get_quote(symbol: str, exp: int, right: str):
    url = os.getenv("BASE_URL") + "/bulk_snapshot/option/quote"
    params = {"root": symbol, "exp": exp}
    response, columns = request_pagination(url, params)
    df = response_to_df(response, columns)
    if right in ["C", "P"]:
        df = df[df["right"] == right]
    df.drop(
        columns=[
            "bid_condition",
            "bid_exchange",
            "ask_condition",
            "ask_exchange",
            "ms_of_day",
        ],
        inplace=True,
    )
    df = oe.calculate_mark(df)
    return df


def get_oi(symbol: str, exp: int, right: str):
    url = os.getenv("BASE_URL") + "/bulk_snapshot/option/open_interest"
    # log.info("Requesting port: %s", url)
    params = {"root": symbol, "exp": exp}
    response, columns = request_pagination(url, params)
    df = response_to_df(response, columns)
    if right in ["C", "P"]:
        df = df[df["right"] == right]
    df.drop(columns=["ms_of_day"], inplace=True)
    return df


# --------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------


def response_to_df(response, columns):
    rows = []
    for item in response:
        ticks = item["ticks"][0]
        contract = item["contract"]
        row = {**contract, **dict(zip(columns, ticks))}
        rows.append(row)
    return pd.DataFrame(rows)


def is_market_open(break_Script=True) -> bool:
    try:
        url = f'https://api.polygon.io/v1/marketstatus/now?apiKey={os.getenv("naughty_hermann")}'
        response = httpx.get(url)
        if response.json()["market"].lower() == "open":
            log.info("Market is open")
            is_open = True
        else:
            log.info("Market is closed")
            is_open = False
    except Exception as e:
        log.error(f"Failed to fetch market status - {e}")
        log.info("Building calendar with assumption that market is open")
        is_open = True

    finally:
        if break_Script and not is_open:
            sys.exit("Market is closed")
        else:
            log.warning("Bypassing market open check...")
            return is_open


def time_checker_ny(target_hour=9, target_minute=34, break_Script=True):
    new_york_tz = pytz.timezone("America/New_York")
    current_time_ny = dt.now(new_york_tz)
    target_time = current_time_ny.replace(
        hour=target_hour, minute=target_minute, second=0, microsecond=0
    )
    if current_time_ny < target_time:
        # datetime.datetime(2024, 12, 28, 6, 57, 10, 157880, tzinfo=<DstTzInfo 'America/New_York' EST-1 day, 19:00:00 STD>)
        log.info(f"{current_time_ny.strftime('%Y-%m-%d %H:%M:%S %Z')} -> Current Time")
        log.info(f"{target_time.strftime('%Y-%m-%d %H:%M:%S %Z')} -> Target Time")
        if break_Script:
            sys.exit("Too early to run the script.")
        else:
            log.warning("Bypassing time check...")
