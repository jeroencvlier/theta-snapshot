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

    def __repr__(self):
        attributes = [
            f"\t{field}={getattr(self, field)!r}"
            for field in self.__dataclass_fields__
            if getattr(self, field) is not None
        ]
        attributes_str = ",\n".join(attributes)
        return f"CalendarSnapData(\n{attributes_str}\n)"


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
# Helper Functions
# --------------------------------------------------------------


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
            log.info("Market is closed")
            sys.exit(0)
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
            log.info("Too early to run the script.")
            sys.exit(0)
        else:
            log.warning("Bypassing time check...")


def main_wrapper(func):
    def wrapper(*args, **kwargs):
        try:
            start = round(dt.now().timestamp())
            func(*args, **kwargs)
            end = round(dt.now().timestamp())
            time_taken = divmod((end - start), 60)
            log.info(f"Time taken: {time_taken[0]} minutes, {time_taken[1]} seconds")
        except Exception as err:
            log.opt(exception=True).error(f"Error in {func.__name__}: {err}")
            # TODO: add telegram alert to notify of error

    return wrapper


# --------------------------------------------------------------
# s3BucketHandler
# --------------------------------------------------------------


# class S3Handler:
#     def __init__(self, bucket_name=None, region=None, env_path=".env", log_level=logging.INFO):
#         """
#         Initialize AWS S3 connection using environment variables or parameters

#         Args:
#             bucket_name (str): Name of the S3 bucket
#                            If not provided, will look for S3_BUCKET_NAME in environment variables
#             region (str): AWS region (e.g., 'us-east-1')
#                          If not provided, will look for AWS_REGION in environment variables
#             env_path (str): Path to .env file (default: ".env")
#             log_level (int): Logging level (default: logging.INFO)
#         """
#         # Set up logging
#         self.logger = log.getLogger("S3Handler")
#         self.logger.setLevel(log_level)

#         if not self.logger.handlers:
#             handler = logging.StreamHandler()
#             formatter = logging.Formatter(
#                 "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#                 datefmt="%Y-%m-%d %H:%M:%S",
#             )
#             handler.setFormatter(formatter)
#             self.logger.addHandler(handler)

#         load_dotenv(env_path)

#         # Get bucket name from parameter or environment
#         self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME")
#         if not self.bucket_name:
#             self.logger.error("Bucket name not provided")
#             raise ValueError(
#                 "Bucket name must be provided either as parameter or S3_BUCKET_NAME environment variable"
#             )

#         # Get region from parameter or environment
#         self.region = region or os.getenv("AWS_REGION")
#         if not self.region:
#             self.logger.error("AWS region not provided")
#             raise ValueError(
#                 "AWS region must be provided either as parameter or AWS_REGION environment variable"
#             )

#         # Get credentials from environment
#         self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
#         self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

#         if not self.access_key or not self.secret_key:
#             self.logger.info("No explicit credentials provided, using AWS credential chain")

#         self._log_config()

#         # Initialize client
#         self.session = boto3.session.Session()
#         self.client = self.session.client(
#             "s3",
#             region_name=self.region,
#             aws_access_key_id=self.access_key,
#             aws_secret_access_key=self.secret_key,
#         )

#         self._verify_connection()
