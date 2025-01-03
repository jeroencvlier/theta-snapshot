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

import boto3
import pyarrow.parquet as pq
import botocore
import io


# --------------------------------------------------------------
# Data Classes
# --------------------------------------------------------------


@dataclass
class CalendarSnapData:
    symbol: str
    rdatedt: dt
    weeks: int
    right: str
    roots: str
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
    f_dates: Optional[list] = field(default=None, init=False)
    b_dates: Optional[list] = field(default=None, init=False)
    trade_dates: Optional[list] = field(default=None, init=False)

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
        log.success(f"DataFrame written successfully. Table: {table_name}, Entries: {df.shape[0]}")
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
            log.success(f"Time taken: {time_taken[0]} minutes, {time_taken[1]} seconds")
        except Exception as err:
            log.opt(exception=True).error(f"Error in {func.__name__}: {err}")
            # TODO: add telegram alert to notify of error

    return wrapper


# --------------------------------------------------------------
# s3BucketHandler
# --------------------------------------------------------------


class S3Handler:
    def __init__(self, bucket_name=None, region=None, env_path=".env"):
        # ), log_level=logging.INFO):
        """
        Initialize AWS S3 connection using environment variables or parameters

        Args:
            bucket_name (str): Name of the S3 bucket
                           If not provided, will look for S3_BUCKET_NAME in environment variables
            region (str): AWS region (e.g., 'us-east-1')
                         If not provided, will look for AWS_REGION in environment variables
            env_path (str): Path to .env file (default: ".env")
            log_level (int): Logging level (default: logging.INFO)
        """
        # # Set up logging
        # log = logging.getlog("S3Handler")
        # log.setLevel(log_level)

        # if not log.handlers:
        #     handler = logging.StreamHandler()
        #     formatter = logging.Formatter(
        #         "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        #         datefmt="%Y-%m-%d %H:%M:%S",
        #     )
        #     handler.setFormatter(formatter)
        #     log.addHandler(handler)

        # load_dotenv(env_path)

        # Get bucket name from parameter or environment
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME")
        if not self.bucket_name:
            log.error("Bucket name not provided")
            raise ValueError(
                "Bucket name must be provided either as parameter or S3_BUCKET_NAME environment variable"
            )

        # Get region from parameter or environment
        self.region = region or os.getenv("AWS_REGION")
        if not self.region:
            log.error("AWS region not provided")
            raise ValueError(
                "AWS region must be provided either as parameter or AWS_REGION environment variable"
            )

        # Get credentials from environment
        self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        if not self.access_key or not self.secret_key:
            log.info("No explicit credentials provided, using AWS credential chain")

        self._log_config()

        # Initialize client
        self.session = boto3.session.Session()
        self.client = self.session.client(
            "s3",
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

        self._verify_connection()

    def _log_config(self):
        """Log configuration with masked secrets"""
        log.info("Configuration:")
        log.info(f"Bucket Name: {self.bucket_name}")
        log.info(f"Region: {self.region}")
        if self.access_key:
            log.debug(f"Access Key: {self.access_key[:4]}...{self.access_key[-4:]}")
        if self.secret_key:
            log.debug(f"Secret Key: {self.secret_key[:4]}...{self.secret_key[-4:]}")

    def _verify_connection(self):
        """Verify connection and bucket access"""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            log.info(f"Successfully connected to bucket: {self.bucket_name}")
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = f"Error accessing bucket: {str(e)}"
            if error_code == "404":
                error_msg = f"Bucket '{self.bucket_name}' does not exist"
            elif error_code == "403":
                error_msg = f"Access denied to bucket '{self.bucket_name}'. Check your credentials."
            log.error(error_msg)
            raise ValueError(error_msg)

    def upload_file(self, local_path, s3_path, make_public=False):
        """Upload a file to S3"""
        if not os.path.exists(local_path):
            log.error(f"Local file not found: {local_path}")
            raise FileNotFoundError(f"Local file not found: {local_path}")

        try:
            extra_args = {"ACL": "public-read" if make_public else "private"}
            s3_path = s3_path.lstrip("/")

            log.info(f"Uploading {local_path} to {self.bucket_name}/{s3_path}")

            with open(local_path, "rb") as f:
                self.client.put_object(
                    Bucket=self.bucket_name, Key=s3_path, Body=f.read(), **extra_args
                )
            log.info(f"Successfully uploaded {local_path} to {s3_path}")

        except botocore.exceptions.ClientError as e:
            error = e.response.get("Error", {})
            error_code = error.get("Code", "Unknown")
            error_message = error.get("Message", str(e))
            log.error(f"Error uploading file ({error_code}): {error_message}")
            log.debug(f"Full error response: {e.response}")
            raise
        except Exception as e:
            log.error(f"Unexpected error uploading file: {str(e)}")
            raise

    def list_files(self, prefix=""):
        """List all files in the bucket"""
        try:
            prefix = prefix.lstrip("/")
            log.info(f"Listing files in {self.bucket_name} with prefix: {prefix}")

            paginator = self.client.get_paginator("list_objects_v2")
            files = []

            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if "Contents" in page:
                    files.extend([obj["Key"] for obj in page["Contents"]])

            log.info(f"Found {len(files)} files")
            return files

        except botocore.exceptions.ClientError as e:
            error = e.response.get("Error", {})
            error_code = error.get("Code", "Unknown")
            error_message = error.get("Message", str(e))
            log.error(f"Error listing files ({error_code}): {error_message}")
            log.debug(f"Full error response: {e.response}")
            raise
        except Exception as e:
            log.error(f"Unexpected error listing files: {str(e)}")
            raise

    def download_file(self, s3_path, local_path):
        """Download a file from S3"""
        try:
            s3_path = s3_path.lstrip("/")
            log.info(f"Downloading {self.bucket_name}/{s3_path} to {local_path}")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            self.client.download_file(self.bucket_name, s3_path, local_path)
            log.info(f"Successfully downloaded {s3_path} to {local_path}")

        except botocore.exceptions.ClientError as e:
            error = e.response.get("Error", {})
            error_code = error.get("Code", "Unknown")
            error_message = error.get("Message", str(e))
            log.error(f"Error downloading file ({error_code}): {error_message}")
            log.debug(f"Full error response: {e.response}")
            raise
        except Exception as e:
            log.error(f"Unexpected error downloading file: {str(e)}")
            raise

    def write_dataframe(self, df, s3_path, format="csv", **kwargs):
        """Write DataFrame directly to S3"""
        try:
            s3_path = s3_path.lstrip("/")
            log.info(f"Writing DataFrame to {self.bucket_name}/{s3_path}")

            buffer = io.BytesIO()

            # Write to buffer based on format
            if format.lower() == "csv":
                df.to_csv(buffer, **kwargs)
            elif format.lower() == "parquet":
                df.to_parquet(buffer, **kwargs)
            elif format.lower() == "json":
                df.to_json(buffer, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")

            buffer.seek(0)

            content_type = {
                "csv": "text/csv",
                "parquet": "application/octet-stream",
                "json": "application/json",
            }.get(format.lower(), "application/octet-stream")

            self.client.put_object(
                Bucket=self.bucket_name,
                Key=s3_path,
                Body=buffer.getvalue(),
                ContentType=content_type,
            )

            log.info(f"Successfully wrote DataFrame to {s3_path}")

        except Exception as e:
            log.error(f"Error writing DataFrame: {str(e)}")
            raise

    def read_dataframe(self, s3_path, format="csv", **kwargs):
        """Read DataFrame from S3"""
        try:
            s3_path = s3_path.lstrip("/")
            log.info(f"Reading DataFrame from {self.bucket_name}/{s3_path}")

            response = self.client.get_object(Bucket=self.bucket_name, Key=s3_path)
            buffer = io.BytesIO(response["Body"].read())

            if format.lower() == "csv":
                df = pd.read_csv(buffer, **kwargs)
            elif format.lower() == "parquet":
                df = pd.read_parquet(buffer, **kwargs)
            elif format.lower() == "json":
                df = pd.read_json(buffer, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")

            log.info(f"Successfully read DataFrame from {s3_path}")
            return df

        except Exception as e:
            log.error(f"Error reading DataFrame: {str(e)}")
            raise

    def append_dataframe(self, df, s3_path, format="csv", **kwargs):
        """Append DataFrame to existing file in S3 (CSV only)"""
        if format.lower() != "csv":
            log.error("Append operation only supported for CSV format")
            raise ValueError("Append operation only supported for CSV format")

        try:
            s3_path = s3_path.lstrip("/")
            log.info(f"Appending DataFrame to {self.bucket_name}/{s3_path}")

            try:
                existing_df = self.read_dataframe(s3_path, format="csv")
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            except botocore.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    log.info("File doesn't exist, creating new file")
                    combined_df = df
                else:
                    raise

            self.write_dataframe(combined_df, s3_path, format="csv", **kwargs)
            log.info(f"Successfully appended DataFrame to {s3_path}")

        except Exception as e:
            log.error(f"Error appending DataFrame: {str(e)}")
            raise

    def file_exists(self, s3_path):
        """
        Check if a file exists in S3

        Args:
            s3_path (str): Path to file in S3

        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            s3_path = s3_path.lstrip("/")
            self.client.head_object(Bucket=self.bucket_name, Key=s3_path)
            log.debug(f"File exists: {s3_path}")
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                log.debug(f"File does not exist: {s3_path}")
                return False
            # If there's a different error (e.g., permissions), let it raise
            log.error(f"Error checking file existence: {str(e)}")
            raise

    def upload_table(self, table, s3_path, make_public=False):
        """
        Upload a PyArrow Table directly to S3 in parquet format

        Args:
            table (pyarrow.Table): PyArrow Table to upload
            s3_path (str): Destination path in S3
            make_public (bool): Whether to make the file publicly readable
        """
        try:
            s3_path = s3_path.lstrip("/")
            log.info(f"Uploading PyArrow Table to {self.bucket_name}/{s3_path}")

            # Create a buffer and write the table to it
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            buffer.seek(0)

            # Upload to S3
            extra_args = {
                "ACL": "public-read" if make_public else "private",
                "ContentType": "application/octet-stream",
            }

            self.client.put_object(
                Bucket=self.bucket_name,
                Key=s3_path,
                Body=buffer.getvalue(),
                **extra_args,
            )

            log.info(f"Successfully uploaded Table to {s3_path}")

        except Exception as e:
            log.error(f"Error uploading Table: {str(e)}")
            raise
