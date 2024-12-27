import httpx
import pandas as pd
from typing import List
import logging
import sys
import option_emporium as oe
from theta_snapshot import CalendarSnapData, snapshot_filter
from datetime import datetime as dt
from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
pd.set_option("display.max_columns", None)

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


def is_list_of_timestamps(lst):
    return all(isinstance(item, (dt, pd.Timestamp)) for item in lst)


def is_list_of_integers(lst):
    return all(isinstance(item, int) for item in lst)


def get_back_expiration_date(
    fexp: pd.Timestamp,
    exp_list: List[pd.Timestamp],
    weeks_between_fb: int,
) -> pd.Timestamp:
    assert weeks_between_fb in [1, 2, 3, 4, 5, 6], "Invalid number of weeks"
    day_offsets = [0, 1, -1, 2, -2, 3, -3]
    if is_list_of_integers(exp_list):
        exp_list = [pd.to_datetime(date, format="%Y%m%d") for date in exp_list]
    else:
        assert is_list_of_timestamps(exp_list), "Invalid list of timestamps"

    for offset in day_offsets:
        # Calculate potential back expiration date
        bexp = fexp + pd.DateOffset(days=(weeks_between_fb * 7 + offset))
        # Check if the potential back expiration date exists in the data
        if bexp in exp_list:
            if offset != 0:
                logging.info(
                    f"Public Holiday Detected, back expiration date is offset {offset} days."
                )
            return bexp.strftime("%Y%m%d")

    return None


# --------------------------------------------------------------
# Theta Data API
# --------------------------------------------------------------

BASE_URL = "http://127.0.0.1:25511/v2"


import httpx
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            logger.info(f"Fetched {len(data.get('response', []))} items from {url}")

            # Handle pagination
            next_page = response.headers.get("Next-Page")
            url = next_page if next_page and next_page != "null" else None

        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            if retries < max_retries:
                retries += 1
                logger.info(f"Retrying... attempt {retries}")
            else:
                logger.error("Max retries exceeded. Exiting pagination.")
                break
        except KeyError as e:
            logger.error(f"Key error: {e}. Response format may have changed.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break

    # Return responses and format header (if available)
    format_header = (
        data.get("header", {}).get("format", None) if "data" in locals() else None
    )
    return responses, format_header


def get_expiry_dates(symbol):
    params = {"root": symbol}
    url = BASE_URL + "/list/expirations"
    responses, _ = request_pagination(url, params)
    return responses


def greeks_snapshot(symbol: str, exp: int, right: str):
    url = BASE_URL + "/bulk_snapshot/option/greeks"
    params = {"root": symbol, "exp": exp, "right": right}
    response, columns = request_pagination(url, params)
    df = response_to_df(response, columns)
    df = df[df["right"] == right]
    df.drop(columns=["ms_of_day2", "bid", "ask", "ms_of_day"], inplace=True)
    return df


def get_greeks_snapshot(so: CalendarSnapData, fb: str):
    if fb == "front":
        so.greeks_front = greeks_snapshot(so.symbol, so.fexp, so.right)
        so.greeks_front.rename(columns={"underlying_price": "underlying"}, inplace=True)
    elif fb == "back":
        so.greeks_back = greeks_snapshot(so.symbol, so.bexp, so.right)
        so.greeks_back.drop(columns="underlying_price", inplace=True)
    else:
        logging.error(f"Invalid front/back value: {fb}")
        raise ValueError


def get_quote(symbol: str, exp: int, right: str):
    url = BASE_URL + "/bulk_snapshot/option/quote"
    params = {"root": symbol, "exp": exp, "right": right}
    response, columns = request_pagination(url, params)
    df = response_to_df(response, columns)
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


def get_quote_snapshot(so: CalendarSnapData, fb: str):
    if fb == "front":
        so.quotes_front = get_quote(so.symbol, so.fexp, so.right)
    elif fb == "back":
        so.quotes_back = get_quote(so.symbol, so.bexp, so.right)
    else:
        logging.error(f"Invalid front/back value: {fb}")
        raise ValueError


def get_oi(symbol: str, exp: int, right: str):
    url = BASE_URL + "/bulk_snapshot/option/open_interest"
    # logging.info("Requesting port: %s", url)
    params = {"root": symbol, "exp": exp, "right": right}
    response, columns = request_pagination(url, params)
    df = response_to_df(response, columns)
    df = df[df["right"] == right]
    df.drop(columns=["ms_of_day"], inplace=True)
    return df


def get_oi_snapshot(so: CalendarSnapData, fb: str):
    if fb == "front":
        so.oi_front = get_oi(so.symbol, so.fexp, so.right)
    elif fb == "back":
        so.oi_back = get_oi(so.symbol, so.bexp, so.right)
    else:
        logging.error(f"Invalid front/back value: {fb}")
        raise ValueError


def merge_snapshot(front, back):
    return pd.merge(
        front,
        back,
        on=[c for c in front if c in ["root", "right", "date", "strike"]],
        how="inner",
        suffixes=("_front", "_back"),
    )


# --------------------------------------------------------------
# Main Function
# --------------------------------------------------------------


def snapshot(symbol: str, rdate: pd.Timestamp, weeks: int, right: str = "C"):
    # symbol = "JPM"
    # date_string = "2025-01-15 00:00:00"
    # weeks = 1
    # rdate = pd.Timestamp(date_string)
    # right = "C"

    # --------------------------------------------------------------
    so = CalendarSnapData(symbol=symbol, rdatedt=rdate, weeks=weeks, right=right)

    # --------------------------------------------------------------
    # Expiration Dates
    # --------------------------------------------------------------
    expirations = get_expiry_dates(so.symbol)
    cal_dates = [d for d in expirations if d >= so.rdate]
    so.fexp = min(cal_dates)

    so.bexp = get_back_expiration_date(
        fexp=so.fexpdt,
        exp_list=cal_dates,
        weeks_between_fb=so.weeks,
    )

    if so.bexp is None:
        logging.error(f"No back expiration date found for {so.symbol}")
        sys.exit(0)

    if (so.fexpdt - so.rdatedt).days >= 7:
        logging.error("Front expiration date is far close to report date")
        sys.exit(0)

    # --------------------------------------------------------------
    # Get Snapshots
    # --------------------------------------------------------------

    inputs = [
        (get_greeks_snapshot, so, "front"),
        (get_greeks_snapshot, so, "back"),
        (get_quote_snapshot, so, "front"),
        (get_quote_snapshot, so, "back"),
        (get_oi_snapshot, so, "front"),
        (get_oi_snapshot, so, "back"),
    ]
    _ = Parallel(n_jobs=6, backend="threading")(
        delayed(func)(*args) for func, *args in inputs
    )

    so.greeks = merge_snapshot(so.greeks_front, so.greeks_back)
    so.quotes = merge_snapshot(so.quotes_front, so.quotes_back)
    so.oi = merge_snapshot(so.oi_front, so.oi_back)

    m_cols = ["root", "right", "date", "strike", "expiration_front", "expiration_back"]
    complete_df = so.quotes.merge(
        so.greeks, on=m_cols, how="inner", suffixes=("_quote", "_ivgreek")
    ).merge(so.oi, on=m_cols, how="inner", suffixes=("", "_oi"))

    # --------------------------------------------------------------
    # Assertions for the final DataFrame
    # --------------------------------------------------------------
    assert len(complete_df["root"].unique()) == 1, "Multiple symbols found"
    assert so.symbol == complete_df["root"].unique()[0], "Symbol mismatch"
    assert len(complete_df["right"].unique()) == 1, "Multiple rights found"
    assert so.right == complete_df["right"].unique()[0], "Right mismatch"
    assert len(complete_df["date"].unique()) == 1, "Multiple dates found"

    # --------------------------------------------------------------
    # Calculations and Column creation
    # --------------------------------------------------------------
    complete_df["strike"] = complete_df["strike"].div(1000).round(2)
    complete_df = oe.calculate_spreads(complete_df)
    complete_df = oe.calendar_calculations(complete_df)
    complete_df["weeks"] = so.weeks
    complete_df["reportDate"] = so.rdate
    complete_df.rename(
        columns={
            "expiration_front": "fexp",
            "expiration_back": "bexp",
            "root": "symbol",
        },
        inplace=True,
    )

    # --------------------------------------------------------------
    # Filters
    # --------------------------------------------------------------
    complete_df = snapshot_filter(df=complete_df, min_oi=20, max_rows=2)

    return complete_df


if __name__ == "__main__":
    snapshot("JPM", pd.Timestamp("2025-01-15"), 1, "C")
    print("Done")
