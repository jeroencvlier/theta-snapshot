import os
import httpx
from loguru import logger as log
import pandas as pd
import option_emporium as oe
from typing import List
from datetime import datetime as dt
from theta_snapshot import CalendarSnapData


# --------------------------------------------------------------
# Pagination Function
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


# --------------------------------------------------------------
# Endpoint Functions
# --------------------------------------------------------------


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
        columns=["bid_condition", "bid_exchange", "ask_condition", "ask_exchange"],
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
    df.drop(columns=["ms_of_day", "date"], inplace=True)
    return df


# --------------------------------------------------------------
# Endpoint Prep Functions
# --------------------------------------------------------------


def get_greeks_snapshot(so: CalendarSnapData, fb: str):
    if fb == "front":
        so.greeks_front = get_greeks(so.symbol, so.fexp, so.right)
    elif fb == "back":
        so.greeks_back = get_greeks(so.symbol, so.bexp, so.right)
        so.greeks_back.drop(columns="underlying", inplace=True)
    else:
        log.error(f"Invalid front/back value: {fb}")
        raise ValueError


def get_quote_snapshot(so: CalendarSnapData, fb: str):
    if fb == "front":
        so.quotes_front = get_quote(so.symbol, so.fexp, so.right)
    elif fb == "back":
        so.quotes_back = get_quote(so.symbol, so.bexp, so.right)
        so.quotes_back.drop(columns="ms_of_day", inplace=True)
    else:
        log.error(f"Invalid front/back value: {fb}")
        raise ValueError


def get_oi_snapshot(so: CalendarSnapData, fb: str):
    if fb == "front":
        so.oi_front = get_oi(so.symbol, so.fexp, so.right)
    elif fb == "back":
        so.oi_back = get_oi(so.symbol, so.bexp, so.right)
    else:
        log.error(f"Invalid front/back value: {fb}")
        raise ValueError


def merge_snapshot(front, back):
    return pd.merge(
        front,
        back,
        on=[c for c in front if c in ["symbol", "right", "date", "strike_milli"]],
        how="inner",
        suffixes=("_front", "_back"),
    )


# --------------------------------------------------------------
# Unpacking Functions
# --------------------------------------------------------------


def response_to_df(response, columns):
    rows = []
    for item in response:
        ticks = item["ticks"][0]
        contract = item["contract"]
        row = {**contract, **dict(zip(columns, ticks))}
        rows.append(row)
    df = pd.DataFrame(rows)
    rename_dict = {
        "strike": "strike_milli",
        "underlying_price": "underlying",
        "expiration": "exp",
        "root": "symbol",
    }
    for k, v in rename_dict.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
    return df


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
# Expiration Seeker Functions
# --------------------------------------------------------------


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
                log.info(f"Public Holiday Detected, back expiration date is offset {offset} days.")
            return int(bexp.strftime("%Y%m%d"))

    return None
