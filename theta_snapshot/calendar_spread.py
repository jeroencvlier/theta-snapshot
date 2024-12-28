import pandas as pd
from typing import List
from loguru import logger
import sys
from datetime import datetime as dt
from joblib import Parallel, delayed
from dotenv import load_dotenv

import option_emporium as oe
from theta_snapshot import (
    CalendarSnapData,
    snapshot_filter,
    get_expiry_dates,
    get_greeks,
    get_quote,
    get_oi,
)


load_dotenv(".env")

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
pd.set_option("display.max_columns", None)

# --------------------------------------------------------------
# Helper Functions
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
                logger.info(
                    f"Public Holiday Detected, back expiration date is offset {offset} days."
                )
            return bexp.strftime("%Y%m%d")

    return None


# --------------------------------------------------------------
# Theta Data API
# --------------------------------------------------------------


def get_greeks_snapshot(so: CalendarSnapData, fb: str):
    if fb == "front":
        so.greeks_front = get_greeks(so.symbol, so.fexp, so.right)
        so.greeks_front.rename(columns={"underlying_price": "underlying"}, inplace=True)
    elif fb == "back":
        so.greeks_back = get_greeks(so.symbol, so.bexp, so.right)
        so.greeks_back.drop(columns="underlying_price", inplace=True)
    else:
        logger.error(f"Invalid front/back value: {fb}")
        raise ValueError


def get_quote_snapshot(so: CalendarSnapData, fb: str):
    if fb == "front":
        so.quotes_front = get_quote(so.symbol, so.fexp, so.right)
    elif fb == "back":
        so.quotes_back = get_quote(so.symbol, so.bexp, so.right)
    else:
        logger.error(f"Invalid front/back value: {fb}")
        raise ValueError


def get_oi_snapshot(so: CalendarSnapData, fb: str):
    if fb == "front":
        so.oi_front = get_oi(so.symbol, so.fexp, so.right)
    elif fb == "back":
        so.oi_back = get_oi(so.symbol, so.bexp, so.right)
    else:
        logger.error(f"Invalid front/back value: {fb}")
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
        logger.error(f"No back expiration date found for {so.symbol}")
        sys.exit(0)

    if (so.fexpdt - so.rdatedt).days >= 7:
        logger.error("Front expiration date is far close to report date")
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
    _ = Parallel(n_jobs=6, backend="threading")(delayed(func)(*args) for func, *args in inputs)

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
