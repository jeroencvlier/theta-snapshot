import pandas as pd
from loguru import logger as log
from joblib import Parallel, delayed

import option_emporium as oe
from theta_snapshot import (
    CalendarSnapData,
    snapshot_filter,
    get_expiry_dates,
    get_back_expiration_date,
    get_greeks_snapshot,
    get_quote_snapshot,
    get_oi_snapshot,
    merge_snapshot,
)

pd.set_option("display.max_columns", None)


def snapshot(symbol: str, rdate: pd.Timestamp, weeks: int, right: str = "C"):
    # symbol = "HUM"
    # date_string = "2025-01-23 00:00:00"
    # weeks = 2
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
        log.warning(f"No back expiration date found for {so.symbol}")
        expiration_sought = (so.fexpdt + pd.DateOffset(days=(so.weeks * 7))).strftime("%Y%m%d")
        log.info(f"Expiration Sought: {expiration_sought}")
        log.info(f"Expirations: {cal_dates}")
        log.info(f"DataClass: \n{so}")
        return pd.DataFrame()

    if (so.fexpdt - so.rdatedt).days >= 7:
        log.error("Front expiration date is far close to report date")
        return pd.DataFrame()

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

    m_cols = ["symbol", "right", "date", "strike_milli", "exp_front", "exp_back"]
    moi_cols = ["symbol", "right", "strike_milli", "exp_front", "exp_back"]

    complete_df = so.quotes.merge(
        so.greeks, on=m_cols, how="inner", suffixes=("_quote", "_ivgreek")
    ).merge(so.oi, on=moi_cols, how="inner", suffixes=("", "_oi"))

    # --------------------------------------------------------------
    # Assertions for the final DataFrame
    # --------------------------------------------------------------
    all_symbols = complete_df["symbol"].unique()
    assert len(all_symbols) == 1, f"Multiple symbols found,  expected {so.symbol}, {all_symbols}"
    assert (
        so.symbol == all_symbols[0]
    ), f"Symbol mismatch, expected {so.symbol}, found {all_symbols[0]}"
    assert len(complete_df["right"].unique()) == 1, "Multiple rights found"
    assert so.right == complete_df["right"].unique()[0], "Right mismatch"
    assert len(complete_df["date"].unique()) == 1, "Multiple dates found"

    # --------------------------------------------------------------
    # Calculations and Column creation
    # --------------------------------------------------------------
    complete_df["strike"] = complete_df["strike_milli"].div(1000).round(2)
    complete_df = oe.calculate_spreads(complete_df)
    complete_df = oe.calendar_calculations(complete_df)
    complete_df["weeks"] = so.weeks
    complete_df["reportDate"] = so.rdate

    # --------------------------------------------------------------
    # Filters
    # --------------------------------------------------------------
    complete_df = snapshot_filter(df=complete_df, min_oi=20, max_rows=2)

    return complete_df
