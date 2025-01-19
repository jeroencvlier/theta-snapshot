import pandas as pd
import logging as log
from joblib import Parallel, delayed
import numpy as np

import option_emporium as oe
from theta_snapshot import (
    CalendarSnapData,
    snapshot_filter,
    get_expiry_dates,
    get_back_expiration_date,
    get_greeks_snapshot,
    get_quote_snapshot,
    get_oi_snapshot,
    merge_snapshot
)

pd.set_option("display.max_columns", None)
log.basicConfig(level=log.INFO, format="%(asctime)s - %(message)s")


def snapshot(symbol: str, rdate: pd.Timestamp, weeks: int, right: str = "C", roots: list = None):
    # symbol = "APO"
    # date_string = '2025-02-04 00:00:00'
    # weeks = 1
    # rdate = pd.Timestamp(date_string)
    # right = "C"
    try:
        # --------------------------------------------------------------
        if roots is None:
            roots = [symbol]

        so = CalendarSnapData(
            symbol=symbol, roots=roots, rdatedt=rdate, weeks=int(weeks), right=right
        )

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
            # log.info(f"No back expiration date found for {so.symbol}")
            return pd.DataFrame()

        if (so.fexpdt - so.rdatedt).days >= 7:
            log.error(
                "Front exp is more than 7 days away for {}, {} days".format(
                    so.symbol, (so.fexpdt - so.rdatedt).days
                )
            )
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

        # get_quote_snapshot(so, "front")
        _ = Parallel(n_jobs=6, backend="threading")(delayed(func)(*args) for func, *args in inputs)

        # if any are None then return empty DataFrame
        if any(
            [
                so.greeks_front is None,
                so.greeks_back is None,
                so.quotes_front is None,
                so.quotes_back is None,
                # so.oi_front is None,
                # so.oi_back is None,
            ]
        ):
            log.warning(f"Snapshot data is missing for {so.symbol}.")
            return pd.DataFrame()

        so.greeks = merge_snapshot(so.greeks_front, so.greeks_back)
        so.quotes = merge_snapshot(so.quotes_front, so.quotes_back)

        m_cols = ["symbol", "right", "date", "strike_milli", "exp_front", "exp_back"]

        complete_df = so.quotes.merge(
            so.greeks, on=m_cols, how="inner", suffixes=("_quote", "_ivgreek")
        )
        # --------------------------------------------------------------
        # Deal with oi (it's ok if missing)
        # --------------------------------------------------------------
        if (so.oi_front is not None) and (so.oi_back is None):
            so.oi_back = so.oi_front
            so.oi_back["open_interest"] = np.nan

        elif (so.oi_back is not None) and (so.oi_front is None):
            so.oi_front = so.oi_back
            so.oi_front["open_interest"] = np.nan
        elif (so.oi_back is None) and (so.oi_front is None):
            cols = ["symbol", "right", "strike_milli", "exp", "open_interest"]
            so.oi_front = pd.DataFrame({}, columns=cols)
            so.oi_back = pd.DataFrame({}, columns=cols)

        so.oi = merge_snapshot(so.oi_front, so.oi_back)
        moi_cols = ["symbol", "right", "strike_milli", "exp_front", "exp_back"]
        complete_df = complete_df.merge(so.oi, on=moi_cols, how="left", suffixes=("", "_oi"))

        # --------------------------------------------------------------
        # Assertions for the final DataFrame
        # --------------------------------------------------------------
        all_symbols = complete_df["symbol"].unique()
        assert len(all_symbols) == 1, (
            f"Multiple symbols found,  expected {so.symbol}, {all_symbols}"
        )
        assert so.symbol == all_symbols[0], (
            f"Symbol mismatch, expected {so.symbol}, found {all_symbols[0]}"
        )
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
        complete_df = snapshot_filter(df=complete_df, min_oi=0, max_rows=2)

        return complete_df

    except Exception as e:
        log.error(f"Error in snapshot for {symbol}: {e}")
        return pd.DataFrame()
