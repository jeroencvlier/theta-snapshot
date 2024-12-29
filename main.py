import os
import sys
from loguru import logger as log
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from datetime import datetime as dt
import option_emporium as oe

from theta_snapshot import (
    snapshot,
    read_from_db,
    write_to_db,
    get_iv_chain,
    is_market_open,
    time_checker_ny,
    main_wrapper,
    iv_features,
    send_telegram_alerts,
)


@main_wrapper
def main():
    is_market_open(break_Script=os.getenv("BREAK_SCRIPT") == "True")
    time_checker_ny(break_Script=os.getenv("BREAK_SCRIPT") == "True")
    # --------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------
    right = "C"
    cpus = max(os.cpu_count() - 1, 20)
    log.info(f"Available CPUs: {os.cpu_count()}, defaulting to {cpus} for parallelism.")

    # --------------------------------------------------------------
    # Read Data
    # --------------------------------------------------------------

    log.info("Reading Grades from DB")
    earnings_query = """select * from "EarningsCalendarCombined" WHERE bdte <= 20"""
    earn_df = read_from_db(query=earnings_query)

    # --------------------------------------------------------------
    log.info("Reading Earnings Calendar from DB")
    grade_query = """
        SELECT * FROM public."StockGrades" 
        WHERE under_avg_trade_class >= 1.0 AND weeks <= 5
    """
    grades = read_from_db(query=grade_query)

    # --------------------------------------------------------------

    snap_df = pd.merge(grades, earn_df, on="symbol", how="inner")

    # --------------------------------------------------------------
    # Generate Snapshot
    # --------------------------------------------------------------
    n_symbols = snap_df["symbol"].unique().shape[0]
    log.info(f"Scrapping Snapshot: {n_symbols} symbols, {snap_df.shape[0]} strategies")

    inputs = [
        (snapshot, row["symbol"], row["reportDate"], row["weeks"], right)
        for _, row in snap_df.iterrows()
    ]

    snap_result = Parallel(n_jobs=cpus, backend="loky", verbose=2)(
        delayed(func)(*args) for func, *args in inputs
    )

    theta_df = pd.concat(snap_result)
    theta_df = theta_df[(theta_df["calCost"] < 5) & (theta_df["calCost"] > 0.5)]
    log.success(f"Snapshot Completed: {theta_df.shape[0]} strategies")

    # --------------------------------------------------------------
    # Merge with Earnings Calendar and Grades
    # --------------------------------------------------------------
    theta_df = theta_df.merge(
        earn_df[["symbol", "noOfEsts", "bdte", "dte"]], on="symbol", how="inner"
    )
    theta_df = theta_df.merge(grades, on=["symbol", "weeks"], how="inner")

    # --------------------------------------------------------------
    # Merge calCost Table
    # --------------------------------------------------------------
    log.info("Fecthing calCostPctMean")
    if len(theta_df) == 0:
        log.warning("No data to process")
        sys.exit("No data to process")

    # TODO: Check if histcalcostmean is the same as calCostPctMean
    calcost_query = """
        SELECT symbol, bdte, "histcalcostmean" AS "calCostPctMean", "weeks", "histearningscount" 
        FROM public."calCostPctHistoricalMeans" 
        WHERE latest = True AND symbol in {}
    """.format(tuple(theta_df["symbol"].unique()))

    calcost_df = read_from_db(query=calcost_query)
    calcost_df["calCostPctMean"] = calcost_df["calCostPctMean"].round(4)

    # NOTE: This might be able to be removed, the cal cost on day zero
    cc_fd = calcost_df[calcost_df["bdte"] == 1].copy()
    cc_fd = cc_fd.rename(columns={"calCostPctMean": "calCostPctMeanDayZero"})
    theta_df = theta_df.merge(
        cc_fd[["symbol", "weeks", "calCostPctMeanDayZero"]], on=["symbol", "weeks"], how="inner"
    )
    # >> end of NOTE

    theta_df = theta_df.merge(calcost_df, on=["symbol", "bdte", "weeks"], how="inner")
    theta_df = oe.calculate_diffs(theta_df)
    theta_df = oe.expected_calendar_price(theta_df)

    # --------------------------------------------------------------
    # Implied Volatility
    # --------------------------------------------------------------
    log.info("Scrapping Implied Volatility")
    remaining_symbols = set(theta_df["symbol"].unique())

    ivs = Parallel(n_jobs=cpus, backend="loky", verbose=2)(
        delayed(get_iv_chain)(symb) for symb in remaining_symbols
    )
    iv_df = pd.concat(ivs)

    theta_df = iv_features(theta_df, iv_df)

    # --------------------------------------------------------------
    # Machine Learning
    # --------------------------------------------------------------

    # TODO: Add ML Code

    # --------------------------------------------------------------
    # Write to DB
    # --------------------------------------------------------------
    theta_df["lastUpdated"] = int(dt.now().timestamp())
    write_to_db(theta_df, "ThetaSnapshot")
    write_to_db(iv_df, "ThetaIVSnapshot")

    log.info("Completed Snapshots and IVs")

    # --------------------------------------------------------------
    # Telegram
    # --------------------------------------------------------------
    time_checker_ny(target_minute=44, break_Script=os.getenv("BREAK_SCRIPT") == "True")
    send_telegram_alerts()


if __name__ == "__main__":
    load_dotenv(".env")
    main()
