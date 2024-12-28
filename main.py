import os
from loguru import logger as log
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from theta_snapshot import (
    snapshot,
    read_from_db,
    write_to_db,
    get_iv_chain,
    is_market_open,
    time_checker_ny,
)

if __name__ == "__main__":
    # --------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------
    load_dotenv(".env")

    is_market_open(break_Script=os.getenv("BREAK_SCRIPT") == "True")
    time_checker_ny(break_Script=os.getenv("BREAK_SCRIPT") == "True")

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
    log.info(
        f"Scrapping Snapshot: {snap_df['symbol'].unique().shape[0]} symbols, {snap_df.shape[0]} strategies"
    )

    inputs = [
        (snapshot, row["symbol"], row["currentDate"], row["weeks"], right)
        for _, row in snap_df.iterrows()
    ]

    snap_result = Parallel(n_jobs=cpus, backend="loky", verbose=2)(
        delayed(func)(*args) for func, *args in inputs
    )

    theta_df = pd.concat(snap_result)
    theta_df = theta_df[(theta_df["calCost"] < 5) & (theta_df["calCost"] > 0.8)]

    # --------------------------------------------------------------
    # Merge with Earnings Calendar
    # --------------------------------------------------------------
    theta_df = theta_df.merge(
        earn_df[["symbol", "noOfEsts", "bdte", "dte"]], on="symbol", how="inner"
    )

    # --------------------------------------------------------------
    # Merge calCost Table
    # --------------------------------------------------------------
    log.info("Fecthing calCostPctMean")

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

    # --------------------------------------------------------------
    # Implied Volatility
    # --------------------------------------------------------------
    log.info("Scrapping Implied Volatility")
    remaining_symbols = set(theta_df["symbol"].unique())

    ivs = Parallel(n_jobs=cpus, backend="loky", verbose=2)(
        delayed(get_iv_chain)(symb) for symb in remaining_symbols
    )
    iv_df = pd.concat(ivs)

    # --------------------------------------------------------------
    # Write to DB
    # --------------------------------------------------------------

    log.info("Completed Snapshots and IVs")
