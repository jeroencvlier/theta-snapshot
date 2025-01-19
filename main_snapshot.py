import os
import sys
import logging as log
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from datetime import datetime as dt
import option_emporium as oe

from theta_snapshot import (
    snapshot,
    read_from_db,
    write_to_db,
    append_to_table,
    delete_old_data,
    get_iv_chain,
    is_market_open,
    time_checker_ny,
    main_wrapper,
    iv_features,
    main_telegram,
    calculate_buisness_days,
    generate_predictions,
)



log.basicConfig(level=log.INFO, format="%(asctime)s - %(message)s")


@main_wrapper
def main():
    is_market_open(break_Script=os.getenv("BREAK_SCRIPT") == "True")
    time_checker_ny(break_Script=os.getenv("BREAK_SCRIPT") == "True")
    # --------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------
    right = "C"

    cpus = max(os.cpu_count(), 20)
    log.info(f"Available CPUs: {os.cpu_count()}, defaulting to {cpus} for parallelism.")

    # --------------------------------------------------------------
    # Read Data
    # --------------------------------------------------------------

    log.info("Reading Grades from DB")
    earnings_query = """select * from "EarningsCalendarCombined" WHERE bdte <= 20"""
    earn_df = read_from_db(query=earnings_query)

    # --------------------------------------------------------------
    log.info("Reading Earnings Calendar from DB")
    maxgrade_query = """SELECT "symbol" FROM public."historicalBacktestGrades" GROUP BY "symbol" HAVING max(undmean_avg_trade_class_combined_grade) >= 1.25"""
    grade_symbols = read_from_db(query=maxgrade_query)["symbol"].unique()
    grade_query = f"""SELECT * FROM public."historicalBacktestGrades" WHERE symbol in {tuple(grade_symbols)} AND "weeks" <= '5' AND "latest" = TRUE"""
    grades = read_from_db(query=grade_query).drop(columns=["latest"])
    grades["weeks"] = grades["weeks"].astype(int)


    # --------------------------------------------------------------

    snap_df = pd.merge(grades, earn_df, on="symbol", how="inner")

    # --------------------------------------------------------------
    # Generate Snapshot
    # --------------------------------------------------------------
    n_symbols = snap_df["symbol"].unique().shape[0]
    log.info(f"Scrapping Snapshot: {n_symbols} symbols, {snap_df.shape[0]} strategies")

    inputs = [
        {
            "symbol": row["symbol"],
            "rdate": row["reportDate"],
            "weeks": row["weeks"],
            "right": right,
        }
        for _, row in snap_df.iterrows()
    ]

    snap_result = Parallel(n_jobs=cpus, backend="loky", verbose=10)(
        delayed(snapshot)(**kwargs) for kwargs in inputs
    )

    theta_df = pd.concat(snap_result)
    log.info(f"Snapshot Completed: {theta_df.shape[0]} strategies")

    # --------------------------------------------------------------
    # Merge with Earnings Calendar and Grades
    # --------------------------------------------------------------
    theta_df = theta_df.merge(
        earn_df[["symbol", "noOfEsts", "epsForecast", "bdte", "dte"]], on="symbol", how="inner"
    )
    theta_df = theta_df.merge(grades, on=["symbol", "weeks"], how="inner")

    # --------------------------------------------------------------
    # Merge calCost Table
    # --------------------------------------------------------------
    log.info("Fecthing calCostPctMean")
    if len(theta_df) == 0:
        log.warning("No data to process")
        return

    calcost_query = """
        SELECT symbol, bdte, "calCostPctMean", "calCostPctStd", "calCostPctMax", "calCostPctMin", "weeks", "histEarningsCount" ,"histCalCostCount"
        FROM public."calCostPctBacktestMeans" 
        WHERE latest = True AND symbol in {}
    """.format(tuple(theta_df["symbol"].unique()))

    calcost_df = read_from_db(query=calcost_query)
    calcost_df["weeks"] = calcost_df["weeks"].astype(int)
    calcost_df["calCostPctMean"] = calcost_df["calCostPctMean"].round(4)

    # --------------------------------------------------------------
    # Calculate Expected Calendar Price
    # --------------------------------------------------------------
    cc_fd = calcost_df[calcost_df["bdte"] == 1].copy()
    cc_fd = cc_fd.rename(columns={"calCostPctMean": "calCostPctMeanDayZero"})
    theta_df = theta_df.merge(
        cc_fd[["symbol", "weeks", "calCostPctMeanDayZero"]], on=["symbol", "weeks"], how="inner"
    )

    theta_df = theta_df.merge(calcost_df, on=["symbol", "bdte", "weeks"], how="inner")
    theta_df = oe.calculate_diffs(theta_df)
    theta_df = oe.expected_calendar_price(theta_df)

    # --------------------------------------------------------------
    # Implied Volatility
    # --------------------------------------------------------------
    log.info("Scrapping Implied Volatility")
    remaining_symbols = set(theta_df["symbol"].unique())
    log.info("Scrapping Implied Volatility for {} symbols".format(len(remaining_symbols)))

    ivs = Parallel(n_jobs=cpus, backend="loky", verbose=10)(
        delayed(get_iv_chain)(symb) for symb in remaining_symbols
    )
    iv_df = pd.concat(ivs)

    theta_df = iv_features(theta_df, iv_df)

    # --------------------------------------------------------------
    # Machine Learning
    # --------------------------------------------------------------

    theta_df = calculate_buisness_days(theta_df)
    theta_df = generate_predictions(theta_df)

    # --------------------------------------------------------------
    # Write to DB
    # --------------------------------------------------------------
    timestamp_update = int(dt.now().timestamp())
    theta_df = theta_df.assign(lastUpdated=timestamp_update)
    iv_df = iv_df.assign(lastUpdated=timestamp_update)
    # write_to_db(theta_df, "ThetaSnapshot",if_exists="replace")
    append_to_table(theta_df, "ThetaSnapshot", indexes=["symbol", "lastUpdated", "bdte", "weeks"])
    append_to_table(iv_df, "ThetaIVSnapshot", indexes=["symbol", "lastUpdated"])
    log.info("Completed Snapshots and IVs")

    # --------------------------------------------------------------
    # Telegram
    # --------------------------------------------------------------
    main_telegram()

    # TODO: Drop data oldert than 5 days
    delete_old_data("ThetaSnapshot", 7)
    delete_old_data("ThetaIVSnapshot", 7)


if __name__ == "__main__":
    load_dotenv(".env")
    main()
