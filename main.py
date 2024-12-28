import os
from loguru import logger
import pandas as pd
from joblib import Parallel, delayed
from theta_snapshot import snapshot, read_from_db, write_to_db, get_iv_chain

if __name__ == "__main__":
    right = "C"
    cpus = max(os.cpu_count() - 1, 20)
    logger.info(
        f"Number of CPUs: {os.cpu_count()}, defaulting to {cpus} cores for parallel processing"
    )

    logger.info("Reading Grades from DB")
    query = (
        """SELECT * FROM public."StockGrades" WHERE under_avg_trade_class >= 1.0 AND weeks <= 5"""
    )
    grades = read_from_db(query=query)

    logger.info("Reading Earnings Calendar from DB")
    earn_df = read_from_db(query="""select * from "EarningsCalendarCombined" WHERE bdte <= 20""")
    snap_df = pd.merge(grades, earn_df, on="symbol", how="inner")

    # --------------------------------------------------------------
    # Snapshot
    # --------------------------------------------------------------
    logger.info("Scrapping Snapshot")

    inputs = [
        (snapshot, row["symbol"], row["currentDate"], row["weeks"], right)
        for _, row in snap_df.iterrows()
    ]

    snapshot_result = Parallel(n_jobs=cpus, backend="loky", verbose=2)(
        delayed(func)(*args) for func, *args in inputs
    )

    snapshot_df = pd.concat(snapshot_result)
    snapshot_df = snapshot_df[(snapshot_df["calCost"] < 5) & (snapshot_df["calCost"] > 0.8)]

    # join the eps
    snapshot_df = snapshot_df.merge(
        earn_df[["symbol", "noOfEsts", "bdte", "dte"]], on="symbol", how="inner"
    )

    # --------------------------------------------------------------
    # Implied Volatility
    # --------------------------------------------------------------
    logger.info("Scrapping Implied Volatility")
    remaining_symbols = set(snapshot_df["symbol"].unique())

    ivs = Parallel(n_jobs=cpus, backend="loky", verbose=2)(
        delayed(get_iv_chain)(symb) for symb in remaining_symbols
    )
    iv_df = pd.concat(ivs)

    # --------------------------------------------------------------
    # Write to DB
    # --------------------------------------------------------------

    logger.info("Completed Snapshots and IVs")
