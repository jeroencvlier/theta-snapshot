import os
from loguru import logger
import pandas as pd
from joblib import Parallel, delayed
from theta_snapshot import snapshot, read_from_db, write_to_db, get_iv_chain
import logging

if __name__ == "__main__":
    # set logger to info
    # logger.remove()
    # logger.add(sys.stdout, level="INFO")
    # logging.getLogger("httpx").setLevel(logging.WARNING)
    # logging.getLogger("http").setLevel(logging.WARNING)
    # logging.getLogger("httpcore").setLevel(logging.WARNING)
    # logging.getLogger("urllib3").setLevel(logging.WARNING)
    # logging.getLogger("asyncio").setLevel(logging.WARNING)

    right = "C"
    cpus = max(os.cpu_count() - 1, 20)

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
