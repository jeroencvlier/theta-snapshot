import os
import sys
from datetime import datetime as dt
from tqdm import tqdm

from loguru import logger
import logging
import pandas as pd
from joblib import Parallel, delayed
from theta_snapshot import snapshot, read_from_db, write_to_db


if __name__ == "__main__":
    right = "C"

    logging.info("Starting snapshot")

    query = """SELECT * FROM public."StockGrades" WHERE under_avg_trade_class >= 1.0 AND weeks <= 5"""
    grades = read_from_db(query=query)

    earn_df = read_from_db(
        query="""select * from "EarningsCalendarCombined" WHERE bdte <= 20"""
    )

    assert (
        len(earn_df["currentDate"].unique()) == 1
    ), "Error in the earnings calendar on the date"
    assert (
        earn_df["currentDate"].unique()[0].date() == dt.today().date()
    ), "Error in the earnings calendar on the date"

    snap_df = pd.merge(grades, earn_df, on="symbol", how="inner")

    inputs = [
        (snapshot, row["symbol"], row["currentDate"], row["weeks"], right)
        for _, row in snap_df.iterrows()
    ]

    snapshot_result = Parallel(n_jobs=os.cpu_count(), backend="loky")(
        delayed(func)(*args) for func, *args in inputs
    )

    snapshot_df = pd.concat(snapshot_result)

    snapshot_df = snapshot_df[
        (snapshot_df["calCost"] < 5) & (snapshot_df["calCost"] > 0.8)
    ]

    # join the eps
    snapshot_df = snapshot_df.merge(
        earn_df[["symbol", "noOfEsts", "bdte", "dte"]], on="symbol", how="inner"
    )
