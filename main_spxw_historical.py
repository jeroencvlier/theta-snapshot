import os
import sys
import datetime as dt
import logging as log
import pandas as pd
from joblib import Parallel, delayed
import pyarrow as pa
from tqdm import tqdm
import itertools
from functools import reduce
import random

from theta_snapshot import (
    CalendarSnapData,
    S3Handler,
    read_from_db,
    get_expiry_dates,
    get_back_expiration_date,
    get_exp_trading_days,
    get_strikes_exp,
    get_greeks_historical,
    get_quotes_historical,
    get_oi_historical,
    get_bulk_oi_historical,
    batched,
    is_market_open,
    get_option_roots,
)

log.basicConfig(level=log.INFO, format="%(asctime)s - %(message)s")
# --------------------------------------------------------------
# S3 Bucket preparation
# --------------------------------------------------------------


def get_folder_name(gaps: str) -> str:
    # return f"theta_calendar_{weeks}_weeks"
    return f"data/index/calendar/{gaps}_gap"


# --------------------------------------------------------------
# Prepare inputs for the multi-processors
# --------------------------------------------------------------


def thread_historical_queries(
    kwargs: dict,
    fb: str,
    gap: str,
    base_params: dict,
    strikes: list,
    func: callable,
    attr_name: str,
    njobs=8,
):
    if fb == "front":
        exp = kwargs["fexp"]
        attr_name = f"{base_params['right']}_{attr_name}_front"
    elif fb == "back":
        exp = kwargs["bexps"][gap - 1]
        attr_name = f"{base_params['right']}_{attr_name}_back_G{gap}"
    else:
        raise ValueError("Invalid front/back value")

    inputs = [
        {
            "symbol": kwargs["symbol"],
            "roots": kwargs["roots"],
            "exp": exp,
            "strike": strike,
            "base_params": base_params,
        }
        for strike in strikes
    ]

    dfs = Parallel(n_jobs=min(len(inputs), njobs), backend="threading", verbose=5)(
        delayed(func)(**kwargs) for kwargs in inputs
    )
    dfs = [df for df in dfs if df is not None]
    if len(dfs) == 0:
        return
    df = pd.concat(dfs, ignore_index=True)
    if attr_name == "greeks_back":
        df.drop(columns="underlying", inplace=True)

    if "right" in df.columns:
        df.drop(columns="right", inplace=True)

    return {attr_name: df}


def merge_historical_snapshot(dfss, df_key, cols=None):
    if cols is None:
        cols = ["symbol", "date", "ms_of_day", "strike_milli"]

    # Get matching DataFrames with their keys
    matching_items = [(k, df) for k, df in dfss.items() if df_key in k]

    if not matching_items:
        return None

    if len(matching_items) == 1:
        return matching_items[0][1]

    # For each DataFrame, rename non-key columns to include the key as prefix
    processed_dfs = []

    for key, df in matching_items:
        df_copy = df.copy()

        # Rename columns that are not in the merge keys
        rename_dict = {}
        for col in df_copy.columns:
            if col not in cols:
                rename_dict[col] = f"{col}_{key}"

        df_copy = df_copy.rename(columns=rename_dict)
        processed_dfs.append(df_copy)

    # Now merge all the processed DataFrames
    return reduce(lambda left, right: pd.merge(left, right, on=cols, how="inner"), processed_dfs)


def find_common_elements(lists):
    common = set(lists[0])
    for lst in lists[1:]:
        common = common.intersection(set(lst))
    return sorted(list(common))


def historical_snapshot(kwargs):
    f_dates = get_exp_trading_days(roots=kwargs["roots"], exp=kwargs["fexp"])
    b_dates = [get_exp_trading_days(roots=kwargs["roots"], exp=d) for d in kwargs["bexps"]]
    trade_dates = find_common_elements(b_dates + [f_dates])

    # print(len(trade_dates))
    if len(trade_dates) < 3:
        return kwargs["filepath"]

    # Strikes
    f_strikes = get_strikes_exp(roots=kwargs["roots"], exp=kwargs["fexp"])
    b_strikes = [get_strikes_exp(roots=kwargs["roots"], exp=d) for d in kwargs["bexps"]]
    strikes = find_common_elements(b_strikes + [f_strikes])

    # Params Defaults
    base_params = {
        "start_date": str(min(trade_dates)),
        "end_date": str(max(trade_dates)),
        "right": "C",
        "ivl": str(kwargs["ivl"]),
    }

    # Underlying
    und_df = get_greeks_historical(
        symbol=kwargs["symbol"],
        roots=kwargs["roots"],
        exp=kwargs["fexp"],
        strike=strikes[len(strikes) // 2],
        base_params=base_params,
    )
    if und_df is None:
        return kwargs["filepath"]

    und_df = und_df[["underlying"]]

    if len(und_df[und_df["underlying"] == 0]) > 0:
        log.info(
            f"Underlying Price contains zero: {kwargs}, Amount: {len(und_df[und_df['underlying'] == 0])}"
        )
        und_df = und_df[und_df["underlying"] != 0]
        if len(und_df) == 0:
            log.error(f"No data to process.")
            return kwargs["filepath"]

    # Find bounds for strikes
    min_und = und_df["underlying"].min()
    max_und = und_df["underlying"].max()
    del und_df

    start_strikes = [s for s in strikes if (s / 1000) > (min_und * 0.94)]
    sliced_strikes = [s for s in start_strikes if (s / 1000) < (max_und * 1.06)]
    if len(sliced_strikes) == 0:
        log.info(f"No strikes found for {kwargs['symbol']}")
        return kwargs["filepath"]

    # Greeks

    combinations = list(
        itertools.product(
            ["P", "C"],
            enumerate(["front", *["back"] * len(kwargs["bexps"])]),
            [
                (get_greeks_historical, "greeks"),
                (get_quotes_historical, "quotes"),
                # (get_oi_historical, "oi"),
            ],
        )
    )
    dfss = {}
    # Single loop with tqdm
    for r, (en, fb), (func, attr_name) in tqdm(
        combinations, desc="Processing queries", total=len(combinations)
    ):
        base_params["right"] = r
        dfs = thread_historical_queries(
            kwargs=kwargs,
            fb=fb,
            gap=en,
            base_params=base_params,
            strikes=sliced_strikes,
            func=func,
            attr_name=attr_name,
            njobs=-1,
        )

        dfss.update(dfs)

    if any([df is None for df in dfss.items()]):
        log.info(f"Snapshot data is missing for {kwargs['fexp']}, dropping the symbol")
        return kwargs["filepath"]

    greeks = merge_historical_snapshot(dfss, "greeks")
    quotes = merge_historical_snapshot(dfss, "quotes")

    cal = pd.merge(
        greeks, quotes, on=["ms_of_day", "date", "strike_milli", "symbol"], how="inner"
    ).sort_values(by=["strike_milli", "date", "ms_of_day"])

    def expiration_organiser(cal: pd.DataFrame):
        for fb in ["front", "back"]:
            if fb == "back":
                exps = [c for c in cal.columns if f"_{fb}_G" in c and "exp_" in c]
            else:
                exps = [c for c in cal.columns if f"_{fb}" in c and "exp_" in c]

            for gap in set([c.split("_")[-1] for c in exps]):
                gap_cols = [c for c in exps if c.endswith(gap)]
                if not cal[gap_cols].nunique(axis=1).eq(1).all():
                    log.error("expiration columns not equal")
                if len(gap_cols) > 1:
                    cal = cal.drop(columns=gap_cols[1:])
                    if fb == "back":
                        cal = cal.rename(columns={gap_cols[0]: f"exp_{fb}_{gap}"})
                    else:
                        cal = cal.rename(columns={gap_cols[0]: f"exp_{fb}"})

        underlying = [c for c in cal.columns if "underlying" in c]
        cal = cal.drop(columns=underlying[1:])
        cal = cal.rename(columns={underlying[0]: "underlying"})
        cal = cal.drop(columns="symbol")

        return cal

    cal = expiration_organiser(cal)

    if cal.empty:
        return kwargs["filepath"]

    if len(cal["date"].unique()) > 5:
        table = pa.Table.from_pandas(cal)
        bucket = S3Handler(bucket_name=os.getenv("S3_BUCKET_NAME"), region="us-east-2")
        bucket.upload_table(table, f"{kwargs['filepath']}")

        return None


# def historical_oi_snapshot(kwargs):
#     f_dates = get_exp_trading_days(roots=kwargs["roots"], exp=kwargs["fexp"])
#     b_dates = [get_exp_trading_days(roots=kwargs["roots"], exp=d) for d in kwargs["bexps"]]
#     trade_dates = find_common_elements(b_dates + [f_dates])

#     # print(len(trade_dates))
#     if len(trade_dates) < 3:
#         return kwargs["filepath"]

#     # Strikes

#     # Params Defaults
#     base_params = {
#         "start_date": str(min(trade_dates)),
#         "end_date": str(max(trade_dates)),
#         "right": "C",
#         "ivl": str(kwargs["ivl"]),
#     }

#     if len(sliced_strikes) == 0:
#         log.info(f"No strikes found for {kwargs['symbol']}")
#         return kwargs["filepath"]

#     # Greeks

#     combinations = list(
#         itertools.product(
#             ["P", "C"],
#             enumerate(["front", *["back"] * len(kwargs["bexps"])]),
#             [
#                 (get_oi_historical, "oi"),
#             ],
#         )
#     )
#     dfss = {}
#     # Single loop with tqdm
#     for r, (en, fb), (func, attr_name) in tqdm(
#         combinations, desc="Processing queries", total=len(combinations)
#     ):
#         base_params["right"] = r
#         dfs = thread_historical_queries(
#             kwargs=kwargs,
#             fb=fb,
#             gap=en,
#             base_params=base_params,
#             strikes=sliced_strikes,
#             func=func,
#             attr_name=attr_name,
#             njobs=-1,
#         )
#         dfss.update(dfs)
#     if any([df is None for df in dfss.items()]):
#         log.info(f"Snapshot data is missing for {kwargs['fexp']}, dropping the symbol")
#         return kwargs["filepath"]

#     greeks = merge_historical_snapshot(dfss, "greeks")
#     quotes = merge_historical_snapshot(dfss, "quotes")

#     cal = pd.merge(
#         greeks, quotes, on=["ms_of_day", "date", "strike_milli", "symbol"], how="inner"
#     ).sort_values(by=["strike_milli", "date", "ms_of_day"])

#     if cal.empty:
#         return kwargs["filepath"]

#     if len(cal["date"].unique()) > 5:
#         table = pa.Table.from_pandas(cal)
#         bucket = S3Handler(bucket_name=os.getenv("S3_BUCKET_NAME"), region="us-east-2")
#         bucket.upload_table(table, f"{kwargs['filepath']}")

#         return None


def expiration_loop(roots, exp):
    exps = get_exp_trading_days(roots=ticker, exp=exp)
    return {exp: exps}


def prepare_inputs(ticker, combinations, existing_files, gaps):
    inputs = []

    for en, comb in enumerate(combinations):
        kwargs = {
            "symbol": ticker,
            "fexp": comb[0],
            "bexps": comb[1:],
            "roots": [ticker],
            "filepath": f"{get_folder_name(gaps)}/{ticker}/{'_'.join(list([str(c) for c in comb]))}.parquet",
            "ivl": ivl,
        }
        if kwargs["filepath"] not in existing_files:
            if len(kwargs["bexps"]) == gaps:
                inputs.append(kwargs)

    return inputs


if __name__ == "__main__":
    # --------------------------------------------------------------
    # Input Parameters
    # --------------------------------------------------------------
    ivl = 900000  # 15 minutes
    n_calendars = 3
    min_trading_dates = 5
    max_date_diff = 90
    ticker = "SPY"

    n_exps = n_calendars + 1  # t include the Front expiration

    # --------------------------------------------------------------
    # Prepare Inputs
    # --------------------------------------------------------------
    expirations = get_expiry_dates(ticker)
    first_dates = pd.Timestamp("2016-01-01").strftime("%Y%m%d")  # Standard subscription
    expirations = [d for d in expirations if d > int(first_dates)]

    exps_list = Parallel(n_jobs=50, backend="threading", verbose=0)(
        delayed(expiration_loop)(roots=ticker, exp=exp) for exp in tqdm(expirations)
    )
    exp_map = {}
    for exp in exps_list:
        exp_map.update(exp)

    last_date = int((dt.datetime.now() - dt.timedelta(days=5)).strftime("%Y%m%d"))

    all_trade_dates = sorted(
        list(
            set(
                [
                    date
                    for e, ds in exp_map.items()
                    for date in ds
                    if date > int(first_dates) and date < last_date
                ]
            )
        )
    )
    expirations_on_dates = {}
    for date in all_trade_dates:
        for exp in exp_map.keys():
            if exp < last_date:
                if date in exp_map[exp]:
                    if date not in expirations_on_dates.keys():
                        expirations_on_dates[date] = [exp]
                    else:
                        expirations_on_dates[date].append(exp)

    valid_combs = []
    for date in expirations_on_dates:
        for combination in [
            tuple(sorted(expirations_on_dates[date])[i : i + n_exps])
            for i in range(len(expirations_on_dates[date]) - n_exps + 1)
        ]:
            common_trading_dates = find_common_elements(
                [exp_map[combination[i]] for i in range(n_exps)]
            )
            if len(common_trading_dates) >= min_trading_dates:
                valid_combs.append(combination)
    valid_combs = sorted(set(valid_combs))

    bucket = S3Handler(bucket_name=os.getenv("S3_BUCKET_NAME"), region="us-east-2")
    existing_files = bucket.list_files(f"{get_folder_name(n_calendars)}")
    inputs = prepare_inputs(
        ticker=ticker, combinations=valid_combs, existing_files=existing_files, gaps=n_calendars
    )

    failed_files_path = f"{get_folder_name(n_calendars)}/failed_files.parquet"
    if bucket.file_exists(failed_files_path):
        failed_df = bucket.read_dataframe(failed_files_path, format="parquet")
        failed_files = failed_df["filepath"].tolist()
    else:
        failed_files = []

    # --------------------------------------------------------------
    # Start the process
    # --------------------------------------------------------------
    random.shuffle(inputs)
    tf = len(inputs)
    inputs = [kwargs for kwargs in inputs if kwargs["filepath"] not in failed_files]

    log.info(f"Total Failed Files: {len(failed_files)}")
    log.info(f"Total Inputs: {len(inputs)}")
    log.info(f"Removed {tf - len(inputs)} failed files")

    for batch in batched(inputs, 5):
        cpus = 4
        if is_market_open(break_Script=False, bypass=True):
            cpus = 2
            log.info(f"Market is open, reducing the number of CPUs to {cpus}")
        for kwargs in batch:
            print(kwargs)
        failed_returns = Parallel(n_jobs=cpus, backend="multiprocessing", verbose=10)(
            delayed(historical_snapshot)(kwargs) for kwargs in batch
        )

        failed_files.extend([f for f in failed_returns if f is not None])
        # write the failed files to the S3 bucket
        failed_files_df = pd.DataFrame(failed_files, columns=["filepath"])
        table = pa.Table.from_pandas(failed_files_df)
        bucket.upload_table(table, failed_files_path)

    log.info("All Done")
    # empty df and upload to S3
    failed_files_df = pd.DataFrame(columns=["filepath"])
    table = pa.Table.from_pandas(failed_files_df)
    bucket.upload_table(table, failed_files_path)
