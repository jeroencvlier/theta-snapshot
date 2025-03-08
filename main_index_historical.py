import os
import sys
import datetime as dt
import logging as log
import pandas as pd
from joblib import Parallel, delayed
import pyarrow as pa
from tqdm import tqdm
import random
import httpx  # install via pip install httpx
import csv
import pytz

from theta_snapshot import (
    S3Handler,
    get_expiry_dates,
    get_exp_trading_days,
    batched,
    is_market_open,
)

log.basicConfig(level=log.INFO, format="%(asctime)s - %(message)s")
# --------------------------------------------------------------
# S3 Bucket preparation
# -------------------------------------------------------------


def get_folder_name() -> str:
    # return f"theta_calendar_{weeks}_weeks"
    return "data/raw/index/"


# --------------------------------------------------------------
# Prepare inputs for the multi-processors
# --------------------------------------------------------------


def bulk_csv_request(url, params):
    dfs = []
    while url is not None:
        response = httpx.get(url, params=params, timeout=60)  # make the request
        response.raise_for_status()  # make sure the request worked
        csv_reader = csv.reader(response.text.split("\n"))
        data = list(csv_reader)
        header = data[0]
        rows = [r for r in data[1:] if len(r) > 0]
        df = pd.DataFrame(rows, columns=header)
        dfs.append(df)

        if "Next-Page" in response.headers and response.headers["Next-Page"] != "null":
            url = response.headers["Next-Page"]
            params = None
            print("Requesting Next Page, ", url)
        else:
            url = None

    if len(dfs) > 0:
        return pd.concat(dfs).reset_index(drop=True)


def get_bulk_oi_historical(params):
    url = os.getenv("BASE_URL") + "/bulk_hist/option/open_interest"
    df = bulk_csv_request(url=url, params=params)
    return df


def get_bulk_quote_historical(params):
    url = os.getenv("BASE_URL") + "/bulk_hist/option/quote"
    df = bulk_csv_request(url=url, params=params)
    return df


def thread_strikes_greeks(params, strike):
    url = os.getenv("BASE_URL") + "/hist/option/greeks"
    params_copy = params.copy()
    dfs = []
    params_copy["strike"] = strike
    for right in ["P", "C"]:
        params_copy["right"] = right
        df = bulk_csv_request(url=url, params=params_copy)
        df = df.assign(strike=strike, right=right)
        if df is not None:
            dfs.append(df)
    if len(dfs) == 2:
        return pd.concat(dfs).reset_index(drop=True)


def get_bulk_greeks_historical(params, strikes):
    strike_dfs = Parallel(n_jobs=20, backend="threading", verbose=0)(
        delayed(thread_strikes_greeks)(params=params, strike=exp) for exp in strikes
    )
    strike_dfs = [df for df in strike_dfs if df is not None]
    if len(strike_dfs) == len(strikes):
        return pd.concat(strike_dfs).reset_index(drop=True)


def historical_snapshot(exp_dict, ticker, ivl, existing_files):
    assert len(exp_dict) == 1
    [(exp, trading_dates)] = exp_dict.items()
    last_date = int((dt.datetime.now() - dt.timedelta(days=5)).strftime("%Y%m%d"))

    for date in trading_dates:
        greek_filename = get_folder_name() + f"{ticker}/{exp}/greeks/{date}.parquet"
        oi_filename = get_folder_name() + f"{ticker}/{exp}/open_interest/{date}.parquet"
        quote_filename = get_folder_name() + f"{ticker}/{exp}/quotes/{date}.parquet"
        greeks = greek_filename in existing_files
        ois = oi_filename in existing_files
        quotes = quote_filename in existing_files
        all_exist = all([greeks, ois, quotes])
        df_quote = None
        df_oi = None
        df_greeks = None

        if (date <= last_date) and (not all_exist):
            try:
                params = {
                    "start_date": str(date),
                    "end_date": str(date),
                    "use_csv": "true",
                    "root": ticker,
                    "exp": exp,
                    "ivl": str(ivl),
                }
                try:
                    df_oi = get_bulk_oi_historical(params)
                except Exception as err:
                    df_oi = None

                df_quote = get_bulk_quote_historical(params)

                if not greeks:
                    strikes = list(set(df_quote["strike"].to_list()))
                    df_greeks = get_bulk_greeks_historical(params, strikes)

                bucket = S3Handler(bucket_name=os.getenv("S3_BUCKET_NAME"), region="us-east-2")
                if df_quote is not None:
                    quote_table = pa.Table.from_pandas(df_quote)
                    bucket.upload_table(quote_table, quote_filename)

                if df_greeks is not None:
                    greeks_table = pa.Table.from_pandas(df_greeks)
                    bucket.upload_table(greeks_table, greek_filename)

                if df_oi is not None:
                    oi_table = pa.Table.from_pandas(df_oi)
                    bucket.upload_table(oi_table, oi_filename)

            except Exception as error:
                log.warning("Failure at exp %s and date %s, error: %s", exp, date, error)


def expiration_loop(ticker, exp):
    trading_days = get_exp_trading_days(roots=ticker, exp=exp)
    return {exp: trading_days}


def true_between_time_ny(start_hour=7, end_hour=10):
    new_york_tz = pytz.timezone("America/New_York")
    current_time_ny = dt.datetime.now(new_york_tz)
    start_time = current_time_ny.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    end_time = current_time_ny.replace(hour=end_hour, minute=0, second=0, microsecond=0)
    # Check if current day is a weekday (Monday=0, Friday=4)
    is_weekday = current_time_ny.weekday() <= 4

    # Return True if it's a weekday AND between the specified hours
    if is_weekday and (current_time_ny > start_time) and (current_time_ny < end_time):
        return True
    else:
        return False


if __name__ == "__main__":
    # --------------------------------------------------------------
    # Input Parameters
    # --------------------------------------------------------------
    ivl = 900000  # 15 minutes
    ticker = "SPY"
    max_trading_days = 45

    # --------------------------------------------------------------
    # Prepare Inputs
    # --------------------------------------------------------------
    expirations = get_expiry_dates(ticker)
    first_dates = pd.Timestamp("2018-01-01").strftime("%Y%m%d")  # Standard subscription
    expirations = [d for d in expirations if d > int(first_dates)]

    exps_list = Parallel(n_jobs=-1, backend="multiprocessing", verbose=0)(
        delayed(expiration_loop)(ticker=ticker, exp=exp) for exp in tqdm(expirations)
    )

    sliced_exp_list = []
    for d in exps_list:
        [(k, v)] = d.items()
        # if max(v) == k:
        nv = sorted(v)[-max_trading_days:]
        sliced_exp_list.append({k: nv})
        # else:
        # break

    sum([len(v) for d in exps_list for k, v in d.items()])
    sum([len(v) for d in sliced_exp_list for k, v in d.items()])

    # --------------------------------------------------------------
    # Start the process
    # --------------------------------------------------------------
    random.shuffle(exps_list)
    random.shuffle(sliced_exp_list)

    for exp_dict in exps_list:
        pass

    bucket = S3Handler(bucket_name=os.getenv("S3_BUCKET_NAME"), region="us-east-2")
    existing_files = bucket.list_files(get_folder_name())

    for batch in batched(sliced_exp_list, 20):
        cpus = -1
        if is_market_open(break_Script=False) or true_between_time_ny():
            cpus = 1
            log.info(f"Reducing the number of CPUs to {cpus}")

        _ = Parallel(n_jobs=cpus, backend="multiprocessing", verbose=0)(
            delayed(historical_snapshot)(
                exp_dict=exp_dict, ticker=ticker, ivl=ivl, existing_files=existing_files
            )
            for exp_dict in tqdm(batch)
        )

    log.info("All Done")
