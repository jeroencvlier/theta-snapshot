import os
import sys
import datetime as dt
import logging as log
import pandas as pd
from joblib import Parallel, delayed
import pyarrow as pa
from tqdm import tqdm
import time
import random
import httpx  # install via pip install httpx
import csv
import pytz
import numpy as np
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
    time.sleep(2)
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
    strike_dfs = Parallel(n_jobs=10, backend="threading", verbose=0)(
        delayed(thread_strikes_greeks)(params=params, strike=exp) for exp in strikes
    )
    strike_dfs = [df for df in strike_dfs if df is not None]
    if len(strike_dfs) == len(strikes):
        return pd.concat(strike_dfs).reset_index(drop=True)


def historical_snapshot(exp_dict, ticker, ivl, rth, existing_files):
    assert len(exp_dict) == 1
    [(exp, trading_dates)] = exp_dict.items()
    last_date = int((dt.datetime.now() - dt.timedelta(days=1)).strftime("%Y%m%d"))
    config_prefix = get_config_prefix(ticker, ivl, rth)

    for date in trading_dates:
        # greek_filename = get_folder_name() + f"{ticker}/{exp}/greeks/{date}.parquet"
        # oi_filename = get_folder_name() + f"{ticker}/{exp}/open_interest/{date}.parquet"
        # quote_filename = get_folder_name() + f"{ticker}/{exp}/quotes/{date}.parquet"

        greek_filename = f"{config_prefix}{exp}/greeks/{date}.parquet"
        oi_filename = f"{config_prefix}{exp}/open_interest/{date}.parquet"
        quote_filename = f"{config_prefix}{exp}/quotes/{date}.parquet"
        greeks = greek_filename in existing_files
        ois = oi_filename in existing_files
        quotes = quote_filename in existing_files
        all_exist = all([greeks, ois, quotes])
        df_quote = None
        df_oi = None
        df_greeks = None

        if (date <= last_date) and (not all_exist):
            # pass
            try:
                params = {
                    "start_date": str(date),
                    "end_date": str(date),
                    "use_csv": "true",
                    "root": ticker,
                    "exp": exp,
                    "ivl": str(ivl),
                    "rth": rth,
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
                return [greek_filename, oi_filename, quote_filename]


dtypes_quotes = {
    "expiration": "int32",  # YYYYMMDD dates fit in int32
    "strike": "int32",  # Strike prices don't need float64 precision
    "right": "category",  # C/P categorical is very efficient
    "ms_of_day": "int32",  # Milliseconds in a day fit in int32
    "bid_size": "int32",  # Trade sizes fit in int32
    "bid": "float32",  # Prices don't need float64 precision
    "ask_size": "int32",  # Trade sizes fit in int32
    "ask": "float32",  # Prices don't need float64 precision
    "date": "int32",  # YYYYMMDD dates fit in int32
}

dtypes_greeks = {
    "ms_of_day": "int32",  # Milliseconds in a day fit in int32
    "delta": "float32",  # Options Greeks don't need float64 precision
    "theta": "float32",  # Options Greeks
    "vega": "float32",  # Options Greeks
    "rho": "float32",  # Options Greeks
    "epsilon": "float32",  # Options Greeks
    "lambda": "float32",  # Options Greeks (note: lambda is a reserved word)
    "implied_vol": "float32",  # Implied volatility
    "iv_error": "float32",  # IV error
    "underlying_price": "float32",  # Stock prices don't need float64 precision
    "date": "int32",  # YYYYMMDD dates fit in int32
    "strike": "int32",  # Strike prices don't need float64 precision
    "right": "category",  # C/P categorical is very memory efficient
}


def optimize_dtypes(df):
    """
    Automatically convert columns to appropriate data types based on content.
    Works with dynamic DataFrames where columns and types may vary.
    """
    result = df.copy()

    for col in result.select_dtypes(include=["object"]).columns:
        if col in ["root"]:
            continue
        try:
            numeric_series = pd.to_numeric(result[col])
            if (numeric_series == numeric_series.astype("int64")).all():
                result[col] = numeric_series.astype("int64")
            else:
                result[col] = numeric_series.astype("float64")
        except:
            if set(result[col].dropna().unique()).issubset({"True", "False", True, False, 0, 1}):
                result[col] = result[col].astype("boolean")
            elif result[col].str.match(r"^\d{4}-\d{2}-\d{2}").all():
                try:
                    result[col] = pd.to_datetime(result[col])
                except:
                    pass
    for col in result.select_dtypes(include=["float"]).columns:
        if result[col].notna().all() and (result[col] == result[col].astype("int64")).all():
            result[col] = result[col].astype("int64")
    return result


def undppctdiff(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        undPricePctDiff=(((df["strike"] / 1000) - df["underlying_price"]) / df["underlying_price"]).astype(
            "Float32"
        )
    )


def calendar_calculations(df: pd.DataFrame, gaps: int) -> pd.DataFrame:
    # Dictionary to store all new columns
    ncs = {}

    for r in ["C", "P"]:
        # Fixed parentheses in mark price calculation
        ncs[f"mark_{r}"] = ((df[f"ask_{r}"] - df[f"bid_{r}"]) / 2) + df[f"bid_{r}"]

        for g in range(1, gaps + 1):
            ncs[f"mark_{r}_G{g}"] = ((df[f"ask_{r}_G{g}"] - df[f"bid_{r}_G{g}"]) / 2) + df[f"bid_{r}_G{g}"]
            ncs[f"calCost_{r}_G{g}"] = ncs[f"mark_{r}_G{g}"] - ncs[f"mark_{r}"]
            ncs[f"calCostPct_{r}_G{g}"] = (ncs[f"calCost_{r}_G{g}"] / df["underlying_price"]) * 100
            ncs[f"calGapPct_{r}_G{g}"] = ncs[f"calCost_{r}_G{g}"] - ncs[f"mark_{r}_G{g}"]
            ncs[f"ask_cal_{r}_G{g}"] = df[f"ask_{r}_G{g}"] - df[f"bid_{r}"]
            ncs[f"bid_cal_{r}_G{g}"] = df[f"bid_{r}_G{g}"] - df[f"ask_{r}"]
            ncs[f"spread_cal_{r}_G{g}"] = ncs[f"ask_cal_{r}_G{g}"] - ncs[f"bid_cal_{r}_G{g}"]
            ncs[f"mark_cal_{r}_G{g}"] = (ncs[f"spread_cal_{r}_G{g}"] / 2) + ncs[f"bid_cal_{r}_G{g}"]
            ask_cal = ncs[f"ask_cal_{r}_G{g}"]
            spread_cal = ncs[f"spread_cal_{r}_G{g}"]
            ncs[f"spreadPct_cal_{r}_G{g}"] = np.where(ask_cal == 0, np.nan, spread_cal / ask_cal)

    keep_columns = [
        col for col in df.columns if not any(ba in col for ba in ["bid_P", "ask_P", "bid_C", "ask_C"])
    ]

    # Create the result DataFrame with a single copy operation
    result_df = pd.DataFrame({**{col: df[col] for col in keep_columns}, **ncs}, index=df.index)

    return result_df


def iv_pct_diff(df: pd.DataFrame, gaps) -> pd.DataFrame:
    result = df.copy()
    for r in ["C", "P"]:
        for g in range(1, gaps + 1):
            mask1 = df[f"implied_vol_{r}"] == 0
            mask2 = df[f"implied_vol_{r}_G{g}"] == 0
            result[f"iv_pct_diff_{r}_G{g}"] = np.nan
            result.loc[mask1 & mask2, f"iv_pct_diff_{r}_G{g}"] = 0.0
            result.loc[mask1 & ~mask2, f"iv_pct_diff_{r}_G{g}"] = 1.0
            result.loc[~mask1, f"iv_pct_diff_{r}_G{g}"] = (
                df.loc[~mask1, f"implied_vol_{r}"] - df.loc[~mask1, f"implied_vol_{r}_G{g}"]
            ) / df.loc[~mask1, f"implied_vol_{r}"]
            result[f"iv_pct_diff_{r}_G{g}"] = result[f"iv_pct_diff_{r}_G{g}"].astype("float32")
    return result


def days_before_calcs(
    df: pd.DataFrame,
    end_date_col: str,
    target: str,
    trading_days: list[pd.Timestamp],
    from_date_col: str = None,
    suffix: str = None,
):
    target_columns = {
        "earnings": ("bdte", "dte"),
        "fexp": ("bdtfexp", "dtfexp"),
        "bexp": ("bdtbexp", "dtbexp"),
        "dit": ("bdit", "dit"),
    }
    bdte_col, dte_col = target_columns.get(target)
    start_dates = pd.to_datetime(df[from_date_col], format="%Y%m%d")
    end_dates = pd.to_datetime(df[end_date_col], format="%Y%m%d")
    start_indices = np.searchsorted(trading_days, start_dates.values)
    end_indices = np.searchsorted(trading_days, end_dates.values)
    df[bdte_col] = end_indices - start_indices
    df[dte_col] = (end_dates - start_dates).dt.days
    if suffix is not None:
        df = df.rename(columns={bdte_col: f"{bdte_col}_{suffix}", dte_col: f"{dte_col}_{suffix}"})

    return df


def historical_snapshot_fix(sliced_exp_list, ticker, ivl):
    dfs = []
    furthest_date = 20251016
    for exp_dict in tqdm(sliced_exp_list):
        [(exp, trading_dates)] = exp_dict.items()
        last_date = int((dt.datetime.now() - dt.timedelta(days=0)).strftime("%Y%m%d"))

        trading_dates = [td for td in trading_dates if td >= furthest_date]

        for date in trading_dates:
            if date <= last_date:
                params = {
                    "start_date": str(date),
                    "end_date": str(date),
                    "use_csv": "true",
                    "root": ticker,
                    "exp": exp,
                    "ivl": str(ivl),
                }

                df_quotes = get_bulk_quote_historical(params)

                strikes = list(set(df_quotes["strike"].to_list()))
                df_greeks = get_bulk_greeks_historical(params, strikes)
                df_quotes = df_quotes.drop(
                    columns=["root", "bid_condition", "bid_exchange", "ask_exchange", "ask_condition"]
                ).astype(dtypes_quotes)

                df_greeks = df_greeks.drop(
                    columns=["bid", "ask", "ms_of_day2", "error_type", "error_msg"], errors="ignore"
                )
                df_greeks = df_greeks.dropna(subset=["ms_of_day"]).astype(dtypes_greeks)

                df = pd.merge(df_greeks, df_quotes, on=["strike", "right", "date", "ms_of_day"])

                assert len(df["date"].unique()) == 1, "duplicated dates!!"
                ms_of_day = df["ms_of_day"].unique()
                ms_of_day_exclude = [ms_of_day.min(), ms_of_day.max()]

                df_puts = df[df["right"] == "P"].drop(columns="right")
                df_calls = df[df["right"] == "C"].drop(columns="right")

                df = pd.merge(
                    df_puts,
                    df_calls,
                    on=["strike", "date", "ms_of_day", "underlying_price", "expiration"],
                    suffixes=["_P", "_C"],
                ).reset_index(drop=True)

                df = df[~df["ms_of_day"].isin(ms_of_day_exclude)]

                dfs.append(df)

                df = pd.concat(dfs)
                df = df.reset_index(drop=True)
                df.to_parquet("recovery_snapshots.parquet")

    df = pd.read_parquet("recovery_snapshots.parquet")

    from collections import namedtuple

    expirations_all = list(df["expiration"].unique())
    max_cal_gap_days = 45

    Calendar = namedtuple("Calendar", ["fexp", "bexp"])
    calendars = []
    expirations = []
    for fexp in tqdm(expirations_all):
        bexps = expirations_all[expirations_all.index(fexp) + 1 :]
        for bexp in bexps:
            count_cal_gap_days = pd.date_range(
                start=pd.to_datetime(fexp, format="%Y%m%d"), end=pd.to_datetime(bexp, format="%Y%m%d")
            )
            count_days_to_front = pd.date_range(
                start=pd.to_datetime(str(furthest_date), format="%Y%m%d"),
                end=pd.to_datetime(fexp, format="%Y%m%d"),
            )
            if (len(count_cal_gap_days) <= max_cal_gap_days) & (len(count_days_to_front) <= max_trading_days):
                calendars.append(Calendar(fexp=fexp, bexp=bexp))
                expirations.extend([fexp, bexp])
    cals = []
    for calendar in calendars:
        front_cal = df[df["expiration"].astype(int) == calendar.fexp]
        back_cal = df[df["expiration"].astype(int) == calendar.bexp]
        cal = front_cal.merge(
            back_cal,
            how="inner",
            on=["date", "strike", "ms_of_day"],
            suffixes=["", "_G1"],
        )
        cals.append(cal)
    cals_df = pd.concat(cals)

    cals_df["strike"] = cals_df["strike"].astype("Int64")
    cals_df["underlying_price"] = cals_df["underlying_price"].astype("Float32")

    cals_df["underlying_price"] = cals_df["underlying_price"].astype("Float32")
    cals_df = optimize_dtypes(cals_df)
    cals_df = undppctdiff(cals_df)

    cals_df = calendar_calculations(cals_df, gaps=1)
    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar("NYSE")
    trading_days = nyse.schedule(
        start_date=dt.datetime.now().date(), end_date=pd.to_datetime(max(expirations_all), format="%Y%m%d")
    ).index

    cals_df = days_before_calcs(
        df=cals_df,
        end_date_col="expiration",
        target="fexp",
        trading_days=trading_days,
        from_date_col="date",
    )
    for g in range(1, 1 + 1):
        cals_df = days_before_calcs(
            df=cals_df,
            end_date_col=f"expiration_G{g}",
            target="bexp",
            trading_days=trading_days,
            from_date_col="date",
            suffix=f"G{g}",
        )
    cals_df = iv_pct_diff(cals_df, gaps=1)
    cals_df = cals_df.reset_index(drop=True)

    cals_df.to_parquet("fully_calculated_recovery.parquet")


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


def get_config_prefix(ticker: str, ivl: int, rth: str) -> str:
    return f"{get_folder_name()}{ticker}/ivl={ivl}_rth={rth}/"


if __name__ == "__main__":
    start_time = time.time()
    # --------------------------------------------------------------
    # Input Parameters
    # --------------------------------------------------------------
    # ivl = 900000  # 15 minutes

    ticker_list = []
    for t in ["SPY", "SPXW", "SPX", "XSP", "QQQ", "IWM", "XLE", "GLD", "DBO"]:
        ticker_list.append({"ticker": t, "ivl": 900000, "max_trading_days": 60, "rth": "true"})

    for t in ["VXZ", "SVXY", "VIXW", "VIX"]:
        ticker_list.append({"ticker": t, "ivl": 900000, "max_trading_days": 180, "rth": "true"})

    for t in ["SPX", "SPXW"]:
        ticker_list.append({"ticker": t, "ivl": 900000, "max_trading_days": 60, "rth": "false"})

    for t in ["SPY"]:
        ticker_list.append({"ticker": t, "ivl": 300000, "max_trading_days": 1, "rth": "true"})

    # --------------------------------------------------------------
    # Prepare Inputs
    # --------------------------------------------------------------
    for ticker_obj in ticker_list:
        # --------------------------------------------------------------
        # Create Failed Files
        # --------------------------------------------------------------
        ticker = ticker_obj["ticker"]
        ivl = ticker_obj["ivl"]
        max_trading_days = ticker_obj["max_trading_days"]
        rth = ticker_obj["rth"]

        bucket = S3Handler(bucket_name=os.getenv("S3_BUCKET_NAME"), region="us-east-2")
        # existing_files = bucket.list_files(get_folder_name() + ticker)
        existing_files = bucket.list_files(get_config_prefix(ticker, ivl, rth))

        config_key = f"{ticker}_ivl{ivl}_rth{rth}"
        failed_files_path = f"{get_folder_name()}.memory/failed_files_{config_key}.parquet"

        # failed_files_path = f"{get_folder_name()}.memory/failed_files_{ticker}.parquet"
        if bucket.file_exists(failed_files_path):
            failed_df = bucket.read_dataframe(failed_files_path, format="parquet")
            failed_files = failed_df["filepath"].tolist()
            log.info(f"Found {len(failed_files)} failed files for {ticker}")
            existing_files.extend(failed_files)
        else:
            failed_files = []

        # --------------------------------------------------------------
        # prepare inputs
        # --------------------------------------------------------------
        expirations = get_expiry_dates(ticker)
        # first_dates = pd.Timestamp("2025-10-15").strftime("%Y%m%d")
        first_dates = pd.Timestamp("2021-01-01").strftime("%Y%m%d")  # Standard subscription
        expirations = [d for d in expirations if d > int(first_dates)]
        exps_list_filename = f"{get_folder_name()}.memory/exps_list_{ticker}.json"
        if bucket.file_exists(exps_list_filename):
            exps_list = bucket.load_json_from_s3(exps_list_filename)
        else:
            exps_list = Parallel(n_jobs=-1, backend="multiprocessing", verbose=0)(
                delayed(expiration_loop)(ticker=ticker, exp=exp) for exp in tqdm(expirations)
            )
            bucket.save_json_to_s3(exps_list, exps_list_filename)

        sliced_exp_list = []
        for d in exps_list:
            [(k, v)] = d.items()
            if k in v:
                nv = sorted(v)[-max_trading_days:]
                sliced_exp_list.append({k: nv})

        for d in sliced_exp_list:
            [(k, v)] = d.items()
            if k not in v:
                print(f"{k} not in {v}")

        sum([len(v) for d in exps_list for k, v in d.items()])
        sum([len(v) for d in sliced_exp_list for k, v in d.items()])
        # --------------------------------------------------------------
        # Start the process
        # --------------------------------------------------------------
        random.shuffle(sliced_exp_list)

        n_jobs = 1
        for batch in batched(sliced_exp_list, n_jobs):
            is_market_open(break_script=False)

            failed_returns = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=10)(
                delayed(historical_snapshot)(
                    exp_dict=exp_dict, ticker=ticker, ivl=ivl, rth=rth, existing_files=existing_files
                )
                for exp_dict in batch
            )

            failed_returns = [f for f in failed_returns if f is not None]
            if len(failed_returns) > 0:
                for fl in failed_returns:
                    failed_files.extend(fl)
                failed_files_df = pd.DataFrame(failed_files, columns=["filepath"])
                table = pa.Table.from_pandas(failed_files_df)
                bucket.upload_table(table, failed_files_path)

            if (time.time() - start_time) > (60 * 60 * 12) - (60 * 60):
                log.info("Shutting Down before failing")
                sys.exit(0)

        log.info(f"Completed {ticker}")
    log.info("All Done")
    # empty df and upload to S3
    for ticker_obj in ticker_list:
        ticker = ticker_obj["ticker"]
        ivl = ticker_obj["ivl"]
        rth = ticker_obj["rth"]
        config_key = f"{ticker}_ivl{ivl}_rth{rth}"
        failed_files_path = f"{get_folder_name()}.memory/failed_files_{config_key}.parquet"
        # failed_files_path = f"{get_folder_name()}.memory/failed_files_{ticker}.parquet"
        failed_files_df = pd.DataFrame(columns=["filepath"])
        table = pa.Table.from_pandas(failed_files_df)
        bucket.upload_table(table, failed_files_path)
        exps_list_filename = f"{get_folder_name()}.memory/exps_list_{ticker}.json"
        bucket.delete_file(exps_list_filename)
