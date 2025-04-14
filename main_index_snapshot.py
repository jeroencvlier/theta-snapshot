import os
import datetime as dt
import logging as log
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import httpx  # install via pip install httpx
import csv
import pytz
import pandas_market_calendars as mcal
from collections import namedtuple
import numpy as np
from theta_snapshot import (
    get_expiry_dates,
    get_exp_trading_days,
    append_to_table,
    is_market_open,
    time_checker_ny,
)

log.basicConfig(level=log.INFO, format="%(asctime)s - %(message)s")
# --------------------------------------------------------------
# S3 Bucket preparation
# -------------------------------------------------------------


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


def optimize_dtypes(df):
    """
    Automatically convert columns to appropriate data types based on content.
    Works with dynamic DataFrames where columns and types may vary.
    """
    result = df.copy()
    for col in result.select_dtypes(include=["object"]).columns:
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


def calculate_fb_spread(df: pd.DataFrame, fb: str) -> pd.DataFrame:
    df[f"spread_{fb}"] = df[f"ask_{fb}"] - df[f"bid_{fb}"]
    df[f"spreadPct_{fb}"] = (df[f"spread_{fb}"] / df[f"mark_{fb}"]).round(2)
    return df


def undppctdiff(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        undPricePctDiff=(((df["strike"] / 1000) - df["underlying_price"]) / df["underlying_price"]).astype(
            "Float32"
        )
    )


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


def get_bulk_quote_historical(params):
    url = os.getenv("BASE_URL") + "/bulk_snapshot/option/quote"
    df = bulk_csv_request(url=url, params=params)
    df = df.drop(columns=["ms_of_day", "bid_condition", "bid_exchange", "ask_exchange", "ask_condition"])
    return "quotes", params["exp"], df


def get_bulk_greeks_historical(params):
    url = os.getenv("BASE_URL") + "/bulk_snapshot/option/greeks"
    df = bulk_csv_request(url=url, params=params)
    df = df.drop(columns=["ms_of_day2", "bid", "ask"])
    return "greeks", params["exp"], df


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


def exp_merge(greek, quote):
    cal = greek.merge(quote, how="inner", on=["root", "strike", "right", "date", "expiration"])

    cal_puts = cal[cal["right"] == "P"].drop(columns="right")
    cal_calls = cal[cal["right"] == "C"].drop(columns="right")

    df = pd.merge(
        cal_puts,
        cal_calls,
        on=["strike", "date", "underlying_price", "expiration", "root"],
        suffixes=["_P", "_C"],
    ).reset_index(drop=True)

    df["ms_of_day"] = ((df["ms_of_day_P"].astype(int) + df["ms_of_day_C"].astype(int)) / 2).astype(int)
    df = df.drop(columns=["ms_of_day_P", "ms_of_day_C"])
    return df


if __name__ == "__main__":
    is_market_open(break_Script=os.getenv("BREAK_SCRIPT") == "True")
    time_checker_ny(break_Script=os.getenv("BREAK_SCRIPT") == "True")
    # --------------------------------------------------------------
    # Input Parameters
    # --------------------------------------------------------------
    nyse = mcal.get_calendar("NYSE")
    tickers = ["SPY"]
    max_trading_days = 45
    max_expirations_days_away = 60
    max_cal_gap_days = 10
    today = dt.datetime.now().date().strftime("%Y%m%d")
    months_away = (dt.datetime.now().date() + dt.timedelta(days=max_expirations_days_away)).strftime("%Y%m%d")
    cpus = max(os.cpu_count(), 20)
    log.info(f"Available CPUs: {os.cpu_count()}, defaulting to {cpus} for parallelism.")

    # --------------------------------------------------------------
    # Prepare Inputs
    # --------------------------------------------------------------
    for ticker in tickers:
        expirations = get_expiry_dates(ticker)

        expirations = [d for d in expirations if d >= int(today)]
        expirations = [d for d in expirations if d <= int(months_away)]

        all_dates = pd.date_range(
            start=dt.datetime.now().date(), end=pd.to_datetime(max(expirations), format="%Y%m%d")
        )

        trading_days = nyse.schedule(start_date=min(all_dates), end_date=max(all_dates)).index

        params = {
            "root": ticker,
            "use_csv": "true",
        }
        inputs = []
        for exp in expirations:
            params_ = params.copy()
            params_["exp"] = exp
            inputs.append((get_bulk_greeks_historical, params_))
            inputs.append((get_bulk_quote_historical, params_))

        returns = Parallel(n_jobs=cpus, backend="threading")(delayed(func)(*args) for func, *args in inputs)
        greeks = [tup for tup in returns if tup[0] == "greeks"]
        quotes = [tup for tup in returns if tup[0] == "quotes"]
        del returns

        underlying_prices = []
        for tup in greeks:
            if "underlying_price" in tup[2].columns:
                underlying_prices.append(tup[2]["underlying_price"].iloc[0])

        underlying_price = np.round(np.mean([float(up) for up in underlying_prices]), 2)
        diffs = [abs(underlying_price - float(up)) for up in underlying_prices]
        assert max(diffs) / underlying_price <= 0.0005, (
            "Price deviation is more than 0.0005 percentage off the mean"
        )

        Calendar = namedtuple("Calendar", ["fexp", "bexp"])
        calendars = []
        for fexp in tqdm(expirations):
            bexps = expirations[expirations.index(fexp) + 1 :]
            for bexp in bexps:
                count_distance = pd.date_range(
                    start=pd.to_datetime(fexp, format="%Y%m%d"), end=pd.to_datetime(bexp, format="%Y%m%d")
                )
                if len(count_distance) <= max_cal_gap_days:
                    calendars.append(Calendar(fexp=fexp, bexp=bexp))

        cals = []
        for calendar in calendars:
            front_greeks = [g[2] for g in greeks if g[1] == calendar.fexp]
            front_quotes = [q[2] for q in quotes if q[1] == calendar.fexp]
            back_greeks = [g[2] for g in greeks if g[1] == calendar.bexp]
            back_quotes = [q[2] for q in quotes if q[1] == calendar.bexp]
            if any(
                [len(front_greeks) != 1, len(front_quotes) != 1, len(back_greeks) != 1, len(back_quotes) != 1]
            ):
                print(f"Error in the amount of calendar for {calendar}")
                continue
            front_greek = front_greeks[0]
            front_quote = front_quotes[0]
            back_greek = back_greeks[0]
            back_quote = back_quotes[0]

            front_cal = exp_merge(front_greek, front_quote)
            back_cal = exp_merge(back_greek, back_quote)

            cal = front_cal.merge(
                back_cal,
                how="inner",
                on=["root", "date", "strike"],
                suffixes=["", "_G1"],
            )
            cal["ms_of_day"] = ((cal["ms_of_day"].astype(int) + cal["ms_of_day_G1"].astype(int)) / 2).astype(
                int
            )
            cal = cal.drop(columns=["ms_of_day_G1"])

            cals.append(cal)

        cals_df = pd.concat(cals)
        for c in ["error_msg_P", "error_msg_C", "error_type_C", "error_type_P"]:
            if c in cals_df.columns:
                cals_df = cals_df.drop(columns=c)
        cals_df["strike"] = cals_df["strike"].astype("Int64")
        cals_df["underlying_price"] = cals_df["underlying_price"].astype("Float32")
        cals_df = optimize_dtypes(cals_df)
        cals_df = undppctdiff(cals_df)

        cals_df = calendar_calculations(cals_df, gaps=1)
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
        # write to db
        timestamp_update = int(dt.datetime.now().timestamp())
        cals_df = cals_df.assign(lastUpdated=timestamp_update)
        cals_df = cals_df.reset_index(drop=True)
        append_to_table(
            cals_df,
            "IndexSnapshot",
            indexes=["symbol", "lastUpdated", "strike", "expiration", "expiration_G1"],
        )

        log.info("All Done")
