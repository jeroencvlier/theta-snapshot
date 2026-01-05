import os
import sys
import logging as log
import pandas as pd
from joblib import Parallel, delayed
import pyarrow as pa

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
    batched,
    is_market_open,
)

log.basicConfig(level=log.INFO, format="%(asctime)s - %(message)s")


def get_quarter(date):
    date = pd.Timestamp(date)
    month = date.month
    year = date.year
    fiscal_quart = (month - 1) // 3 + 1
    return f"{year}Q{fiscal_quart}"


# --------------------------------------------------------------
# S3 Bucket preparation
# --------------------------------------------------------------


def get_folder_name(weeks, right) -> str:
    # return f"theta_calendar_{weeks}_weeks"
    return f"data/raw/calendar/calendar_{right}_{weeks}_weeks"


# --------------------------------------------------------------
# Get priority symbols
# --------------------------------------------------------------


def get_priority_symbols():
    # we can keep this as the old table for now
    grade_query = '''SELECT symbol, under_avg_trade_class, weeks FROM public."StockGrades"'''
    grades = read_from_db(query=grade_query)
    grades = (
        grades.groupby("symbol")["under_avg_trade_class"].mean().sort_values(ascending=False).reset_index()
    )

    # sorted_symbols = ["AAPL","AMZN","AZO","BABA","BIB","BIDU","CF","CMG","CRM","DE","FB","FDX","FFIV","GMCR","GME","GOOG","IBM","JOY","LNKD","LULU","MA","NAG","NFLX","PCLN","SCTY","SINA","TIF","TSLA","ULTA","WFM"]
    sorted_symbols = []
    for idx, row in grades.iterrows():
        if row["symbol"] in symbols:
            if row["symbol"] not in sorted_symbols:
                sorted_symbols.append(row["symbol"])

    for symbol in symbols:
        if symbol not in sorted_symbols:
            sorted_symbols.append(symbol)

    return sorted_symbols


# --------------------------------------------------------------
# Prepare inputs for the multi-processors
# --------------------------------------------------------------
def prepare_inputs():
    inputs = []

    for symbol in get_priority_symbols():
        cdf = sc_df[sc_df["symbol"] == symbol]
        roots = list(
            set([symbol] + cdf["symbol"].unique().tolist() + cdf["symbol_changes"].unique().tolist())
        )
        edf = earnings[earnings["symbol"].isin(roots)]

        if edf["symbol"].nunique() > 1:
            log.info(f"Multiple symbols found for {roots}")

        for idx, row in edf.iterrows():
            fiscal_quart = get_quarter(row["fiscalQuarterEnding"])
            if (pd.Timestamp.now() - row["reportDate"]).days < 14:
                log.info(f"Skipping {symbol} {fiscal_quart}: Date is less than 2 weeks: {row['reportDate']}")
            else:
                for weeks in week_list:
                    kwargs = {
                        "symbol": symbol,
                        "rdate": row["reportDate"].strftime("%Y-%m-%d"),
                        "weeks": weeks,
                        "right": right,
                        "roots": roots,
                        "filepath": f"{get_folder_name(weeks, right)}/{symbol}/{symbol}_{fiscal_quart}.parquet",
                        "fQuarter": fiscal_quart,
                        "noOfEsts": row["noOfEsts"],
                        "epsForecast": row["epsForecast"],
                        "fiscalQuarterEnding": row["fiscalQuarterEnding"],
                        "rQuarter": get_quarter(row["reportDate"]),
                        "ivl": ivl,
                    }
                    if kwargs["filepath"] not in existing_files[weeks]:
                        inputs.append(kwargs)

    return inputs


def thread_historical_queries(
    so: CalendarSnapData, fb: str, base_params: dict, strikes: list, func: callable, attr_name: str
):
    if fb == "front":
        exp = so.fexp
        attr_name = f"{attr_name}_front"
    elif fb == "back":
        exp = so.bexp
        attr_name = f"{attr_name}_back"
    else:
        raise ValueError("Invalid front/back value")

    inputs = [
        {
            "symbol": so.symbol,
            "roots": so.roots,
            "exp": exp,
            "strike": strike,
            "base_params": base_params,
        }
        for strike in strikes
    ]

    dfs = Parallel(n_jobs=min(len(inputs), 8), backend="threading", verbose=0)(
        delayed(func)(**kwargs) for kwargs in inputs
    )
    dfs = [df for df in dfs if df is not None]
    if len(dfs) == 0:
        return
    df = pd.concat(dfs, ignore_index=True)
    if attr_name == "greeks_back":
        df.drop(columns="underlying", inplace=True)

    setattr(so, attr_name, df)


def merge_historical_snapshot(front, back, cols=None):
    if cols is None:
        cols = ["symbol", "right", "date", "ms_of_day", "strike_milli"]
    return pd.merge(
        front,
        back,
        on=[c for c in front if c in cols],
        how="inner",
        suffixes=("_front", "_back"),
    )


def historical_snapshot(kwargs):
    so = CalendarSnapData(
        symbol=kwargs["symbol"],
        roots=kwargs["roots"],
        rdatedt=pd.Timestamp(kwargs["rdate"]),
        weeks=kwargs["weeks"],
        right=kwargs["right"],
    )

    expirations = get_expiry_dates(so.roots)
    cal_dates = [d for d in expirations if d >= so.rdate]
    so.fexp = min(cal_dates)
    so.bexp = get_back_expiration_date(
        fexp=so.fexpdt,
        exp_list=cal_dates,
        weeks_between_fb=so.weeks,
    )
    if so.bexp is None:
        return kwargs["filepath"]

    if (so.fexpdt - so.rdatedt).days >= 7:
        return kwargs["filepath"]

    # Trading dates
    so.f_dates = get_exp_trading_days(roots=so.roots, exp=so.fexp)
    so.b_dates = get_exp_trading_days(roots=so.roots, exp=so.bexp)
    so.trade_dates = sorted(list(set(so.f_dates) & set(so.b_dates)))
    if len(so.trade_dates) < 5:
        return kwargs["filepath"]

    # Strikes
    so.f_strikes = get_strikes_exp(roots=so.roots, exp=so.fexp)
    so.b_strikes = get_strikes_exp(roots=so.roots, exp=so.bexp)
    so.strikes = sorted(list(set(so.f_strikes) & set(so.b_strikes)))

    # Params Defaults
    base_params = {
        "start_date": str(min(so.trade_dates)),
        "end_date": str(max(so.trade_dates)),
        "right": so.right,
        "ivl": str(kwargs["ivl"]),
    }

    # Underlying
    und_df = get_greeks_historical(
        symbol=so.symbol,
        roots=so.roots,
        exp=so.fexp,
        strike=so.strikes[len(so.strikes) // 2],
        base_params=base_params,
    )
    if und_df is None:
        return kwargs["filepath"]

    und_df = und_df[["underlying"]]

    if len(und_df[und_df["underlying"] == 0]) > 0:
        log.info(
            f"Underlying Price contains zero: {so.symbol}, {so.rdate}, {so.fexp}, Amount: {len(und_df[und_df['underlying'] == 0])}"
        )
        und_df = und_df[und_df["underlying"] != 0]
        if len(und_df) == 0:
            log.error(f"No data to process for {so.symbol}, {so.rdate}, {so.fexp}")
            return kwargs["filepath"]

    # Find bounds for strikes
    min_und = und_df["underlying"].min()
    max_und = und_df["underlying"].max()
    del und_df

    start_strikes = [s for s in so.strikes if (s / 1000) > (min_und * 0.94)]
    sliced_strikes = [s for s in start_strikes if (s / 1000) < (max_und * 1.06)]
    if len(sliced_strikes) == 0:
        log.info(f"No strikes found for {so.symbol}, {so.rdate}, {so.fexp}")
        return kwargs["filepath"]

    # Greeks
    for fb in ["front", "back"]:
        for func, attr_name in [
            (get_greeks_historical, "greeks"),
            (get_quotes_historical, "quotes"),
            (get_oi_historical, "oi"),
        ]:
            thread_historical_queries(
                so=so,
                fb=fb,
                base_params=base_params,
                strikes=sliced_strikes,
                func=func,
                attr_name=attr_name,
            )

    if any(
        [
            so.greeks_front is None,
            so.greeks_back is None,
            so.quotes_front is None,
            so.quotes_back is None,
            so.oi_front is None,
            so.oi_back is None,
        ]
    ):
        log.info(f"Snapshot data is missing for {so.symbol}, dropping the symbol")
        return kwargs["filepath"]

    so.greeks = merge_historical_snapshot(so.greeks_front, so.greeks_back)
    so.quotes = merge_historical_snapshot(so.quotes_front, so.quotes_back)

    for attr_name, fb in [("oi_front", "Front"), ("oi_back", "Back")]:
        attr_data = getattr(so, attr_name, None)
        if attr_data[["strike_milli", "date"]].value_counts().max() > 1:
            log.info(f"Duplicate values found in OI {fb}: {so.symbol}, {so.rdate}, {so.fexp}")
            attr_data = attr_data.sort_values(["ms_of_day", "date"]).drop_duplicates(
                subset=["strike_milli", "date"], keep="first"
            )
            setattr(so, attr_name, attr_data)
        setattr(so, attr_name, getattr(so, attr_name, None).drop(columns="ms_of_day"))

    so.oi = merge_historical_snapshot(
        so.oi_front, so.oi_back, cols=["symbol", "right", "date", "strike_milli"]
    )
    # final merge
    m_cols = ["symbol", "right", "date", "ms_of_day", "strike_milli", "exp_front", "exp_back"]
    moi_cols = ["symbol", "date", "right", "strike_milli", "exp_front", "exp_back"]

    for col in so.quotes_front.columns:
        if (col in so.greeks_front.columns) and (not m_cols):
            log.warning(f"Column {col} already exists in m_cols")

    df = so.quotes.merge(so.greeks, on=m_cols, how="inner", suffixes=("_quote", "_ivgreek")).merge(
        so.oi, on=moi_cols, how="left", suffixes=("", "_oi")
    )

    columns_to_fill = ["open_interest_front", "open_interest_back"]
    df[columns_to_fill] = df[columns_to_fill].fillna(0)

    complete_df = df.sort_values(by=["strike_milli", "date", "ms_of_day"])

    df = df.assign(
        strike=complete_df["strike_milli"].div(1000).round(2),
        weeks=so.weeks,
        reportDate=so.rdate,
        fQuarter=kwargs["fQuarter"],
        noOfEsts=kwargs["noOfEsts"],
        epsForecast=kwargs["epsForecast"],
        fiscalQuarterEnding=kwargs["fiscalQuarterEnding"],
        rQuarter=kwargs["rQuarter"],
    )
    if df.empty:
        return kwargs["filepath"]

    if len(df["date"].unique()) > 5:
        table = pa.Table.from_pandas(df)
        bucket = S3Handler(bucket_name=os.getenv("S3_BUCKET_NAME"), region="us-east-2")
        bucket.upload_table(table, f"{kwargs['filepath']}")

        return None


if __name__ == "__main__":
    # --------------------------------------------------------------
    # Input Parameters
    # --------------------------------------------------------------
    week_list = [1, 2, 3, 4, 5]
    first_dates = pd.Timestamp("2016-01-01")  # Standard subscription
    ivl = 900000  # 15 minutes
    right = "C"

    bucket = S3Handler(bucket_name=os.getenv("S3_BUCKET_NAME"), region="us-east-2")
    existing_files = {w: bucket.list_files(f"{get_folder_name(w, right)}/") for w in week_list}

    failed_files_path = "data/raw/calendar/failed_files.parquet"
    if bucket.file_exists(failed_files_path):
        failed_df = bucket.read_dataframe(failed_files_path, format="parquet")
        failed_files = failed_df["filepath"].tolist()
    else:
        failed_files = []

    # --------------------------------------------------------------
    # Setup historical earnings dates for qualified symbols
    # --------------------------------------------------------------

    symbols_df = read_from_db(query='SELECT symbol FROM public."qualifiedSymbols"')
    symbols = symbols_df["symbol"].unique().tolist()
    earnings = read_from_db(
        query=f'SELECT * FROM "nasdaqHistoricalEarnings" WHERE "symbol" IN {tuple(symbols)}'
    )
    earnings["reportDate"] = pd.to_datetime(earnings["reportDate"])
    earnings = earnings[earnings["reportDate"] >= first_dates]

    # TODO: Increase the historical earnings dates to 2016

    # --------------------------------------------------------------
    # Prepare symbols that have changed
    # --------------------------------------------------------------
    sc_df = read_from_db(query=f"""SELECT * FROM "changesPolygon" WHERE "symbol" IN {tuple(symbols)}""")
    sc_df["entries"] = sc_df.groupby("symbol")["symbol"].transform("count")
    sc_df = sc_df[sc_df["entries"] >= 2]

    sc_df["last_entry"] = sc_df.groupby("symbol")["date"].transform("max")
    sc_df["last_entry"] = pd.to_datetime(sc_df["last_entry"], format="%Y-%m-%d")
    sc_df = sc_df[sc_df["last_entry"] >= first_dates]

    # --------------------------------------------------------------
    # Start the process
    # --------------------------------------------------------------

    inputs = prepare_inputs()

    tf = len(inputs)
    inputs = [kwargs for kwargs in inputs if kwargs["filepath"] not in failed_files]

    log.info(f"Total Failed Files: {len(failed_files)}")
    log.info(f"Total Inputs: {len(inputs)}")
    log.info(f"Removed {tf - len(inputs)} failed files")

    for batch in batched(inputs, 80):
        cpus = 4
        if is_market_open(break_Script=False):
            cpus = 1
            log.info(f"Market is open, reducing the number of CPUs to {cpus}")

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
