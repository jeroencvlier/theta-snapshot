import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
from theta_snapshot.utils import read_from_db
import logging as log
from theta_snapshot import (
    read_from_db,
    calculate_buisness_days
)
import time 
from sklearn.metrics import matthews_corrcoef
import joblib
import numpy as np
from typing import List
import pandas_market_calendars as mcal
log.basicConfig(level=log.INFO, format="%(asctime)s - %(message)s")


def create_time_features(df):
    df["date_"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["cos_day"] = np.cos(2 * np.pi * df["date_"].dt.dayofyear / 365)
    df["sin_day"] = np.sin(2 * np.pi * df["date_"].dt.dayofyear / 365)
    df["cos_month"] = np.cos(2 * np.pi * df["date_"].dt.month / 12)
    df["sin_month"] = np.sin(2 * np.pi * df["date_"].dt.month / 12)
    df["cos_weekday"] = np.cos(2 * np.pi * df["date_"].dt.weekday / 7)
    df["sin_weekday"] = np.sin(2 * np.pi * df["date_"].dt.weekday / 7)
    df["year"] = df["date_"].dt.year
    df.drop(columns=["date_"], inplace=True)
    return df

def sector_industry(df):
    symbol_industries = read_from_db(
        query="""SELECT symbol,"longName",sector FROM "qualifiedSymbols" where symbol in {}""".format(
            tuple(df["symbol"].unique())
        )
    )
    df = df.merge(
        symbol_industries[["symbol", "sector"]],
        on="symbol",
        how="inner",
    )
    return df

def histroical_grade(df, latest=False):
    query_grades = """
            SELECT *    
            FROM public."historicalBacktestGrades" 
            WHERE symbol in {}
            """.format(tuple(df["symbol"].unique()))
    if latest:
        query_grades += "AND latest = TRUE"
    grades = read_from_db(query=query_grades)
    grades = grades[grades["weeks"] != "Average"]
    grades["weeks"] = grades["weeks"].astype(int)
    grades.drop(columns=["latest", "Grade"], inplace=True)

    if latest:
        grades = grades.drop(columns=["targetFiscalQuarter"])
        duplicated_columns = [
            c for c in grades.columns if c in df.columns and c not in ["symbol", "weeks"]
        ]
        log.info("func: histroical_grade. Duplicated colums to drop: %s", duplicated_columns)
        grades.drop(columns=duplicated_columns, inplace=True)
        df = df.merge(grades, on=["symbol", "weeks"], how="inner")
    else:
        grades = grades.rename(columns={"targetFiscalQuarter": "fQuarter"})
        df = df.merge(grades, on=["symbol", "fQuarter", "weeks"], how="inner")
    return df

def historical_backtest(df, latest=False):
    query_aggs = """
        SELECT *
        FROM public."historicalBacktestAggs" 
        WHERE symbol in {}
        """.format(tuple(df["symbol"].unique()))
    if latest:
        query_aggs += "AND latest = TRUE"

    aggs = read_from_db(query=query_aggs)
    aggs.drop(
        columns=["latest", "overmean_str_ratio", "undmean_str_ratio", "all_str_ratio"],
        inplace=True,
    )
    aggs["weeks"] = aggs["weeks"].astype(int)
    if latest:
        aggs.drop(columns=["targetFiscalQuarter"], inplace=True)
        duplicated_columns = [
            c for c in aggs.columns if c in df.columns and c not in ["symbol", "bdte", "weeks"]
        ]
        log.info("func: historical_backtest. Duplicated colums to drop: %s", duplicated_columns)
        aggs.drop(columns=duplicated_columns, inplace=True)
        df = df.merge(aggs, on=["symbol", "weeks", "bdte"], how="inner")
    else:
        aggs.rename(columns={"targetFiscalQuarter": "fQuarter"}, inplace=True)
        df = df.merge(aggs, on=["symbol", "fQuarter", "weeks", "bdte"], how="inner")

    return df


# def generate_predictions(theta_df):
#     try:
#         pred_df = theta_df.copy()
#         pred_df = create_time_features(pred_df)
#         pred_df = sector_industry(pred_df)
#         pred_df = historical_backtest(pred_df, latest=True)
#         pred_df = calculate_buisness_days(pred_df)
#         pred_df = pred_df.replace([np.inf, -np.inf], np.nan)
#         model_path = os.path.join(os.getcwd(), "models", "xg_pipeline")
#         model = load_model(model_path)
#         preds = predict_model(model, data=pred_df.dropna(), probability_threshold=0.9, raw_score=True)
#         theta_df = theta_df.merge(preds[["symbol", "strike", "right", "exp_front", "weeks","prediction_label", "prediction_score_0", "prediction_score_1"]], on=["symbol", "strike", "right", "exp_front", "weeks"], how="left")
#     except Exception as e:
#         log.error("Error in generate_predictions: %s", e)
#         theta_df.assign(prediction_label=np.nan, prediction_score_0=np.nan, prediction_score_1=np.nan)
#     return theta_df        

def custom_mcc_metric(
    X_val, y_val, estimator, labels,
    X_train, y_train, weight_val=None, weight_train=None,
    *args,
):
    start = time.time()
    y_pred_val = estimator.predict(X_val)  
    pred_time = (time.time() - start) / len(X_val)
    val_mcc = matthews_corrcoef(y_val, y_pred_val)
    
    
    y_pred_train = estimator.predict(X_train)
    train_mcc = matthews_corrcoef(y_train, y_pred_train)
    # If FLAML minimizes the metric, return -MCC (because we want to maximize MCC)
    return -val_mcc, {
        "val_mcc": val_mcc,
        "train_mcc": train_mcc,
        "pred_time": pred_time,
    }
    


def generate_predictions(theta_df):
    try:
        pred_df = theta_df.copy()
        pred_df = create_time_features(pred_df)
        pred_df = sector_industry(pred_df)
        pred_df = historical_backtest(pred_df, latest=True)
        pred_df = calculate_buisness_days(pred_df)
        pred_df = min_max_size_feature(pred_df)
        pred_df = pred_df.replace([np.inf, -np.inf], np.nan)
        pred_df = cal_gap_feature(pred_df)
        trend_df = price_trend(pred_df['symbol'].unique().tolist())
        trend_slice = trend_slicer(pred_df, trend_df)
        pred_df = pred_df.merge(trend_slice, on=['symbol', 'weeks', 'bdte'], how="inner")  
        model_path = os.path.join(os.getcwd(), "models", "flaml_pipeline.joblib")
        ml_pipe = joblib.load(model_path)
        
        preds = pd.DataFrame(ml_pipe.predict_proba(pred_df)).rename(columns={0:"prediction_score_0", 1:"prediction_score_1"})
        preds['prediction_label'] = preds['prediction_score_1'].apply(lambda x: 1 if x > 0.8 else 0)
        pred_df = pd.concat([preds,pred_df],axis=1) 
        theta_df = theta_df.merge(pred_df[["symbol", "strike", "right", "exp_front", "weeks","prediction_label", "prediction_score_0", "prediction_score_1"]], on=["symbol", "strike", "right", "exp_front", "weeks"], how="left")

    
    except Exception as e:
        log.error("Error in generate_predictions: %s", e)
        theta_df.assign(prediction_label=np.nan, prediction_score_0=np.nan, prediction_score_1=np.nan)
    return theta_df        


# --------------------------------------------------------------
# Auto Flaml
# --------------------------------------------------------------

def custom_mcc_metric(
    X_val, y_val, estimator, labels,
    X_train, y_train, weight_val=None, weight_train=None,
    *args,
):
    start = time.time()
    y_pred_val = estimator.predict(X_val)  
    pred_time = (time.time() - start) / len(X_val)
    val_mcc = matthews_corrcoef(y_val, y_pred_val)
    
    
    y_pred_train = estimator.predict(X_train)
    train_mcc = matthews_corrcoef(y_train, y_pred_train)
    # If FLAML minimizes the metric, return -MCC (because we want to maximize MCC)
    return -val_mcc, {
        "val_mcc": val_mcc,
        "train_mcc": train_mcc,
        "pred_time": pred_time,
    }
    

def days_before_calcs(
    df: pd.DataFrame,
    end_date_col: str,
    target: str,
    trading_days: List[pd.Timestamp],
    from_date_col: str = None,
):
    target_columns = {
        "earnings": ("bdte", "dte"),
        "fexp": ("bdtfexp", "dtfexp"),
        "dit": ("bdit", "dit"),
    }
    bdte_col, dte_col = target_columns.get(target)
    start_dates = pd.to_datetime(df[from_date_col], format="%Y%m%d")
    end_dates = pd.to_datetime(df[end_date_col], format="%Y%m%d")
    start_indices = np.searchsorted(trading_days, start_dates.values)
    end_indices = np.searchsorted(trading_days, end_dates.values)
    df[bdte_col] = end_indices - start_indices
    df[dte_col] = (end_dates - start_dates).dt.days
    return df



def min_max_size_feature(df:pd.DataFrame):
    # Convert relevant columns to NumPy arrays first (reduces overhead)
    ask_front = df['ask_size_front'].values
    bid_back = df['bid_size_back'].values
    ask_back = df['ask_size_back'].values
    bid_front = df['bid_size_front'].values

    # Use NumPy's element-wise min/max (2x faster than pandas)
    df['min_entry_size'] = np.minimum(ask_front, bid_back)
    df['max_entry_size'] = np.maximum(ask_front, bid_back)
    df['min_exit_size'] = np.minimum(ask_back, bid_front)
    df['max_exit_size'] = np.maximum(ask_back, bid_front)
    return df



def adjust_calgap_pct(calgappct, weeks):
    cal_gap_pct = {"1": 0.1, "2": 0.175, "3": 0.25, "4": 0.325, "5": 0.40, "6": 0.475}
    return calgappct - cal_gap_pct[str(weeks)]

def cal_gap_feature(df:pd.DataFrame):
    df["calGapPctAdjusted"] = df.apply(lambda x: adjust_calgap_pct(x["calGapPct"], x["weeks"]), axis=1)
    return df


def get_trading_days(df):
    nyse = mcal.get_calendar("NYSE")
    trading_days = nyse.schedule(
        start_date=pd.to_datetime(df["date"].min(), format="%Y%m%d"),
        end_date=pd.to_datetime(df["reportDate"].max(), format="%Y%m%d"),
    ).index

    df = days_before_calcs(
        df=df,
        end_date_col="reportDate",
        target="earnings",
        trading_days=trading_days,
        from_date_col="date",
    )
    return df


def price_trend(symbols):
    df = read_from_db(query=f"""SELECT "symbol", "weeks", "date", "calCostPct", "reportDate", "ask_size_front","bid_size_back","ask_size_back", "bid_size_front"  FROM public."ThetaSnapshot" WHERE symbol in {tuple(symbols)};""")
    df = min_max_size_feature(df)
    df = get_trading_days(df)
    df = df.drop(columns=["ask_size_front","bid_size_back","ask_size_back", "bid_size_front","reportDate" ,"date", "dte"])
    df = df.groupby(["bdte", "symbol", "weeks"]).agg(stddev=("calCostPct", "std"),max=("calCostPct", "max"),min=("calCostPct", "min"),avg=("calCostPct", "mean"),count=("calCostPct", "count"), min_entry_size_mean=("min_entry_size", "mean"), max_entry_size_mean=("max_entry_size", "mean"), min_exit_size_mean=("min_exit_size", "mean"), max_exit_size_mean=("max_exit_size", "mean")).reset_index()
    df["upper_ci"] = df["avg"] + df["stddev"]
    df["lower_ci"] = df["avg"] - df["stddev"]        
    return df

def trend_slicer(pred_df, trend_df):
    
    final_slice = []
    for idx, row in pred_df[['symbol','bdte', 'weeks']].drop_duplicates().iterrows():
        
        trade_slice = {'symbol':row.symbol, "weeks":row.weeks, 'bdte':row.bdte}
        bdtes = [*range(row.bdte,row.bdte+4)]
        pre_slice = trend_df[(trend_df['symbol']==row.symbol)&(trend_df['weeks']==row.weeks)]
        pre_slice = pre_slice[pre_slice["bdte"].isin(bdtes)]
        pre_slice = pre_slice[pre_slice["count"] > 4].sort_values(by="bdte", ascending=True)   
        if (len(pre_slice) >= 3):
            if all([True if x in pre_slice['bdte'].tolist() else False for x in bdtes[1:]]) or all([True if x in pre_slice['bdte'].tolist() else False for x in bdtes[:-1]]):
                pre_slice = pre_slice.iloc[:3].reset_index(drop=True)
                nas = pre_slice.isna().sum().sum()
                if nas > 0:
                    log.info(f"NA values found in grouped_pre_slice: {nas}")    
                for index, row2 in pre_slice.iterrows():
                    row2.pop("bdte")
                    row2.pop("symbol")
                    row2.pop("weeks")
                    rename_cols = {k: f"pre_{int(index)}_{k}" for k in row2.keys()}
                    row2 = row2.rename(rename_cols)
                    trade_slice.update(row2.to_dict())
                final_slice.append(trade_slice)
                
    return pd.DataFrame(final_slice)
                    