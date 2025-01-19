import os
import numpy as np
import pandas as pd
from pycaret.classification import load_model, predict_model
from datetime import datetime as dt
from theta_snapshot.utils import read_from_db
import logging as log
from theta_snapshot import (
    read_from_db,
    calculate_buisness_days
)

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


def generate_predictions(theta_df):
    try:
        pred_df = theta_df.copy()
        pred_df = create_time_features(pred_df)
        pred_df = sector_industry(pred_df)
        pred_df = historical_backtest(pred_df, latest=True)
        pred_df = calculate_buisness_days(pred_df)
        pred_df = pred_df.replace([np.inf, -np.inf], np.nan)
        model_path = os.path.join(os.getcwd(), "models", "xg_pipeline")
        model = load_model(model_path)
        preds = predict_model(model, data=pred_df.dropna(), probability_threshold=0.9, raw_score=True)
        theta_df = theta_df.merge(preds[["symbol", "strike", "right", "exp_front", "weeks","prediction_label", "prediction_score_0", "prediction_score_1"]], on=["symbol", "strike", "right", "exp_front", "weeks"], how="left")
    except Exception as e:
        log.error("Error in generate_predictions: %s", e)
        theta_df.assign(prediction_label=np.nan, prediction_score_0=np.nan, prediction_score_1=np.nan)
    return theta_df        
