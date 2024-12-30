import os
import sys
from loguru import logger as log
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from datetime import datetime as dt
import option_emporium as oe
from loguru import logger as log

from theta_snapshot import (
    snapshot,
    read_from_db,
    write_to_db,
    get_iv_chain,
    is_market_open,
    time_checker_ny,
    main_wrapper,
    iv_features,
    send_telegram_alerts,
)


symbols_df = read_from_db(query='SELECT symbol FROM "qualifiedSymbols"')
symbols = symbols_df["symbol"].unique().tolist()
earnings = read_from_db(
    query=f'SELECT * FROM "nasdaqHistoricalEarnings" where "symbol" in {tuple(symbols)}'
)
stocks_symbol_change = read_from_db(query='SELECT * FROM "changesPolygon"')

symbols_tracked = []
for symbol in stocks_symbol_change["symbol"]:
    a = stocks_symbol_change[stocks_symbol_change["symbol"] == symbol]
    if len(a) >= 2:
        for date in a["date"]:
            days_ago_ago = dt.now() - dt.strptime(date, "%Y-%m-%d")
            months_ago = days_ago_ago.days // 30
            if months_ago <= 50:
                if symbol in symbols:
                    symbols_tracked.append(symbol)
                    symbols_tracked = list(set(symbols_tracked))

log.info(
    "There are %s symbols with name changes within the last 4 years",
    len(symbols_tracked),
)
