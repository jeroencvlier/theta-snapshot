from collections import Counter

import datetime as dt
import pandas as pd
from joblib import Parallel, delayed
from theta_snapshot import (
    get_expiry_dates,
    get_greeks,
)

from loguru import logger as log


def get_most_common_strikes(ivs_list, symbol):
    """Get strikes that appear most frequently across all IV dataframes"""
    strike_counts = Counter(strike for iv in ivs_list for strike in iv["strike"].unique())
    max_count = max(strike_counts.values())
    return_strikes = {strike for strike, count in strike_counts.items() if count == max_count}
    if len(return_strikes) < 3:
        log.error(
            "Less than 3 strikes found in the IV dataframes for symbol %s. %s",
            symbol,
            return_strikes,
        )
    return return_strikes


def find_closest_strike(strikes, target):
    strikes_list = sorted(list(strikes))
    closest_strike = min(strikes_list, key=lambda x: abs(x - target))
    return closest_strike


def get_iv_chain(symb):
    # symb = "AAPL"
    try:
        expirations = get_expiry_dates(symb)
        # slice between today and 60 days from now
        expirations = sorted([pd.to_datetime(date, format="%Y%m%d") for date in expirations])
        cal_dates = [
            exp
            for exp in expirations
            if exp >= dt.datetime.now() and exp <= dt.datetime.now() + dt.timedelta(days=90)
        ]
        expirations_str = [exp.strftime("%Y%m%d") for exp in cal_dates]
        # get the iv and greeks for the symbol
        symb_ivs_puts = []
        symb_ivs_calls = []
        underlying = []

        inputs = [(get_greeks, symb, exp, "PC") for exp in expirations_str]
        greeks = Parallel(n_jobs=len(expirations_str), backend="threading")(
            delayed(func)(*args) for func, *args in inputs
        )

        underlying = pd.concat(greeks)["underlying_price"].mean().round(2)

        symb_ivs_puts = [df[df["right"] == "P"] for df in greeks]
        symb_ivs_calls = [df[df["right"] == "C"] for df in greeks]

        strikes_puts = get_most_common_strikes(symb_ivs_puts, symb)
        strikes_calls = get_most_common_strikes(symb_ivs_calls, symb)
        putstrike = find_closest_strike(strikes_puts, underlying * 1000)
        callstrike = find_closest_strike(strikes_calls, underlying * 1000)
        symb_ivs_puts = pd.concat(symb_ivs_puts)
        symb_ivs_puts = symb_ivs_puts[symb_ivs_puts["strike"] == putstrike]
        symb_ivs_puts["right"] = "P"

        symb_ivs_calls = pd.concat(symb_ivs_calls)
        symb_ivs_calls = symb_ivs_calls[symb_ivs_calls["strike"] == callstrike]
        symb_ivs_calls["right"] = "C"

        symb_ivs = pd.concat([symb_ivs_puts, symb_ivs_calls])
        symb_ivs["symbol"] = symb

        symb_ivs["underlying"] = underlying
        symb_ivs["strike_relative"] = round(symb_ivs["strike"] / 1000, 2)
        symb_ivs["strike_pct_diff"] = round(
            (symb_ivs["strike_relative"] - symb_ivs["underlying"]) / symb_ivs["underlying"],
            3,
        )

        return symb_ivs

    except Exception as err:
        log.error("Symbol: %s, Error: %s", symb, err)
