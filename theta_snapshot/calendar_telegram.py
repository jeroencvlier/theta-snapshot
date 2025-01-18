import os
import re
import sys
from collections import Counter
from datetime import datetime as dt
import pandas as pd
import logging as log
import telebot

from theta_snapshot import (
    read_from_db,
    write_to_db,
)

log.basicConfig(level=log.INFO, format="%(asctime)s - %(message)s")


# --------------------------------------------------------------
# Filter presets
# --------------------------------------------------------------
min_dbte = 6
max_dbte = 14
min_histearningscount = 7
min_class = 1.25
cal_gap_pct = {"1": 0.1, "2": 0.175, "3": 0.25, "4": 0.325, "5": 0.40, "6": 0.475}
min_oi = 5
min_size = 10
pct_under_over_mean = -0.05
min_spread = 0.3
min_cal_spread = 0.6
min_cost = 0.5
max_cost = 5


# --------------------------------------------------------------
# Load Data
# --------------------------------------------------------------
def send_telegram_alerts():
    cols_iv = [
        "implied_vol_back",
        "implied_vol_front",
        "iv_pct_diff",
        "iv_consensus",
        "iv_max_exp_c",
        "iv_max_exp_p",
    ]
    cols_spread = ["spreadPct_front", "spreadPct_back", "spreadPct_cal"]
    cols_earn = ["symbol", "reportDate", "exp_front", "exp_back", "weeks", "bdte"]
    cols_oi = [
        "open_interest_front",
        "open_interest_back",
        "ask_size_back",
        "bid_size_back",
        "ask_size_front",
        "bid_size_front",
    ]
    cols_cal = ["calCost", "avgCalCost", "calGapPct", "pct_under_over_mean"]
    cols_und = ["underlying", "strike", "undPricePctDiff"]
    cols_hist = ["undmean_avg_trade_class", "histEarningsCount", "Grade"]
    cols_extra = ["lastUpdated"]
    cols_all = (
        cols_iv + cols_spread + cols_earn + cols_oi + cols_cal + cols_und + cols_hist + cols_extra
    )
    cols_alert = (
        cols_earn + cols_und + cols_cal + cols_iv + cols_spread + cols_hist + cols_oi + cols_extra
    )

    cols_duplicated = [item for item, count in Counter(cols_all).items() if count > 1]
    if len(cols_duplicated) > 0:
        log.warning(f"Duplicate columns found: {cols_duplicated}")

    trade_query = f"""
        SELECT "{'", "'.join(cols_all)}"
        FROM "ThetaSnapshot"
        WHERE "undmean_avg_trade_class" >= 1.25
        AND "lastUpdated" = (
            SELECT MAX("lastUpdated")
            FROM "ThetaSnapshot"
        );
        """

    alert_df = read_from_db(query=trade_query)

    current_date = dt.now().strftime("%Y%m%d")
    historical_query = (
        f"""SELECT * FROM public."ThetaTelegramAlerts" WHERE "alert_date" = '{current_date}'"""
    )
    try:
        hist_alerts = read_from_db(query=historical_query)
    except Exception as e:
        hist_alerts = pd.DataFrame()
        log.error(f"Error reading historical alerts: {e}")

    log.info(f"Loaded Previous Alerts: {len(hist_alerts)}")

    alert_df[cols_iv] = alert_df[cols_iv].round(3)
    alert_df[cols_spread] = alert_df[cols_spread].round(2)
    alert_df[cols_cal] = alert_df[cols_cal].round(3)

    # --------------------------------------------------------------
    # Filter Alerts
    # --------------------------------------------------------------

    if os.getenv("ENV").upper() not in ["DEV", "TEST"]:
        alert_df = alert_df[
            (alert_df["histEarningsCount"] >= min_histearningscount)
            & (alert_df["bdte"] <= max_dbte)
            & (alert_df["bdte"] >= min_dbte)
            & (alert_df["pct_under_over_mean"] < pct_under_over_mean)
            # & (alert_df["open_interest_front"] >= min_oi)
            # & (alert_df["open_interest_back"] >= min_oi)
            & (alert_df["ask_size_back"] >= min_size)
            & (alert_df["bid_size_back"] >= min_size)
            & (alert_df["ask_size_front"] >= min_size)
            & (alert_df["bid_size_front"] >= min_size)
            & (alert_df["spreadPct_front"] <= min_spread)
            & (alert_df["spreadPct_back"] <= min_spread)
            & (alert_df["spreadPct_cal"] <= min_cal_spread)
            & (alert_df["calCost"] > min_cost)
            & (alert_df["calCost"] < max_cost)
        ]

        alert_df = alert_df[
            alert_df.apply(
                lambda row: row["calGapPct"] < cal_gap_pct.get(str(row["weeks"])), axis=1
            )
        ]

    alert_df = alert_df.sort_values("undPricePctDiff").drop_duplicates(
        subset=["symbol", "weeks"], keep="first"
    )

    if alert_df.empty:
        log.info("No new alerts found")
        return

    alert_df = alert_df.assign(
        alert_time=alert_df["lastUpdated"],
        alert_date=alert_df["lastUpdated"].apply(
            lambda x: int(dt.fromtimestamp(x).strftime("%Y%m%d"))
        ),
    )

    # alerts_df = alert_df[cols_alert]
    # get unique alerts only for symbol and week
    alert_df = alert_df.drop_duplicates(subset=["symbol", "weeks"], keep="first")
    if len(alert_df) == 0:
        log.info("No new alerts found")
        return

    # --------------------------------------------------------------
    # Logic to check for previous alerts
    # --------------------------------------------------------------
    new_alerts = []
    alert_df.sort_values(by=["symbol", "weeks"], inplace=True)

    for alert in alert_df.iterrows():
        symbol = alert[1]["symbol"]
        week = alert[1]["weeks"]

        if len(hist_alerts) > 0:
            previous_alert_sublist = hist_alerts[
                (hist_alerts["symbol"] == symbol) & (hist_alerts["weeks"] == week)
            ]
        else:
            previous_alert_sublist = pd.DataFrame()

        if len(previous_alert_sublist) == 0:
            new_alerts.append(alert[1])
        else:
            alert_time = alert[1]["alert_time"]
            old_alert_time = previous_alert_sublist["alert_time"].max()
            # check that the time of the new alert is an hour after the old alert
            if (alert_time - old_alert_time) > 3600:
                new_alerts.append(alert[1])
            else:
                log.info("Skipping Alert due to recent alert: %s", alert[1]["symbol"])

    new_alerts = pd.DataFrame(new_alerts)

    if len(new_alerts) == 0:
        log.info("No new alerts found")
        return

    log.info(f"New Alerts: {len(new_alerts)}")

    # --------------------------------------------------------------
    # Send alerts to telegram
    # --------------------------------------------------------------

    if len(new_alerts) > 0:
        muted_alerts = read_from_db(query='''SELECT * FROM "mutedTelegramTickers"''')
        muted_alerts["mute_end_date"] = pd.to_datetime(muted_alerts["mute_end_date"]).dt.normalize()

        symbols = new_alerts["symbol"].unique()

        # If there's only one symbol, format the query without a tuple
        if len(symbols) == 1:
            symbol_query = f"('{symbols[0]}')"
        else:
            symbol_query = str(tuple(symbols))

        earn_dates = read_from_db(
            query=f"""SELECT * FROM "EarningsCalendarCombined" WHERE "symbol" IN {symbol_query}"""
        )

        new_alerts = new_alerts.merge(
            earn_dates[["symbol", "comment", "sourceUsed"]], on="symbol", how="left"
        )

        chat_ids = [c for c in os.getenv("TELEGRAM_BOT_TRADEALERT_IDS").split(",") if len(c) > 0]
        log.info(f"Sending alerts to chat_ids: {chat_ids}")

        bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TRADEALERT_TOKEN"))

        comment_symbols = {
            "samedate": "🟢",
            "sameweek": "🟡",
            "differentweek": "🔴",
            "default": "⚪",
        }
        iv_consensus = {"High": "🟢", "Medium": "🟡", "Low": "🔴", "No Data": "⚪"}

        for chat_id in chat_ids:
            messages = []
            message = ""
            seperator = "----------------------------------------\n"

            muted_alerts_ids = muted_alerts[muted_alerts["chat_id"] == chat_id]
            current_date = pd.to_datetime("today").normalize()
            muted_alerts_ids = muted_alerts_ids[muted_alerts_ids["mute_end_date"] > current_date]
            it = 1
            for idx, row in new_alerts.iterrows():
                symbol = row["symbol"]
                if symbol not in muted_alerts_ids["ticker"].to_list():
                    comment = row["comment"].strip().lower().replace(" ", "")
                    if comment not in ["samedate", "sameweek", "differentweek"]:
                        comment = "default"
                        row["symbol"] = "No Data"

                    right = ""
                    if row["iv_consensus"]:
                        iv_comment = "High"
                    elif row["iv_max_exp_c"] == row["exp_front"]:
                        right = "Calls "
                        iv_comment = "Medium"
                    elif row["iv_max_exp_p"] == row["exp_front"]:
                        right = "Puts "
                        iv_comment = "Medium"
                    else:
                        iv_comment = "Low"

                    exp_dates = "{} / {}".format(
                        pd.to_datetime(row["exp_front"], format="%Y%m%d").strftime("%b%d'%y"),
                        pd.to_datetime(row["exp_back"], format="%Y%m%d").strftime("%b%d'%y"),
                    )

                    report_date = "{}".format(
                        pd.to_datetime(row["reportDate"], format="%Y%m%d").strftime("%b%d'%y")
                    )
                    # Then construct the alert with explicit markdown link formatting
                    link_nasdaq = (
                        "https://www.nasdaq.com/market-activity/stocks/{}/earnings".format(
                            row["symbol"].lower()
                        )
                    )
                    link_zacks = (
                        "https://www.zacks.com/stock/quote/{}/detailed-earning-estimates".format(
                            row["symbol"]
                        )
                    )
                    link_salt = "{}/alertpage?ticker={}&weeks={}&timestamp={}&strike={}#trade-details".format(
                        os.getenv("SALT_URL"),
                        row["symbol"],
                        row["weeks"],
                        row["lastUpdated"],
                        row["strike"],
                    )

                    alert = (
                        "{}) {} - ({}W) {}".format(it, row["symbol"], row["weeks"], exp_dates)
                        + '\n      Date: {} {} (<a href="{}">Nasdaq</a> / <a href="{}">Zacks</a>)'.format(
                            report_date, comment_symbols[comment], link_nasdaq, link_zacks
                        )
                        # + '\n      View Earnings on <a href="{}">Nasdaq</a> / <a href="{}">Zacks</a>'.format(
                        #     link_nasdaq, link_zacks
                        # )
                        + "\n      Comment: {}".format(row["comment"])
                        + "\n      IV Consensus: {} {}".format(
                            iv_comment, right + iv_consensus[iv_comment]
                        )
                        + "\n      BDTE: {}".format(row["bdte"])
                        + "\n      Strike: {}".format(row["strike"])
                        + "\n      Underlying: {}".format(row["underlying"])
                        + "\n      CalCost: {}".format(row["calCost"])
                        + "\n      AvgCalCost: {}".format(row["avgCalCost"])
                        + "\n      Grade: {}".format(row["Grade"])
                        + "\n      CalGapPct: {}".format(row["calGapPct"])
                        + "\n      Spread % (F/B/Cal): {} / {} / {}".format(
                            row["spreadPct_front"], row["spreadPct_back"], row["spreadPct_cal"]
                        )
                        + "\n      OI (F/B): {} / {}".format(
                            row["open_interest_front"], row["open_interest_back"]
                        )
                        + "\n      IV (F/B/Diff): {} / {} / {}".format(
                            row["implied_vol_front"], row["implied_vol_back"], row["iv_pct_diff"]
                        )
                        + '\n      Salt Link: <a href="{}">Take me to SALT</a>'.format(link_salt)
                        + "\n{}".format(seperator)
                    )

                    message += alert
                    it += 1
                    if len(message) > 3600:
                        messages.append(message)
                        message = ""
            if len(message) > 0:
                messages.append(message)
            if len(messages) > 0:
                messages[0] = f"🚨 {len(new_alerts)} Trade Alerts 🚨\n{seperator}\n" + messages[0]
                for message in messages:
                    bot.send_message(
                        chat_id,
                        message,
                        parse_mode="HTML",  # Changed from MarkdownV2 to HTML
                        disable_web_page_preview=True,
                    )
                    log.info("Alert sent to chat_id: {}".format(chat_id))

            if os.getenv("ENV") not in ["dev", "test"]:
                write_to_db(new_alerts, "ThetaTelegramAlerts", if_exists="append")
