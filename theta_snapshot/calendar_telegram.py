import os
import re
import sys
from collections import Counter
from datetime import datetime as dt
import pandas as pd
from loguru import logger as log
import telebot

from theta_snapshot import (
    read_from_db,
    write_to_db,
)


# def escape_markdown_v2(text):
#     """
#     Escapes special characters for Telegram's MarkdownV2.
#     """
#     # Escape all special MarkdownV2 characters, including '-'
#     escape_chars = r"_*[]()~`>#+-=|{}.!"
#     return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)


# def escape_markdown_v2(text):
#     """
#     Escapes special characters for Telegram's MarkdownV2.
#     Preserves any existing markdown links in the format [text](url)
#     """
#     # First, temporarily replace any existing markdown links
#     link_pattern = r"\[(.*?)\]\((.*?)\)"
#     links = re.findall(link_pattern, text)
#     placeholder_text = text

#     # Store links with their complete original form
#     original_links = []
#     for match in re.finditer(link_pattern, text):
#         original_links.append(match.group(0))

#     for i, original_link in enumerate(original_links):
#         placeholder = f"LINK_PLACEHOLDER_{i}"
#         placeholder_text = placeholder_text.replace(original_link, placeholder)

#     # Escape all special MarkdownV2 characters except in placeholders
#     escape_chars = r"_*[]()~`>#+-=|{}.!"
#     parts = []
#     for part in re.split(r"(LINK_PLACEHOLDER_\d+)", placeholder_text):
#         if not part.startswith("LINK_PLACEHOLDER_"):
#             part = re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", part)
#         parts.append(part)
#     escaped_text = "".join(parts)

#     # Restore the links
#     for i, (link_text, url) in enumerate(links):
#         placeholder = f"LINK_PLACEHOLDER_{i}"
#         # Only escape the text portion of the link, leave markdown syntax intact
#         escaped_link_text = re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", link_text)
#         escaped_url = url.replace(")", "\\)")  # Only escape closing parenthesis in URL
#         escaped_text = escaped_text.replace(placeholder, f"[{escaped_link_text}]({escaped_url})")

#     return escaped_text


def escape_markdown_v2(text):
    """
    Escapes special characters for Telegram's MarkdownV2.
    Preserves any existing markdown links in the format [text](url)
    """
    # First, temporarily replace any existing markdown links
    link_pattern = r"\[(.*?)\]\((.*?)\)"
    links = []

    def replace_link(match):
        links.append(match.groups())
        return "LINK_PLACEHOLDER_{}".format(len(links) - 1)

    placeholder_text = re.sub(link_pattern, replace_link, text)

    # Escape special characters
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    parts = []
    for part in re.split(r"(LINK_PLACEHOLDER_\d+)", placeholder_text):
        if not part.startswith("LINK_PLACEHOLDER_"):
            part = re.sub("([{}])".format(re.escape(escape_chars)), r"\\\1", part)
        parts.append(part)
    escaped_text = "".join(parts)

    # Restore links with proper escaping
    for i, (text, url) in enumerate(links):
        placeholder = "LINK_PLACEHOLDER_{}".format(i)
        # Escape special characters in link text, but handle URL differently
        escaped_link_text = re.sub("([{}])".format(re.escape(escape_chars)), r"\\\1", text)
        escaped_text = escaped_text.replace(placeholder, "[{}]({})".format(escaped_link_text, url))

    return escaped_text


# --------------------------------------------------------------
# Filter presets
# --------------------------------------------------------------
min_dbte = 6
max_dbte = 14
min_histearningscount = 7
min_class = 1.25
cal_gap_pct = {"1": 0.1, "2": 0.175, "3": 0.25, "4": 0.325, "5": 0.40, "6": 0.475}
min_oi = 5
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
    cols_oi = ["open_interest_front", "open_interest_back"]
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
        WHERE "undmean_avg_trade_class" >= {min_class}
    """

    # trade_query = f"""
    #     SELECT *
    #     FROM "ThetaSnapshot"
    #     WHERE "undmean_avg_trade_class" >= {min_class}
    # """

    alert_df = read_from_db(query=trade_query)

    current_date = dt.now().strftime("%Y-%m-%d")
    historical_query = (
        f"""SELECT * FROM public."thetaTelegramAlerts" WHERE "alert_date" = '{current_date}'"""
    )
    hist_alerts = read_from_db(query=historical_query)

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
            & (alert_df["open_interest_front"] >= min_oi)
            & (alert_df["open_interest_back"] >= min_oi)
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

    alert_df["alert_time"] = alert_df["lastUpdated"]
    alert_df["alert_date"] = alert_df["alert_time"].apply(
        lambda x: int(dt.fromtimestamp(x).strftime("%Y%m%d"))
    )

    alerts_df = alert_df[cols_alert]
    # get unique alerts only for symbol and week
    alerts_df = alerts_df.drop_duplicates(subset=["symbol", "weeks"], keep="first")
    if len(alerts_df) == 0:
        log.info("No new alerts found")
        return

    # --------------------------------------------------------------
    # Logic to check for previous alerts
    # --------------------------------------------------------------
    new_alerts = []
    alerts_df.sort_values(by=["symbol", "weeks"], inplace=True)

    for alert in alerts_df.iterrows():
        symbol = alert[1]["symbol"]
        week = alert[1]["weeks"]

        previous_alert_sublist = hist_alerts[
            (hist_alerts["symbol"] == symbol) & (hist_alerts["weeks"] == week)
        ]
        # display(alert[1])
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
                    elif any([row["iv_max_exp_c"], row["iv_max_exp_p"]]):
                        right = "Calls " if row["iv_max_exp_c"] else "Puts "
                        iv_comment = "Medium"
                    elif not any([row["iv_max_exp_c"], row["iv_max_exp_p"]]):
                        iv_comment = "Low"
                    else:
                        iv_comment = "No Data"

                    exp_dates = "{} / {}".format(
                        pd.to_datetime(row["exp_front"], format="%Y%m%d").strftime("%b%d'%y"),
                        pd.to_datetime(row["exp_back"], format="%Y%m%d").strftime("%b%d'%y"),
                    )

                    report_date = "{}".format(
                        pd.to_datetime(row["reportDate"], format="%Y%m%d").strftime("%b%d'%y")
                    )
                    # link_nasdaq = f"https://www.nasdaq.com/market-activity/stocks/{row['symbol'].lower()}/earnings"
                    # link_zacks = f"https://www.zacks.com/stock/quote/{row['symbol']}/detailed-earning-estimates"
                    # # [Go to MSFT Research](/researchpage?ticker=MSFT)
                    # link_salt = f"{os.getenv('SALT_URL')}/researchpage"  # /researchpage?ticker={row['symbol']}&weeks={row['weeks']}&timestamp={row['lastUpdated']}&strike={row['strike']}"
                    # alert = (
                    #     f"{it}) {row['symbol']} - ({row['weeks']}W) {exp_dates}\n"
                    #     f"      Reported Date: {report_date} {comment_symbols[comment]}\n"
                    #     f"      View Earnings on [Nasdaq]({link_nasdaq}) / [Zacks]({link_zacks})\n"
                    #     f"      Comment: {row['comment']}\n"
                    #     f"      IV Consensus: {iv_comment} {right}{iv_consensus[iv_comment]}\n"
                    #     f"      BDTE: {row['bdte']}\n"
                    #     f"      Strike: {row['strike']}\n"
                    #     f"      Underlying: {row['underlying']}\n"
                    #     f"      CalCost: {row['calCost']}\n"
                    #     f"      AvgCalCost: {row['avgCalCost']}\n"
                    #     f"      Grade: {row['Grade']}\n"
                    #     f"      CalGapPct: {row['calGapPct']}\n"
                    #     f"      Spread % (F/B/Cal): {row['spreadPct_front']} / {row['spreadPct_back']} / {row['spreadPct_cal']}\n"
                    #     f"      OI (F/B): {row['open_interest_front']} / {row['open_interest_back']}\n"
                    #     f"      IV (F/B/Diff): {row['implied_vol_front']} / {row['implied_vol_back']} / {row['iv_pct_diff']}\n"
                    #     f"      Salt Link: [Take me to SALT]({link_salt})\n"
                    #     f"{seperator}\n"
                    # )

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
                    # message = escape_markdown_v2(message)
                    # bot.send_message(
                    #     chat_id, message, parse_mode="MarkdownV2", disable_web_page_preview=True
                    # )
                    bot.send_message(
                        chat_id,
                        message,
                        parse_mode="HTML",  # Changed from MarkdownV2 to HTML
                        disable_web_page_preview=True,
                    )
                    log.info("Alert sent to chat_id: {}".format(chat_id))

        if os.getenv("ENV") not in ["dev", "test"]:
            write_to_db(hist_alerts, "thetaTelegramAlerts", if_exists="append")
