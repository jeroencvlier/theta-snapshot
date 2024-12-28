current_date = dt.datetime.now().strftime("%Y-%m-%d")
        previous_alerts = read_from_db(
            query=f"""SELECT * FROM "thetaGradeAlerts" WHERE "alert_date" = '{current_date}'"""
        )
        logging.info(f"Loaded Previous Alerts: {len(previous_alerts)}")
        potential_alerts = theta_df[theta_df["Grade"].str.contains("A")]
        potential_alerts = create_iv_pct_diff(potential_alerts)

        potential_alerts[["implied_vol_back", "implied_vol_front", "iv_pct_diff"]] = (
            potential_alerts[
                ["implied_vol_back", "implied_vol_front", "iv_pct_diff"]
            ].round(3)
        )
        # first set of filters:
        potential_alerts = potential_alerts[
            (potential_alerts["histearningscount"] >= 7)
            & (potential_alerts["bdte"] <= 14)
            & (potential_alerts["bdte"] >= 6)
        ]

        alerts = []
        for week, calgappct in zip(
            [1, 2, 3, 4, 5, 6], [0.1, 0.175, 0.25, 0.325, 0.40, 0.475]
        ):
            # break
            sub_pa = potential_alerts[(potential_alerts["weeks"] == week)]
            sub_pa = sub_pa.sort_values(["symbol", "undPricePctDiff"]).drop_duplicates(
                subset=["symbol", "weeks"], keep="first"
            )
            sub_pa = sub_pa[(sub_pa["calGapPct"] <= calgappct)]
            if os.getenv("ENV") not in ["dev", "test"]:
                potential_alerts = sub_pa[
                    (sub_pa["pct_under_over_mean"] < -0.05)
                    & (sub_pa["open_interest_front"] >= 5)
                    & (sub_pa["open_interest_back"] >= 5)
                    & (sub_pa["spreadPct_front"] <= 0.3)
                    & (sub_pa["spreadPct_back"] <= 0.3)
                ]

            if len(sub_pa) > 0:
                alerts.append(sub_pa)
        if len(alerts) == 0:
            logging.info("No new alerts found")
            sys.exit(0)

        alerts_df = pd.concat(alerts)
        alerts_df["alert_time"] = alerts_df["lastUpdated"]
        alerts_df["alert_date"] = alerts_df["alert_time"].apply(
            lambda x: dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d")
        )

        # potential_alerts[['strike', 'underlying', 'undPricePctDiff']]

        cols_alert = [
            "symbol",
            "underlying",
            "strike",
            "weeks",
            "calCost",
            "avgCalCost",
            "Grade",
            "calGapPct",
            # "spread_front",
            # "spread_back",
            "spreadPct_front",
            "spreadPct_back",
            "open_interest_front",
            "open_interest_back",
            "alert_time",
            "alert_date",
            "exp_front",
            "exp_back",
            "bdte",
            "reportedDate",
            "implied_vol_back",
            "implied_vol_front",
            "iv_pct_diff",
        ]

        alerts_df = alerts_df[cols_alert]
        # get unique alerts only for symbol and week
        alerts_df = alerts_df.drop_duplicates(subset=["symbol", "weeks"], keep="first")
        if len(alerts_df) == 0:
            logging.info("No new alerts found")
            sys.exit(0)
        # --------------------------------------------------------------
        # Add logic to check for previous alerts
        # --------------------------------------------------------------
        new_alerts = []
        alerts_df.sort_values(by=["symbol", "weeks"], inplace=True)
        for alert in alerts_df.iterrows():
            symbol = alert[1]["symbol"]
            week = alert[1]["weeks"]

            previous_alert_sublist = previous_alerts[
                (previous_alerts["symbol"] == symbol)
                & (previous_alerts["weeks"] == week)
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
                    logging.info(
                        "Skipping Alert due to recent alert: %s", alert[1]["symbol"]
                    )

        new_alerts = pd.DataFrame(new_alerts)
        new_alerts = new_alerts[new_alerts["calCost"] > 0.5]
        if len(new_alerts) == 0:
            logging.info("No new alerts found")
            sys.exit(0)
        logging.info(f"New Alerts: {len(new_alerts)}")

        # --------------------------------------------------------------
        # Send alerts to telegram
        # --------------------------------------------------------------

        if len(new_alerts) > 0:

            muted_alerts = read_from_db(
                query='''SELECT * FROM "mutedTelegramTickers"'''
            )
            muted_alerts["mute_end_date"] = pd.to_datetime(
                muted_alerts["mute_end_date"]
            ).dt.normalize()

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

            chat_ids = json.loads(os.getenv("TELEGRAM_BOT_TRADEALERT_IDS"))
            # chatids seperated by comma
            bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TRADEALERT_TOKEN"))
            for chat_id in chat_ids:
                messages = []
                message = ""
                seperator = "------------------------------------------"

                muted_alerts_ids = muted_alerts[muted_alerts["chat_id"] == chat_id]
                current_date = pd.to_datetime("today").normalize()
                muted_alerts_ids = muted_alerts_ids[
                    muted_alerts_ids["mute_end_date"] > current_date
                ]
                it = 1
                for idx, row in new_alerts.iterrows():
                    symbol = row["symbol"]
                    if symbol not in muted_alerts_ids["ticker"].to_list():
                        comment = row["comment"].strip().lower().replace(" ", "")
                        if comment == "samedate":
                            comment_symbol = "🟢"
                        elif comment == "sameweek":
                            comment_symbol = "🟡"
                        elif comment == "differentweek":
                            comment_symbol = "🔴"
                        else:
                            row["symbol"] = "No Data"
                            comment_symbol = "⚪"

                        exp_dates = f"{pd.to_datetime(row['exp_front']).strftime('%b%d')}'{pd.to_datetime(row['exp_front']).strftime('%y')}/ {pd.to_datetime(row['exp_back']).strftime('%b%d')}'{pd.to_datetime(row['exp_back']).strftime('%y')}"
                        report_date = (
                            f"{pd.to_datetime(row['reportedDate']).strftime('%b%d')}"
                        )
                        alert = (
                            f"{it}) {row['symbol']} - ({row['weeks']}W) {exp_dates}\n"
                            f"      Reported Date: {row['reportedDate'].strftime('%b%d')} {comment_symbol}\n"
                            f"      Comment: {row['comment']}\n"
                            f"      BDTE: {row['bdte']}\n"
                            f"      Strike: {row['strike']}\n"
                            f"      Underlying: {row['underlying']}\n"
                            f"      CalCost: {row['calCost']}\n"
                            f"      AvgCalCost: {row['avgCalCost']}\n"
                            f"      Grade: {row['Grade']}\n"
                            f"      CalGapPct: {row['calGapPct']}\n"
                            f"      Spread % (F/B): {row['spreadPct_front']} / {row['spreadPct_back']}\n"
                            f"      OI (F/B): {row['open_interest_front']} / {row['open_interest_back']}\n"
                            f"      IV (F/B/Diff): {row['implied_vol_front']} / {row['implied_vol_back']} / {row['iv_pct_diff']}\n"
                            f"{seperator}\n"
                        )
                        message += alert
                        it += 1
                        if len(message) > 3600:
                            messages.append(message)
                            message = ""
                if len(message) > 0:
                    messages.append(message)
                if len(messages) > 0:
                    messages[0] = (
                        f"🚨 {len(new_alerts)} Trade Alerts 🚨\n{seperator}\n"
                        + messages[0]
                    )
                    for message in messages:
                        message = escape_markdown_v2(message)
                        bot.send_message(chat_id, message, parse_mode="MarkdownV2")
                        logging.info("Alert sent to chat_id: %s", chat_id)

            if os.getenv("ENV") not in ["dev", "test"]:
                write_to_db(new_alerts, "thetaGradeAlerts", if_exists="append")
                # write_to_db(previous_alerts, "thetaGradeAlerts", if_exists="replace")
