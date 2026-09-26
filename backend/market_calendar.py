"""NYSE sessions for daily close forecasts; all instants are timezone aware."""
from functools import lru_cache
import pandas as pd
import pandas_market_calendars as mcal

BAR_DELAY = pd.Timedelta(minutes=20)


def utc_now(now=None):
    value = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if value.tzinfo is None:
        raise ValueError("An explicit timezone is required")
    return value.tz_convert("UTC")


@lru_cache(maxsize=64)
def schedule(start, end):
    return mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)


def latest_completed_session(now=None):
    now = utc_now(now)
    sessions = schedule(str((now - pd.Timedelta(days=30)).date()), str(now.date()))
    complete = sessions[sessions.market_close + BAR_DELAY <= now]
    if complete.empty:
        raise ValueError("No completed session available")
    return complete.index[-1].strftime("%Y-%m-%d")


def forecast_window(session_date, horizon=5):
    day = pd.Timestamp(session_date).normalize()
    sessions = schedule(str(day.date()), str((day + pd.Timedelta(days=40)).date()))
    if day not in sessions.index or horizon < 1 or horizon >= len(sessions):
        raise ValueError("Invalid forecast session or horizon")
    return {
        "target_date": sessions.index[horizon].strftime("%Y-%m-%d"),
        "target_close_at": sessions.iloc[horizon].market_close.isoformat(),
        "data_cutoff_at": sessions.iloc[0].market_close.isoformat(),
        "earliest_record_at": (sessions.iloc[0].market_close + BAR_DELAY).isoformat(),
        "record_before": sessions.iloc[1].market_open.isoformat(),
    }


def eligible_to_record(session_date, now=None):
    now = utc_now(now)
    window = forecast_window(session_date)
    return pd.Timestamp(window["earliest_record_at"]) <= now < pd.Timestamp(window["record_before"])


def news_session(available_at):
    """First session whose close is at/after a known availability timestamp."""
    instant = utc_now(available_at)
    sessions = schedule(str((instant - pd.Timedelta(days=1)).date()),
                        str((instant + pd.Timedelta(days=14)).date()))
    return sessions[sessions.market_close >= instant].index[0].strftime("%Y-%m-%d")
