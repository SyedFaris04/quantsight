"""Version 2 forward evaluation. Legacy rows remain in live_predictions.

A forecast uses a completed close and is recorded before the next open.
Its outcome is the close exactly five NYSE sessions later, even if the job
misses a day. Both outcome prices are retrieved on the same adjusted basis.
"""
import os
import logging
from datetime import timedelta
import numpy as np
import pandas as pd
from live_signals import get_live_signal
from market_calendar import utc_now, latest_completed_session, forecast_window, eligible_to_record
from prediction_contract import LABEL_HORIZON_TRADING_DAYS
from forward_metrics import build_evidence

logger = logging.getLogger("nuroquant-api")
TABLE = "live_predictions_v2"
PROTOCOL = "nyse-close-5-v2"
SUMMARY_FIELDS = ",".join([
    "id", "ticker", "predicted_date", "target_date", "target_close_at", "horizon_sessions",
    "protocol_version", "model_version", "feature_version", "calibration_method",
    "predicted_signal", "confidence", "probability_up", "raw_probability_up",
    "created_at", "data_cutoff_at", "generated_at", "record_before", "price_basis",
    "resolved", "actual_signal", "correct", "actual_return", "resolved_at",
])
_admin_client = None


def get_admin_client():
    global _admin_client
    if _admin_client is None:
        url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            return None
        from supabase import create_client
        _admin_client = create_client(url, key)
    return _admin_client


def read_pages(query):
    """Read all pages in stable order; never silently cap the accuracy sample."""
    rows, offset = [], 0
    while True:
        page = query.range(offset, offset + 499).execute().data or []
        rows.extend(page)
        if not page:
            return rows
        offset += len(page)


def fetch_outcome_prices(ticker, start, end):
    import yfinance as yf
    prices = yf.download(ticker, start=start,
                         end=str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()),
                         auto_adjust=True, progress=False, threads=False)
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    if prices.empty:
        return pd.Series(dtype=float)
    close = prices["Close"].copy()
    close.index = pd.to_datetime(close.index).strftime("%Y-%m-%d")
    return close


def outcome_for(row, prices):
    """Do not substitute a later close for a missing target-session quote."""
    start, target = row["predicted_date"], row["target_date"]
    if start not in prices.index or target not in prices.index:
        return None
    entry, exit_price = float(prices.loc[start]), float(prices.loc[target])
    if not np.isfinite([entry, exit_price]).all() or min(entry, exit_price) <= 0:
        return None
    actual = "BUY" if exit_price > entry else "SELL"
    return {"resolved": True, "resolved_date": target, "actual_close": exit_price,
            "resolution_entry_close": entry, "actual_return": exit_price / entry - 1,
            "actual_signal": actual, "correct": actual == row["predicted_signal"]}


def run_daily_job(tickers, now=None):
    sb = get_admin_client()
    if sb is None:
        raise RuntimeError("Live tracker requires backend Supabase configuration.")
    supplied_now = now
    now = utc_now(now)
    completed = latest_completed_session(now)
    # Fail explicitly before fetching the universe if the migration is missing.
    try:
        pending = read_pages(sb.table(TABLE).select("*").eq("resolved", False)
                             .lte("target_date", completed).order("id"))
    except Exception as exc:
        logger.warning("Tracker schema/read unavailable: %s", type(exc).__name__)
        raise RuntimeError("Live tracker v2 unavailable. Apply 002_live_predictions_v2.sql and check database connectivity.") from exc
    counts = {"logged": 0, "resolved": 0, "duplicates": 0, "skipped": 0,
              "tickers_processed": len(tickers), "errors": [], "protocol_version": PROTOCOL}
    # Resolution is independent of whether new inference is available today.
    for ticker in sorted({r["ticker"] for r in pending}):
        rows = [r for r in pending if r["ticker"] == ticker]
        try:
            prices = fetch_outcome_prices(ticker, min(r["predicted_date"] for r in rows),
                                          max(r["target_date"] for r in rows))
            for row in rows:
                update = outcome_for(row, prices)
                if update is None:
                    counts["errors"].append(f"{ticker}: missing exact outcome prices for {row['target_date']}")
                    continue
                update["resolved_at"] = utc_now().isoformat()
                result = sb.table(TABLE).update(update).eq("id", row["id"]).eq("resolved", False).execute()
                counts["resolved"] += len(result.data or [])
        except Exception as exc:
            logger.warning("Outcome resolution failed for %s: %s", ticker, type(exc).__name__)
            counts["errors"].append(f"{ticker}: outcome resolution failed")
    if not eligible_to_record(completed, now):
        counts["skipped"] = len(tickers)
        counts["skip_reason"] = "Outside the after-close, before-next-open recording window."
        counts["status"] = "partial_failure" if counts["errors"] else "skipped"
        return counts
    for ticker in tickers:
        try:
            sig = get_live_signal(ticker)
            if sig.get("source") != "live" or sig.get("date") != completed or sig.get("signal_label") not in ("BUY", "SELL"):
                counts["skipped"] += 1
                counts["errors"].append(f"{ticker}: current complete signal unavailable")
                continue
            # Inference may have taken long enough to cross the next open.
            record_time = utc_now(supplied_now)
            if not eligible_to_record(completed, record_time):
                counts["skipped"] += 1
                continue
            window = forecast_window(completed, LABEL_HORIZON_TRADING_DAYS)
            row = {"ticker": ticker, "predicted_date": completed,
                   "target_date": window["target_date"], "horizon_sessions": LABEL_HORIZON_TRADING_DAYS,
                   "target_close_at": window["target_close_at"],
                   "protocol_version": PROTOCOL, "model_version": sig["model_version"],
                   "feature_version": sig["feature_version"], "data_cutoff_at": window["data_cutoff_at"],
                   "feature_values": sig["feature_values"],
                   "record_before": window["record_before"], "generated_at": sig["generated_at"],
                   "data_fetched_at": sig["data_fetched_at"],
                   "predicted_signal": sig["signal_label"], "confidence": sig["direction_confidence"],
                   "probability_up": sig["probability_up"], "raw_probability_up": sig["raw_probability_up"],
                   "calibration_method": sig["calibration_method"],
                   "price_basis": "yahoo_auto_adjust", "price_at_prediction": sig["close_price"]}
            result = sb.table(TABLE).upsert(row, on_conflict="ticker,predicted_date,protocol_version",
                                           ignore_duplicates=True).execute()
            inserted = len(result.data or [])
            counts["logged"] += inserted
            counts["duplicates"] += int(not inserted)
        except Exception as exc:
            logger.warning("Forecast recording failed for %s: %s", ticker, type(exc).__name__)
            counts["errors"].append(f"{ticker}: forecast recording failed")
    counts["status"] = "partial_failure" if counts["errors"] else "ok"
    return counts


def get_summary(days=30):
    sb = get_admin_client()
    if sb is None:
        return {"available": False, "reason": "Live tracker is not configured."}
    now = utc_now()
    since = str((now - timedelta(days=days)).date())
    try:
        rows = read_pages(sb.table(TABLE).select(SUMMARY_FIELDS).gte("predicted_date", since)
                          .lte("predicted_date", str(now.date()))
                          .eq("protocol_version", PROTOCOL).order("predicted_date", desc=True).order("id"))
    except Exception as exc:
        logger.warning("Live tracker read failed: %s", type(exc).__name__)
        return {"available": False, "reason": "Live tracker v2 unavailable. Database migration or connectivity needs attention."}
    resolved = [r for r in rows if r["resolved"]]
    try:
        evidence = build_evidence(rows, now)
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning("Forward evidence integrity failure: %s", type(exc).__name__)
        return {"available": False, "reason": "Forward records could not be verified. Check recorded directions, probabilities and forecast dates."}
    correct = sum(bool(r["correct"]) for r in resolved)
    accuracy = correct / len(resolved) if resolved else None
    always_up = sum(r["actual_signal"] == "BUY" for r in resolved) / len(resolved) if resolved else None
    brier = sum((float(r["probability_up"]) - (r["actual_signal"] == "BUY"))**2 for r in resolved) / len(resolved) if resolved else None
    return {"available": True, "days": days, "since_date": since, "as_of": now.isoformat(),
            "protocol_version": PROTOCOL, "horizon_sessions": LABEL_HORIZON_TRADING_DAYS,
            "legacy_excluded": True, "total_logged": len(rows), "total_resolved": len(resolved),
            "total_pending": len(rows) - len(resolved),
            "accuracy_pct": round(accuracy * 100, 1) if accuracy is not None else None,
            "always_up_accuracy_pct": round(always_up * 100, 1) if always_up is not None else None,
            "brier_score": round(brier, 4) if brier is not None else None,
            "evidence": evidence,
            "recent_resolved": resolved[:20],
            "recent_pending": [r for r in rows if not r["resolved"]][:20],
            "recent": rows[:20]}
