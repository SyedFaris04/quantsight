"""
backend/prediction_tracker.py
─────────────────────────────────────────────────────────────────────────────
Live prediction track record — the forward-looking counterpart to
copilot_engine.get_accuracy_track_record()'s historical backtest. Every
trading day, run_daily_job():

  1. Fetches a fresh live signal for every ticker (one yfinance call each,
     via live_signals.get_live_signal — the same function GET /live/{ticker}
     already uses).
  2. Resolves any prediction logged on an earlier day that's still pending,
     by comparing today's close against the price at prediction time —
     exactly the same next-day-direction labeling the models were trained
     on (see build_features.py's target definition).
  3. Logs today's signal as a new pending prediction, if one for this
     (ticker, date) doesn't already exist.

Writes go through the Supabase *service role* key, which bypasses Row Level
Security entirely — this is deliberate and the only way any row in
live_predictions ever gets written. No client-side code (browser, mobile,
whatever) can insert/update a row with the anon key; the table's only RLS
policy is public SELECT. That's what makes the resulting track record mean
something — nobody, including QuantSight's own frontend, can fake a result.

Triggered by a scheduled GitHub Actions workflow calling
POST /admin/run-daily-predictions (see .github/workflows/daily-predictions.yml),
since Render's free tier doesn't run background/cron processes on its own.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
from supabase import create_client

from live_signals import get_live_signal

logger = logging.getLogger("nuroquant-api")

_admin_client = None


def get_admin_client():
    """
    Service-role Supabase client — backend-only, never exposed to the
    frontend. Returns None (not raises) if unconfigured, so callers can
    degrade gracefully rather than crash the whole request.
    """
    global _admin_client
    if _admin_client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            return None
        _admin_client = create_client(url, key)
    return _admin_client


def run_daily_job(tickers: list[str]) -> dict:
    sb = get_admin_client()
    if sb is None:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set — the live "
            "prediction tracker needs the service role key (Supabase "
            "Settings → API → service_role secret), not the anon key."
        )

    logged, resolved, errors = 0, 0, []

    for ticker in tickers:
        try:
            sig = get_live_signal(ticker)
        except Exception as e:
            errors.append(f"{ticker}: {e}")
            continue

        if sig.get("source") != "live" or sig.get("signal_label") not in ("BUY", "SELL"):
            continue  # yfinance/model unavailable for this ticker today — skip, don't log garbage

        today_date  = sig["date"]
        today_close = sig["close_price"]

        # ── Resolve anything still pending from an earlier day ──────────────
        pending = (
            sb.table("live_predictions")
            .select("id, predicted_signal, price_at_prediction")
            .eq("ticker", ticker)
            .eq("resolved", False)
            .lt("predicted_date", today_date)
            .execute()
        ).data

        for p in pending:
            actual_signal = "BUY" if today_close > p["price_at_prediction"] else "SELL"
            correct = actual_signal == p["predicted_signal"]
            sb.table("live_predictions").update({
                "resolved"      : True,
                "resolved_date" : today_date,
                "actual_close"  : today_close,
                "actual_signal" : actual_signal,
                "correct"       : correct,
            }).eq("id", p["id"]).execute()
            resolved += 1

        # ── Log today's prediction (unique constraint makes this idempotent
        # if the job somehow runs twice the same day) ───────────────────────
        existing = (
            sb.table("live_predictions")
            .select("id")
            .eq("ticker", ticker)
            .eq("predicted_date", today_date)
            .execute()
        ).data
        if not existing:
            sb.table("live_predictions").insert({
                "ticker"              : ticker,
                "predicted_date"      : today_date,
                "predicted_signal"    : sig["signal_label"],
                "confidence"          : sig["confidence"],
                "price_at_prediction" : today_close,
            }).execute()
            logged += 1

    if errors:
        logger.warning(f"Live prediction job had {len(errors)} ticker errors: {errors[:5]}")

    return {"logged": logged, "resolved": resolved, "tickers_processed": len(tickers), "errors": errors}


def get_summary(days: int = 30) -> dict:
    """
    Aggregate live accuracy over resolved predictions in the last N days —
    used by GET /live-track-record and the chatbot's get_live_track_record
    tool. Reads through the same service-role client since it's already
    wired for writes; RLS would allow this via the anon key too (public
    SELECT), this just avoids standing up a second client.
    """
    sb = get_admin_client()
    if sb is None:
        return {"available": False, "reason": "Live tracker not configured yet."}

    rows = (
        sb.table("live_predictions")
        .select("ticker, predicted_date, predicted_signal, confidence, resolved, resolved_date, actual_signal, correct")
        .order("predicted_date", desc=True)
        .limit(500)
        .execute()
    ).data

    resolved_rows = [r for r in rows if r["resolved"]]
    pending_rows  = [r for r in rows if not r["resolved"]]
    correct_count = sum(1 for r in resolved_rows if r["correct"])

    return {
        "available"        : True,
        "total_logged"     : len(rows),
        "total_resolved"   : len(resolved_rows),
        "total_pending"    : len(pending_rows),
        "accuracy_pct"     : round(correct_count / len(resolved_rows) * 100, 1) if resolved_rows else None,
        "recent"           : rows[:20],
    }
