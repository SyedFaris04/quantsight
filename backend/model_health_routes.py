"""Read-only model diagnostics, separate from inference and database job routes."""
import logging
from datetime import timedelta
from fastapi import APIRouter, Query
from market_calendar import utc_now
from model_health_reference import load_reference
from model_health import build_health
import prediction_tracker

router = APIRouter()
logger = logging.getLogger(__name__)
FIELDS = ",".join(["id", "ticker", "predicted_date", "protocol_version", "model_version", "feature_version",
                   "calibration_method", "feature_values", "raw_probability_up", "probability_up", "predicted_signal",
                   "data_cutoff_at", "data_fetched_at", "generated_at", "created_at", "record_before"])


@router.get("/model-health")
def model_health(days: int = Query(30, ge=7, le=90)):
    try:
        reference = load_reference()
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        return {"available": False, "reason": "The historical input reference is unavailable or does not match current artifacts."}
    client = prediction_tracker.get_admin_client()
    if client is None:
        return {"available": False, "reason": "Saved forecast monitoring is not configured."}
    now = utc_now()
    since = str((now - timedelta(days=days)).date())
    try:
        rows = prediction_tracker.read_pages(client.table(prediction_tracker.TABLE).select(FIELDS)
                    .eq("protocol_version", prediction_tracker.PROTOCOL).gte("predicted_date", since)
                    .lte("predicted_date", str(now.date())).order("predicted_date", desc=True).order("id"))
    except Exception as exc:
        logger.warning("Model-health forecast read failed: %s", type(exc).__name__)
        return {"available": False, "reason": "Saved forecasts could not be read. Please retry."}
    try:
        return {"available": True, "days": days, "since_date": since, **build_health(rows, reference, now)}
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        logger.warning("Model-health integrity check failed: %s", type(exc).__name__)
        return {"available": False, "reason": "Forecast keys or versions could not be verified. Check the saved records."}
