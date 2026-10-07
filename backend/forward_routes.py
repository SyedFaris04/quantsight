"""Public forward evaluation routes, isolated from application startup dependencies."""
from fastapi import APIRouter, Query

import prediction_tracker

router = APIRouter()


@router.get("/live-track-record")
def live_track_record(days: int = Query(30, ge=7, le=365)):
    """Read-only rolling-window evidence and recent forecast records."""
    return prediction_tracker.get_summary(days=days)
