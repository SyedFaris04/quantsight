"""Exact native TreeSHAP for the latest saved XGBoost predictions.

Only trusted, repository-owned model bundles are loaded. Contributions explain
the binary classifier's raw log-odds, before the separate calibration step.
"""
import hashlib
import json
import logging
import pickle
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from cv_utils import apply_calibrator

DATA = Path(__file__).resolve().parent / "data"
VARIANTS = ("xgb_finance", "xgb_sentiment")
router = APIRouter()
_load_lock = Lock()
logger = logging.getLogger(__name__)


class AttributionUnavailable(ValueError):
    """Evidence cannot be verified against the saved prediction."""


def explain_row(bundle, row, prediction, model_sha256):
    """Reconstruct one output and refuse mismatched or incomplete evidence."""
    import xgboost as xgb

    features = list(bundle["features"])
    if not features or len(set(features)) != len(features):
        raise AttributionUnavailable("Invalid model feature list.")
    if any(name not in row for name in features):
        raise AttributionUnavailable("The saved input is missing model features.")
    values = np.asarray([row[name] for name in features], dtype=np.float64)
    if not np.isfinite(values).all():
        raise AttributionUnavailable("The saved input contains non-finite values.")
    transformed = bundle["scaler"].transform(values.reshape(1, -1))
    if transformed.shape != (1, len(features)) or not np.isfinite(transformed).all():
        raise AttributionUnavailable("The transformed input is invalid.")
    model = bundle["model"]
    booster = model.get_booster()
    best = getattr(model, "best_iteration", None)
    iterations = (0, int(best) + 1) if best is not None else (0, 0)
    matrix = xgb.DMatrix(transformed, feature_names=booster.feature_names)
    contributions = booster.predict(
        matrix, pred_contribs=True, approx_contribs=False, iteration_range=iterations,
    )[0].astype(np.float64)
    margin = float(booster.predict(matrix, output_margin=True, iteration_range=iterations)[0])
    raw_probability = float(model.predict_proba(transformed)[0, 1])
    if contributions.shape != (len(features) + 1,) or not np.isfinite(contributions).all():
        raise AttributionUnavailable("The model returned invalid contributions.")
    reconstructed = float(contributions.sum())
    margin_error = abs(reconstructed - margin)
    # Tree traversal sums have float32 rounding error; do not round the evidence.
    if not np.isclose(reconstructed, margin, atol=1e-5, rtol=1e-5):
        raise AttributionUnavailable("The contributions do not reconstruct the model score.")
    sigmoid = float(1 / (1 + np.exp(-np.clip(margin, -700, 700))))
    if not np.isclose(sigmoid, raw_probability, atol=1e-6, rtol=1e-6):
        raise AttributionUnavailable("The explanation does not match classifier inference.")
    calibrated = float(apply_calibrator(
        bundle["calibrator"], [raw_probability], bundle["calibration_method"],
    )[0])
    saved_probability = float(prediction["confidence"]) / 100
    direction = int(raw_probability >= 0.5)
    # CSV confidence is a percentage rounded to two decimal places (0.00005 in P).
    if (not np.isfinite(calibrated) or not 0 <= calibrated <= 1
            or not np.isfinite(saved_probability)
            or abs(calibrated - saved_probability) > 0.00005001
            or direction != int(prediction["predicted_signal"])):
        raise AttributionUnavailable("Model and input no longer reproduce the saved prediction.")
    inputs = [{"feature": name, "value": float(value),
               "scaled_value": float(transformed[0, index]),
               "contribution": float(contributions[index])}
              for index, (name, value) in enumerate(zip(features, values))]
    input_hash = hashlib.sha256(json.dumps(
        {"features": features, "values": values.tolist()},
        separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    return {
        "ticker": str(prediction["ticker"]), "date": str(prediction["date"]),
        "source": "historical_saved_prediction", "method": "XGBoost native TreeSHAP",
        "scale": "raw_log_odds", "model_sha256": model_sha256,
        "input_sha256": input_hash, "xgboost_version": xgb.__version__,
        "trees_used": iterations[1] or booster.num_boosted_rounds(),
        "calibration_method": bundle["calibration_method"],
        "baseline": float(contributions[-1]), "raw_margin": margin,
        "contribution_sum": float(contributions[:-1].sum()),
        "reconstructed_margin": reconstructed, "reconstruction_error": margin_error,
        "raw_probability_up": raw_probability, "calibrated_probability_up": calibrated,
        "saved_probability_up": saved_probability, "direction": "BUY" if direction else "SELL",
        "verified": True, "features": inputs,
    }


def _paths(variant):
    return (DATA / "models" / f"{variant}.pkl",
            DATA / "processed" / f"features_{variant.removeprefix('xgb_')}.csv",
            DATA / "predictions" / f"{variant}_predictions.csv")


@lru_cache(maxsize=2)
def _latest_evidence(variant, signatures):
    """Cache only the latest input per ticker, not another full feature panel."""
    model_path, feature_path, prediction_path = _paths(variant)
    model_bytes = model_path.read_bytes()
    bundle = pickle.loads(model_bytes)
    predictions = pd.read_csv(prediction_path, dtype={"ticker": str, "date": str})
    if predictions.duplicated(["ticker", "date"]).any():
        raise AttributionUnavailable("Duplicate saved predictions.")
    latest = predictions.sort_values("date").groupby("ticker", sort=False).tail(1)
    keys = set(zip(latest.ticker, latest.date))
    rows = {}
    for chunk in pd.read_csv(feature_path, usecols=["ticker", "date", *bundle["features"]],
                             dtype={"ticker": str, "date": str}, chunksize=10000):
        matches = chunk.loc[[(t, d) in keys for t, d in zip(chunk.ticker, chunk.date)]]
        for record in matches.to_dict("records"):
            key = (record["ticker"], record["date"])
            if key in rows:
                raise AttributionUnavailable("Duplicate feature inputs.")
            rows[key] = record
    # Refuse files changed during loading rather than cache a mixed generation.
    if signatures != _signatures(variant):
        raise AttributionUnavailable("Evidence files changed during loading. Retry the request.")
    return bundle, hashlib.sha256(model_bytes).hexdigest(), latest.set_index("ticker", drop=False), rows


def _signatures(variant):
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in _paths(variant))


def get_attribution(ticker, variant="xgb_finance"):
    if variant not in VARIANTS:
        raise AttributionUnavailable("Only the two saved XGBoost models are supported.")
    with _load_lock:
        bundle, digest, latest, rows = _latest_evidence(variant, _signatures(variant))
    ticker = ticker.upper()
    if ticker not in latest.index:
        raise KeyError(ticker)
    prediction = latest.loc[ticker].to_dict()
    row = rows.get((ticker, prediction["date"]))
    if row is None:
        raise AttributionUnavailable("No exact feature row exists for the saved prediction date.")
    result = explain_row(bundle, row, prediction, digest)
    return {**result, "model": variant}


@router.get("/feature-attribution/{ticker}")
def feature_attribution(ticker: str, model: Literal["xgb_finance", "xgb_sentiment"] = "xgb_finance"):
    try:
        return get_attribution(ticker, model)
    except KeyError as exc:
        raise HTTPException(404, "No saved prediction exists for this ticker.") from exc
    except Exception as exc:
        logger.exception("Feature attribution unavailable for %s / %s", ticker, model)
        raise HTTPException(503, "Verified feature attribution is unavailable. Please retry later.") from exc
