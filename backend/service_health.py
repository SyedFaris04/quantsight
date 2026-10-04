"""Local saved-data readiness. This does not probe external services."""
import os
import re

import numpy as np
import pandas as pd


def release_id():
    value = os.getenv("RENDER_GIT_COMMIT", "")
    return value if re.fullmatch(r"[0-9a-fA-F]{7,40}", value) else "unknown"


def inspect_data(cache, expected_models):
    """Validate the minimum saved-data contract once, during startup."""
    problems = []
    predictions = cache.get("predictions", {})
    models = {}
    for name in expected_models:
        frame = predictions.get(name, pd.DataFrame())
        # LSTM exports omit actual_signal; track records join labels from features.
        required = {"date", "ticker", "predicted_signal", "confidence"}
        valid = not frame.empty and required.issubset(frame.columns)
        if valid:
            valid = (frame[list(required)].notna().all().all()
                     and not frame.duplicated(["ticker", "date"]).any()
                     and frame.predicted_signal.isin([0, 1]).all()
                     and ("actual_signal" not in frame or frame.actual_signal.isin([0, 1]).all())
                     and pd.to_numeric(frame.confidence, errors="coerce").between(0, 100).all())
        models[name] = {"rows": len(frame), "available": bool(valid)}
        if not valid:
            problems.append(f"predictions:{name}")
    features = cache.get("features", pd.DataFrame())
    required = {"date", "ticker", "Open", "High", "Low", "Close", "Volume"}
    valid_features = not features.empty and required.issubset(features.columns)
    if valid_features:
        numbers = features[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
        valid_features = (features[["date", "ticker"]].notna().all().all()
                          and not features.duplicated(["ticker", "date"]).any()
                          and np.isfinite(numbers.to_numpy()).all()
                          and numbers.Close.gt(0).all()
                          and set(cache.get("tickers", [])).issubset(set(features.ticker)))
    if not valid_features:
        problems.append("features:finance")
    if not cache.get("tickers"):
        problems.append("tickers")
    news = cache.get("news", pd.DataFrame())
    return {"ready": not problems, "problems": problems, "datasets": {
        "predictions": models,
        "features": {"rows": len(features), "available": bool(valid_features)},
        "news": {"rows": len(news), "available": not news.empty, "required": False},
    }}
