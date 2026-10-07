"""Descriptive forward evidence and guarded date-block uncertainty; no inference."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_calendar import BAR_DELAY, schedule, utc_now

PROTOCOL = json.loads(Path(__file__).with_name("forward_protocol.json").read_text(encoding="utf-8"))


def metrics(frame):
    if frame.empty:
        return {"rows": 0, "accuracy_pct": None, "always_up_accuracy_pct": None,
                "accuracy_difference_pp": None, "balanced_accuracy_pct": None,
                "brier_score": None, "flat_probability_brier": None,
                "brier_improvement_over_flat": None}
    y = frame.actual_signal.eq("BUY").to_numpy()
    predicted = frame.predicted_signal.eq("BUY").to_numpy()
    p = frame.probability_up.to_numpy(float)
    accuracy, up_rate = float(np.mean(predicted == y)), float(np.mean(y))
    balanced = float((np.mean(predicted[y]) + np.mean(~predicted[~y])) / 2) if y.any() and (~y).any() else None
    brier = float(np.mean((p - y) ** 2))
    return {"rows": len(frame), "accuracy_pct": 100 * accuracy,
            "always_up_accuracy_pct": 100 * up_rate, "accuracy_difference_pp": 100 * (accuracy - up_rate),
            "balanced_accuracy_pct": 100 * balanced if balanced is not None else None,
            "brier_score": brier, "flat_probability_brier": .25,
            "brier_improvement_over_flat": .25 - brier}


def interval_report(resolved, partial_dates, signatures):
    config = PROTOCOL["intervals"]
    result = {"available": False, "method": config["method"], "block_sessions": config["block_sessions"],
              "repetitions": config["repetitions"], "seed": config["seed"], "level": config["level"],
              "minimum_resolved_forecast_dates": config["minimum_resolved_forecast_dates"],
              "scope": config["scope"]}
    reasons = []
    if resolved.predicted_date.nunique() < config["minimum_resolved_forecast_dates"]:
        reasons.append(f"At least {config['minimum_resolved_forecast_dates']} distinct resolved forecast dates are required.")
    if partial_dates:
        reasons.append("Some forecast dates still have a mix of resolved and pending outcomes.")
    if len(signatures) != 1 or any("Unknown" in signature for signature in signatures):
        reasons.append("Intervals require one known model, feature and calibration version.")
    if reasons:
        return dict(result, reasons=reasons)
    # Never compress missing weekdays into consecutive observed dates.
    sessions = schedule(resolved.predicted_date.min(), resolved.predicted_date.max()).index.strftime("%Y-%m-%d")
    daily = resolved.assign(
        count=1, correct_sum=resolved.correct.astype(int), up_sum=resolved.actual_signal.eq("BUY").astype(int),
        loss_sum=(resolved.probability_up.astype(float) - resolved.actual_signal.eq("BUY")) ** 2,
    ).groupby("predicted_date")[["count", "correct_sum", "up_sum", "loss_sum"]].sum().reindex(sessions, fill_value=0)
    config_rng = np.random.Generator(np.random.PCG64(config["seed"]))
    span, block = len(daily), config["block_sessions"]
    starts = config_rng.integers(0, span, size=(config["repetitions"], int(np.ceil(span / block))))
    indices = ((starts[:, :, None] + np.arange(block)) % span).reshape(config["repetitions"], -1)[:, :span]
    sums = daily.to_numpy(float)[indices].sum(axis=1)
    valid = sums[:, 0] > 0
    if int(valid.sum()) < config["minimum_valid_repetitions"]:
        return dict(result, reasons=["Too many bootstrap draws contained no resolved forecasts."], valid_repetitions=int(valid.sum()))
    estimates = {"accuracy_pct": 100 * sums[valid, 1] / sums[valid, 0],
                 "accuracy_difference_pp": 100 * (sums[valid, 1] - sums[valid, 2]) / sums[valid, 0],
                 "brier_score": sums[valid, 3] / sums[valid, 0]}
    alpha = (1 - config["level"]) / 2
    return dict(result, available=True, reasons=[], session_span=span, valid_repetitions=int(valid.sum()),
                bounds={key: np.quantile(values, [alpha, 1 - alpha]).tolist() for key, values in estimates.items()})


def build_evidence(rows, now=None):
    now = utc_now(now)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["ticker", "predicted_date", "resolved", "correct", "actual_signal",
                                      "predicted_signal", "probability_up"])
    if frame[["ticker", "predicted_date", "resolved"]].isna().any().any() or frame.duplicated(["ticker", "predicted_date"]).any():
        raise ValueError("Missing or duplicate forecast keys")
    if not frame.resolved.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("Invalid resolution flags")
    if not frame.empty:
        dates = pd.to_datetime(frame.predicted_date, format="%Y-%m-%d", errors="raise")
        session_dates = schedule(frame.predicted_date.min(), frame.predicted_date.max()).index
        if dates.isna().any() or not dates.isin(session_dates).all():
            raise ValueError("Forecast dates are not NYSE sessions")
    resolved = frame[frame.resolved.eq(True)].copy()
    if not resolved.empty:
        if not resolved.actual_signal.isin(["BUY", "SELL"]).all() or not resolved.predicted_signal.isin(["BUY", "SELL"]).all():
            raise ValueError("Invalid resolved direction")
        if not resolved.correct.map(lambda x: isinstance(x, (bool, np.bool_))).all() or not resolved.correct.eq(resolved.actual_signal.eq(resolved.predicted_signal)).all():
            raise ValueError("Stored correctness disagrees with recorded directions")
        values = resolved.probability_up.to_numpy(float)
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError("Invalid resolved probability")
    for column in ["model_version", "feature_version", "calibration_method"]:
        resolved[column] = resolved.get(column, pd.Series(index=resolved.index, dtype=str)).replace("", None).fillna("Unknown")
    groups, signatures = [], []
    for signature, part in resolved.groupby(["model_version", "feature_version", "calibration_method"], sort=True):
        signatures.append(signature)
        groups.append({"model_version": signature[0], "feature_version": signature[1], "calibration_method": signature[2],
                       "forecast_dates": int(part.predicted_date.nunique()), "first_date": part.predicted_date.min(),
                       "last_date": part.predicted_date.max(), "metrics": metrics(part)})
    cohorts, partial_dates = [], []
    for date, part in frame.groupby("predicted_date", sort=True):
        outcomes = part[part.resolved.eq(True)]
        pending = len(part) - len(outcomes)
        status = "complete" if not pending else "partial" if len(outcomes) else "pending"
        if status == "partial":
            partial_dates.append(date)
        cohorts.append({"date": date, "logged": len(part), "resolved": len(outcomes), "pending": pending,
                        "status": status, "metrics": metrics(outcomes)})
    bins = []
    for i in range(5):
        lower, upper = i / 5, (i + 1) / 5
        part = resolved[resolved.probability_up.ge(lower) &
                        (resolved.probability_up.le(upper) if i == 4 else resolved.probability_up.lt(upper))]
        bins.append({"lower": lower, "upper": upper, "rows": len(part),
                     "mean_probability_up": float(part.probability_up.mean()) if len(part) else None,
                     "actual_up_fraction": float(part.actual_signal.eq("BUY").mean()) if len(part) else None})
    overdue = 0
    for row in frame[frame.resolved.eq(False)].to_dict("records"):
        if row.get("target_close_at"):
            overdue += int(utc_now(row["target_close_at"]) + BAR_DELAY <= now)
    # The existing scheduler is due at 22:00 UTC. Allow one hour for queueing
    # and execution before flagging an unrecorded session. This is a data-gap
    # diagnostic, not a claim that GitHub's scheduler necessarily failed.
    collection = {"available": False, "expected_sessions": None, "recorded_sessions": None,
                  "missing_sessions": [], "through_session": None,
                  "rule": "NYSE sessions since the first record in this window, due after 23:00 UTC (22:00 scheduler plus one-hour grace). Missing records do not identify the failure cause."}
    if not frame.empty:
        expected_calendar = schedule(frame.predicted_date.min(), str(now.date())).index
        deadlines = expected_calendar.tz_localize("UTC") + pd.Timedelta(hours=23)
        expected_dates = expected_calendar[deadlines <= now].strftime("%Y-%m-%d").tolist()
        missing = sorted(set(expected_dates) - set(frame.predicted_date))
        collection.update(available=True, expected_sessions=len(expected_dates),
                          recorded_sessions=len(set(expected_dates) & set(frame.predicted_date)),
                          missing_sessions=missing, through_session=expected_dates[-1] if expected_dates else None)
    return {"version": PROTOCOL["version"], "protocol": PROTOCOL, "metrics": metrics(resolved),
            "logged_forecast_dates": int(frame.predicted_date.nunique()),
            "resolved_forecast_dates": int(resolved.predicted_date.nunique()),
            "first_resolved_date": resolved.predicted_date.min() if len(resolved) else None,
            "last_resolved_date": resolved.predicted_date.max() if len(resolved) else None,
            "latest_logged_date": frame.predicted_date.max() if len(frame) else None,
            "partial_forecast_dates": partial_dates, "overdue_pending": overdue,
            "collection": collection,
            "model_groups": groups, "mixed_versions": len(signatures) > 1,
            "cohorts": cohorts, "probability_bins": bins,
            "intervals": interval_report(resolved, partial_dates, signatures)}
