"""Descriptive monitoring of recorded inputs and probabilities; no inference or writes."""
from collections import defaultdict
import math
import numpy as np
import pandas as pd
from market_calendar import utc_now, latest_completed_session, forecast_window, eligible_to_record, schedule
from model_health_reference import PROTOCOL


def finite_number(value):
    return (isinstance(value, (int, float, np.integer, np.floating)) and
            not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value)))


def valid_probability(value):
    return finite_number(value) and 0 <= value <= 1


def probability_summary(values):
    config = PROTOCOL["probability_flag"]
    if not values:
        return {"n": 0, "min": None, "max": None, "span": None, "mean": None, "flat": None,
                "distinct_values": 0, "largest_tie_count": 0, "bins": []}
    values = np.asarray(values, dtype=float)
    edges = config["histogram_edges"]
    counts, _ = np.histogram(values, bins=edges)
    span = float(values.max() - values.min())
    unique, tie_counts = np.unique(values, return_counts=True)
    return {"n": len(values), "min": float(values.min()), "max": float(values.max()),
            "span": span, "mean": float(values.mean()),
            "distinct_values": len(unique), "largest_tie_count": int(tie_counts.max()),
            "flat": bool(span <= config["maximum_flat_span"] + 1e-12) if len(values) >= config["minimum_forecasts"] else None,
            "bins": [{"lower": a, "upper": b, "count": int(n)} for a, b, n in zip(edges[:-1], edges[1:], counts)]}


def timestamp_problem(row, now):
    try:
        window = forecast_window(row["predicted_date"])
        fields = ["data_cutoff_at", "data_fetched_at", "generated_at", "created_at", "record_before"]
        if any(row.get(field) is None or row.get(field) == "" for field in fields):
            return True
        cutoff, fetched, generated, created, deadline = [utc_now(row[field]) for field in fields]
        return not (cutoff == utc_now(window["data_cutoff_at"]) and
                    deadline == utc_now(window["record_before"]) and
                    cutoff <= fetched <= generated <= created < deadline and
                    utc_now(window["earliest_record_at"]) <= created <= now)
    except (ValueError, TypeError, KeyError, OverflowError):
        return True


def group_summary(rows, reference, now):
    first = rows[0]
    features = reference["features"]
    supported = first["model_version"] == reference["model_sha256"]
    quality = {"unknown_ticker_rows": 0, "invalid_probability_rows": 0,
               "direction_mismatch_rows": 0, "timestamp_problem_rows": 0,
               "input_problem_rows": 0, "missing_or_nonfinite_feature_cells": 0}
    raw, calibrated, directions = [], [], []
    comparable = []
    for row in rows:
        ticker_known = row["ticker"] in reference["tickers"]
        probability_valid = valid_probability(row.get("raw_probability_up")) and valid_probability(row.get("probability_up"))
        direction_valid = row.get("predicted_signal") in ("BUY", "SELL") and probability_valid and row["predicted_signal"] == ("BUY" if row["raw_probability_up"] >= .5 else "SELL")
        timing_valid = not timestamp_problem(row, now)
        quality["unknown_ticker_rows"] += int(not ticker_known)
        quality["invalid_probability_rows"] += int(not probability_valid)
        quality["direction_mismatch_rows"] += int(probability_valid and not direction_valid)
        quality["timestamp_problem_rows"] += int(not timing_valid)
        values = row.get("feature_values")
        values = values if isinstance(values, dict) else {}
        if supported:
            missing = sum(not finite_number(values.get(name)) for name in features)
            quality["missing_or_nonfinite_feature_cells"] += missing
            quality["input_problem_rows"] += int(missing > 0)
        if ticker_known and timing_valid and direction_valid:
            raw.append(float(row["raw_probability_up"]))
            calibrated.append(float(row["probability_up"]))
            directions.append(row["predicted_signal"])
            if supported:
                comparable.append((row["ticker"], values))
    raw_stats, calibrated_stats = probability_summary(raw), probability_summary(calibrated)
    if calibrated_stats["flat"] is True and raw_stats["flat"] is False:
        probability_status = "calibration_compression"
    elif calibrated_stats["flat"] is True or raw_stats["flat"] is True:
        probability_status = "limited_spread"
    elif raw_stats["flat"] is None:
        probability_status = "insufficient_sample"
    else:
        probability_status = "no_spread_flag"
    range_config = PROTOCOL["range_flag"]
    input_stats = []
    for name in features if supported else []:
        observations = [(ticker, float(values[name])) for ticker, values in comparable if finite_number(values.get(name))]
        outside = sum(value < reference["tickers"][ticker]["features"][name]["quantiles"][0] or
                      value > reference["tickers"][ticker]["features"][name]["quantiles"][-1]
                      for ticker, value in observations)
        n = len(observations)
        fraction = outside / n if n else None
        enough = n >= range_config["minimum_finite_forecasts"]
        flag = bool(fraction >= range_config["minimum_outside_fraction"]) if enough else None
        input_stats.append({"feature": name, "n": n, "outside": outside,
                            "outside_pct": 100 * fraction if fraction is not None else None,
                            "review": flag})
    input_stats.sort(key=lambda item: (item["review"] is True, item["outside_pct"] or 0, item["feature"]), reverse=True)
    return {"date": first["predicted_date"], "model_version": first["model_version"],
            "feature_version": first["feature_version"], "calibration_method": first["calibration_method"],
            "recorded": len(rows), "expected": len(reference["tickers"]),
            "complete_cohort": {r["ticker"] for r in rows} == set(reference["tickers"]),
            "reference_supported": supported, "quality": quality,
            "raw_probability": raw_stats, "calibrated_probability": calibrated_stats,
            "probability_status": probability_status,
            "buy_count": directions.count("BUY"), "sell_count": directions.count("SELL"),
            "calibrated_half_disagreements": sum((p >= .5) != (d == "BUY") for p, d in zip(calibrated, directions)),
            "input_ranges": input_stats, "range_review_count": sum(s["review"] is True for s in input_stats)}


def build_health(rows, reference, now=None):
    now = utc_now(now)
    completed = latest_completed_session(now)
    groups = defaultdict(list)
    seen = set()
    if rows:
        dates = [r["predicted_date"] for r in rows]
        sessions = set(schedule(min(dates), max(dates)).index.strftime("%Y-%m-%d"))
        for row in rows:
            key = (row["ticker"], row["predicted_date"])
            if key in seen or row["predicted_date"] not in sessions:
                raise ValueError("Duplicate or non-session forecast keys")
            seen.add(key)
            signature = tuple(row.get(field) for field in PROTOCOL["grouping"])
            if any(not isinstance(value, str) or not value or len(value) > 160 for value in signature):
                raise ValueError("Missing forecast version metadata")
            groups[signature].append(row)
    summaries = [group_summary(part, reference, now) for _, part in sorted(groups.items(), reverse=True)]
    expected = set(reference["tickers"])
    latest_tickers = {row["ticker"] for row in rows if row["predicted_date"] == completed} & expected
    latest_recorded = max((r["predicted_date"] for r in rows), default=None)
    return {"protocol_id": PROTOCOL["protocol_id"], "as_of": now.isoformat(), "source": "saved_forecasts",
            "latest_completed_session": completed, "latest_recorded_date": latest_recorded,
            "latest_session_coverage": len(latest_tickers), "expected_tickers": len(expected),
            "recording_window_open": eligible_to_record(completed, now),
            "records": len(rows), "forecast_dates": len({r["predicted_date"] for r in rows}),
            "groups": summaries, "rules": {"range_flag": PROTOCOL["range_flag"], "probability_flag": PROTOCOL["probability_flag"]},
            "reference": {"description": PROTOCOL["reference"]["description"], "start": reference["start"],
                          "end": reference["end"], "rows": reference["rows"], "features": len(reference["features"]),
                          "tickers": len(expected), "model_sha256": reference["model_sha256"],
                          "source_sha256": reference["source_sha256"], "protocol_sha256": reference["protocol_sha256"]}}
