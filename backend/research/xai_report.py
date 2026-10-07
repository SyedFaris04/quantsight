"""Prepared XAI evidence validation; standard library only in the API."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "backend/data/research_reports/xai_comparison.json"
EXPECTED_SOURCES = {
    "backend/research/protocols/xai_comparison_v1.json", "backend/research/xai_comparison.py",
    "backend/research/xai_metrics.py", "backend/train_lstm.py", "backend/cv_utils.py",
    "backend/data/models/xgb_finance.pkl", "backend/data/models/lstm_finance.pt",
    "backend/data/processed/features_finance.csv",
    "backend/data/predictions/xgb_finance_predictions.csv", "backend/data/predictions/lstm_finance_predictions.csv",
}
METHODS = {"xgb_finance": {"native_treeshap", "lime_historical_neighborhood"},
           "lstm_finance": {"integrated_gradients", "day_window_occlusion"}}


def source_digest(path):
    if path.suffix in {".pkl", ".pt"}:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    # Git may convert LF/CRLF on checkout; values must still match exactly.
    with path.open("r", encoding="utf-8", newline=None) as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), ""):
            digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


@lru_cache(maxsize=32)
def _cached_digest(path, modified_ns, size):
    return source_digest(Path(path))


def summarize(rows):
    summaries = []
    for model, keys in METHODS.items():
        subset = [row for row in rows if row["model"] == model]
        for key in sorted(keys):
            pairs = [(row, next(m for m in row["methods"] if m["key"] == key)) for row in subset]
            methods = [m for _, m in pairs]
            def average(field):
                values = [m[field] for m in methods if m.get(field) is not None]
                return mean(values) if values else None
            summaries.append({"model": model, "method": key, "name": methods[0]["name"], "n": len(methods),
                              "removal_response": average("removal_response"),
                              "random_response": mean(row["random_response"] for row, _ in pairs),
                              "response_above_random": mean(m["removal_response"] - row["random_response"] for row, m in pairs),
                              "alternative_removal_response": average("alternative_removal_response"),
                              "runtime_ms": average("runtime_ms"), "reference_top3_overlap": average("reference_top3_overlap"),
                              "seed_top3_overlap": average("seed_top3_overlap"), "weighted_r2": average("weighted_r2"),
                              "point_error": average("point_error"),
                              "max_completeness_error": max((m["completeness_error"] for m in methods if m["completeness_error"] is not None), default=None)})
    return summaries


def validate(report):
    def require(condition):
        if not condition: raise ValueError("Invalid XAI comparison evidence.")
    def finite(value):
        require(type(value) in (int, float) and math.isfinite(value))
    def probability(value):
        finite(value)
        require(0 <= value <= 1)
    require(report["schema_version"] == 1 and report["status"] == "exploratory")
    require(report["holdout_files_opened"] is False and report["serving_models_changed"] is False)
    require(report["scale"] == "raw_log_odds" and report["protocol"]["protocol_id"] == "xai_comparison_v1")
    require(len(report["sources"]) == len(EXPECTED_SOURCES) and {s["path"] for s in report["sources"]} == EXPECTED_SOURCES)
    for source in report["sources"]:
        require(len(source["sha256"]) == 64 and all(c in "0123456789abcdef" for c in source["sha256"]))
    require(len(report["rows"]) == 88)
    instruments = {model: set() for model in METHODS}
    for row in report["rows"]:
        require(row["model"] in METHODS and row["date"] == report["prediction_date"])
        require(row["ticker"] not in instruments[row["model"]])
        instruments[row["model"]].add(row["ticker"])
        require(len(row["units"]) == (10 if row["model"] == "lstm_finance" else 30))
        require(len(set(row["units"])) == len(row["units"]))
        probability(row["raw_probability_up"])
        probability(row["calibrated_probability_up"])
        probability(row["saved_probability_up"])
        require(abs(row["saved_probability_up"] - row["calibrated_probability_up"]) <= .00005001)
        finite(row["raw_logit"])
        require(math.isclose(1 / (1 + math.exp(-row["raw_logit"])), row["raw_probability_up"], abs_tol=1e-6))
        threshold = row.get("decision_threshold", .5)
        probability(threshold)
        require(row["direction"] == ("BUY" if row["raw_probability_up"] >= threshold else "SELL"))
        if row["model"] == "lstm_finance":
            require(row["units"] == sorted(row["units"]) and row["window_start"] == row["units"][0])
            require(row["window_end"] == row["units"][-1] and row["window_end"] < row["date"])
        require(len(row["methods"]) == 2 and {m["key"] for m in row["methods"]} == METHODS[row["model"]])
        finite(row["random_response"])
        require(row["random_response"] >= 0)
        for method in row["methods"]:
            require(len(method["values"]) == len(row["units"]))
            for value in method["values"]: finite(value)
            for key in ["runtime_ms", "removal_response", "alternative_removal_response"]:
                finite(method[key])
                require(method[key] >= 0)
            for key in ["reference_top3_overlap", "seed_top3_overlap"]:
                if method[key] is not None: probability(method[key])
            if method["key"] in {"native_treeshap", "integrated_gradients"}:
                finite(method["completeness_error"])
                error = abs(sum(method["values"]) + method["baseline_logit"] - row["raw_logit"])
                require(math.isclose(error, method["completeness_error"], abs_tol=2e-6))
                if method["key"] == "native_treeshap": require(error <= 1e-5)
            else: require(method["completeness_error"] is None)
            if method["key"] == "lime_historical_neighborhood":
                finite(method["weighted_r2"])
                finite(method["point_error"])
                require(method["point_error"] >= 0)
    require(len(instruments["xgb_finance"]) == 44 and instruments["xgb_finance"] == instruments["lstm_finance"])
    require(report["summaries"] == summarize(report["rows"]))
    return report


def load():
    if not REPORT.exists(): return None
    report = validate(json.loads(REPORT.read_text(encoding="utf-8")))
    for source in report["sources"]:
        path = ROOT / source["path"]
        stat = path.stat()
        if source["sha256"] != _cached_digest(str(path), stat.st_mtime_ns, stat.st_size):
            raise ValueError("XAI evidence does not match current files.")
    protocol = json.loads((ROOT / "backend/research/protocols/xai_comparison_v1.json").read_text(encoding="utf-8"))
    if report["protocol"] != protocol: raise ValueError("XAI evidence uses a different protocol.")
    return report
