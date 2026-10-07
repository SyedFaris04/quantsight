"""Load a verified historical input reference without model fitting or CSV parsing."""
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "model_health_protocol.json"
REFERENCE_PATH = ROOT / "data/serving/model_health_reference.json"
SOURCE_PATH = ROOT / "data/processed/features_finance.csv"
MODEL_PATH = ROOT / "data/models/xgb_finance.pkl"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def digest(path, binary=False):
    result = hashlib.sha256()
    if binary:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(chunk)
    else:
        with Path(path).open("r", encoding="utf-8", newline=None) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), ""):
                result.update(chunk.encode("utf-8"))
    return result.hexdigest()


def validate(reference):
    def require(condition):
        if not condition:
            raise ValueError("Historical input reference is invalid or stale.")
    require(reference["schema_version"] == 1 and reference["protocol_id"] == PROTOCOL["protocol_id"])
    features, tickers = reference["features"], reference["tickers"]
    require(isinstance(features, list) and 0 < len(features) <= 100 and len(set(features)) == len(features))
    require(all(isinstance(name, str) and name for name in features) and isinstance(tickers, dict) and bool(tickers))
    config = PROTOCOL["reference"]
    for ticker, item in tickers.items():
        require(isinstance(ticker, str) and bool(ticker))
        require(type(item["rows"]) is int and item["rows"] >= config["minimum_rows_per_ticker"])
        require(config["start"] <= item["start"] <= item["end"] <= config["end"])
        require(set(item["features"]) == set(features))
        for stats in item["features"].values():
            require(type(stats["n"]) is int and config["minimum_rows_per_ticker"] <= stats["n"] <= item["rows"])
            values = stats["quantiles"]
            require(isinstance(values, list) and len(values) == len(config["quantiles"]))
            require(all(type(v) in (int, float) and math.isfinite(v) for v in values))
            require(values == sorted(values))
    require(reference["rows"] == sum(t["rows"] for t in tickers.values()))
    require(reference["start"] == min(t["start"] for t in tickers.values()))
    require(reference["end"] == max(t["end"] for t in tickers.values()))
    return reference


@lru_cache(maxsize=1)
def _load(signatures):
    reference = validate(json.loads(REFERENCE_PATH.read_text(encoding="utf-8")))
    if (reference["source_sha256"] != digest(SOURCE_PATH) or
            reference["model_sha256"] != digest(MODEL_PATH, binary=True) or
            reference["protocol_sha256"] != digest(PROTOCOL_PATH)):
        raise ValueError("Historical input reference does not match current artifacts.")
    return reference


def load_reference():
    paths = (REFERENCE_PATH, SOURCE_PATH, MODEL_PATH, PROTOCOL_PATH)
    signatures = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
    return _load(signatures)
