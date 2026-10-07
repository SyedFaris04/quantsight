"""Build/check historical feature percentiles. No fits, predictions or holdout reads."""
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from model_health_reference import (PROTOCOL, PROTOCOL_PATH, REFERENCE_PATH,
                                    SOURCE_PATH, MODEL_PATH, digest, validate)


def build():
    model_hash, source_hash = digest(MODEL_PATH, binary=True), digest(SOURCE_PATH)
    with MODEL_PATH.open("rb") as stream:
        features = list(pickle.load(stream)["features"])
    frame = pd.read_csv(SOURCE_PATH, usecols=["ticker", "date", *features])
    config = PROTOCOL["reference"]
    frame = frame[frame.date.between(config["start"], config["end"])].copy()
    if frame.empty or frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Historical reference keys are missing or duplicated.")
    tickers = {}
    for ticker, part in frame.groupby("ticker", sort=True):
        stats = {}
        for name in features:
            values = part[name].to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values) < config["minimum_rows_per_ticker"]:
                raise ValueError("Insufficient finite historical feature observations.")
            stats[name] = {"n": len(values), "quantiles": np.quantile(values, config["quantiles"], method=config["quantile_method"]).tolist()}
        tickers[ticker] = {"rows": len(part), "start": part.date.min(), "end": part.date.max(), "features": stats}
    reference = validate({"schema_version": 1, "protocol_id": PROTOCOL["protocol_id"],
                         "source_sha256": source_hash, "model_sha256": model_hash,
                         "protocol_sha256": digest(PROTOCOL_PATH), "features": features,
                         "start": frame.date.min(), "end": frame.date.max(),
                         "rows": len(frame), "tickers": tickers})
    if source_hash != digest(SOURCE_PATH) or model_hash != digest(MODEL_PATH, binary=True):
        raise ValueError("Reference source changed during generation.")
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    reference = build()
    if args.check:
        if json.loads(REFERENCE_PATH.read_text(encoding="utf-8")) != reference:
            raise ValueError("Historical input reference needs rebuilding.")
    else:
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE_PATH.write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"checked": args.check, "rows": reference["rows"], "tickers": len(reference["tickers"]),
                      "features": len(reference["features"]), "start": reference["start"], "end": reference["end"]}))


if __name__ == "__main__":
    main()
