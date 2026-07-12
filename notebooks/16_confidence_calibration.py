"""
notebooks/16_confidence_calibration.py
─────────────────────────────────────────────────────────────────────────────
Confidence-bucketed accuracy evaluation.

The proposal's Decision Support Layer (§6.5.2) already defines a Prediction
Reliability label — High (>=70% confidence), Medium (50-69%), Low (<50%) —
but nothing previously computed *accuracy conditional on that label*. This
script does exactly that: for each of the 4 model variants, it reports
accuracy separately within each confidence bucket, alongside the overall
(blanket) accuracy already in model_metrics.json.

This is the legitimate way to answer "does any model hit >=70% accuracy":
daily stock-direction accuracy of ~70% blanket is not realistic per the
efficient-markets framing the proposal itself cites (Section 5.3), but a
model's accuracy *within its own High-reliability bucket* is a standard,
citable calibration statistic — not a leakage-prone claim.

The stored "confidence" column in the prediction CSVs is raw P(BUY)*100,
not confidence-in-the-actual-prediction — it is transformed here to a
symmetric 50-100 scale via max(p, 100-p) before bucketing.
"""

import pandas as pd
import json
from pathlib import Path

BASE_DIR       = Path(__file__).resolve().parent.parent / "backend"
PROCESSED_DIR  = BASE_DIR / "data" / "processed"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"

PREDICTION_FILES = {
    "xgb_finance"    : (PREDICTIONS_DIR / "xgb_finance_predictions.csv",    PROCESSED_DIR / "features_finance.csv"),
    "xgb_sentiment"  : (PREDICTIONS_DIR / "xgb_sentiment_predictions.csv",  PROCESSED_DIR / "features_sentiment.csv"),
    "lstm_finance"   : (PREDICTIONS_DIR / "lstm_finance_predictions.csv",   PROCESSED_DIR / "features_finance.csv"),
    "lstm_sentiment" : (PREDICTIONS_DIR / "lstm_sentiment_predictions.csv", PROCESSED_DIR / "features_sentiment.csv"),
}

BUCKETS = [
    ("Low (<50%)",    0.0,  50.0),
    ("Medium (50-69%)", 50.0, 70.0),
    ("High (>=70%)",  70.0, 100.0001),
]


def bucket_label(conf):
    for label, lo, hi in BUCKETS:
        if lo <= conf < hi:
            return label
    return "Unknown"


def evaluate(model_key, pred_path, feature_path):
    pred = pd.read_csv(pred_path)
    pred["date"] = pd.to_datetime(pred["date"]).dt.normalize()
    pred["ticker"] = pred["ticker"].astype(str)

    feat = pd.read_csv(feature_path, usecols=["ticker", "date", "signal"])
    feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()
    feat["ticker"] = feat["ticker"].astype(str)

    df = pred.merge(feat, on=["ticker", "date"], how="left")
    missing = df["signal"].isna().sum()
    df = df.dropna(subset=["signal"])

    # Symmetric confidence-in-own-prediction (50-100 scale), not raw P(BUY)
    df["pred_confidence"] = df.apply(
        lambda r: r["confidence"] if r["predicted_signal"] == 1 else 100 - r["confidence"],
        axis=1,
    )
    df["correct"] = (df["predicted_signal"] == df["signal"]).astype(int)
    df["bucket"] = df["pred_confidence"].apply(bucket_label)

    overall_acc = round(df["correct"].mean() * 100, 2)

    bucket_rows = []
    for label, lo, hi in BUCKETS:
        sub = df[df["bucket"] == label]
        n = len(sub)
        acc = round(sub["correct"].mean() * 100, 2) if n > 0 else None
        pct_of_total = round(100 * n / len(df), 2) if len(df) > 0 else 0.0
        bucket_rows.append({
            "bucket": label,
            "n_predictions": int(n),
            "pct_of_total": pct_of_total,
            "accuracy": acc,
        })

    return {
        "model": model_key,
        "n_total": int(len(df)),
        "n_dropped_no_ground_truth": int(missing),
        "overall_accuracy": overall_acc,
        "buy_rate_pct": round(100 * (df["predicted_signal"] == 1).mean(), 2),
        "buckets": bucket_rows,
    }


def main():
    results = {}
    for model_key, (pred_path, feature_path) in PREDICTION_FILES.items():
        if not pred_path.exists():
            print(f"SKIP {model_key}: {pred_path.name} not found")
            continue
        results[model_key] = evaluate(model_key, pred_path, feature_path)

    out_path = PREDICTIONS_DIR / "confidence_calibration.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 88)
    print("CONFIDENCE-BUCKETED ACCURACY — per model variant")
    print("=" * 88)
    for model_key, r in results.items():
        flag = "  [WARNING] DEGENERATE (near-constant BUY, see buy_rate_pct)" if r["buy_rate_pct"] > 95 else ""
        print(f"\n{model_key}  (overall accuracy: {r['overall_accuracy']}%, BUY rate: {r['buy_rate_pct']}%){flag}")
        for b in r["buckets"]:
            acc_str = f"{b['accuracy']}%" if b["accuracy"] is not None else "n/a"
            print(f"    {b['bucket']:<18} n={b['n_predictions']:>6}  ({b['pct_of_total']:>5}% of preds)  accuracy={acc_str}")

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
