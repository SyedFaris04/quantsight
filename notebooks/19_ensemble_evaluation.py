"""
notebooks/19_ensemble_evaluation.py
─────────────────────────────────────────────────────────────────────────────
Evaluates a simple probability-averaging ensemble across all 4 trained model
variants (research memo, Tier 2 item 5): averages each model's CALIBRATED
confidence for matching (ticker, date) rows in the test set, thresholds at
0.5, and reports the same statistical metrics as the other 4 variants for a
fair, apples-to-apples comparison.

This is complementary to — not a replacement for — the existing majority-vote
Consensus Engine in copilot_engine.py: the consensus engine gives a discrete
Strong/Moderate/Mixed agreement label, this gives a single continuous
ensembled probability. Saves an ensemble_predictions.csv in the same schema
as the other 4 prediction files so 15_backtesting.py can pick it up as a
5th strategy.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

BASE_DIR        = Path(__file__).resolve().parent.parent / "backend"
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
METRICS_FILE    = PREDICTIONS_DIR / "model_metrics.json"

PREDICTION_FILES = {
    "xgb_finance"    : PREDICTIONS_DIR / "xgb_finance_predictions.csv",
    "xgb_sentiment"  : PREDICTIONS_DIR / "xgb_sentiment_predictions.csv",
    "lstm_finance"   : PREDICTIONS_DIR / "lstm_finance_predictions.csv",
    "lstm_sentiment" : PREDICTIONS_DIR / "lstm_sentiment_predictions.csv",
}


def main():
    dfs = {}
    for key, path in PREDICTION_FILES.items():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["ticker"] = df["ticker"].astype(str)
        dfs[key] = df[["ticker", "date", "confidence"]].rename(columns={"confidence": f"conf_{key}"})

    # Inner join on (ticker, date) — LSTM's test window starts ~2 weeks later
    # than XGBoost's (10-day sequence warmup), so only overlapping rows are
    # kept, same as every other cross-model comparison already in this app.
    merged = dfs["xgb_finance"]
    for key in ["xgb_sentiment", "lstm_finance", "lstm_sentiment"]:
        merged = merged.merge(dfs[key], on=["ticker", "date"], how="inner")

    conf_cols = [c for c in merged.columns if c.startswith("conf_")]
    merged["ensemble_prob"] = merged[conf_cols].mean(axis=1) / 100.0
    merged["predicted_signal"] = (merged["ensemble_prob"] >= 0.5).astype(int)

    print(f"Ensemble rows (all 4 variants overlap): {len(merged):,}")

    # Ground truth from the feature file (same pattern as 16_confidence_calibration.py)
    feat = pd.read_csv(PROCESSED_DIR / "features_finance.csv", usecols=["ticker", "date", "signal"])
    feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()
    feat["ticker"] = feat["ticker"].astype(str)
    merged = merged.merge(feat, on=["ticker", "date"], how="left").dropna(subset=["signal"])

    y_true = merged["signal"].values
    y_pred = merged["predicted_signal"].values
    y_prob = merged["ensemble_prob"].values

    metrics = {
        "model"     : "Ensemble",
        "variant"   : "All 4 Variants (avg. calibrated probability)",
        "accuracy"  : round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1"        : round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "precision" : round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall"    : round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "auc_roc"   : round(roc_auc_score(y_true, y_prob) * 100, 2),
        "n_rows"    : int(len(merged)),
    }

    print("\n-- Ensemble (avg. of all 4 calibrated probabilities) --------------")
    for k, v in metrics.items():
        print(f"  {k:<10}: {v}")
    print("---------------------------------------------------------------------")

    # Compare against the 4 individual variants for context
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            all_metrics = json.load(f)
        print("\n-- All variants, for comparison --------------------------------------")
        for key, m in all_metrics.items():
            print(f"  {key:<45} acc={m['accuracy']:>6.2f}%  f1={m['f1']:>6.2f}%  auc={m['auc_roc']:>6.2f}%")
        print(f"  {'Ensemble (this script)':<45} acc={metrics['accuracy']:>6.2f}%  "
              f"f1={metrics['f1']:>6.2f}%  auc={metrics['auc_roc']:>6.2f}%")
        print("-------------------------------------------------------------------------")

        all_metrics["Ensemble_All4Variants"] = metrics
        with open(METRICS_FILE, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved metrics -> {METRICS_FILE}")

    # Save predictions in the same schema as the other 4 files so
    # 15_backtesting.py can pick this up as a 5th strategy.
    out_df = pd.DataFrame({
        "ticker"           : merged["ticker"],
        "date"             : merged["date"].dt.strftime("%Y-%m-%d"),
        "predicted_signal" : merged["predicted_signal"],
        "confidence"       : np.round(merged["ensemble_prob"] * 100, 2),
        "signal_label"     : merged["predicted_signal"].map({1: "BUY", 0: "SELL"}),
    })
    out_path = PREDICTIONS_DIR / "ensemble_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions -> {out_path}")


if __name__ == "__main__":
    main()
