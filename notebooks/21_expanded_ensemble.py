"""
notebooks/21_expanded_ensemble.py
─────────────────────────────────────────────────────────────────────────────
Tests whether averaging more of the trained model variants together beats
the current 4-core-model ensemble — a low-risk way to push accuracy since
it combines already-validated predictors rather than introducing a new
technique. All rows compared on the exact same overlapping (ticker, date)
set for a fair, apples-to-apples comparison.
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

BASE_DIR        = Path(__file__).resolve().parent.parent / "backend"
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
METRICS_FILE    = PREDICTIONS_DIR / "model_metrics.json"

ALL_MODELS = {
    "xgb_finance"         : "xgb_finance_predictions.csv",
    "xgb_sentiment"       : "xgb_sentiment_predictions.csv",
    "lstm_finance"        : "lstm_finance_predictions.csv",
    "lstm_sentiment"      : "lstm_sentiment_predictions.csv",
    "rf_finance"          : "rf_finance_predictions.csv",
    "rf_sentiment"        : "rf_sentiment_predictions.csv",
    "logreg_finance"      : "logreg_finance_predictions.csv",
    "logreg_sentiment"    : "logreg_sentiment_predictions.csv",
    "gru_finance"         : "gru_finance_predictions.csv",
    "gru_sentiment"       : "gru_sentiment_predictions.csv",
    "transformer_finance"   : "transformer_finance_predictions.csv",
    "transformer_sentiment" : "transformer_sentiment_predictions.csv",
}

CORE_4 = ["xgb_finance", "xgb_sentiment", "lstm_finance", "lstm_sentiment"]


def load_all():
    dfs = {}
    for key, fname in ALL_MODELS.items():
        df = pd.read_csv(PREDICTIONS_DIR / fname)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["ticker"] = df["ticker"].astype(str)
        dfs[key] = df[["ticker", "date", "confidence"]].rename(columns={"confidence": f"conf_{key}"})
    return dfs


def evaluate(merged, model_keys, y_true):
    conf_cols = [f"conf_{k}" for k in model_keys]
    prob = (merged[conf_cols] / 100.0).mean(axis=1)
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy"  : round(accuracy_score(y_true, pred) * 100, 2),
        "f1"        : round(f1_score(y_true, pred, zero_division=0) * 100, 2),
        "precision" : round(precision_score(y_true, pred, zero_division=0) * 100, 2),
        "recall"    : round(recall_score(y_true, pred, zero_division=0) * 100, 2),
        "auc_roc"   : round(roc_auc_score(y_true, prob) * 100, 2),
    }, int(len(merged))


def main():
    dfs = load_all()

    # Inner join across ALL 12 — the common denominator so every ensemble
    # size tested below is evaluated on the exact same rows.
    keys = list(ALL_MODELS.keys())
    merged = dfs[keys[0]]
    for k in keys[1:]:
        merged = merged.merge(dfs[k], on=["ticker", "date"], how="inner")

    feat = pd.read_csv(PROCESSED_DIR / "features_finance.csv", usecols=["ticker", "date", "signal"])
    feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()
    feat["ticker"] = feat["ticker"].astype(str)
    merged = merged.merge(feat, on=["ticker", "date"], how="left").dropna(subset=["signal"])
    y_true = merged["signal"].values

    print(f"Common rows across all 12 variants: {len(merged):,}\n")

    combos = {
        "Core 4 (XGB x2, LSTM x2)"        : CORE_4,
        "Core 4 + RF x2"                  : CORE_4 + ["rf_finance", "rf_sentiment"],
        "Core 4 + LogReg x2"              : CORE_4 + ["logreg_finance", "logreg_sentiment"],
        "Core 4 + GRU x2"                 : CORE_4 + ["gru_finance", "gru_sentiment"],
        "Core 4 + Transformer x2"         : CORE_4 + ["transformer_finance", "transformer_sentiment"],
        "All 12"                          : keys,
    }

    results = {}
    print(f"  {'Combination':<32} {'n':>6} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'AUC':>8}")
    for name, model_keys in combos.items():
        metrics, n = evaluate(merged, model_keys, y_true)
        results[name] = metrics
        print(f"  {name:<32} {n:>6} {metrics['accuracy']:>7.2f}% {metrics['f1']:>7.2f}% "
              f"{metrics['precision']:>7.2f}% {metrics['recall']:>7.2f}% {metrics['auc_roc']:>7.2f}%")

    best_acc = max(results, key=lambda k: results[k]["accuracy"])
    best_auc = max(results, key=lambda k: results[k]["auc_roc"])
    print(f"\nBest accuracy : {best_acc} ({results[best_acc]['accuracy']}%)")
    print(f"Best AUC-ROC  : {best_auc} ({results[best_auc]['auc_roc']}%)")

    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            all_metrics = json.load(f)
        for name, metrics in results.items():
            all_metrics[f"Ensemble_Expanded_{name}"] = {
                "model": "Ensemble", "variant": name, **metrics, "n_rows": len(merged),
            }
        with open(METRICS_FILE, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved all combinations -> {METRICS_FILE}")


if __name__ == "__main__":
    main()
