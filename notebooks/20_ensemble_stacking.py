"""
notebooks/20_ensemble_stacking.py
─────────────────────────────────────────────────────────────────────────────
Replaces 19_ensemble_evaluation.py's naive average of the 4 core models'
calibrated probabilities with a trained stacking meta-learner (logistic
regression over the 4 models' probabilities) — research memo, Tier 1
item 2.

Leakage discipline: the meta-learner is fit ONLY on the first ~70% of the
test period (2023-02-06 to the split date) and evaluated on the last ~30%
(split date to 2024-12-20), which it never sees during fitting. The naive
average and each individual model are recomputed on that SAME held-out
slice for a fair, apples-to-apples comparison — none of the numbers here
are directly comparable to the full-test-period numbers in model_metrics.json
for the 4 individual models, by design, since this uses less data to leave
a genuinely clean evaluation slice for the ensemble specifically.

A logistic regression meta-learner is used deliberately over anything more
complex: its 4 coefficients are directly interpretable as "how much the
ensemble trusts each base model," which is more transparent than SHAP-on-
XGBoost, not less — in keeping with this project's explainability focus.

Does NOT touch ensemble_predictions.csv / 15_backtesting.py's naive-average
5th strategy — that stays as-is; this is a separate, additive analysis.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
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

META_TRAIN_FRAC = 0.70


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy"  : round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1"        : round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "precision" : round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall"    : round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "auc_roc"   : round(roc_auc_score(y_true, y_prob) * 100, 2),
    }


def main():
    dfs = {}
    for key, path in PREDICTION_FILES.items():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["ticker"] = df["ticker"].astype(str)
        dfs[key] = df[["ticker", "date", "confidence"]].rename(columns={"confidence": f"conf_{key}"})

    merged = dfs["xgb_finance"]
    for key in ["xgb_sentiment", "lstm_finance", "lstm_sentiment"]:
        merged = merged.merge(dfs[key], on=["ticker", "date"], how="inner")

    feat = pd.read_csv(PROCESSED_DIR / "features_finance.csv", usecols=["ticker", "date", "signal"])
    feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()
    feat["ticker"] = feat["ticker"].astype(str)
    merged = merged.merge(feat, on=["ticker", "date"], how="left").dropna(subset=["signal"])
    merged = merged.sort_values("date").reset_index(drop=True)

    conf_cols = [c for c in merged.columns if c.startswith("conf_")]
    X = (merged[conf_cols] / 100.0).values
    y = merged["signal"].values

    split_date = merged["date"].quantile(META_TRAIN_FRAC)
    meta_train_mask = (merged["date"] <= split_date).values
    meta_test_mask  = (merged["date"] >  split_date).values

    print(f"Meta-train: {meta_train_mask.sum():,} rows (up to {split_date.date()})")
    print(f"Meta-test : {meta_test_mask.sum():,} rows (after {split_date.date()}) — the ONLY rows scored below")

    # ── Stacking meta-learner ──────────────────────────────────────────────
    meta = LogisticRegression()
    meta.fit(X[meta_train_mask], y[meta_train_mask])

    meta_prob = meta.predict_proba(X[meta_test_mask])[:, 1]
    meta_pred = (meta_prob >= 0.5).astype(int)
    y_test    = y[meta_test_mask]

    stacking_metrics = compute_metrics(y_test, meta_pred, meta_prob)

    print("\n-- Meta-learner coefficients (higher = more trusted) --------------")
    for col, coef in zip(conf_cols, meta.coef_[0]):
        print(f"  {col:<20}: {coef:+.3f}")
    print(f"  {'intercept':<20}: {meta.intercept_[0]:+.3f}")

    # ── Naive average, recomputed on the SAME held-out slice ────────────────
    avg_prob = X[meta_test_mask].mean(axis=1)
    avg_pred = (avg_prob >= 0.5).astype(int)
    naive_metrics = compute_metrics(y_test, avg_pred, avg_prob)

    # ── Each individual model, recomputed on the SAME held-out slice ───────
    individual_metrics = {}
    for i, col in enumerate(conf_cols):
        prob_i = X[meta_test_mask][:, i]
        pred_i = (prob_i >= 0.5).astype(int)
        individual_metrics[col.replace("conf_", "")] = compute_metrics(y_test, pred_i, prob_i)

    print(f"\n-- Fair comparison, all evaluated on the same {meta_test_mask.sum():,}-row held-out slice --")
    print(f"  {'Model':<24} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'AUC':>8}")
    for name, m in individual_metrics.items():
        print(f"  {name:<24} {m['accuracy']:>7.2f}% {m['f1']:>7.2f}% {m['precision']:>7.2f}% {m['recall']:>7.2f}% {m['auc_roc']:>7.2f}%")
    print(f"  {'Naive average':<24} {naive_metrics['accuracy']:>7.2f}% {naive_metrics['f1']:>7.2f}% {naive_metrics['precision']:>7.2f}% {naive_metrics['recall']:>7.2f}% {naive_metrics['auc_roc']:>7.2f}%")
    print(f"  {'Stacking meta-learner':<24} {stacking_metrics['accuracy']:>7.2f}% {stacking_metrics['f1']:>7.2f}% {stacking_metrics['precision']:>7.2f}% {stacking_metrics['recall']:>7.2f}% {stacking_metrics['auc_roc']:>7.2f}%")

    result = {
        "model"   : "Ensemble",
        "variant" : "Stacking meta-learner (LogisticRegression over 4 core models), "
                    f"evaluated on the last {int((1-META_TRAIN_FRAC)*100)}% of the test period only",
        **stacking_metrics,
        "n_rows"  : int(meta_test_mask.sum()),
        "coefficients": {col: round(float(c), 4) for col, c in zip(conf_cols, meta.coef_[0])},
        "naive_average_same_slice": naive_metrics,
        "individual_models_same_slice": individual_metrics,
    }

    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            all_metrics = json.load(f)
        all_metrics["Ensemble_Stacking"] = result
        with open(METRICS_FILE, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved -> {METRICS_FILE} (key: Ensemble_Stacking)")


if __name__ == "__main__":
    main()
