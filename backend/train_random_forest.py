"""
backend/train_random_forest.py
─────────────────────────────────────────────────────────────────────────────
Trains TWO Random Forest model variants and evaluates both:

    Variant A — Random Forest (Finance Only)
        Input : features_finance.csv
        Output: data/predictions/rf_finance_predictions.csv
                data/models/rf_finance.pkl

    Variant B — Random Forest (Finance + Sentiment)
        Input : features_sentiment.csv
        Output: data/predictions/rf_sentiment_predictions.csv
                data/models/rf_sentiment.pkl

Structurally identical to train_xgboost.py — same load/split, same purged
walk-forward CV + calibration pipeline, same shared model_metrics.json.
Random Forest is the single most common baseline across the XAI-in-finance
literature reviewed for this project, and being less sensitive to
hyperparameters than XGBoost, needs no eval_set/early-stopping machinery.

METRICS SAVED:
    data/predictions/model_metrics.json
    (appended — does not overwrite the other variants already present)

HOW TO RUN:
    python train_random_forest.py

RUN THIS AFTER:  build_features.py
─────────────────────────────────────────────────────────────────────────────
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)
import logging

from cv_utils import (
    pick_best_calibration, apply_calibrator,
    make_purged_folds, summarize_cv_metrics,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("quantsight-rf")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR      = BASE_DIR / "data" / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"

FINANCE_CSV     = PROCESSED_DIR / "features_finance.csv"
SENTIMENT_CSV   = PROCESSED_DIR / "features_sentiment.csv"
METRICS_FILE    = PREDICTIONS_DIR / "model_metrics.json"

# Columns that are NOT features — always exclude these
NON_FEATURE_COLS = {"ticker", "date", "signal", "Date", "Ticker"}

# ── Random Forest Hyperparameters ──────────────────────────────────────────────
# Same for both variants — fair comparison. RF is less sensitive to tuning
# than XGBoost, so these are reasonable defaults rather than a tuned search.
RF_PARAMS = {
    "n_estimators"      : 300,
    "max_depth"         : 8,
    "min_samples_leaf"  : 5,
    "class_weight"      : "balanced",
    "random_state"      : 42,
    "n_jobs"            : -1,
}

# ── Utility Functions ──────────────────────────────────────────────────────────
# (identical to train_xgboost.py — same split/metrics/save conventions so the
#  comparison across model families is apples-to-apples)

def load_and_split(csv_path: Path):
    """
    Load a feature CSV and split into train/test using time-based split.
    We use the last 20% of dates as the test set — no data leakage.

    Returns:
        X_train, X_test, y_train, y_test, feature_cols, full_df, train_df, test_df
    """
    logger.info(f"Loading {csv_path.name} ...")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    logger.info(f"  Features: {len(feature_cols)} columns")
    logger.info(f"  Rows    : {len(df):,}")
    logger.info(f"  Tickers : {df['ticker'].nunique()}")

    split_date = df["date"].quantile(0.80)
    train_df = df[df["date"] <= split_date]
    test_df  = df[df["date"] >  split_date]

    logger.info(
        f"  Train: {train_df['date'].min().date()} → {train_df['date'].max().date()} "
        f"({len(train_df):,} rows)"
    )
    logger.info(
        f"  Test : {test_df['date'].min().date()} → {test_df['date'].max().date()} "
        f"({len(test_df):,} rows)"
    )

    X_train = train_df[feature_cols].values
    X_test  = test_df[feature_cols].values
    y_train = train_df["signal"].values
    y_test  = test_df["signal"].values

    return X_train, X_test, y_train, y_test, feature_cols, df, train_df, test_df


def compute_metrics(y_true, y_pred, y_prob, variant_name: str) -> dict:
    """Compute all evaluation metrics for one model variant."""
    return {
        "model"     : "RandomForest",
        "variant"   : variant_name,
        "accuracy"  : round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1"        : round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "precision" : round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall"    : round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "auc_roc"   : round(roc_auc_score(y_true, y_prob) * 100, 2),
    }


def save_predictions(df_test: pd.DataFrame, y_pred: np.ndarray,
                     y_prob: np.ndarray, out_path: Path):
    """
    Save per-row predictions. `y_prob` is the CALIBRATED probability so
    downstream consumers get calibrated confidence automatically.
    """
    out_df = df_test[["ticker", "date", "signal"]].copy()
    out_df = out_df.rename(columns={"signal": "actual_signal"})
    out_df["predicted_signal"] = y_pred
    out_df["confidence"]       = np.round(y_prob * 100, 2)
    out_df["signal_label"]     = out_df["predicted_signal"].map({1: "BUY", 0: "SELL"})
    out_df["date"]             = out_df["date"].dt.strftime("%Y-%m-%d")

    out_df.to_csv(out_path, index=False)
    logger.info(f"  Saved predictions → {out_path.name}")


def update_metrics_file(new_metrics: dict):
    """
    Load existing metrics JSON (if any), update/insert the new variant,
    then save back — additive, never clobbers other models' keys.
    """
    all_metrics = {}
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r") as f:
            all_metrics = json.load(f)

    key = f"{new_metrics['model']}_{new_metrics['variant']}"
    all_metrics[key] = new_metrics

    with open(METRICS_FILE, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"  Updated metrics → {METRICS_FILE.name}")


def run_purged_cv(train_df: pd.DataFrame, feature_cols: list, n_folds: int = 5) -> dict:
    """
    Purged expanding-window walk-forward CV, entirely within the training
    period. Retrains a fresh Random Forest per fold for a distribution of
    out-of-sample estimates, and returns pooled out-of-fold (probability,
    label) pairs used to fit the confidence calibrator.
    """
    folds = make_purged_folds(train_df["date"], n_folds=n_folds, embargo_days=5)
    fold_metrics = []
    oof_probs, oof_labels = [], []

    for i, (train_mask, val_mask) in enumerate(folds, start=1):
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        fold_train = train_df.loc[train_mask]
        fold_val   = train_df.loc[val_mask]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(fold_train[feature_cols].values)
        X_va = scaler.transform(fold_val[feature_cols].values)
        y_tr = fold_train["signal"].values
        y_va = fold_val["signal"].values

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue

        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_va)[:, 1]
        pred = (prob >= 0.5).astype(int)

        fold_metrics.append({
            "accuracy": round(accuracy_score(y_va, pred) * 100, 2),
            "f1":       round(f1_score(y_va, pred, zero_division=0) * 100, 2),
            "auc_roc":  round(roc_auc_score(y_va, prob) * 100, 2),
        })
        logger.info(f"    CV fold {i}: acc={fold_metrics[-1]['accuracy']}% "
                    f"f1={fold_metrics[-1]['f1']}% auc={fold_metrics[-1]['auc_roc']}% "
                    f"(train={train_mask.sum():,}, val={val_mask.sum():,})")

        oof_probs.append(prob)
        oof_labels.append(y_va)

    cv_summary = summarize_cv_metrics(fold_metrics)
    oof_probs  = np.concatenate(oof_probs)  if oof_probs  else np.array([])
    oof_labels = np.concatenate(oof_labels) if oof_labels else np.array([])
    return cv_summary, oof_probs, oof_labels


def get_feature_importance(model, feature_cols: list, top_n: int = 15) -> pd.DataFrame:
    """Return top-N feature importances as a DataFrame."""
    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature"    : feature_cols,
        "importance" : importance,
    }).sort_values("importance", ascending=False).head(top_n)
    return fi_df


# ── Train One Variant ──────────────────────────────────────────────────────────

def train_variant(csv_path: Path, variant_name: str,
                  model_out: Path, pred_out: Path):
    """
    Full train → evaluate → save pipeline for one Random Forest variant.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Random Forest — {variant_name}")
    logger.info(f"{'='*60}")

    X_train, X_test, y_train, y_test, feature_cols, full_df, train_df, test_df = \
        load_and_split(csv_path)

    logger.info("Running purged walk-forward CV (robustness check + OOF calibration data) ...")
    cv_summary, oof_probs, oof_labels = run_purged_cv(train_df, feature_cols, n_folds=5)
    if cv_summary:
        logger.info(
            f"  CV accuracy: {cv_summary['accuracy']['mean']}% ± {cv_summary['accuracy']['std']}%  |  "
            f"CV AUC: {cv_summary['auc_roc']['mean']}% ± {cv_summary['auc_roc']['std']}%"
        )

    scaler  = StandardScaler()
    X_train_fit = scaler.fit_transform(X_train)
    X_test      = scaler.transform(X_test)

    logger.info("Training model ...")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train_fit, y_train)

    calib_method, calibrator, brier_scores = pick_best_calibration(oof_probs, oof_labels)
    logger.info(f"  Calibration method chosen: {calib_method} (Brier scores: {brier_scores}) "
                f"— fit on {len(oof_probs):,} out-of-fold predictions")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    calibrated_prob = apply_calibrator(calibrator, y_prob, calib_method)

    metrics = compute_metrics(y_test, y_pred, y_prob, variant_name)
    metrics["calibration_method"] = calib_method
    metrics["cv_robustness"] = cv_summary

    raw_conf   = np.where(y_pred == 1, y_prob, 1 - y_prob)
    calib_conf = np.where(y_pred == 1, calibrated_prob, 1 - calibrated_prob)
    correct    = (y_pred == y_test).astype(int)
    metrics["calibration_check"] = {
        "raw_confidence_vs_correctness_corr":       round(float(np.corrcoef(raw_conf, correct)[0, 1]), 3),
        "calibrated_confidence_vs_correctness_corr": round(float(np.corrcoef(calib_conf, correct)[0, 1]), 3),
    }

    logger.info(f"  Accuracy  : {metrics['accuracy']}%")
    logger.info(f"  F1 Score  : {metrics['f1']}%")
    logger.info(f"  Precision : {metrics['precision']}%")
    logger.info(f"  Recall    : {metrics['recall']}%")
    logger.info(f"  AUC-ROC   : {metrics['auc_roc']}%")

    print(f"\nClassification Report — Random Forest ({variant_name}):")
    print(classification_report(y_test, y_pred, target_names=["SELL", "BUY"]))

    fi_df = get_feature_importance(model, feature_cols)
    logger.info(f"\nTop 10 features ({variant_name}):")
    for _, row in fi_df.head(10).iterrows():
        bar = "#" * int(row["importance"] * 200)
        logger.info(f"  {row['feature']:30s} {bar} {row['importance']:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(model_out, "wb") as f:
        pickle.dump({
            "model"            : model,
            "scaler"           : scaler,
            "features"         : feature_cols,
            "calibrator"       : calibrator,
            "calibration_method": calib_method,
        }, f)
    logger.info(f"  Saved model → {model_out.name}")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    save_predictions(test_df, y_pred, calibrated_prob, pred_out)

    update_metrics_file(metrics)

    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if not FINANCE_CSV.exists():
        logger.error(f"Missing {FINANCE_CSV} — run build_features.py first")
        return

    metrics_a = train_variant(
        csv_path     = FINANCE_CSV,
        variant_name = "Finance Only",
        model_out    = MODELS_DIR / "rf_finance.pkl",
        pred_out     = PREDICTIONS_DIR / "rf_finance_predictions.csv",
    )

    if not SENTIMENT_CSV.exists():
        logger.error(f"Missing {SENTIMENT_CSV} — run build_features.py first")
        return

    metrics_b = train_variant(
        csv_path     = SENTIMENT_CSV,
        variant_name = "Finance + Sentiment",
        model_out    = MODELS_DIR / "rf_sentiment.pkl",
        pred_out     = PREDICTIONS_DIR / "rf_sentiment_predictions.csv",
    )

    print("\n-- Random Forest Model Comparison ---------------------------------")
    print(f"  {'Metric':<12} {'Finance Only':>16} {'Finance+Sentiment':>18} {'Delta':>15}")
    print(f"  {'-'*65}")
    for key in ["accuracy", "f1", "precision", "recall", "auc_roc"]:
        a_val = metrics_a[key]
        b_val = metrics_b[key]
        delta = b_val - a_val
        arrow = "UP" if delta > 0 else "DOWN"
        print(f"  {key:<12} {a_val:>15.2f}% {b_val:>17.2f}% {arrow}{abs(delta):>13.2f}%")
    print("--------------------------------------------------------------------\n")
    print(f"All outputs saved to:")
    print(f"  {PREDICTIONS_DIR}/rf_finance_predictions.csv")
    print(f"  {PREDICTIONS_DIR}/rf_sentiment_predictions.csv")
    print(f"  {PREDICTIONS_DIR}/model_metrics.json")
    print(f"  {MODELS_DIR}/rf_finance.pkl")
    print(f"  {MODELS_DIR}/rf_sentiment.pkl\n")


if __name__ == "__main__":
    main()
