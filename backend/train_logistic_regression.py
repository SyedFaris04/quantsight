"""
backend/train_logistic_regression.py
─────────────────────────────────────────────────────────────────────────────
Trains TWO Logistic Regression model variants and evaluates both:

    Variant A — Logistic Regression (Finance Only)
        Input : features_finance.csv
        Output: data/predictions/logreg_finance_predictions.csv
                data/models/logreg_finance.pkl

    Variant B — Logistic Regression (Finance + Sentiment)
        Input : features_sentiment.csv
        Output: data/predictions/logreg_sentiment_predictions.csv
                data/models/logreg_sentiment.pkl

The simplest possible baseline in the model roster, and the only one that's
INTRINSICALLY interpretable — its coefficients are the explanation, no
post-hoc SHAP/LIME needed. This is the concrete demonstration of the
intrinsic-vs-post-hoc XAI taxonomy (Khan et al. 2025, cited in README.md)
that every other model here (XGBoost/RF via SHAP, LSTM/GRU/Transformer via
attention) sits on the other side of.

METRICS SAVED:
    data/predictions/model_metrics.json
    (appended — does not overwrite the other variants already present)

HOW TO RUN:
    python train_logistic_regression.py

RUN THIS AFTER:  build_features.py
─────────────────────────────────────────────────────────────────────────────
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
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
logger = logging.getLogger("quantsight-logreg")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR      = BASE_DIR / "data" / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"

FINANCE_CSV     = PROCESSED_DIR / "features_finance.csv"
SENTIMENT_CSV   = PROCESSED_DIR / "features_sentiment.csv"
METRICS_FILE    = PREDICTIONS_DIR / "model_metrics.json"

NON_FEATURE_COLS = {"ticker", "date", "signal", "Date", "Ticker"}

# ── Logistic Regression Hyperparameters ────────────────────────────────────────
LOGREG_PARAMS = {
    "max_iter"      : 1000,
    "class_weight"  : "balanced",
    "random_state"  : 42,
}

# ── Utility Functions ──────────────────────────────────────────────────────────

def load_and_split(csv_path: Path):
    """Time-based 80/20 split — identical convention to train_xgboost.py."""
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
    return {
        "model"     : "LogisticRegression",
        "variant"   : variant_name,
        "accuracy"  : round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1"        : round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "precision" : round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall"    : round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "auc_roc"   : round(roc_auc_score(y_true, y_prob) * 100, 2),
    }


def save_predictions(df_test: pd.DataFrame, y_pred: np.ndarray,
                     y_prob: np.ndarray, out_path: Path):
    out_df = df_test[["ticker", "date", "signal"]].copy()
    out_df = out_df.rename(columns={"signal": "actual_signal"})
    out_df["predicted_signal"] = y_pred
    out_df["confidence"]       = np.round(y_prob * 100, 2)
    out_df["signal_label"]     = out_df["predicted_signal"].map({1: "BUY", 0: "SELL"})
    out_df["date"]             = out_df["date"].dt.strftime("%Y-%m-%d")

    out_df.to_csv(out_path, index=False)
    logger.info(f"  Saved predictions → {out_path.name}")


def update_metrics_file(new_metrics: dict):
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
    Purged walk-forward CV — kept for consistency with the other tabular
    scripts (same shared cv_utils folds), even though LR trains fast enough
    that this isn't strictly necessary for runtime reasons.
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

        model = LogisticRegression(**LOGREG_PARAMS)
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


def get_coefficient_importance(model, feature_cols: list, top_n: int = 15) -> pd.DataFrame:
    """
    Logistic Regression's "feature importance" is its coefficients — the
    magnitude shows how much each standardized feature moves the log-odds of
    BUY, and the sign shows the direction. This IS the explanation: no
    separate XAI method (SHAP/LIME/attention) is applied to this model.
    """
    coefs = model.coef_[0]
    fi_df = pd.DataFrame({
        "feature"     : feature_cols,
        "coefficient" : coefs,
        "abs_coef"    : np.abs(coefs),
    }).sort_values("abs_coef", ascending=False).head(top_n)
    return fi_df


# ── Train One Variant ──────────────────────────────────────────────────────────

def train_variant(csv_path: Path, variant_name: str,
                  model_out: Path, pred_out: Path):
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Logistic Regression — {variant_name}")
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
    model = LogisticRegression(**LOGREG_PARAMS)
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

    print(f"\nClassification Report — Logistic Regression ({variant_name}):")
    print(classification_report(y_test, y_pred, target_names=["SELL", "BUY"]))

    fi_df = get_coefficient_importance(model, feature_cols)
    logger.info(f"\nTop 10 coefficients by |magnitude| ({variant_name}):")
    for _, row in fi_df.head(10).iterrows():
        direction = "+BUY" if row["coefficient"] > 0 else "-BUY"
        logger.info(f"  {row['feature']:30s} {direction} {row['abs_coef']:.4f}")

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
        model_out    = MODELS_DIR / "logreg_finance.pkl",
        pred_out     = PREDICTIONS_DIR / "logreg_finance_predictions.csv",
    )

    if not SENTIMENT_CSV.exists():
        logger.error(f"Missing {SENTIMENT_CSV} — run build_features.py first")
        return

    metrics_b = train_variant(
        csv_path     = SENTIMENT_CSV,
        variant_name = "Finance + Sentiment",
        model_out    = MODELS_DIR / "logreg_sentiment.pkl",
        pred_out     = PREDICTIONS_DIR / "logreg_sentiment_predictions.csv",
    )

    print("\n-- Logistic Regression Model Comparison ----------------------------")
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
    print(f"  {PREDICTIONS_DIR}/logreg_finance_predictions.csv")
    print(f"  {PREDICTIONS_DIR}/logreg_sentiment_predictions.csv")
    print(f"  {PREDICTIONS_DIR}/model_metrics.json")
    print(f"  {MODELS_DIR}/logreg_finance.pkl")
    print(f"  {MODELS_DIR}/logreg_sentiment.pkl\n")


if __name__ == "__main__":
    main()
