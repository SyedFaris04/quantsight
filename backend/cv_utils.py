"""
backend/cv_utils.py
─────────────────────────────────────────────────────────────────────────────
Shared utilities for probability calibration and purged walk-forward
cross-validation, used by both train_xgboost.py and train_lstm.py.

Why this exists (see research memo, Tier 1 items 1-2):
  - Confidence calibration: raw model probabilities are not the same thing
    as "how often is this prediction actually right." Fitting a calibrator
    (isotonic regression or Platt/logistic scaling) on a held-out validation
    slice — never the final test set — corrects this. AUC-ROC is unaffected
    (both methods are monotonic, so ranking is preserved); accuracy/F1/
    precision/recall computed post-calibration reflect a genuinely
    recalibrated decision boundary, not just a cosmetic change.
  - Purged walk-forward CV: a single train/val/test split gives one noisy
    estimate of performance. Expanding-window folds with an embargo gap
    (dropping rows within a few days of each fold boundary, since technical
    indicators use trailing rolling windows) give a distribution of
    out-of-sample estimates instead of one point estimate — the standard
    fix for the leakage/instability risk in naive time-series CV.

Both utilities operate purely within the TRAINING period (before the
final 2023-2024 test boundary) — the proposal's headline train/test split
is untouched; this adds robustness analysis on top of it, not a
replacement for it.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ── Calibration ────────────────────────────────────────────────────────────────

def make_train_calib_split(dates: pd.Series, calib_frac: float = 0.15, embargo_days: int = 5):
    """
    Split a chronologically-sorted date series into a training mask and a
    calibration-fitting mask, with an embargo gap between them so rolling-
    window technical indicators can't leak across the boundary.

    Returns (train_mask, calib_mask) as boolean numpy arrays.
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    split_date = dates.quantile(1 - calib_frac)
    embargo_start = split_date - pd.Timedelta(days=embargo_days)
    embargo_end   = split_date + pd.Timedelta(days=embargo_days)

    train_mask = (dates <= embargo_start).values
    calib_mask = (dates > embargo_end).values
    return train_mask, calib_mask


def fit_calibrator(probs, labels, method: str = "isotonic"):
    """Fit a calibrator mapping raw model probability -> calibrated probability."""
    probs  = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(probs, labels)
        return cal
    elif method == "platt":
        cal = LogisticRegression()
        cal.fit(probs.reshape(-1, 1), labels)
        return cal
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(calibrator, probs, method: str = "isotonic"):
    """Apply a fitted calibrator to raw probabilities."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if method == "isotonic":
        return calibrator.predict(probs)
    elif method == "platt":
        return calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
    raise ValueError(f"Unknown calibration method: {method}")


def pick_best_calibration(probs, labels):
    """
    Fit both isotonic and Platt calibrators on the same validation slice,
    keep whichever gives the better Brier score (mean squared error between
    calibrated probability and actual outcome — the standard calibration
    quality metric). Returns (method_name, fitted_calibrator).
    """
    probs  = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)

    results = {}
    for method in ("isotonic", "platt"):
        cal = fit_calibrator(probs, labels, method)
        calibrated = apply_calibrator(cal, probs, method)
        brier = float(np.mean((calibrated - labels) ** 2))
        results[method] = (brier, cal)

    best_method = min(results, key=lambda k: results[k][0])
    return best_method, results[best_method][1], {m: results[m][0] for m in results}


# ── Purged walk-forward CV ──────────────────────────────────────────────────────

def make_purged_folds(dates: pd.Series, n_folds: int = 5, embargo_days: int = 5):
    """
    Expanding-window walk-forward folds over a date range: fold i trains on
    everything up to (with an embargo gap before) the start of fold i's
    validation window, and validates on that window only.

    Returns a list of (train_mask, val_mask) boolean numpy array pairs,
    aligned to the input `dates` Series' index.
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    unique_dates = np.sort(dates.unique())
    # First chunk is warm-up (always in-fold-1's training set); remaining
    # n_folds chunks are each fold's validation window.
    chunks = np.array_split(unique_dates, n_folds + 1)

    folds = []
    for i in range(1, n_folds + 1):
        val_dates = chunks[i]
        val_start, val_end = pd.Timestamp(val_dates[0]), pd.Timestamp(val_dates[-1])
        embargo_start = val_start - pd.Timedelta(days=embargo_days)

        train_mask = (dates <= embargo_start).values
        val_mask   = ((dates >= val_start) & (dates <= val_end)).values
        folds.append((train_mask, val_mask))
    return folds


def summarize_cv_metrics(fold_metrics: list) -> dict:
    """
    fold_metrics: list of dicts, each with the same keys (e.g. accuracy, f1, auc_roc).
    Returns {key: {"mean": ..., "std": ...}} across folds.
    """
    if not fold_metrics:
        return {}
    keys = fold_metrics[0].keys()
    summary = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        summary[k] = {
            "mean": round(float(np.mean(vals)), 2),
            "std":  round(float(np.std(vals)), 2),
        }
    return summary
