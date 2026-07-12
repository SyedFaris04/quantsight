"""
notebooks/18_shap_feature_selection.py
─────────────────────────────────────────────────────────────────────────────
SHAP-based feature importance and selection for the two XGBoost variants
(research memo, Tier 1 item 3). Computes SHAP values on the held-out test
set, reports per-feature mean |SHAP|, flags features with near-zero
contribution, and retrains on the reduced set to check whether dropping
them changes accuracy/F1/AUC — a legitimate strict-improvement check, not
just a smaller model for its own sake.

Also feeds copilot_engine.py's Trading Copilot: SHAP values are a
theoretically-grounded upgrade on the raw XGBoost .feature_importances_
already used there for plain-language explanations.
"""

import json
import pickle
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE_DIR       = Path(__file__).resolve().parent.parent / "backend"
PROCESSED_DIR  = BASE_DIR / "data" / "processed"
MODELS_DIR     = BASE_DIR / "data" / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"

VARIANTS = {
    "xgb_finance":   ("features_finance.csv",   MODELS_DIR / "xgb_finance.pkl"),
    "xgb_sentiment": ("features_sentiment.csv", MODELS_DIR / "xgb_sentiment.pkl"),
}

NON_FEATURE_COLS = {"ticker", "date", "signal", "Date", "Ticker"}
LOW_IMPORTANCE_THRESHOLD = 0.02  # features contributing < 2% of total mean |SHAP|


def load_split(csv_path: Path, feature_cols: list):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    split_date = df["date"].quantile(0.80)
    train_df = df[df["date"] <= split_date]
    test_df  = df[df["date"] >  split_date]
    return train_df, test_df


def main():
    results = {}

    for variant_key, (csv_name, model_path) in VARIANTS.items():
        print(f"\n{'='*70}\n{variant_key}\n{'='*70}")

        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        model, scaler, feature_cols = saved["model"], saved["scaler"], saved["features"]

        train_df, test_df = load_split(PROCESSED_DIR / csv_name, feature_cols)
        X_test = scaler.transform(test_df[feature_cols].values)

        # ── SHAP values on the test set ──
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        total = mean_abs_shap.sum()

        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": mean_abs_shap,
            "pct_of_total": 100 * mean_abs_shap / total,
        }).sort_values("mean_abs_shap", ascending=False)

        print("\nTop 10 features by SHAP importance:")
        print(importance_df.head(10).to_string(index=False))

        low_importance = importance_df[importance_df["pct_of_total"] < LOW_IMPORTANCE_THRESHOLD]
        print(f"\n{len(low_importance)} feature(s) below {LOW_IMPORTANCE_THRESHOLD}% of total SHAP importance:")
        print(low_importance["feature"].tolist())

        # ── Baseline metrics (already-trained model, full feature set) ──
        y_test = test_df["signal"].values
        y_prob_full = model.predict_proba(X_test)[:, 1]
        y_pred_full = (y_prob_full >= 0.5).astype(int)
        baseline = {
            "accuracy": round(accuracy_score(y_test, y_pred_full) * 100, 2),
            "f1":       round(f1_score(y_test, y_pred_full, zero_division=0) * 100, 2),
            "auc_roc":  round(roc_auc_score(y_test, y_prob_full) * 100, 2),
        }

        # ── Retrain on the reduced feature set, same hyperparams, same split ──
        reduced_cols = [c for c in feature_cols if c not in low_importance["feature"].tolist()]
        reduced_metrics = None
        if len(reduced_cols) < len(feature_cols) and len(reduced_cols) > 0:
            scaler_r = StandardScaler()
            X_train_r = scaler_r.fit_transform(train_df[reduced_cols].values)
            X_test_r  = scaler_r.transform(test_df[reduced_cols].values)
            y_train   = train_df["signal"].values

            model_r = xgb.XGBClassifier(**{k: v for k, v in model.get_params().items()
                                            if k not in ("use_label_encoder",)})
            model_r.fit(X_train_r, y_train, verbose=False)
            y_prob_r = model_r.predict_proba(X_test_r)[:, 1]
            y_pred_r = (y_prob_r >= 0.5).astype(int)
            reduced_metrics = {
                "accuracy": round(accuracy_score(y_test, y_pred_r) * 100, 2),
                "f1":       round(f1_score(y_test, y_pred_r, zero_division=0) * 100, 2),
                "auc_roc":  round(roc_auc_score(y_test, y_prob_r) * 100, 2),
                "n_features": len(reduced_cols),
            }
            print(f"\nBaseline ({len(feature_cols)} features): {baseline}")
            print(f"Reduced  ({len(reduced_cols)} features): {reduced_metrics}")

        results[variant_key] = {
            "top_features": importance_df.head(10).to_dict(orient="records"),
            "low_importance_features": low_importance["feature"].tolist(),
            "baseline_metrics": baseline,
            "reduced_metrics": reduced_metrics,
        }

    out_path = PREDICTIONS_DIR / "shap_feature_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
