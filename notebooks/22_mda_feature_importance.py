"""
notebooks/22_mda_feature_importance.py
─────────────────────────────────────────────────────────────────────────────
Mean Decrease Accuracy (MDA) feature importance for the trained XGBoost
models — research memo, Tier 2 item 4 (hudson-and-thames/mlfinlab, the
reference implementation of López de Prado's purge/embargo scheme this
project already uses). Complements the existing SHAP analysis
(18_shap_feature_selection.py): SHAP explains the model's own internal
logic (in-sample attribution), MDA measures something different — how much
held-out AUC/accuracy actually degrades when a feature is shuffled, which
is more robust to collinearity between correlated indicators (RSI/MACD/
Bollinger/SMAs all move together) than in-sample attribution.

Diagnostic only — does not retrain or change any model. Run on the
already-trained xgb_finance.pkl / xgb_sentiment.pkl against the held-out
test set, since this is a post-hoc audit, not a decision that itself needs
walk-forward validation.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

BASE_DIR        = Path(__file__).resolve().parent.parent / "backend"
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR      = BASE_DIR / "data" / "models"

MODELS = {
    "finance"   : (MODELS_DIR / "xgb_finance.pkl",   PROCESSED_DIR / "features_finance.csv"),
    "sentiment" : (MODELS_DIR / "xgb_sentiment.pkl", PROCESSED_DIR / "features_sentiment.csv"),
}

N_REPEATS = 5
SEED = 42


def compute_mda(model, scaler, feature_cols, X_test_raw, y_test, calibrator=None, calib_method=None):
    rng = np.random.default_rng(SEED)
    X_test = scaler.transform(X_test_raw)

    baseline_prob = model.predict_proba(X_test)[:, 1]
    baseline_pred = (baseline_prob >= 0.5).astype(int)
    baseline_acc  = accuracy_score(y_test, baseline_pred)
    baseline_auc  = roc_auc_score(y_test, baseline_prob)

    results = []
    for i, col in enumerate(feature_cols):
        acc_drops, auc_drops = [], []
        for _ in range(N_REPEATS):
            X_shuffled = X_test.copy()
            rng.shuffle(X_shuffled[:, i])
            prob = model.predict_proba(X_shuffled)[:, 1]
            pred = (prob >= 0.5).astype(int)
            acc_drops.append(baseline_acc - accuracy_score(y_test, pred))
            auc_drops.append(baseline_auc - roc_auc_score(y_test, prob))
        results.append({
            "feature"       : col,
            "mda_accuracy"  : round(float(np.mean(acc_drops)) * 100, 3),
            "mda_auc"       : round(float(np.mean(auc_drops)) * 100, 3),
        })

    return pd.DataFrame(results).sort_values("mda_auc", ascending=False), baseline_acc, baseline_auc


def main():
    for variant, (model_path, csv_path) in MODELS.items():
        print(f"\n{'='*70}\nXGBoost — {variant}\n{'='*70}")

        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        model, scaler, feature_cols = bundle["model"], bundle["scaler"], bundle["features"]

        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        split_date = df["date"].quantile(0.80)
        test_df = df[df["date"] > split_date]

        X_test_raw = test_df[feature_cols].values
        y_test = test_df["signal"].values

        mda_df, base_acc, base_auc = compute_mda(model, scaler, feature_cols, X_test_raw, y_test)

        print(f"Baseline: accuracy={base_acc*100:.2f}%  AUC={base_auc*100:.2f}%\n")
        print("Top 10 most important (biggest AUC drop when shuffled):")
        for _, row in mda_df.head(10).iterrows():
            print(f"  {row['feature']:<28} MDA-AUC: {row['mda_auc']:+.3f}  MDA-Acc: {row['mda_accuracy']:+.3f}")

        print("\nBottom 10 (near-zero or negative = candidates for removal — shuffling barely hurt, or helped):")
        for _, row in mda_df.tail(10).iterrows():
            print(f"  {row['feature']:<28} MDA-AUC: {row['mda_auc']:+.3f}  MDA-Acc: {row['mda_accuracy']:+.3f}")

        noise_features = mda_df[mda_df["mda_auc"] <= 0]["feature"].tolist()
        print(f"\n{len(noise_features)} feature(s) with MDA-AUC <= 0 (no evidence they help): {noise_features}")

        out_path = BASE_DIR / "data" / "predictions" / f"mda_importance_{variant}.csv"
        mda_df.to_csv(out_path, index=False)
        print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
