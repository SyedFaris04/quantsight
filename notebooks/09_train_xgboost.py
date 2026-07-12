import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score)
import os
import pickle

# ── Settings ──────────────────────────────────────────────
PROCESSED_FOLDER = "../data/processed/"
MODELS_FOLDER    = "../models/"
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Only use stocks where we have real sentiment coverage
SENTIMENT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                     "TSLA", "NVDA", "JPM", "JNJ", "SPY"]

FEATURE_COLS = [
    "SMA_5", "SMA_10", "SMA_20",
    "EMA_12", "EMA_26",
    "MACD", "MACD_signal", "MACD_hist",
    "RSI",
    "Return_5", "Return_10",
    "BB_upper", "BB_lower", "BB_mid", "BB_width",
    "Volatility_20", "ATR",
    "Volume_change", "Volume_MA_10",
    "news_sentiment", "news_count",
    "wsb_sentiment", "wsb_avg_score",
    "combined_sentiment"
]

TARGET_COL = "Target"
# ──────────────────────────────────────────────────────────

# ── Load final dataset ────────────────────────────────────
print("Loading final dataset...")
df = pd.read_csv(os.path.join(PROCESSED_FOLDER, "final_dataset.csv"))
df["Date"] = pd.to_datetime(df["Date"])

# ── Filter to sentiment covered tickers ───────────────────
print(f"\nFiltering to sentiment-covered tickers...")
df = df[df["Ticker"].isin(SENTIMENT_TICKERS)]
print(f"Rows after ticker filter: {len(df):,}")

print(f"\nTotal rows: {len(df):,}")
print(f"Target distribution:\n{df[TARGET_COL].value_counts()}")

# Best performing split — recent data only
train = df[(df["Date"] >= "2021-01-01") & (df["Date"] < "2023-01-01")]
val   = df[(df["Date"] >= "2023-01-01") & (df["Date"] < "2024-01-01")]
test  = df[df["Date"] >= "2024-01-01"]

print(f"Train : {len(train):,} rows (2021–2022)")
print(f"Val   : {len(val):,}   rows (2023)")
print(f"Test  : {len(test):,}  rows (2024)")

# ── Prepare features ──────────────────────────────────────
X_train = train[FEATURE_COLS]
y_train = train[TARGET_COL]

X_val   = val[FEATURE_COLS]
y_val   = val[TARGET_COL]

X_test  = test[FEATURE_COLS]
y_test  = test[TARGET_COL]

# ── Scale features ────────────────────────────────────────
print("\nScaling features...")
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ── Train XGBoost ─────────────────────────────────────────
print("\nTraining XGBoost model...")

# Handle class imbalance
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale     = neg_count / pos_count
print(f"Class balance ratio: {scale:.3f}")

model = XGBClassifier(
    n_estimators     = 500,    # more trees
    max_depth        = 4,
    learning_rate    = 0.01,   # slower = more careful
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    gamma            = 0.1,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    scale_pos_weight = scale,
    eval_metric      = "auc",
    random_state     = 42
)

model.fit(
    X_train, y_train,
    eval_set = [(X_val, y_val)],
    verbose  = 100
)

# ── Evaluate on validation set ────────────────────────────
print("\n── Validation Results ───────────────────────────────")
val_preds    = model.predict(X_val)
val_probs    = model.predict_proba(X_val)[:, 1]
val_accuracy = accuracy_score(y_val, val_preds)
val_auc      = roc_auc_score(y_val, val_probs)

print(f"Accuracy : {val_accuracy:.4f}")
print(f"AUC ROC  : {val_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, val_preds))

# ── Evaluate on test set ──────────────────────────────────
print("\n── Test Results (2022–2024) ─────────────────────────")
test_preds    = model.predict(X_test)
test_probs    = model.predict_proba(X_test)[:, 1]
test_accuracy = accuracy_score(y_test, test_preds)
test_auc      = roc_auc_score(y_test, test_probs)

print(f"Accuracy : {test_accuracy:.4f}")
print(f"AUC ROC  : {test_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, test_preds))

# ── Feature importance ────────────────────────────────────
print("\n── Top 10 Most Important Features ───────────────────")
importance = pd.DataFrame({
    "feature"   : FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print(importance.head(10).to_string(index=False))

# ── Save model and scaler ─────────────────────────────────
print("\nSaving model and scaler...")
with open(os.path.join(MODELS_FOLDER, "xgboost_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODELS_FOLDER, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ── Save test predictions for backtesting ─────────────────
test_results = test[["Date", "Ticker", "Close", TARGET_COL]].copy()
test_results["xgb_pred"] = test_preds
test_results["xgb_prob"] = test_probs
test_results.to_csv(
    os.path.join(PROCESSED_FOLDER, "xgb_predictions.csv"), index=False)

print("\n✅ XGBoost training complete!")
print(f"   Model saved  → models/xgboost_model.pkl")
print(f"   Scaler saved → models/scaler.pkl")
print(f"   Predictions  → data/processed/xgb_predictions.csv")