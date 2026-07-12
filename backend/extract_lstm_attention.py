"""
backend/extract_lstm_attention.py
─────────────────────────────────────────────────────────────────────────────
Local, per-prediction explainability for the LSTM+Attention models.

LSTMAttention (train_lstm.py) already computes, internally, an attention
weight for each of the SEQUENCE_LEN trading days in its lookback window —
this is the exact mechanism the model itself uses to decide which recent
days matter most before making a prediction (see LSTMAttention.forward,
the `self.attention` sub-network), not a post-hoc approximation like
SHAP/LIME. This script loads the ALREADY-TRAINED checkpoints exactly as
train_lstm.py saved them, rebuilds the same test set with the same saved
scaler, and runs one more forward pass with those weights exposed.

No retraining, no change to predictions or metrics — this only reads out
an internal value the model already computes for every prediction it makes.

OUTPUT (one file per variant):
    data/predictions/lstm_finance_attention.json
    data/predictions/lstm_sentiment_attention.json

    {
      "TICKER|YYYY-MM-DD": [
        {"date": "YYYY-MM-DD", "weight": 0.0-1.0},   # oldest of the 10 days
        ...
        {"date": "YYYY-MM-DD", "weight": 0.0-1.0}    # most recent day, i.e. the day before the prediction date
      ],
      ...
    }

    The 10 weights for a given prediction sum to 1.0 (softmax output).

HOW TO RUN:
    python extract_lstm_attention.py

RUN THIS AFTER: train_lstm.py (needs lstm_finance.pt / lstm_sentiment.pt to exist)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import logging

from train_lstm import LSTMAttention, StockSequenceDataset, SEQUENCE_LEN, BATCH_SIZE, DEVICE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nuroquant-lstm-attention")

BASE_DIR        = Path(__file__).resolve().parent
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR      = BASE_DIR / "data" / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"

VARIANTS = {
    "lstm_finance":   ("features_finance.csv",   MODELS_DIR / "lstm_finance.pt"),
    "lstm_sentiment": ("features_sentiment.csv", MODELS_DIR / "lstm_sentiment.pt"),
}


def extract_variant(csv_name: str, model_path: Path) -> dict:
    checkpoint   = torch.load(model_path, map_location=DEVICE, weights_only=False)
    feature_cols = checkpoint["feature_cols"]
    scaler       = checkpoint["scaler"]

    model = LSTMAttention(n_features=checkpoint["n_features"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Rebuild the exact same test split train_lstm.py used (80th-percentile date cut).
    df = pd.read_csv(PROCESSED_DIR / csv_name)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    split_test = df["date"].quantile(0.80)
    test_df    = df[df["date"] > split_test].copy()
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    test_dataset = StockSequenceDataset(test_df, feature_cols, SEQUENCE_LEN)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    logger.info(f"  Test sequences: {len(test_dataset):,}")

    all_weights = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            _, weights = model(X_batch, return_logits=True, return_attention=True)
            all_weights.append(weights.cpu().numpy())
    all_weights = np.concatenate(all_weights, axis=0)  # (N, seq_len)

    result = {}
    for (ticker, pred_date), win_dates, w in zip(test_dataset.meta, test_dataset.window_dates, all_weights):
        key = f"{ticker}|{pred_date[:10]}"
        result[key] = [
            {"date": d, "weight": round(float(wi), 4)}
            for d, wi in zip(win_dates, w)
        ]
    return result


def main():
    for variant_key, (csv_name, model_path) in VARIANTS.items():
        if not model_path.exists():
            logger.warning(f"Skipping {variant_key} — {model_path.name} not found. Run train_lstm.py first.")
            continue

        logger.info(f"\n{'='*60}\n{variant_key}\n{'='*60}")
        result = extract_variant(csv_name, model_path)

        out_path = PREDICTIONS_DIR / f"{variant_key}_attention.json"
        with open(out_path, "w") as f:
            json.dump(result, f)
        logger.info(f"  Saved {len(result):,} predictions' attention weights -> {out_path.name}")

        # Spot-check: print one prediction's day-by-day weights
        sample_key = next(iter(result))
        logger.info(f"  Sample ({sample_key}):")
        for day in result[sample_key]:
            bar = "#" * int(day["weight"] * 50)
            logger.info(f"    {day['date']}  {day['weight']:.4f}  {bar}")


if __name__ == "__main__":
    main()
