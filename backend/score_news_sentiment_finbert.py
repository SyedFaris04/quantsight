"""
backend/score_news_sentiment_finbert.py
─────────────────────────────────────────────────────────────────────────────
FinBERT UPGRADE — replaces score_news_sentiment.py (VADER).

Reads GDELT news parquet files (output of fetch_news.py), runs FinBERT
(a BERT model pre-trained specifically on financial text) on each headline,
then aggregates to a DAILY sentiment score per ticker.

WHY FINBERT INSTEAD OF VADER:
    VADER is a general-purpose lexicon scorer — it doesn't understand
    financial language. A headline like "Company slashes guidance" reads
    as neutral to VADER (no obviously negative words) but is clearly bad
    news to a financial reader. FinBERT was pre-trained on financial
    text specifically, so it understands this kind of domain language.
    This is the exact upgrade described in the proposal (Section 6.4,
    citing Zheng et al., 2024 — FinBERT-LSTM).

OUTPUT (identical schema to the VADER version — safe drop-in replacement):
    data/processed/gdelt_sentiment.csv

    Columns:
        ticker              | str   — stock ticker e.g. AAPL
        date                | str   — YYYY-MM-DD
        gdelt_pos           | float — avg positive probability (0–1)
        gdelt_neg           | float — avg negative probability (0–1)
        gdelt_compound      | float — avg (positive - negative), -1 to +1
        gdelt_article_count | int   — number of articles that day

    Because the column names are unchanged, build_features.py and every
    downstream script needs ZERO changes — this file is a direct swap.

HOW TO RUN:
    pip install transformers torch pandas pyarrow
    python score_news_sentiment_finbert.py

RUN THIS AFTER:  fetch_news.py
RUN THIS BEFORE: build_features.py

NOTE: FinBERT inference is much slower than VADER (a neural network vs.
a lexicon lookup). On CPU, expect roughly 5–15 minutes for a few thousand
headlines. A GPU (CUDA) speeds this up significantly if available.
─────────────────────────────────────────────────────────────────────────────
"""

import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("nuroquant-finbert")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
NEWS_DIR    = BASE_DIR / "data" / "raw" / "news"
OUTPUT_DIR  = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "gdelt_sentiment.csv"

# ── Model config ───────────────────────────────────────────────────────────────
MODEL_NAME = "ProsusAI/finbert"   # BERT pre-trained on financial text (Zheng et al., 2024 style)
BATCH_SIZE = 32
MAX_LENGTH = 64                    # headlines are short; keeps inference fast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loading ──────────────────────────────────────────────────────────────

def load_finbert():
    """Downloads (first run only, then cached) and loads FinBERT."""
    logger.info(f"Loading FinBERT ({MODEL_NAME}) on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    # FinBERT label order from the model config: 0=positive, 1=negative, 2=neutral
    logger.info(f"  Model label order: {model.config.id2label}")
    return tokenizer, model


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_headlines_finbert(df: pd.DataFrame, tokenizer, model) -> pd.DataFrame:
    """
    Runs FinBERT on every headline in batches.
    Returns the same DataFrame with 3 new columns:
        finbert_pos | finbert_neg | finbert_neu
    """
    titles = df["title"].fillna("").astype(str).tolist()
    n = len(titles)

    all_pos, all_neg, all_neu = [], [], []
    id2label = model.config.id2label  # e.g. {0: 'positive', 1: 'negative', 2: 'neutral'}
    label2idx = {v.lower(): k for k, v in id2label.items()}

    start_time = time.time()
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, n, BATCH_SIZE):
        batch = titles[i : i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()

        all_pos.extend(probs[:, label2idx["positive"]])
        all_neg.extend(probs[:, label2idx["negative"]])
        all_neu.extend(probs[:, label2idx["neutral"]])

        batch_num = i // BATCH_SIZE + 1
        if batch_num % 20 == 0 or batch_num == n_batches:
            elapsed = time.time() - start_time
            rate    = (i + len(batch)) / elapsed if elapsed > 0 else 0
            logger.info(
                f"  Batch {batch_num}/{n_batches} "
                f"({i + len(batch):,}/{n:,} headlines, {rate:.0f} headlines/sec)"
            )

    df = df.reset_index(drop=True).copy()
    df["finbert_pos"] = all_pos
    df["finbert_neg"] = all_neg
    df["finbert_neu"] = all_neu
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse article-level FinBERT sentiment to one row per (ticker, date).
    Compound score = mean(positive) - mean(negative), same convention as
    the VADER version, so downstream code needs no changes.
    """
    daily = (
        df.groupby(["ticker", "date"])
        .agg(
            gdelt_pos           = ("finbert_pos", "mean"),
            gdelt_neg           = ("finbert_neg", "mean"),
            gdelt_article_count = ("title",       "count"),
        )
        .reset_index()
    )
    daily["gdelt_compound"] = daily["gdelt_pos"] - daily["gdelt_neg"]
    daily = daily[["ticker", "date", "gdelt_compound", "gdelt_pos", "gdelt_neg", "gdelt_article_count"]]
    return daily


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_news_path = NEWS_DIR / "all_news.parquet"
    if not all_news_path.exists():
        logger.error(
            f"Could not find {all_news_path}\n"
            "Please run fetch_news.py first to download GDELT headlines."
        )
        return

    logger.info(f"Loading news from {all_news_path} ...")
    df = pd.read_parquet(all_news_path)
    logger.info(f"  Loaded {len(df):,} articles across {df['ticker'].nunique()} tickers")

    before = len(df)
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip() != ""]
    logger.info(f"  Dropped {before - len(df)} rows with empty titles")

    tokenizer, model = load_finbert()

    logger.info(f"Running FinBERT sentiment analysis on {len(df):,} headlines ...")
    df = score_headlines_finbert(df, tokenizer, model)
    logger.info(f"  Scored {len(df):,} headlines")

    logger.info("Aggregating to daily sentiment per ticker ...")
    daily = aggregate_daily(df)
    logger.info(f"  Result: {len(daily):,} rows ({daily['ticker'].nunique()} tickers x dates)")

    daily.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"  Saved -> {OUTPUT_FILE}")

    print("\n-- Sample output (first 10 rows) ----------------------")
    print(daily.head(10).to_string(index=False))

    print("\n-- Compound score range --------------------------------")
    print(f"  Min : {daily['gdelt_compound'].min():.4f}")
    print(f"  Max : {daily['gdelt_compound'].max():.4f}")
    print(f"  Mean: {daily['gdelt_compound'].mean():.4f}")

    print("\n-- Articles per ticker (top 10) -------------------------")
    summary = (
        daily.groupby("ticker")["gdelt_article_count"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    print(summary.to_string())
    print("----------------------------------------------------------\n")

    print(
        "FinBERT scoring complete. gdelt_sentiment.csv has been updated with\n"
        "financial-domain sentiment scores. No other files need to change --\n"
        "just re-run build_features.py, train_xgboost.py, and train_lstm.py\n"
        "to retrain the sentiment-enhanced model variants with the improved data.\n"
    )


if __name__ == "__main__":
    main()
