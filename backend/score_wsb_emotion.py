"""
backend/score_wsb_emotion.py
─────────────────────────────────────────────────────────────────────────────
Multi-label EMOTION detection over WallStreetBets post titles, using a
RoBERTa encoder pre-trained on GoEmotions (58k Reddit comments, 27 emotions
+ neutral) as a feature extractor — exactly the design in the proposal
(Section 6.5.2): no training from scratch, the model just scores each post,
producing an independent probability per emotion via a sigmoid output layer.

This is a parallel, second pass over the SAME per-post text
(wsb_sentiment.csv's `title_clean` column) that score_news_sentiment_finbert.py
and notebooks 06/12/13 already run FinBERT sentiment over — sentiment and
emotion are complementary views of the same underlying text, not two
different datasets.

Only the 6 finance-relevant emotions named in the proposal are kept:
    fear, optimism, anger, excitement, confusion, disappointment

OUTPUT (per-post, NOT yet aggregated to daily — build_features.py aggregates
this the same way it already aggregates wsb_sentiment.csv):
    data/processed/wsb_emotion.csv

    Columns:
        ticker              | str
        date                | str   — YYYY-MM-DD
        title_clean         | str   — kept for spot-check validation
        emo_fear            | float — 0-1 probability
        emo_optimism        | float
        emo_anger           | float
        emo_excitement      | float
        emo_confusion       | float
        emo_disappointment  | float

HOW TO RUN:
    pip install transformers torch pandas
    python score_wsb_emotion.py

RUN THIS AFTER:  the wsb_sentiment.csv pipeline (notebooks 05/06/12/13)
RUN THIS BEFORE: build_features.py

Model choice was validated against the official GoEmotions test split first
(see notebooks/17_goemotions_validation.py) rather than assumed to work.
─────────────────────────────────────────────────────────────────────────────
"""

import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import pipeline
import logging

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("nuroquant-emotion")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
INPUT_FILE  = BASE_DIR / "data" / "processed" / "wsb_sentiment.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "wsb_emotion.csv"

# ── Model config ───────────────────────────────────────────────────────────────
MODEL_NAME = "SamLowe/roberta-base-go_emotions"
BATCH_SIZE = 32
FINANCE_EMOTIONS = ["fear", "optimism", "anger", "excitement", "confusion", "disappointment"]

DEVICE = 0 if torch.cuda.is_available() else -1


def load_emotion_pipeline():
    logger.info(f"Loading {MODEL_NAME} on {'GPU' if DEVICE == 0 else 'CPU'} ...")
    clf = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,          # return all 28 label scores per text, not just top-1
        truncation=True,
        device=DEVICE,
    )
    return clf


def score_posts(df: pd.DataFrame, clf) -> pd.DataFrame:
    """Runs the emotion classifier on every post title in batches."""
    titles = df["title_clean"].fillna("").astype(str).tolist()
    n = len(titles)

    scores = {emo: np.zeros(n, dtype=np.float32) for emo in FINANCE_EMOTIONS}

    start_time = time.time()
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, n, BATCH_SIZE):
        batch = titles[i : i + BATCH_SIZE]
        # Empty strings crash the tokenizer's attention mask in some edge cases
        batch = [t if t.strip() else "neutral" for t in batch]
        results = clf(batch)

        for row_i, row_scores in enumerate(results):
            for item in row_scores:
                if item["label"] in scores:
                    scores[item["label"]][i + row_i] = item["score"]

        batch_num = i // BATCH_SIZE + 1
        if batch_num % 20 == 0 or batch_num == n_batches:
            elapsed = time.time() - start_time
            rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
            logger.info(f"  Batch {batch_num}/{n_batches} ({i + len(batch):,}/{n:,} posts, {rate:.0f} posts/sec)")

    df = df.reset_index(drop=True).copy()
    for emo in FINANCE_EMOTIONS:
        df[f"emo_{emo}"] = scores[emo]
    return df


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        logger.error(f"Could not find {INPUT_FILE} — run the WSB sentiment pipeline first (notebooks 05/06/12/13).")
        return

    logger.info(f"Loading WSB posts from {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"  Loaded {len(df):,} posts across {df['ticker'].nunique()} tickers")

    before = len(df)
    df = df.dropna(subset=["title_clean"])
    df = df[df["title_clean"].str.strip() != ""]
    logger.info(f"  Dropped {before - len(df)} rows with empty text")

    clf = load_emotion_pipeline()

    logger.info(f"Running emotion detection on {len(df):,} posts ...")
    scored = score_posts(df, clf)

    out_cols = ["ticker", "date", "title_clean"] + [f"emo_{e}" for e in FINANCE_EMOTIONS]
    out_df = scored[out_cols]
    out_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"  Saved -> {OUTPUT_FILE}")

    print("\n-- Sample output (first 5 rows) --------------------------")
    print(out_df.head(5).to_string(index=False))

    print("\n-- Mean emotion intensity across all posts ----------------")
    for emo in FINANCE_EMOTIONS:
        col = f"emo_{emo}"
        print(f"  {emo:<15}: mean={out_df[col].mean():.4f}  max={out_df[col].max():.4f}")

    print("\n-- Dominant emotion per post (share of posts) --------------")
    emo_cols = [f"emo_{e}" for e in FINANCE_EMOTIONS]
    dominant = out_df[emo_cols].idxmax(axis=1).str.replace("emo_", "")
    print(dominant.value_counts(normalize=True).round(3).to_string())
    print("-------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
