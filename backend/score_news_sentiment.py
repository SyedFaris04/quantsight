"""
backend/score_news_sentiment.py
─────────────────────────────────────────────────────────────────────────────
[LEGACY — superseded by score_news_sentiment_finbert.py]

This is the original VADER-based sentiment scorer. It has been replaced
by score_news_sentiment_finbert.py, which uses FinBERT (a finance-domain
BERT model) for more accurate financial sentiment scoring, as planned in
the proposal (Section 6.4 / WBS item 8).

Kept here for reference / fallback only — run score_news_sentiment_finbert.py
instead for the current pipeline.

Reads GDELT news parquet files (output of fetch_news.py),
runs VADER sentiment scoring on each headline,
then aggregates to a DAILY sentiment score per ticker.

OUTPUT:
    data/processed/gdelt_sentiment.csv

    Columns:
        ticker     | str  — stock ticker e.g. AAPL
        date       | str  — YYYY-MM-DD
        gdelt_pos  | float — avg positive sentiment score (0–1)
        gdelt_neg  | float — avg negative sentiment score (0–1)
        gdelt_compound | float — avg compound score (-1 to +1)
        gdelt_article_count | int — number of articles that day

HOW TO RUN:
    pip install vaderSentiment pandas pyarrow
    python score_news_sentiment.py

RUN THIS AFTER:  fetch_news.py
RUN THIS BEFORE: build_features.py
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("nuroquant-sentiment")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
NEWS_DIR    = BASE_DIR / "data" / "raw" / "news"
OUTPUT_DIR  = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "gdelt_sentiment.csv"

# ── Scoring ────────────────────────────────────────────────────────────────────

def score_headlines(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """
    Run VADER on each headline in the DataFrame.
    Returns the same DataFrame with 4 new columns:
        vader_pos | vader_neg | vader_neu | vader_compound
    """
    scores = df["title"].fillna("").apply(
        lambda text: analyzer.polarity_scores(str(text))
    )
    scores_df = pd.DataFrame(list(scores))
    scores_df.columns = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
    return pd.concat([df.reset_index(drop=True), scores_df], axis=1)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse article-level sentiment to one row per (ticker, date).
    Uses mean of compound / pos / neg scores across all articles that day.
    Also counts how many articles were seen — more articles = more signal confidence.
    """
    daily = (
        df.groupby(["ticker", "date"])
        .agg(
            gdelt_compound      = ("vader_compound", "mean"),
            gdelt_pos           = ("vader_pos",      "mean"),
            gdelt_neg           = ("vader_neg",      "mean"),
            gdelt_article_count = ("title",          "count"),
        )
        .reset_index()
    )
    return daily


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check that news data exists
    all_news_path = NEWS_DIR / "all_news.parquet"
    if not all_news_path.exists():
        logger.error(
            f"Could not find {all_news_path}\n"
            "Please run fetch_news.py first to download GDELT headlines."
        )
        return

    # Load all news
    logger.info(f"Loading news from {all_news_path} ...")
    df = pd.read_parquet(all_news_path)
    logger.info(f"  Loaded {len(df):,} articles across {df['ticker'].nunique()} tickers")

    # Drop rows with no title
    before = len(df)
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip() != ""]
    logger.info(f"  Dropped {before - len(df)} rows with empty titles")

    # Initialise VADER
    logger.info("Running VADER sentiment analysis ...")
    analyzer = SentimentIntensityAnalyzer()
    df = score_headlines(df, analyzer)
    logger.info(f"  Scored {len(df):,} headlines")

    # Aggregate to daily scores
    logger.info("Aggregating to daily sentiment per ticker ...")
    daily = aggregate_daily(df)
    logger.info(f"  Result: {len(daily):,} rows ({daily['ticker'].nunique()} tickers × dates)")

    # Save
    daily.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"  Saved → {OUTPUT_FILE}")

    # ── Quick sanity check ─────────────────────────────────────────────────────
    print("\n── Sample output (first 10 rows) ────────────────────────")
    print(daily.head(10).to_string(index=False))

    print("\n── Compound score range ──────────────────────────────────")
    print(f"  Min : {daily['gdelt_compound'].min():.4f}")
    print(f"  Max : {daily['gdelt_compound'].max():.4f}")
    print(f"  Mean: {daily['gdelt_compound'].mean():.4f}")

    print("\n── Articles per ticker (top 10) ──────────────────────────")
    summary = (
        daily.groupby("ticker")["gdelt_article_count"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    print(summary.to_string())
    print("──────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
