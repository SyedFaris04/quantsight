"""
backend/fetch_news.py
─────────────────────────────────────────────────────────────────────────────
Fetches news headlines from GDELT for each stock ticker.
Saves one parquet file per ticker + a combined all_news.parquet.

HOW TO RUN:
    pip install gdeltdoc pandas pyarrow
    python fetch_news.py

OUTPUT:
    data/raw/news/AAPL_news.parquet
    data/raw/news/MSFT_news.parquet
    ...
    data/raw/news/all_news.parquet
─────────────────────────────────────────────────────────────────────────────
"""

import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from gdeltdoc import GdeltDoc, Filters
from market_calendar import news_session
import logging

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("nuroquant-news")

# ── Config ─────────────────────────────────────────────────────────────────────
# Path is relative to this file so it works both locally and on Render
BASE_DIR      = Path(__file__).resolve().parent
OUTPUT_FOLDER = BASE_DIR / "data" / "raw" / "news"

# GDELT's DOC 2.0 API (this library) only searches a rolling recent window —
# it is NOT a full historical archive (that requires BigQuery/bulk GKG files,
# a separate integration). Finnhub's free tier has the same ~1-year cap.
# The existing news snapshot is from 2026; training prices end in 2024.
# WSB ends in August 2021, leaving the 2023-2024 test period without text.
# The current live XGBoost model uses FINANCE features only. News feeds on
# the dashboard are not evidence of news features entering that model.
# See docs/DATA_AND_LIVE_EVALUATION.md before designing sentiment experiments.
#
# A prior version hardcoded a fixed 2026-01-22..2026-04-22 window instead of
# computing it relative to "today" — combined with the unconditional
# skip-if-file-exists caching below, that meant a ticker's news, once
# fetched, silently never refreshed again, contradicting the "real-time"
# framing used elsewhere (main.py's /market-sentiment, /market-news).
ROLLING_WINDOW_DAYS = 90
END_DATE   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
START_DATE = (datetime.now(timezone.utc) - timedelta(days=ROLLING_WINDOW_DAYS)).strftime("%Y-%m-%d")

# Re-fetch a ticker once its cached parquet is older than this, instead of
# caching forever — keeps "live" news actually live without re-fetching all
# 44 tickers (GDELT rate limits) on every single run.
CACHE_MAX_AGE_HOURS = 20

MAX_RECORDS = 250  # max articles per ticker per fetch (GDELT cap is 250)

# ── Ticker → Company Name Mapping ─────────────────────────────────────────────
# The production 44-ticker universe (backend/data/processed/features_finance.csv
# — the same set frontend/src/data/companyNames.js documents). A prior version
# of this dict was a stale ~49-ticker list from before the universe was
# expanded/changed (notebooks/14_expand_stocks.py) — only 23 tickers
# overlapped, so GDELT news was silently never fetched for the other 21
# production tickers (including SPY, NFLX, TSLA's overlap was fine but e.g.
# GOOGL/GOOG duplication and 26 non-production tickers like PG/V/GE wasted
# fetch effort). Use descriptive keywords (not bare tickers) for better GDELT
# search precision — longer/more unique names = fewer false positives.
TICKER_TO_NAME = {
    "AAPL":  "Apple",
    "ABBV":  "AbbVie",
    "ADBE":  "Adobe",
    "AMD":   "AMD semiconductor",
    "AMZN":  "Amazon",
    "BAC":   "Bank of America",
    "C":     "Citigroup",
    "COP":   "ConocoPhillips",
    "COST":  "Costco",
    "CRM":   "Salesforce",
    "CVS":   "CVS Health",
    "CVX":   "Chevron",
    "DIA":   "Dow Jones Industrial Average",
    "GLD":   "gold price bullion",
    "GOOGL": "Google Alphabet",
    "GS":    "Goldman Sachs",
    "INTC":  "Intel",
    "IWM":   "Russell 2000 small cap",
    "JNJ":   "Johnson Johnson pharmaceutical",
    "JPM":   "JPMorgan",
    "MCD":   "McDonalds restaurant",
    "META":  "Meta Platforms Facebook",
    "MRNA":  "Moderna",
    "MS":    "Morgan Stanley bank",
    "MSFT":  "Microsoft",
    "NFLX":  "Netflix",
    "NKE":   "Nike",
    "NVDA":  "Nvidia",
    "ORCL":  "Oracle",
    "OXY":   "Occidental Petroleum",
    "PFE":   "Pfizer",
    "PYPL":  "PayPal",
    "QQQ":   "Nasdaq 100",
    "SLB":   "Schlumberger",
    "SNAP":  "Snap Snapchat",
    "SPY":   "S&P 500",
    "TGT":   "Target retailer",
    "TLT":   "Treasury bond yield",
    "TSLA":  "Tesla",
    "UBER":  "Uber",
    "UNH":   "UnitedHealth",
    "WFC":   "Wells Fargo",
    "WMT":   "Walmart",
    "XOM":   "ExxonMobil",
}


# ── Fetch Function ─────────────────────────────────────────────────────────────
def fetch_news(ticker: str, keyword: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch news articles from GDELT for a single ticker.
    Retries up to 3 times on failure with exponential backoff.

    Returns a cleaned DataFrame with columns:
        ticker | date | title | url | source
    Returns empty DataFrame if no articles found or all retries fail.
    """
    gd = GdeltDoc()

    for attempt in range(1, 4):
        try:
            f = Filters(keyword=keyword, start_date=start, end_date=end)
            articles = gd.article_search(f)

            if articles is None or articles.empty:
                logger.warning(f"No articles found for {ticker} ({keyword})")
                return pd.DataFrame()

            # GDELT seendate is provider discovery time, not publication time.
            # For prospective evaluation we cannot use an article before our
            # collector actually received it, even if discovery predates retrieval.
            retrieved = pd.Timestamp.now(tz="UTC")
            articles["ticker"] = ticker
            articles["source_seen_at"] = pd.to_datetime(articles["seendate"], utc=True, errors="coerce")
            articles["retrieved_at"] = retrieved.isoformat()
            articles["availability_basis"] = "collector_retrieval"
            articles["feature_session"] = news_session(retrieved)
            articles["date"] = articles["source_seen_at"].dt.strftime("%Y-%m-%d")
            articles["source_seen_at"] = articles["source_seen_at"].astype(str)
            return articles[["ticker", "date", "title", "url", "domain", "source_seen_at",
                             "retrieved_at", "availability_basis", "feature_session"]].rename(
                columns={"domain": "source"})

        except Exception as e:
            logger.warning(f"Attempt {attempt}/3 failed for {ticker}: {e}")
            if attempt < 3:
                time.sleep(5 * attempt)  # wait 5s, then 10s before retrying
            else:
                logger.error(f"All retries failed for {ticker}, skipping.")
                return pd.DataFrame()

    return pd.DataFrame()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    all_frames = []
    skipped    = []
    failed     = []

    for ticker, keyword in TICKER_TO_NAME.items():
        parquet_path = OUTPUT_FOLDER / f"{ticker}_news.parquet"

        # Skip tickers fetched recently — caching to avoid re-fetching every
        # run (and hitting GDELT rate limits), but not forever: a cache that
        # never expires means "live" news silently goes stale indefinitely.
        if parquet_path.exists():
            age_hours = (time.time() - parquet_path.stat().st_mtime) / 3600
            if age_hours < CACHE_MAX_AGE_HOURS:
                logger.info(f"Skipping {ticker} — fetched {age_hours:.1f}h ago (< {CACHE_MAX_AGE_HOURS}h cache)")
                df = pd.read_parquet(parquet_path)
                all_frames.append(df)
                skipped.append(ticker)
                continue
            logger.info(f"{ticker} cache is {age_hours:.1f}h old — refreshing")

        logger.info(f"Fetching {ticker} ({keyword}) ...")
        df = fetch_news(ticker, keyword, START_DATE, END_DATE)

        if not df.empty:
            if parquet_path.exists():
                previous = pd.read_parquet(parquet_path)
                # Prefer earliest observed timestamped row. Legacy date-only
                # rows retain unknown availability until an actual retrieval.
                df = pd.concat([previous, df], ignore_index=True)
                df = df.sort_values("retrieved_at", na_position="last").drop_duplicates(["ticker", "url"], keep="first")
            df.to_parquet(parquet_path, engine="pyarrow", index=False)
            logger.info(f"  Saved {len(df)} rows → {parquet_path.name}")
            all_frames.append(df)
        else:
            failed.append(ticker)
            if parquet_path.exists():
                all_frames.append(pd.read_parquet(parquet_path))

        # Small delay between requests to avoid rate limiting
        time.sleep(1)

    # ── Save combined file ─────────────────────────────────────────────────────
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)

        # Remove duplicates (same article can appear for different date windows)
        combined = combined.drop_duplicates(subset=["ticker", "url"])

        combined_path = OUTPUT_FOLDER / "all_news.parquet"
        combined.to_parquet(combined_path, engine="pyarrow", index=False)

        logger.info(
            f"\nDone! {combined.shape[0]} articles across "
            f"{combined['ticker'].nunique()} tickers"
        )
        logger.info(f"Saved combined → {combined_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── Summary ───────────────────────────────────────────")
    print(f"  Fetched / loaded : {len(TICKER_TO_NAME) - len(failed)} tickers")
    print(f"  Skipped (cached) : {len(skipped)} tickers")
    print(f"  Failed           : {len(failed)} tickers {failed if failed else ''}")
    print("──────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
