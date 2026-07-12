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
from gdeltdoc import GdeltDoc, Filters
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
# This means GDELT/Finnhub news sentiment is only ever available for the
# LIVE/current-day signal module (main.py's live_signals.py), never for
# historical model training — the historical 2015-2024 training set's
# sentiment signal comes from WSB (data/processed/wsb_sentiment.csv), which
# genuinely has that depth. See build_features.py for how the two are merged;
# gdelt_* columns are correctly all-zero for historical (pre-2026) rows.
START_DATE = "2026-01-22"
END_DATE   = "2026-04-22"

MAX_RECORDS = 250  # max articles per ticker per fetch (GDELT cap is 250)

# ── Ticker → Company Name Mapping ─────────────────────────────────────────────
# Use descriptive company names (not just tickers) for better GDELT search results
# Longer/more unique names = fewer false positives
TICKER_TO_NAME = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "NVDA":  "Nvidia",
    "AMZN":  "Amazon",
    "META":  "Meta Platforms Facebook",
    "GOOGL": "Google Alphabet",
    "GOOG":  "Google Alphabet",
    "TSLA":  "Tesla",
    "BRK-B": "Berkshire Hathaway",
    "UNH":   "UnitedHealth",
    "LLY":   "Eli Lilly",
    "JPM":   "JPMorgan",
    "V":     "Visa payment card",
    "XOM":   "ExxonMobil",
    "MA":    "Mastercard credit card",
    "AVGO":  "Broadcom",
    "PG":    "Procter Gamble",
    "HD":    "Home Depot",
    "COST":  "Costco",
    "MRK":   "Merck pharmaceutical",
    "ABBV":  "AbbVie",
    "CVX":   "Chevron",
    "ADBE":  "Adobe",
    "CRM":   "Salesforce",
    "PEP":   "PepsiCo",
    "KO":    "Coca-Cola",
    "WMT":   "Walmart",
    "MCD":   "McDonalds restaurant",
    "BAC":   "Bank of America",
    "TMO":   "Thermo Fisher Scientific",
    "CSCO":  "Cisco",
    "ACN":   "Accenture",
    "ABT":   "Abbott Laboratories",
    "LIN":   "Linde industrial gas",
    "DHR":   "Danaher",
    "NEE":   "NextEra Energy",
    "TXN":   "Texas Instruments",
    "NKE":   "Nike",
    "PM":    "Philip Morris tobacco",
    "ORCL":  "Oracle",
    "RTX":   "Raytheon",
    "MS":    "Morgan Stanley bank",
    "QCOM":  "Qualcomm",
    "HON":   "Honeywell",
    "UPS":   "United Parcel Service",
    "AMGN":  "Amgen",
    "INTC":  "Intel",
    "IBM":   "IBM corporation",
    "AMD":   "AMD semiconductor",
    "GE":    "General Electric",
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

            # Add ticker and clean up the date column
            articles["ticker"] = ticker
            articles["date"] = (
                pd.to_datetime(articles["seendate"], errors="coerce")
                .dt.date
                .astype(str)
            )

            # Keep only the columns we need
            return articles[["ticker", "date", "title", "url", "domain"]].rename(
                columns={"domain": "source"}
            )

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

        # Skip tickers we already fetched (caching — avoids re-fetching on re-run)
        if parquet_path.exists():
            logger.info(f"Skipping {ticker} — already fetched")
            df = pd.read_parquet(parquet_path)
            all_frames.append(df)
            skipped.append(ticker)
            continue

        logger.info(f"Fetching {ticker} ({keyword}) ...")
        df = fetch_news(ticker, keyword, START_DATE, END_DATE)

        if not df.empty:
            df.to_parquet(parquet_path, engine="pyarrow", index=False)
            logger.info(f"  Saved {len(df)} rows → {parquet_path.name}")
            all_frames.append(df)
        else:
            failed.append(ticker)

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
