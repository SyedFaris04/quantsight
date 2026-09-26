"""
backend/fetch_sector_data.py
─────────────────────────────────────────────────────────────────────────────
Fetches daily history for the sector SPDR ETFs the 44-ticker universe spans,
and saves their 5- and 20-day returns per date. Feeds build_features.py's
add_sector_relative_features(), which computes each stock's return RELATIVE
to its own sector — genuinely new information a single ticker's OHLCV can't
contain, and the one lever (new info, not new modeling) that can move a
near-efficient signal (industry momentum is a robust, cost-surviving effect;
Moskowitz & Grinblatt 1999).

Complements the existing market-relative features (return5_rel_market/_spy),
which measure a stock vs the WHOLE market; this measures it vs its SECTOR,
a finer comparison (is AAPL leading tech, not just leading the market).

Output: data/processed/sector_returns.csv  (date, sector_etf, sret_5, sret_20)

RUN ONCE (or to refresh) BEFORE: build_features.py
"""

import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nuroquant-sector")

BASE_DIR      = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUT_FILE      = PROCESSED_DIR / "sector_returns.csv"

# Only the 6 sector ETFs the universe actually spans, all with full 2015
# coverage. Comm-services names (GOOGL/META/NFLX/SNAP) are mapped to XLK
# (tech) rather than XLC because XLC only lists from 2018-06 — XLK is a
# fully-covered, defensible growth/tech proxy for the 2015-2024 window.
SECTOR_ETFS = ["XLK", "XLY", "XLP", "XLF", "XLV", "XLE"]

START = "2015-01-01"
END   = "2025-01-01"


def main():
    import yfinance as yf

    logger.info(f"Fetching {len(SECTOR_ETFS)} sector ETFs {START}→{END} ...")
    raw = yf.download(SECTOR_ETFS, start=START, end=END, interval="1d",
                      group_by="ticker", progress=False, auto_adjust=True, threads=True)

    frames = []
    for etf in SECTOR_ETFS:
        try:
            close = raw[etf]["Close"].dropna()
        except Exception as e:
            logger.warning(f"  {etf}: fetch failed ({e}) — skipping")
            continue
        df = pd.DataFrame({"date": close.index})
        df["sector_etf"] = etf
        df["sret_5"]  = close.pct_change(5).values
        df["sret_20"] = close.pct_change(20).values
        frames.append(df)
        logger.info(f"  {etf}: {len(df):,} days")

    if not frames:
        logger.error("No sector data fetched — aborting.")
        return

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["sret_5", "sret_20"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    logger.info(f"Saved {len(out):,} rows → {OUT_FILE.name} ({out['sector_etf'].nunique()} sectors)")


if __name__ == "__main__":
    main()
