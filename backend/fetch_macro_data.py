"""
backend/fetch_macro_data.py
─────────────────────────────────────────────────────────────────────────────
Fetches market-wide macro context (VIX, credit spread proxy, yield-curve
proxy) via yfinance and saves one row per trading date — genuinely new
information no single ticker's own OHLCV/technicals can encode, since it's
identical across all 44 tickers on a given date (unlike the price-level
indicators fixed in build_features.py, there's no cross-ticker scaling
concern here: every ticker gets the same macro value on the same date).

Columns:
    date, vix_level, vix_change_pct, credit_spread_proxy, yield_curve_proxy

    vix_level             — CBOE Volatility Index, raw close
    vix_change_pct        — day-over-day % change in VIX (shifts in fear)
    credit_spread_proxy   — HYG/LQD price ratio (high-yield vs investment-
                             grade corporate bond ETFs); falling = widening
                             credit spreads = risk aversion
    yield_curve_proxy     — TLT/SHY price ratio (long vs short-duration
                             Treasury ETFs); captures the rate environment

RUN THIS BEFORE: build_features.py
HOW TO RUN:      python fetch_macro_data.py
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nuroquant-macro")

BASE_DIR = Path(__file__).resolve().parent
OUT_FILE = BASE_DIR / "data" / "processed" / "macro_context.csv"

START = "2015-01-01"
END   = "2024-12-31"

TICKERS = {
    "vix" : "^VIX",
    "hyg" : "HYG",
    "lqd" : "LQD",
    "tlt" : "TLT",
    "shy" : "SHY",
}


def fetch_close(symbol: str) -> pd.Series:
    df = yf.download(symbol, start=START, end=END, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"]
    s.index = pd.to_datetime(s.index).strftime("%Y-%m-%d")
    return s


def main():
    logger.info("Fetching VIX + macro ETF history via yfinance ...")
    closes = {}
    for key, symbol in TICKERS.items():
        logger.info(f"  {symbol} ...")
        closes[key] = fetch_close(symbol)

    df = pd.DataFrame(closes)
    df.index.name = "date"
    df = df.reset_index()

    df["vix_level"]           = df["vix"]
    df["vix_change_pct"]      = df["vix"].pct_change()
    df["credit_spread_proxy"] = df["hyg"] / df["lqd"]
    df["yield_curve_proxy"]   = df["tlt"] / df["shy"]

    out = df[["date", "vix_level", "vix_change_pct", "credit_spread_proxy", "yield_curve_proxy"]].dropna()
    out.to_csv(OUT_FILE, index=False)
    logger.info(f"Saved {len(out):,} rows -> {OUT_FILE}")
    print(out.head())
    print(out.tail())


if __name__ == "__main__":
    main()
