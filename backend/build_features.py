"""
backend/build_features.py
─────────────────────────────────────────────────────────────────────────────
Merges all data sources into two clean feature datasets:

    features_finance.csv   — OHLCV + technical indicators ONLY
                             (used to train finance-only model variants)

    features_sentiment.csv — OHLCV + technical indicators
                             + WSB sentiment + GDELT sentiment
                             (used to train sentiment model variants)

TECHNICAL INDICATORS COMPUTED HERE:
    RSI(14), MACD, MACD Signal, Bollinger Bands (upper/mid/lower),
    SMA(10), SMA(50), EMA(12), EMA(26), Daily Return, Volume Change

TARGET COLUMN:
    signal — 1 if next day close > today close (BUY), else 0 (SELL)

HOW TO RUN:
    python build_features.py

RUN THIS AFTER:  score_news_sentiment_finbert.py (or score_news_sentiment.py for the legacy VADER version)
RUN THIS BEFORE: train_xgboost.py  AND  train_lstm.py
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("nuroquant-features")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
PROCESSED_DIR  = BASE_DIR / "data" / "processed"
OUTPUT_DIR     = BASE_DIR / "data" / "processed"

# Input files
WSB_FILE       = PROCESSED_DIR / "wsb_sentiment.csv"
GDELT_FILE     = PROCESSED_DIR / "gdelt_sentiment.csv"
EMOTION_FILE   = PROCESSED_DIR / "wsb_emotion.csv"
FINANCE_EMOTIONS = ["fear", "optimism", "anger", "excitement", "confusion", "disappointment"]

# Output files
FINANCE_OUT    = OUTPUT_DIR / "features_finance.csv"
SENTIMENT_OUT  = OUTPUT_DIR / "features_sentiment.csv"

# features_stock.csv already has all indicators pre-computed by notebook 07
# We use this as the base instead of recomputing from raw OHLCV
FEATURES_STOCK = PROCESSED_DIR / "features_stock.csv"


# ── Technical Indicator Functions ─────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — momentum oscillator (0–100)."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    """
    MACD = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD, signal)
    Returns (macd, signal_line, histogram)
    """
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd       = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram  = macd - signal_line
    return macd, signal_line, histogram


def compute_bollinger(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """
    Bollinger Bands — volatility bands around SMA.
    Returns (upper, middle, lower)
    """
    middle = series.rolling(window=period).mean()
    std    = series.rolling(window=period).std()
    upper  = middle + std_dev * std
    lower  = middle - std_dev * std
    return upper, middle, lower


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a per-ticker OHLCV DataFrame and adds all technical indicators.
    Expects columns: Date, Open, High, Low, Close, Volume
    """
    df = df.sort_values("Date").copy()

    close  = df["Close"]
    volume = df["Volume"]

    # RSI
    df["rsi"] = compute_rsi(close)

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(close)

    # Bollinger Bands
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = compute_bollinger(close)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # Moving averages
    df["sma_10"]  = close.rolling(10).mean()
    df["sma_50"]  = close.rolling(50).mean()
    df["ema_12"]  = close.ewm(span=12, adjust=False).mean()
    df["ema_26"]  = close.ewm(span=26, adjust=False).mean()

    # Price-based features
    df["daily_return"]   = close.pct_change()
    df["high_low_range"] = (df["High"] - df["Low"]) / close
    df["close_open_gap"] = (df["Close"] - df["Open"]) / df["Open"]

    # Volume features
    df["volume_change"]  = volume.pct_change()
    df["volume_sma_10"]  = volume.rolling(10).mean()
    df["volume_ratio"]   = volume / df["volume_sma_10"]

    # Target: 1 if NEXT day's close > today's close (BUY signal)
    df["signal"] = (close.shift(-1) > close).astype(int)

    return df


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds market-context features computed ACROSS tickers on the same date.
    Every other feature in this pipeline is computed per-ticker in total
    isolation — RSI, MACD, etc. never see the other 43 tickers — so the
    model has no way to distinguish "this stock moved" from "the whole
    market moved that day" (research memo, Tier 2 item 7). This is the only
    feature-engineering change that adds genuinely new information rather
    than re-deriving what a single ticker's own OHLCV already encodes.
    """
    df = df.copy()

    # How overbought/oversold this ticker is relative to the day's cross-
    # sectional median, rather than in absolute terms.
    df["rsi_rel_market"] = df["rsi"] - df.groupby("date")["rsi"].transform("median")

    # 5-day return relative to the day's median across all 44 tickers, and
    # relative to SPY specifically (is this stock leading or lagging the
    # benchmark, not just "is it up").
    df["return5_rel_market"] = df["return_5"] - df.groupby("date")["return_5"].transform("median")
    spy_return5 = df[df["ticker"] == "SPY"].set_index("date")["return_5"]
    df["spy_return_5"]    = df["date"].map(spy_return5)
    df["return5_rel_spy"] = df["return_5"] - df["spy_return_5"]

    # Unusual volume relative to the rest of the market that day.
    df["volchg_rel_market"] = df["volume_change"] - df.groupby("date")["volume_change"].transform("median")

    # Market-wide turbulence that day (cross-sectional dispersion of returns) —
    # context a single ticker's own volatility_20 can't capture.
    df["market_volatility"] = df.groupby("date")["return_5"].transform("std")

    return df


# ── Load OHLCV files ───────────────────────────────────────────────────────────

def load_all_ohlcv() -> pd.DataFrame:
    """
    Reads all per-ticker CSV files from data/processed/.
    Accepts two formats:
      - Single-ticker: Date,Open,High,Low,Close,Volume  (filename = TICKER_clean.csv)
      - Multi-ticker:  Date,Open,High,Low,Close,Volume,Ticker
    Returns a single DataFrame with a 'Ticker' column.
    """
    all_frames = []

    # Only load per-ticker clean files (AAPL_clean.csv etc.)
    # Exclude everything that is NOT a raw OHLCV file
    exclude = {
        "gdelt_sentiment.csv", "wsb_sentiment.csv", "wsb_clean.csv",
        "features_finance.csv", "features_sentiment.csv", "features_stock.csv",
        "final_dataset.csv", "xgb_predictions.csv", "lstm_predictions.csv",
        "yahoo_news_clean.csv", "yahoo_news_sentiment.csv",
    }
    csv_files = list(PROCESSED_DIR.glob("*_clean.csv"))  # only TICKER_clean.csv files
    csv_files = [f for f in csv_files if f.name not in exclude]

    if not csv_files:
        logger.error(
            f"No OHLCV CSV files found in {PROCESSED_DIR}.\n"
            "Expected files like AAPL_clean.csv with columns: Date,Open,High,Low,Close,Volume,Ticker"
        )
        return pd.DataFrame()

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"])

            # If the file already has a Ticker column (all your _clean.csv files do)
            if "Ticker" in df.columns:
                all_frames.append(df)
                logger.info(f"  Loaded {csv_path.name} ({len(df):,} rows)")

            # Fallback: infer ticker from filename
            else:
                ticker = csv_path.stem.replace("_clean", "").upper()
                df["Ticker"] = ticker
                all_frames.append(df)
                logger.info(f"  Loaded {csv_path.name} → ticker={ticker} ({len(df):,} rows)")

        except Exception as e:
            logger.warning(f"  Skipping {csv_path.name}: {e}")

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    return combined


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load feature data ──────────────────────────────────────────────
    # Prefer features_stock.csv (already has indicators from notebook 07)
    # Fall back to computing from raw OHLCV if not available
    if FEATURES_STOCK.exists():
        logger.info(f"Found features_stock.csv — using pre-computed indicators ...")
        data = pd.read_csv(FEATURES_STOCK)
        data["Date"] = pd.to_datetime(data["Date"])

        # Standardise column names to lowercase for consistency
        rename_map = {
            "Date": "date", "Ticker": "ticker",
            "SMA_5": "sma_5", "SMA_10": "sma_10", "SMA_20": "sma_20",
            "EMA_12": "ema_12", "EMA_26": "ema_26",
            "MACD": "macd", "MACD_signal": "macd_signal", "MACD_hist": "macd_hist",
            "RSI": "rsi", "BB_upper": "bb_upper", "BB_lower": "bb_lower",
            "BB_mid": "bb_mid", "BB_width": "bb_width",
            "Volatility_20": "volatility_20", "ATR": "atr",
            "Volume_change": "volume_change", "Volume_MA_10": "volume_sma_10",
            "Return_5": "return_5", "Return_10": "return_10",
            "Forward_return": "forward_return", "Target": "signal",
        }
        data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})
        data["date"] = data["date"].dt.strftime("%Y-%m-%d")

        logger.info(f"  Loaded {len(data):,} rows, {data['ticker'].nunique()} tickers")

    else:
        # ── Fallback: compute from raw OHLCV ──────────────────────────────────
        logger.info("features_stock.csv not found — computing indicators from raw OHLCV ...")
        ohlcv = load_all_ohlcv()

        if ohlcv.empty:
            logger.error("No OHLCV data found. Aborting.")
            return

        logger.info(f"Total OHLCV rows: {len(ohlcv):,} across {ohlcv['Ticker'].nunique()} tickers")

        logger.info("Computing technical indicators ...")
        ticker_frames = []
        for ticker, group in ohlcv.groupby("Ticker"):
            enriched = add_indicators(group)
            ticker_frames.append(enriched)

        data = pd.concat(ticker_frames, ignore_index=True)
        data = data.rename(columns={"Date": "date", "Ticker": "ticker"})
        data["date"] = data["date"].dt.strftime("%Y-%m-%d")

        before = len(data)
        data = data.dropna(subset=["rsi", "macd", "bb_mid", "signal"])
        logger.info(f"Dropped {before - len(data):,} rows with NaN indicators")

    # ── Step 2b: Cross-sectional (market-relative) features ───────────────────
    logger.info("Computing cross-sectional market-relative features ...")
    data = add_cross_sectional_features(data)

    # ── Step 3: Save features_finance.csv ─────────────────────────────────────
    # Keep all available indicator columns (flexible — works with both sources)
    always_exclude = {"forward_return", "news_sentiment", "news_count",
                      "wsb_sentiment", "wsb_count", "wsb_avg_score", "combined_sentiment"}
    finance_cols = [c for c in data.columns if c not in always_exclude]
    finance_df = data[finance_cols].copy()
    finance_df.to_csv(FINANCE_OUT, index=False)
    logger.info(f"Saved features_finance.csv → {len(finance_df):,} rows, {finance_df.shape[1]} columns")

    # ── Step 4: Load WSB sentiment ─────────────────────────────────────────────
    wsb_df = pd.DataFrame()
    if WSB_FILE.exists():
        logger.info("Loading WSB sentiment ...")
        wsb_df = pd.read_csv(WSB_FILE)
        # Normalise column names — handle different possible formats
        wsb_df.columns = wsb_df.columns.str.lower().str.strip()
        if "ticker" not in wsb_df.columns or "date" not in wsb_df.columns:
            logger.warning("WSB file missing 'ticker' or 'date' column — skipping")
            wsb_df = pd.DataFrame()
        else:
            wsb_df["date"] = pd.to_datetime(wsb_df["date"]).dt.strftime("%Y-%m-%d")
            logger.info(f"  Loaded {len(wsb_df):,} WSB rows")
    else:
        logger.warning(f"WSB file not found at {WSB_FILE} — sentiment dataset will use GDELT only")

    # ── Step 5: Load GDELT sentiment ──────────────────────────────────────────
    gdelt_df = pd.DataFrame()
    if GDELT_FILE.exists():
        logger.info("Loading GDELT sentiment ...")
        gdelt_df = pd.read_csv(GDELT_FILE)
        gdelt_df["date"] = pd.to_datetime(gdelt_df["date"]).dt.strftime("%Y-%m-%d")
        logger.info(f"  Loaded {len(gdelt_df):,} GDELT rows")
    else:
        logger.warning(
            f"GDELT sentiment not found at {GDELT_FILE}\n"
            "Run score_news_sentiment_finbert.py first (or score_news_sentiment.py for VADER)."
        )

    # ── Step 5b: Load WSB multi-label emotion (GoEmotions) ────────────────────
    emotion_df = pd.DataFrame()
    if EMOTION_FILE.exists():
        logger.info("Loading WSB emotion (GoEmotions) ...")
        emotion_df = pd.read_csv(EMOTION_FILE)
        emotion_df["date"] = pd.to_datetime(emotion_df["date"]).dt.strftime("%Y-%m-%d")
        logger.info(f"  Loaded {len(emotion_df):,} emotion-scored posts")
    else:
        logger.warning(
            f"WSB emotion file not found at {EMOTION_FILE}\n"
            "Run score_wsb_emotion.py first if emotion features are wanted."
        )

    # ── Step 6: Merge and save features_sentiment.csv ─────────────────────────
    sentiment_df = finance_df.copy()

    if not wsb_df.empty:
        # Your wsb_sentiment.csv columns: ticker, date, timestamp, title_clean, score, sentiment_score
        # Rename sentiment_score → wsb_sentiment so it's clear in the merged dataset
        if "sentiment_score" in wsb_df.columns:
            wsb_df = wsb_df.rename(columns={"sentiment_score": "wsb_sentiment"})

        # Aggregate to one row per (ticker, date) — take mean if multiple posts same day
        wsb_agg = (
            wsb_df.groupby(["ticker", "date"])
            .agg(wsb_sentiment=("wsb_sentiment", "mean"),
                 wsb_count=("wsb_sentiment", "count"))
            .reset_index()
        )

        sentiment_df = sentiment_df.merge(wsb_agg, on=["ticker", "date"], how="left")
        logger.info("Merged WSB sentiment into dataset")

    if not gdelt_df.empty:
        gdelt_cols = ["ticker", "date", "gdelt_compound", "gdelt_pos", "gdelt_neg", "gdelt_article_count"]
        gdelt_cols = [c for c in gdelt_cols if c in gdelt_df.columns]
        sentiment_df = sentiment_df.merge(
            gdelt_df[gdelt_cols],
            on=["ticker", "date"],
            how="left",
        )
        logger.info("Merged GDELT sentiment into dataset")

    if not emotion_df.empty:
        # Same aggregation pattern as WSB sentiment above: mean per (ticker, date)
        # across however many posts mentioned that ticker that day.
        emo_cols = [f"emo_{e}" for e in FINANCE_EMOTIONS if f"emo_{e}" in emotion_df.columns]
        agg_map = {c: (c, "mean") for c in emo_cols}
        emotion_agg = (
            emotion_df.groupby(["ticker", "date"])
            .agg(**agg_map)
            .reset_index()
        )
        sentiment_df = sentiment_df.merge(emotion_agg, on=["ticker", "date"], how="left")
        logger.info("Merged WSB emotion (GoEmotions) into dataset")

    # Fill missing sentiment values with 0 (neutral) — not all dates will have news
    sentiment_cols = [c for c in sentiment_df.columns if c not in finance_cols]
    sentiment_df[sentiment_cols] = sentiment_df[sentiment_cols].fillna(0)

    sentiment_df.to_csv(SENTIMENT_OUT, index=False)
    logger.info(f"Saved features_sentiment.csv → {len(sentiment_df):,} rows, {sentiment_df.shape[1]} columns")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n-- Summary -----------------------------------------------------------")
    print(f"  features_finance.csv   : {len(finance_df):,} rows | {finance_df.shape[1]} columns")
    print(f"  features_sentiment.csv : {len(sentiment_df):,} rows | {sentiment_df.shape[1]} columns")
    print(f"  Tickers                : {finance_df['ticker'].nunique()}")
    print(f"  Date range             : {finance_df['date'].min()} -> {finance_df['date'].max()}")
    print(f"  Signal balance         : {finance_df['signal'].mean():.1%} BUY days")
    print("------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
