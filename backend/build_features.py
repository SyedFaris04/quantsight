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
    signal — 1 if close after five trading sessions > today's close, else 0

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
from prediction_contract import forward_direction
from technical_indicators import compute_rsi, compute_macd, compute_bollinger, compute_atr

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
MACRO_FILE     = PROCESSED_DIR / "macro_context.csv"
SECTOR_FILE    = PROCESSED_DIR / "sector_returns.csv"
FINANCE_EMOTIONS = ["fear", "optimism", "anger", "excitement", "confusion", "disappointment"]

# Ticker → sector-SPDR-ETF map for the sector-relative features (built by
# fetch_sector_data.py). Comm-services growth names (GOOGL/META/NFLX/SNAP)
# use XLK (tech) as a fully-covered proxy — XLC only lists from 2018. The 6
# broad index/commodity/bond ETFs in the universe (SPY/QQQ/DIA/IWM/GLD/TLT)
# have no single sector, so they get no sector-relative feature (fills 0).
SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "ADBE": "XLK", "CRM": "XLK",
    "ORCL": "XLK", "INTC": "XLK", "AMD": "XLK",
    "GOOGL": "XLK", "META": "XLK", "NFLX": "XLK", "SNAP": "XLK",
    "AMZN": "XLY", "TSLA": "XLY", "MCD": "XLY", "NKE": "XLY", "TGT": "XLY", "UBER": "XLY",
    "WMT": "XLP", "COST": "XLP",
    "JPM": "XLF", "BAC": "XLF", "C": "XLF", "GS": "XLF", "MS": "XLF", "WFC": "XLF", "PYPL": "XLF",
    "UNH": "XLV", "JNJ": "XLV", "PFE": "XLV", "MRNA": "XLV", "ABBV": "XLV", "CVS": "XLV",
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE", "OXY": "XLE",
}

# Output files
FINANCE_OUT    = OUTPUT_DIR / "features_finance.csv"
SENTIMENT_OUT  = OUTPUT_DIR / "features_sentiment.csv"

# features_stock.csv already has all indicators pre-computed by notebook 07
# We use this as the base instead of recomputing from raw OHLCV
FEATURES_STOCK = PROCESSED_DIR / "features_stock.csv"


# ── Technical Indicator Functions ─────────────────────────────────────────────

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
    df["sma_5"]   = close.rolling(5).mean()
    df["sma_20"]  = close.rolling(20).mean()
    df["sma_10"]  = close.rolling(10).mean()
    df["sma_50"]  = close.rolling(50).mean()
    df["ema_12"]  = close.ewm(span=12, adjust=False).mean()
    df["ema_26"]  = close.ewm(span=26, adjust=False).mean()

    df["return_5"] = close.pct_change(5)
    df["return_10"] = close.pct_change(10)
    df["volatility_20"] = close.rolling(20).std()
    df["atr"] = compute_atr(df)

    # Price-based features
    df["daily_return"]   = close.pct_change()
    df["high_low_range"] = (df["High"] - df["Low"]) / close
    df["close_open_gap"] = (df["Close"] - df["Open"]) / df["Open"]

    # Volume features
    df["volume_change"]  = volume.pct_change()
    df["volume_sma_10"]  = volume.rolling(10).mean()
    df["volume_ratio"]   = volume / df["volume_sma_10"]

    # Same five-session target as notebook 07 and the saved models. Unknown
    # tail outcomes remain NaN and are excluded from training below.
    _, df["signal"] = forward_direction(close)

    return df


def add_normalized_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    sma_10/sma_50/ema_12/ema_26/bb_upper/bb_mid/bb_lower/macd/macd_signal/
    macd_hist are absolute dollar values, kept as-is because copilot_engine.py
    formats them directly into explanation text (e.g. "SMA10 ($150.23)").
    But train_xgboost.py / train_lstm.py fit ONE StandardScaler across all
    44 pooled tickers, so a $700 stock's sma_50 and a $20 stock's sma_50
    share that scale — these features partly encode "which ticker is this"
    rather than pure trend. Adds *_norm ratio-to-Close counterparts (the
    same normalization bb_width/close_open_gap/high_low_range already use)
    for the models to actually train on; NON_FEATURE_COLS in the training
    scripts excludes the raw versions so both stay real, neither is lost.
    """
    df = df.copy()
    close = df["Close"] if "Close" in df.columns else df["close"]

    # sma_5/sma_20 (not sma_10/sma_50) is what the features_stock.csv path
    # actually produces, despite this file's own docstring and the fallback
    # add_indicators() above both saying sma_10/sma_50 — cover both naming
    # variants so whichever path is active gets fully normalized, not just
    # the documented-but-not-actually-used one.
    for col in ["sma_5", "sma_10", "sma_20", "sma_50", "ema_12", "ema_26", "bb_upper", "bb_mid", "bb_lower"]:
        if col in df.columns:
            df[f"{col}_norm"] = df[col] / close - 1

    for col in ["macd", "macd_signal", "macd_hist"]:
        if col in df.columns:
            df[f"{col}_norm"] = df[col] / close

    return df


def orderflow_columns(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series) -> dict:
    """
    Core order-flow proxy math for ONE ticker's chronological OHLCV series.
    Returns {clv, cmf_20, svi_10, mfi_14} as Series aligned to the input.

    Shared by add_orderflow_features() (training, applied per ticker) AND
    live_signals.py (inference, one ticker at a time) so the two can never
    drift numerically — the same discipline that keeps live RSI/ATR in sync
    with training after an earlier drift bug. Inputs must be sorted oldest→
    newest and belong to a single ticker (rolling windows assume contiguity).
    """
    rng = (high - low).replace(0, np.nan)
    # Close Location Value: where the close sits in the day's range.
    # +1 = closed at the high (buyers dominated), -1 = closed at the low.
    clv = (((close - low) - (high - close)) / rng).fillna(0.0)

    # Chaikin Money Flow (20d): volume-weighted CLV — sustained accumulation
    # (positive) vs distribution (negative) over the last ~month.
    cmf_20 = (clv * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)

    # Signed-volume imbalance (10d): daily-return sign × volume, normalized by
    # total volume — a daily-frequency tick-rule proxy for net order flow.
    signed_vol = np.sign(close.pct_change()) * vol
    svi_10 = signed_vol.rolling(10).sum() / vol.rolling(10).sum().replace(0, np.nan)

    # Money Flow Index (14d): volume-weighted RSI on the typical price — a
    # bounded [0,100] gauge that (unlike plain RSI) needs volume to confirm.
    tp  = (high + low + close) / 3.0
    rmf = tp * vol
    tp_change = tp.diff()
    pos_sum = rmf.where(tp_change > 0, 0.0).rolling(14).sum()
    neg_sum = rmf.where(tp_change < 0, 0.0).rolling(14).sum().replace(0, np.nan)
    # neg_sum == 0 (a 14-day window with no down-days) → MFI undefined via
    # divide-by-zero; the correct value there is 100 (fully bullish flow).
    mfi_14 = (100.0 - (100.0 / (1.0 + pos_sum / neg_sum))).fillna(100.0)

    return {"clv": clv, "cmf_20": cmf_20, "svi_10": svi_10, "mfi_14": mfi_14}


def add_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order-flow / buying-pressure proxies from daily OHLCV — the one signal
    family the existing features don't capture. RSI/MACD/Bollinger are
    price-only; volume_change/volume_sma_10 are volume-only; none combine
    WHERE the price closed in its range and in WHICH direction with HOW MUCH
    volume confirmed it. These estimate net buy-vs-sell pressure ("low-
    frequency order imbalance", Ha & Hu 2017) purely from daily bars — no
    intraday/tick data needed.

    Deliberately all naturally bounded (CLV/CMF/SVI in [-1,1], MFI in
    [0,100]) so, unlike raw price levels, they don't smuggle "which ticker is
    this" into the shared StandardScaler — the same scale-confound lesson
    that had spy_return_5 and raw OHLCV excluded from training.

    Computed on full per-ticker history BEFORE any warm-up row-drop, so the
    rolling windows see complete history; the ~20-row warm-up NaNs are then
    removed by the sma_50 50-day drop in main() (50 > every window here).
    Delegates the math to orderflow_columns() so live inference stays in sync.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    parts = {c: [] for c in ("clv", "cmf_20", "svi_10", "mfi_14")}
    for _, g in df.groupby("ticker", sort=False):
        cols = orderflow_columns(g["High"], g["Low"], g["Close"], g["Volume"])
        for c, s in cols.items():
            parts[c].append(s)
    for c, pieces in parts.items():
        df[c] = pd.concat(pieces)
    return df


def add_sector_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each stock's return RELATIVE to its own sector ETF — is it leading or
    lagging its sector, not just the market. Genuinely new information a
    single ticker's OHLCV can't contain (industry momentum, Moskowitz &
    Grinblatt 1999), and finer than the existing return5_rel_market/_spy
    (which compare to the whole market). Reads sector_returns.csv from
    fetch_sector_data.py; if absent, the feature is all-zero and harmless.

    ret5_rel_sector = ticker's 5-day return − its sector ETF's 5-day return.
    Only the 5-day horizon is used (matches the pipeline's return_5); the
    universe's broad index/commodity/bond ETFs (SPY/QQQ/... ) have no sector
    in SECTOR_MAP, so their value is 0 (neutral).
    """
    df = df.copy()
    df["ret5_rel_sector"] = 0.0

    if not SECTOR_FILE.exists():
        logger.warning(f"Sector returns not found at {SECTOR_FILE} — ret5_rel_sector left 0. "
                       "Run fetch_sector_data.py to enable it.")
        return df

    sec = pd.read_csv(SECTOR_FILE)
    sec["date"] = pd.to_datetime(sec["date"]).dt.strftime("%Y-%m-%d")
    sret5 = sec.set_index(["date", "sector_etf"])["sret_5"]

    if "return_5" not in df.columns:
        logger.warning("return_5 missing — ret5_rel_sector left 0.")
        return df

    sector_etf = df["ticker"].map(SECTOR_MAP)
    matched_sret5 = pd.Series(sret5.reindex(list(zip(df["date"], sector_etf))).values, index=df.index)

    has_sector = sector_etf.notna() & matched_sret5.notna()
    df.loc[has_sector, "ret5_rel_sector"] = (df["return_5"] - matched_sret5)[has_sector]
    df["ret5_rel_sector"] = df["ret5_rel_sector"].fillna(0.0)
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

    # ── Step 1a2: Order-flow / buying-pressure proxies from OHLCV. Computed
    # here on FULL per-ticker history, before the sma_50 warm-up drop below,
    # so the rolling windows (≤20 days) see complete history and their own
    # warm-up NaNs are cleaned by that 50-day drop. ────────────────────────────
    logger.info("Computing order-flow / buying-pressure proxies (CLV/CMF/SVI/MFI) ...")
    data = add_orderflow_features(data)

    # ── Step 1b: sma_50 — the features_stock.csv path never actually produced
    # this column (only sma_5/sma_10/sma_20 exist there), despite
    # copilot_engine.py's analyse_sma() reading row.get("sma_50", ...) and
    # this file's own module docstring documenting "SMA(10), SMA(50)" as
    # the intent. Silently NaN on the Detail page's SMA-crossing analysis
    # this whole time — compute it directly from Close, same as the
    # fallback OHLCV path already correctly does. ──────────────────────────────
    if "sma_50" not in data.columns:
        logger.info("sma_50 missing from source data — computing directly ...")
        data = data.sort_values(["ticker", "date"])
        data["sma_50"] = data.groupby("ticker")["Close"].transform(
            lambda s: s.rolling(50, min_periods=50).mean()
        )
        before = len(data)
        data = data.dropna(subset=["sma_50"])
        logger.info(f"  Dropped {before - len(data):,} rows in each ticker's first 49 days (no 50-day history yet)")

    # ── Step 2a: Normalized (ratio-to-Close) versions of price-level indicators ──
    logger.info("Computing normalized indicator ratios ...")
    data = add_normalized_indicators(data)

    # ── Step 2b: Cross-sectional (market-relative) features ───────────────────
    logger.info("Computing cross-sectional market-relative features ...")
    data = add_cross_sectional_features(data)

    # ── Step 2b2: Sector-relative return (ticker vs its sector ETF) ───────────
    logger.info("Computing sector-relative feature (ret5_rel_sector) ...")
    data = add_sector_relative_features(data)

    # ── Step 2c: Macro context (VIX, credit spread, yield curve) — TRIED AND
    # REVERTED. fetch_macro_data.py can still build macro_context.csv, but
    # this merge stays off by default: validated against the purged CV/
    # holdout split and it collapsed the LSTM's F1 (59%→18%, heavy all-SELL
    # bias) on both variants, most likely because these 4 features are
    # identical across all 44 tickers on a date (unlike everything else,
    # which is per-ticker) and have a very different scale/variance — VIX
    # alone spans roughly 10-80 with sharp spikes — from the rest of the
    # feature set, and a small (~11K-parameter) LSTM sharing one
    # StandardScaler over everything let them dominate the learned
    # representation. XGBoost's headline numbers looked fine with this on,
    # but AUC stayed just under 50%, consistent with the same feature
    # driving majority-class bias rather than genuine signal. Left here
    # (opt-in via ENABLE_MACRO_FEATURES) rather than deleted, since the
    # underlying idea may still be worth a properly-scoped retry — e.g. a
    # per-feature scaler, or feeding it only to XGBoost — not a repeat of
    # this exact attempt.
    ENABLE_MACRO_FEATURES = False
    if ENABLE_MACRO_FEATURES and MACRO_FILE.exists():
        logger.info("Merging macro context (VIX/credit/yield-curve) ...")
        macro_df = pd.read_csv(MACRO_FILE)
        before_cols = set(data.columns)
        data = data.merge(macro_df, on="date", how="left")
        new_cols = [c for c in data.columns if c not in before_cols]
        data[new_cols] = data[new_cols].ffill()
        logger.info(f"  Added: {new_cols}")

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

    # Missing text is not neutral sentiment. Retain observation flags before
    # numeric imputation. Legacy same-day joins have UNVERIFIED timing; these
    # flags describe coverage, not permission to claim causal news effects.
    for source, count in [("wsb", "wsb_count"), ("gdelt", "gdelt_article_count")]:
        if count in sentiment_df:
            sentiment_df[source + "_observed"] = sentiment_df[count].fillna(0).gt(0).astype(int)
    emotion_cols = [c for c in sentiment_df if c.startswith("emo_")]
    sentiment_df["emotion_observed"] = sentiment_df[emotion_cols].notna().any(axis=1).astype(int) if emotion_cols else 0
    sentiment_df["text_availability_verified"] = 0
    logger.warning("Legacy sentiment joins have unverified intraday availability; do not use them for confirmatory news-effect claims.")

    # Numeric imputation for estimator compatibility, with missingness retained.
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
