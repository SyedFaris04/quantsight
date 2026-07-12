"""
backend/live_signals.py
─────────────────────────────────────────────────────────────────────────────
Live Signal Generator — fetches today's stock data from Yahoo Finance,
computes fresh technical indicators, and runs the trained XGBoost model
to generate a current BUY/SELL signal.

IMPORTANT ACADEMIC NOTE:
    The model was trained on data from 2015–2023 and tested on 2023–2024.
    Signals generated here are based on pattern recognition from that period.
    Market conditions change over time — treat these as educational signals,
    not guaranteed financial advice.

USED BY:
    main.py → GET /live/{ticker}
    main.py → GET /overview  (when live=true query param)
─────────────────────────────────────────────────────────────────────────────
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("nuroquant-live")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "data" / "models"

# We use XGBoost finance model for live signals — most reliable on fresh data
# XGBoost generalises better than LSTM on out-of-distribution dates
XGB_MODEL_PATH = MODELS_DIR / "xgb_finance.pkl"


# ── Technical Indicator Functions (same as build_features.py) ─────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd        = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal,  adjust=False).mean()
    histogram   = macd - signal_line
    return macd, signal_line, histogram


def compute_bollinger(series: pd.Series, period=20, std_dev=2.0):
    middle = series.rolling(window=period).mean()
    std    = series.rolling(window=period).std()
    return middle + std_dev * std, middle, middle - std_dev * std


def build_features_from_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw OHLCV DataFrame from yfinance and computes
    the same technical indicators used during model training.

    Returns a DataFrame with the most recent row of features ready
    to feed into the XGBoost model.
    """
    df = df.copy().sort_index()

    close  = df["Close"]
    volume = df["Volume"]

    df["rsi"]          = compute_rsi(close)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(close)
    df["bb_upper"], df["bb_mid"], df["bb_lower"]   = compute_bollinger(close)
    df["bb_width"]     = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["sma_10"]       = close.rolling(10).mean()
    df["sma_20"]       = close.rolling(20).mean()
    df["ema_12"]       = close.ewm(span=12, adjust=False).mean()
    df["ema_26"]       = close.ewm(span=26, adjust=False).mean()
    df["daily_return"] = close.pct_change()
    df["return_5"]     = close.pct_change(5)
    df["return_10"]    = close.pct_change(10)
    df["volume_change"]= volume.pct_change()
    df["volume_sma_10"]= volume.rolling(10).mean()
    df["volume_ratio"] = volume / df["volume_sma_10"]
    df["volatility_20"]= close.rolling(20).std()
    df["atr"]          = (df["High"] - df["Low"]).rolling(14).mean()
    df["high_low_range"]   = (df["High"] - df["Low"]) / close
    df["close_open_gap"]   = (df["Close"] - df["Open"]) / df["Open"]

    return df.dropna()


# ── Model Loader ───────────────────────────────────────────────────────────────

_model_cache = {}

def load_xgb_model():
    """Load the trained XGBoost model from disk (cached after first load)."""
    if "xgb" in _model_cache:
        return _model_cache["xgb"]

    if not XGB_MODEL_PATH.exists():
        logger.warning(f"XGBoost model not found at {XGB_MODEL_PATH}")
        return None

    try:
        with open(XGB_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        _model_cache["xgb"] = bundle
        logger.info(f"Loaded XGBoost model from {XGB_MODEL_PATH.name}")
        return bundle
    except Exception as e:
        logger.error(f"Failed to load XGBoost model: {e}")
        return None


# ── Main Live Signal Function ──────────────────────────────────────────────────

def get_live_signal(ticker: str) -> dict:
    """
    Fetches live OHLCV data for a ticker, computes fresh indicators,
    and returns a BUY/SELL signal from the trained XGBoost model.

    Returns a dict with:
        ticker, signal_label, confidence, date, source,
        indicators (rsi, macd, bb_position),
        disclaimer
    """
    ticker = ticker.upper()

    # ── Step 1: Fetch live data from Yahoo Finance ─────────────────────────────
    try:
        import yfinance as yf

        # Download 90 days so rolling indicators (SMA50, BB20) are stable
        raw = yf.download(ticker, period="90d", interval="1d", progress=False)

        if raw.empty or len(raw) < 30:
            return _fallback_signal(ticker, reason="Not enough live data from Yahoo Finance")

        # yfinance returns MultiIndex columns sometimes — flatten
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        raw.index = pd.to_datetime(raw.index)

    except ImportError:
        return _fallback_signal(ticker, reason="yfinance not installed")
    except Exception as e:
        logger.warning(f"yfinance fetch failed for {ticker}: {e}")
        return _fallback_signal(ticker, reason=f"Live data fetch failed: {str(e)[:60]}")

    # ── Step 2: Compute technical indicators ───────────────────────────────────
    try:
        featured = build_features_from_ohlcv(raw)
        if featured.empty:
            return _fallback_signal(ticker, reason="Could not compute indicators from live data")
    except Exception as e:
        logger.warning(f"Indicator computation failed for {ticker}: {e}")
        return _fallback_signal(ticker, reason="Indicator computation failed")

    # ── Step 3: Load model and predict ────────────────────────────────────────
    bundle = load_xgb_model()
    if bundle is None:
        return _fallback_signal(ticker, reason="Model not loaded — run train_xgboost.py first")

    model        = bundle["model"]
    scaler       = bundle["scaler"]
    feature_cols = bundle["features"]

    # Get the most recent row of features
    latest = featured.iloc[-1]
    latest_date = featured.index[-1].strftime("%Y-%m-%d")

    # Build feature vector — use only columns the model was trained on
    # Fill missing columns with 0 (neutral)
    feature_vector = []
    missing_cols   = []
    for col in feature_cols:
        val = latest.get(col, np.nan)
        if pd.isna(val):
            feature_vector.append(0.0)
            missing_cols.append(col)
        else:
            feature_vector.append(float(val))

    if missing_cols:
        logger.debug(f"{ticker}: {len(missing_cols)} missing features filled with 0 — {missing_cols[:5]}")

    X = np.array(feature_vector).reshape(1, -1)

    try:
        X_scaled    = scaler.transform(X)
        probability = float(model.predict_proba(X_scaled)[0][1])
        signal      = 1 if probability >= 0.50 else 0
        confidence  = round(probability * 100, 1)
        signal_label = "BUY" if signal == 1 else "SELL"
    except Exception as e:
        logger.warning(f"Model prediction failed for {ticker}: {e}")
        return _fallback_signal(ticker, reason="Model prediction failed")

    # ── Step 4: Extract key indicators for display ─────────────────────────────
    rsi      = round(float(latest.get("rsi", np.nan)), 1) if not pd.isna(latest.get("rsi", np.nan)) else None
    macd_val = float(latest.get("macd", 0))
    macd_sig = float(latest.get("macd_signal", 0))
    bb_upper = float(latest.get("bb_upper", 0))
    bb_lower = float(latest.get("bb_lower", 0))
    bb_mid   = float(latest.get("bb_mid", 0))
    close    = float(latest.get("Close", 0))

    macd_direction = "bullish" if macd_val > macd_sig else "bearish"
    bb_width       = bb_upper - bb_lower
    bb_position    = round(((close - bb_lower) / bb_width) * 100, 1) if bb_width > 0 else 50.0

    return {
        "ticker"        : ticker,
        "signal_label"  : signal_label,
        "signal"        : signal,
        "confidence"    : confidence,
        "date"          : latest_date,
        "source"        : "live",
        "close_price"   : round(close, 2),
        "indicators"    : {
            "rsi"             : rsi,
            "macd_direction"  : macd_direction,
            "bb_position_pct" : bb_position,
        },
        "disclaimer"    : (
            "Signal generated from live market data using a model trained on "
            "2015–2023 historical patterns. For educational purposes — not financial advice."
        ),
    }


def _fallback_signal(ticker: str, reason: str = "") -> dict:
    """
    Returns a neutral/unknown signal when live data cannot be fetched.
    Tells the frontend to fall back to the cached prediction.
    """
    return {
        "ticker"      : ticker,
        "signal_label": "N/A",
        "signal"      : None,
        "confidence"  : None,
        "date"        : None,
        "source"      : "fallback",
        "error"       : reason,
        "disclaimer"  : "Live signal unavailable — showing cached prediction.",
    }


def get_live_signals_batch(tickers: list) -> dict:
    """
    Generate live signals for multiple tickers.
    Returns dict of ticker → signal result.
    Used by the overview endpoint to refresh all signals at once.
    """
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = get_live_signal(ticker)
        except Exception as e:
            logger.warning(f"Live signal failed for {ticker}: {e}")
            results[ticker] = _fallback_signal(ticker, reason=str(e)[:80])
    return results
