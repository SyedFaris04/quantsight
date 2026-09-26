"""Daily XGBoost finance inference on the complete trained universe.

Uses completed NYSE bars, the saved calibrator and explicit provenance.
Raw probability selects direction, matching the existing training export.
"""

import pickle
import hashlib
from functools import lru_cache
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from cv_utils import apply_calibrator
from technical_indicators import compute_rsi, compute_macd, compute_bollinger, compute_atr
from market_calendar import latest_completed_session, forecast_window, utc_now, schedule
from prediction_contract import LABEL_HORIZON_TRADING_DAYS
import logging

from build_features import add_normalized_indicators, add_cross_sectional_features, orderflow_columns, SECTOR_MAP

logger = logging.getLogger("nuroquant-live")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
MODELS_DIR     = BASE_DIR / "data" / "models"
PROCESSED_DIR  = BASE_DIR / "data" / "processed"

# The live endpoint serves the saved XGBoost finance model.
XGB_MODEL_PATH         = MODELS_DIR / "xgb_finance.pkl"
FEATURES_FINANCE_PATH  = PROCESSED_DIR / "features_finance.csv"


@lru_cache(maxsize=1)
def feature_code_version():
    digest = hashlib.sha256()
    for filename in ["live_signals.py", "technical_indicators.py", "build_features.py", "market_calendar.py"]:
        digest.update(filename.encode())
        digest.update((BASE_DIR / filename).read_text(encoding="utf-8").replace("\r\n", "\n").encode())
    return "daily-v2:" + digest.hexdigest()


# ── Technical Indicator Functions (same formulas as notebooks/07_feature_engineering.py,
#    which is the actual source of the trained features_stock.csv indicators) ──

def build_features_from_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw OHLCV DataFrame from yfinance for ONE ticker and computes
    every single-ticker-computable indicator, including the ratio-to-Close
    "_norm" versions (add_normalized_indicators, imported from
    build_features.py). Cross-sectional (market-relative) features are NOT
    computable here — they require the full ticker panel and are added
    separately by _fetch_market_panel().
    """
    df = df.copy().sort_index()

    close  = df["Close"]
    volume = df["Volume"]

    df["rsi"]          = compute_rsi(close)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(close)
    df["bb_upper"], df["bb_mid"], df["bb_lower"]   = compute_bollinger(close)
    df["bb_width"]     = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["sma_5"]        = close.rolling(5).mean()
    df["sma_10"]       = close.rolling(10).mean()
    df["sma_20"]       = close.rolling(20).mean()
    df["sma_50"]       = close.rolling(50).mean()
    df["ema_12"]       = close.ewm(span=12, adjust=False).mean()
    df["ema_26"]       = close.ewm(span=26, adjust=False).mean()
    df["daily_return"] = close.pct_change()
    df["return_5"]     = close.pct_change(5)
    df["return_10"]    = close.pct_change(10)
    df["volume_change"]= volume.pct_change()
    df["volume_sma_10"]= volume.rolling(10).mean()
    df["volume_ratio"] = volume / df["volume_sma_10"]
    df["volatility_20"]= close.rolling(20).std()
    df["atr"]          = compute_atr(df)
    df["high_low_range"]   = (df["High"] - df["Low"]) / close
    df["close_open_gap"]   = (df["Close"] - df["Open"]) / df["Open"]

    # Order-flow proxies via the SAME helper training uses (orderflow_columns,
    # imported from build_features.py) — this df is already one ticker sorted
    # oldest→newest, exactly what the helper expects, so live values match the
    # trained model's clv/cmf_20/svi_10/mfi_14 exactly instead of being
    # zero-filled at inference (the fabrication-bug class fixed earlier).
    for col, series in orderflow_columns(df["High"], df["Low"], df["Close"], df["Volume"]).items():
        df[col] = series

    df = add_normalized_indicators(df)

    return df.dropna()


# ── Ticker universe ────────────────────────────────────────────────────────────

_ticker_universe_cache = None

def get_ticker_universe() -> list:
    """
    The exact ticker universe the models were trained on. Cross-sectional
    features (rsi_rel_market etc.) are only meaningful when computed across
    this same set — a different or partial universe would shift the
    medians/std relative to what the model learned from.
    """
    global _ticker_universe_cache
    if _ticker_universe_cache is not None:
        return _ticker_universe_cache
    try:
        df = pd.read_csv(FEATURES_FINANCE_PATH, usecols=["ticker"])
        _ticker_universe_cache = sorted(df["ticker"].unique().tolist())
    except Exception as e:
        logger.warning(f"Could not load ticker universe from {FEATURES_FINANCE_PATH}: {e}")
        _ticker_universe_cache = []
    return _ticker_universe_cache


# ── Market panel (batched fetch + cross-sectional features) ───────────────────

_panel_cache = {"data": None, "fetched_at": None}
PANEL_TTL = timedelta(minutes=15)


def validate_daily_history(frame, completed, minimum=300):
    """Reject missing/invalid bars instead of treating gaps as trading sessions."""
    dates = pd.to_datetime(frame.index).strftime("%Y-%m-%d")
    if len(dates) < minimum or dates.has_duplicates or dates[-1] != completed:
        raise ValueError("Insufficient or stale daily history")
    expected = schedule(dates[-minimum], completed).index.strftime("%Y-%m-%d")
    if list(dates[-minimum:]) != list(expected):
        raise ValueError("Daily history contains missing or unexpected sessions")
    values = frame.iloc[-minimum:][["Open", "High", "Low", "Close", "Volume"]]
    if not np.isfinite(values.to_numpy()).all() or (values.iloc[:, :4] <= 0).any().any():
        raise ValueError("Invalid OHLCV values")
    if (values.Volume < 0).any() or (values.High < values.Low).any():
        raise ValueError("Invalid volume or price range")


def _fetch_market_panel(tickers: list) -> pd.DataFrame:
    """
    Fetches live OHLCV for the whole ticker universe in ONE batched
    yfinance call, computes per-ticker + normalized indicators for each,
    concatenates into a single multi-ticker/multi-date panel, then layers
    the cross-sectional features on top (add_cross_sectional_features
    needs multiple tickers on the same date to compute medians/std against,
    exactly like build_features.py does at training time).
    """
    import yfinance as yf

    # Fetch the universe AND the sector ETFs in one batched call, so the live
    # panel can compute ret5_rel_sector (ticker's 5-day return minus its
    # sector ETF's) using matching completed sessions. Missing sector quotes
    # leave a missing feature and prevent prediction for the affected stock.
    sector_etfs = sorted(set(SECTOR_MAP.values()))
    raw = yf.download(
        list(tickers) + sector_etfs, period="2y", interval="1d",
        group_by="ticker", progress=False, threads=True, auto_adjust=True,
    )

    # Yahoo can return today's unfinished daily candle. Exclude it before features.
    completed = latest_completed_session()
    raw = raw.loc[pd.to_datetime(raw.index).strftime("%Y-%m-%d") <= completed]
    is_multi = isinstance(raw.columns, pd.MultiIndex)

    # Per-sector-ETF 5-day return, keyed by date string, for the lookup below.
    sector_ret5 = {}
    for etf in sector_etfs:
        try:
            sector_frame = raw[etf].dropna(how="all") if is_multi else raw.dropna(how="all")
            validate_daily_history(sector_frame, completed)
            close = sector_frame["Close"]
            r = close.pct_change(5)
            r.index = pd.to_datetime(r.index).strftime("%Y-%m-%d")
            sector_ret5[etf] = r
        except Exception as e:
            logger.debug(f"Sector ETF {etf} fetch skipped: {e}")

    frames = []

    for ticker in tickers:
        try:
            if is_multi:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                sub = raw[ticker].copy()
            else:
                sub = raw.copy()

            sub = sub.dropna(how="all")
            if sub.empty or len(sub) < 60:
                continue

            sub = sub[["Open", "High", "Low", "Close", "Volume"]].copy()
            sub.index = pd.to_datetime(sub.index)
            validate_daily_history(sub, completed)

            featured = build_features_from_ohlcv(sub)
            if featured.empty:
                continue

            featured = featured.reset_index()
            featured = featured.rename(columns={featured.columns[0]: "date"})
            featured["date"]   = pd.to_datetime(featured["date"]).dt.strftime("%Y-%m-%d")
            featured["ticker"] = ticker
            frames.append(featured)

        except Exception as e:
            logger.debug(f"Panel fetch skipped {ticker}: {e}")
            continue

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    # Cross-sectional statistics are invalid for a partial universe.
    complete_dates = panel.groupby("date")["ticker"].nunique()
    panel = panel[panel["date"].isin(complete_dates[complete_dates == len(tickers)].index)].copy()
    if panel.empty:
        return panel
    panel = add_cross_sectional_features(panel)

    # Sector-relative return: ticker's return_5 minus its sector ETF's, looked
    # up per (date, sector). Universe ETFs (SPY/QQQ/...) aren't in SECTOR_MAP
    # so they stay 0, matching training. Missing sector quotes remain missing.
    def _sret(row):
        etf = SECTOR_MAP.get(row["ticker"])
        if etf is None:
            return 0.0
        if etf not in sector_ret5:
            return np.nan
        v = sector_ret5[etf].get(row["date"], np.nan)
        return np.nan if pd.isna(v) else float(row["return_5"] - v)
    panel["ret5_rel_sector"] = panel.apply(_sret, axis=1)
    return panel


def _get_cached_panel() -> pd.DataFrame:
    """Panel is rebuilt from a fresh batched fetch at most once per
    PANEL_TTL — these are daily-bar indicators, they don't need to be
    recomputed on every request, and refetching 44+ tickers per request
    would be needlessly slow and hammer Yahoo Finance."""
    now = datetime.now(timezone.utc)
    cached, fetched_at = _panel_cache["data"], _panel_cache["fetched_at"]
    ttl = timedelta(seconds=60) if cached is not None and cached.empty else PANEL_TTL
    if cached is not None and fetched_at is not None and (now - fetched_at) < ttl:
        return cached

    # Cache failures briefly to avoid one failed full-universe download per ticker.
    _panel_cache["data"], _panel_cache["fetched_at"] = pd.DataFrame(), now
    tickers = get_ticker_universe()
    if not tickers:
        return pd.DataFrame()

    try:
        panel = _fetch_market_panel(tickers)
    except Exception as e:
        logger.warning(f"Market panel fetch failed: {e}")
        return pd.DataFrame()

    if panel.empty:
        return pd.DataFrame()

    _panel_cache["data"], _panel_cache["fetched_at"] = panel, now
    return panel


def _latest_row_per_ticker(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_values(["ticker", "date"]).groupby("ticker").tail(1).set_index("ticker")


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
        bundle["artifact_sha256"] = hashlib.sha256(XGB_MODEL_PATH.read_bytes()).hexdigest()
        _model_cache["xgb"] = bundle
        logger.info(f"Loaded XGBoost model from {XGB_MODEL_PATH.name}")
        return bundle
    except Exception as e:
        logger.error(f"Failed to load XGBoost model: {e}")
        return None


# ── Main Live Signal Function ──────────────────────────────────────────────────

def _predict_from_row(ticker: str, latest: pd.Series, latest_date: str, source: str) -> dict:
    bundle = load_xgb_model()
    if bundle is None:
        return _fallback_signal(ticker, reason="Model not loaded — run train_xgboost.py first")

    model, scaler, feature_cols = bundle["model"], bundle["scaler"], bundle["features"]

    feature_vector = []
    missing_cols   = []
    for col in feature_cols:
        val = latest.get(col, np.nan)
        if not np.isfinite(val):
            feature_vector.append(0.0)
            missing_cols.append(col)
        else:
            feature_vector.append(float(val))

    if missing_cols:
        return _fallback_signal(ticker, reason="Required model features unavailable: " + ", ".join(missing_cols))

    X = np.array(feature_vector).reshape(1, -1)

    try:
        X_scaled    = scaler.transform(X)
        probability = float(model.predict_proba(X_scaled)[0][1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            return _fallback_signal(ticker, reason="Invalid model probability")
        signal      = 1 if probability >= 0.50 else 0
        if bundle.get("calibrator") is None:
            return _fallback_signal(ticker, reason="Saved probability calibrator unavailable")
        calibrated = float(apply_calibrator(bundle["calibrator"], np.array([probability]),
                                             bundle["calibration_method"])[0])
        if not np.isfinite(calibrated) or not 0 <= calibrated <= 1:
            return _fallback_signal(ticker, reason="Invalid calibrated probability")
        confidence = round(calibrated * 100, 1)
        direction_confidence = round((calibrated if signal else 1 - calibrated) * 100, 1)
        signal_label = "BUY" if signal == 1 else "SELL"
    except Exception as e:
        logger.warning(f"Model prediction failed for {ticker}: {e}")
        return _fallback_signal(ticker, reason="Model prediction failed")

    rsi      = latest.get("rsi", np.nan)
    macd_val = float(latest.get("macd", 0))
    macd_sig = float(latest.get("macd_signal", 0))
    bb_upper = float(latest.get("bb_upper", 0))
    bb_lower = float(latest.get("bb_lower", 0))
    close    = float(latest.get("Close", 0))
    if not np.isfinite(close) or close <= 0:
        return _fallback_signal(ticker, reason="Invalid closing price")

    macd_direction = "bullish" if macd_val > macd_sig else "bearish"
    bb_width       = bb_upper - bb_lower
    bb_position    = round(((close - bb_lower) / bb_width) * 100, 1) if bb_width > 0 else 50.0

    return {
        "ticker"        : ticker,
        "signal_label"  : signal_label,
        "signal"        : signal,
        "confidence"    : confidence,
        "date"          : latest_date,
        "source"        : source,
        "close_price"   : close,
        "probability_up": calibrated,
        "raw_probability_up": probability,
        "direction_confidence": direction_confidence,
        "confidence_definition": "calibrated probability of price rising, in percent",
        "calibration_method": bundle["calibration_method"],
        "model_version": bundle["artifact_sha256"],
        "feature_version": feature_code_version(),
        "feature_values": dict(zip(feature_cols, feature_vector)),
        "horizon_sessions": LABEL_HORIZON_TRADING_DAYS,
        "target_date": forecast_window(latest_date)["target_date"],
        "data_cutoff_at": forecast_window(latest_date)["data_cutoff_at"],
        "generated_at": utc_now().isoformat(),
        "data_fetched_at": _panel_cache["fetched_at"].isoformat() if _panel_cache["fetched_at"] else None,
        "indicators"    : {
            "rsi"             : round(float(rsi), 1) if not pd.isna(rsi) else None,
            "macd_direction"  : macd_direction,
            "bb_position_pct" : bb_position,
        },
        "disclaimer"    : (
            "Signal generated from live market data using a model trained on "
            "2015–2023 historical patterns. For educational purposes — not financial advice."
        ),
    }


def get_live_signal(ticker: str) -> dict:
    """No prediction is issued from stale, partial or fabricated features."""
    ticker = ticker.upper()
    try:
        panel = _get_cached_panel()
        if panel.empty:
            return _fallback_signal(ticker, "Complete market panel unavailable")
        latest_panel = _latest_row_per_ticker(panel)
        if ticker not in latest_panel.index:
            return _fallback_signal(ticker, "Ticker unavailable in trained universe")
        row = latest_panel.loc[ticker]
        if row["date"] != latest_completed_session():
            return _fallback_signal(ticker, "Latest completed session is missing")
        return _predict_from_row(ticker, row, row["date"], source="live")
    except Exception as exc:
        logger.warning("Live inference failed for %s: %s", ticker, type(exc).__name__)
        return _fallback_signal(ticker, "Live inference unavailable")


def _fallback_signal(ticker: str, reason: str = "") -> dict:
    """
    Returns a neutral/unknown signal when live data cannot be fetched.
    Historical predictions must remain explicitly separate in clients.
    """
    return {
        "ticker"      : ticker,
        "signal_label": "N/A",
        "signal"      : None,
        "confidence"  : None,
        "date"        : None,
        "source"      : "fallback",
        "error"       : reason,
        "disclaimer"  : "Live signal unavailable. Historical results are a separate dataset.",
    }


def get_live_signals_batch(tickers: list) -> dict:
    """
    Generate live signals for multiple tickers. Shares one cached market
    panel fetch across the whole batch instead of one yfinance call per
    ticker — both faster and the reason each ticker gets correct
    cross-sectional features.
    """
    _get_cached_panel()  # warm the shared cache once before the loop

    results = {}
    for ticker in tickers:
        try:
            results[ticker] = get_live_signal(ticker)
        except Exception as e:
            logger.warning(f"Live signal failed for {ticker}: {e}")
            results[ticker] = _fallback_signal(ticker, reason=str(e)[:80])
    return results
