"""Shared daily indicator primitives for historical builds and live inference."""
import numpy as np
import pandas as pd

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style exponential smoothing; verified against saved indicators after burn-in."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.mask((avg_loss == 0) & (avg_gain > 0), 100).mask((avg_loss == 0) & (avg_gain == 0), 50)


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


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True range with exponential Wilder smoothing, including overnight gaps."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
