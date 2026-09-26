"""Shared definition of the prediction task; distinct from strategy holding time."""

import pandas as pd

LABEL_HORIZON_TRADING_DAYS = 5
TARGET_DESCRIPTION = "Close higher after five observed trading sessions"


def forward_direction(close: pd.Series, horizon: int = LABEL_HORIZON_TRADING_DAYS):
    """Per-ticker, ascending-session labels. Unknown future outcomes stay missing."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    future = close.shift(-horizon)
    returns = future / close - 1
    labels = (returns > 0).astype(float).where(future.notna() & close.notna())
    return returns, labels
