"""
backend/copilot_engine.py
─────────────────────────────────────────────────────────────────────────────
The AI Copilot Engine — generates human-readable explanations for why
each model variant produced a BUY or SELL signal for a given ticker.

This is the XAI (Explainable AI) layer of NuroQuant. Instead of showing
your lecturer a black-box prediction, this engine breaks down:

    1. What the technical indicators say        (RSI, MACD, Bollinger)
    2. What the sentiment data says             (WSB + GDELT)
    3. What the model's confidence level means
    4. How all 4 model variants compare         (do they agree or disagree?)

USED BY:
    main.py  →  GET /explain/{ticker}
    GET /compare/{ticker}

─────────────────────────────────────────────────────────────────────────────
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger("nuroquant-copilot")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
METRICS_FILE    = PREDICTIONS_DIR / "model_metrics.json"

# Prediction file mapping — new 4-variant files first, old files as fallback
PREDICTION_FILES = {
    "xgb_finance"    : PREDICTIONS_DIR / "xgb_finance_predictions.csv",
    "xgb_sentiment"  : PREDICTIONS_DIR / "xgb_sentiment_predictions.csv",
    "lstm_finance"   : PREDICTIONS_DIR / "lstm_finance_predictions.csv",
    "lstm_sentiment" : PREDICTIONS_DIR / "lstm_sentiment_predictions.csv",
}

# Old prediction files from the original training notebooks (fallback)
OLD_PREDICTION_FILES = {
    "xgb_finance"   : PREDICTIONS_DIR / "xgb_predictions.csv",
    "lstm_finance"  : PREDICTIONS_DIR / "lstm_predictions.csv",
}

# Per-prediction LSTM attention weights (extract_lstm_attention.py) — the model's
# own internal "which of the last 10 days mattered most" reasoning, not a
# post-hoc approximation. Only exists for the two LSTM variants.
ATTENTION_FILES = {
    "lstm_finance"   : PREDICTIONS_DIR / "lstm_finance_attention.json",
    "lstm_sentiment" : PREDICTIONS_DIR / "lstm_sentiment_attention.json",
}

FEATURE_FILES = {
    "finance"   : PROCESSED_DIR / "features_finance.csv",
    "sentiment" : PROCESSED_DIR / "features_sentiment.csv",
}

# In-memory cache of loaded CSVs, keyed by resolved file path. get_latest_features()
# and get_latest_prediction() are called once per ticker (44x) per /overview request —
# without this, each call re-read and re-parsed the whole CSV from disk, which is what
# made /overview and /dashboard take 17-18s per request.
_CSV_CACHE: dict[str, pd.DataFrame] = {}


def _read_csv_cached(path: Path) -> pd.DataFrame:
    key = str(path)
    if key not in _CSV_CACHE:
        _CSV_CACHE[key] = pd.read_csv(path)
    return _CSV_CACHE[key]


_JSON_CACHE: dict[str, dict] = {}


def _read_json_cached(path: Path) -> dict:
    key = str(path)
    if key not in _JSON_CACHE:
        with open(path) as f:
            _JSON_CACHE[key] = json.load(f)
    return _JSON_CACHE[key]


def load_lstm_attention(ticker: str, date: str, model_key: str = "lstm_sentiment") -> Optional[list]:
    """
    Looks up the per-day attention weights the LSTM used for one ticker's
    prediction on one date (see extract_lstm_attention.py). Returns a list
    of {date, weight} for the 10 trading days leading up to `date`, oldest
    first, or None if no attention data exists for that exact key (e.g. the
    ticker/date wasn't in the LSTM's test set).
    """
    attention_file = ATTENTION_FILES.get(model_key)
    if not attention_file or not attention_file.exists():
        return None
    try:
        data = _read_json_cached(attention_file)
    except Exception as e:
        logger.warning(f"Failed to load attention weights from {attention_file.name}: {e}")
        return None
    return data.get(f"{ticker}|{date[:10]}")

# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class IndicatorSignal:
    """Represents one technical indicator's contribution to the signal."""
    name       : str
    value      : float
    signal     : str          # "bullish" | "bearish" | "neutral"
    reason     : str          # plain English explanation
    weight     : float        # contribution weight 0–1 (for the bar chart)


@dataclass
class SentimentSignal:
    """Represents sentiment data contribution."""
    source     : str          # "WSB" | "GDELT"
    score      : float        # compound score -1 to +1
    signal     : str          # "positive" | "negative" | "neutral"
    reason     : str
    weight     : float


@dataclass
class ModelPrediction:
    """One model variant's prediction for a ticker on a date."""
    model_key       : str     # e.g. "xgb_finance"
    model_label     : str     # e.g. "XGBoost (Finance Only)"
    signal_label    : str     # "BUY" | "SELL"
    signal          : int     # 1 | 0
    confidence      : float   # 0–100
    uses_sentiment  : bool


@dataclass
class CopilotExplanation:
    """Full explanation object returned to the frontend."""
    ticker              : str
    date                : str
    overall_signal      : str           # majority vote across all 4 models
    overall_confidence  : float         # average confidence
    agreement_level     : str           # "Strong" | "Moderate" | "Mixed"
    models              : list          # list of ModelPrediction dicts
    indicator_signals   : list          # list of IndicatorSignal dicts
    sentiment_signals   : list          # list of SentimentSignal dicts
    summary             : str           # 2–3 sentence plain English summary
    technical_score     : float         # 0–100 (how bullish technicals are)
    sentiment_score     : float         # 0–100 (how bullish sentiment is)
    model_score         : float         # 0–100 (avg model confidence for BUY)


# ── Label Helpers ──────────────────────────────────────────────────────────────

MODEL_LABELS = {
    "xgb_finance"    : "XGBoost (Finance Only)",
    "xgb_sentiment"  : "XGBoost (Finance + Sentiment)",
    "lstm_finance"   : "LSTM+Transformer (Finance Only)",
    "lstm_sentiment" : "LSTM+Transformer (Finance + Sentiment)",
}

USES_SENTIMENT = {
    "xgb_finance"    : False,
    "xgb_sentiment"  : True,
    "lstm_finance"   : False,
    "lstm_sentiment" : True,
}


# ── Indicator Analysis ─────────────────────────────────────────────────────────

def analyse_rsi(rsi: float) -> IndicatorSignal:
    """
    RSI interpretation:
        < 30  → oversold  → bullish (price likely to bounce up)
        > 70  → overbought → bearish (price likely to drop)
        30–70 → neutral
    """
    if pd.isna(rsi):
        return IndicatorSignal("RSI", 0, "neutral", "RSI data not available", 0.0)

    if rsi < 30:
        signal = "bullish"
        reason = f"RSI at {rsi:.1f} — stock is oversold, historically signals a price rebound"
        weight = 0.85
    elif rsi < 45:
        signal = "bullish"
        reason = f"RSI at {rsi:.1f} — mildly oversold, slight upward pressure expected"
        weight = 0.55
    elif rsi > 70:
        signal = "bearish"
        reason = f"RSI at {rsi:.1f} — stock is overbought, historically signals a pullback"
        weight = 0.15
    elif rsi > 55:
        signal = "bearish"
        reason = f"RSI at {rsi:.1f} — approaching overbought territory, momentum slowing"
        weight = 0.40
    else:
        signal = "neutral"
        reason = f"RSI at {rsi:.1f} — within neutral zone (30–55), no strong directional signal"
        weight = 0.50

    return IndicatorSignal("RSI (14)", rsi, signal, reason, weight)


def analyse_macd(macd: float, macd_signal: float) -> IndicatorSignal:
    """
    MACD interpretation:
        MACD > Signal → bullish crossover (momentum accelerating upward)
        MACD < Signal → bearish crossover (momentum slowing)
    """
    if pd.isna(macd) or pd.isna(macd_signal):
        return IndicatorSignal("MACD", 0, "neutral", "MACD data not available", 0.0)

    diff = macd - macd_signal

    if diff > 0.5:
        signal = "bullish"
        reason = (
            f"MACD ({macd:.3f}) is above signal line ({macd_signal:.3f}) — "
            "strong bullish momentum crossover"
        )
        weight = 0.80
    elif diff > 0:
        signal = "bullish"
        reason = (
            f"MACD ({macd:.3f}) slightly above signal ({macd_signal:.3f}) — "
            "mild bullish momentum"
        )
        weight = 0.60
    elif diff < -0.5:
        signal = "bearish"
        reason = (
            f"MACD ({macd:.3f}) below signal line ({macd_signal:.3f}) — "
            "strong bearish momentum crossover"
        )
        weight = 0.20
    else:
        signal = "bearish"
        reason = (
            f"MACD ({macd:.3f}) slightly below signal ({macd_signal:.3f}) — "
            "mild bearish pressure"
        )
        weight = 0.40

    return IndicatorSignal("MACD", macd, signal, reason, weight)


def analyse_bollinger(close: float, bb_upper: float,
                      bb_mid: float, bb_lower: float) -> IndicatorSignal:
    """
    Bollinger Band interpretation:
        Price near lower band → potentially oversold, bullish
        Price near upper band → potentially overbought, bearish
        Price near middle     → neutral
    """
    if any(pd.isna(v) for v in [close, bb_upper, bb_mid, bb_lower]):
        return IndicatorSignal("Bollinger Bands", 0, "neutral", "BB data not available", 0.0)

    band_width = bb_upper - bb_lower
    if band_width == 0:
        return IndicatorSignal("Bollinger Bands", close, "neutral", "Bands too narrow to interpret", 0.5)

    # Position within bands: 0 = at lower, 1 = at upper
    position = (close - bb_lower) / band_width

    if position < 0.20:
        signal = "bullish"
        reason = (
            f"Price (${close:.2f}) is near the lower Bollinger Band (${bb_lower:.2f}) — "
            "oversold condition, mean reversion likely upward"
        )
        weight = 0.80
    elif position < 0.40:
        signal = "bullish"
        reason = (
            f"Price (${close:.2f}) in lower half of bands — "
            "below midline (${bb_mid:.2f}), mild bullish bias"
        )
        weight = 0.60
    elif position > 0.80:
        signal = "bearish"
        reason = (
            f"Price (${close:.2f}) near upper Bollinger Band (${bb_upper:.2f}) — "
            "overbought condition, potential pullback"
        )
        weight = 0.20
    elif position > 0.60:
        signal = "bearish"
        reason = (
            f"Price (${close:.2f}) in upper half of bands — "
            "above midline, mild bearish bias"
        )
        weight = 0.40
    else:
        signal = "neutral"
        reason = (
            f"Price (${close:.2f}) near Bollinger midline (${bb_mid:.2f}) — "
            "consolidating, no strong directional signal"
        )
        weight = 0.50

    return IndicatorSignal("Bollinger Bands", close, signal, reason, weight)


def analyse_sma(close: float, sma_10: float, sma_50: float) -> IndicatorSignal:
    """
    SMA cross interpretation:
        Price > SMA50 and SMA10 > SMA50 → uptrend
        Price < SMA50 → downtrend
    """
    if any(pd.isna(v) for v in [close, sma_10, sma_50]):
        return IndicatorSignal("SMA Cross", 0, "neutral", "SMA data not available", 0.0)

    if close > sma_50 and sma_10 > sma_50:
        signal = "bullish"
        reason = (
            f"Price (${close:.2f}) and SMA10 (${sma_10:.2f}) both above SMA50 (${sma_50:.2f}) — "
            "confirmed uptrend"
        )
        weight = 0.75
    elif close < sma_50 and sma_10 < sma_50:
        signal = "bearish"
        reason = (
            f"Price (${close:.2f}) and SMA10 (${sma_10:.2f}) both below SMA50 (${sma_50:.2f}) — "
            "confirmed downtrend"
        )
        weight = 0.25
    elif close > sma_50:
        signal = "bullish"
        reason = (
            f"Price (${close:.2f}) above SMA50 (${sma_50:.2f}) — "
            "long-term uptrend intact"
        )
        weight = 0.65
    else:
        signal = "bearish"
        reason = (
            f"Price (${close:.2f}) below SMA50 (${sma_50:.2f}) — "
            "price under long-term average, bearish bias"
        )
        weight = 0.35

    return IndicatorSignal("SMA (10/50)", close, signal, reason, weight)


# ── Sentiment Analysis ─────────────────────────────────────────────────────────

def analyse_gdelt_sentiment(compound: float, article_count: int) -> SentimentSignal:
    """Interpret GDELT daily compound score."""
    if pd.isna(compound):
        return SentimentSignal(
            "GDELT News", 0.0, "neutral",
            "No news articles found for this ticker on this date", 0.5
        )

    confidence_note = (
        f"Based on {article_count} news article{'s' if article_count != 1 else ''}"
    )

    if compound > 0.2:
        signal = "positive"
        reason = f"GDELT news sentiment is positive (score: {compound:+.3f}). {confidence_note} with broadly positive coverage"
        weight = min(0.5 + compound * 0.4, 0.95)
    elif compound < -0.2:
        signal = "negative"
        reason = f"GDELT news sentiment is negative (score: {compound:+.3f}). {confidence_note} with broadly negative coverage"
        weight = max(0.5 + compound * 0.4, 0.05)
    else:
        signal = "neutral"
        reason = f"GDELT news sentiment is neutral (score: {compound:+.3f}). {confidence_note} with mixed or minimal coverage"
        weight = 0.50

    return SentimentSignal("GDELT News", compound, signal, reason, weight)


def analyse_wsb_sentiment(row: pd.Series) -> Optional[SentimentSignal]:
    """
    Interpret WSB/Reddit sentiment.
    Handles multiple possible column name formats.
    Returns None if no WSB data available.
    """
    # Try to find WSB sentiment column — handle different naming conventions
    wsb_col = None
    for col in ["wsb_sentiment", "sentiment_score", "wsb_score",
                "compound", "wsb_compound", "reddit_sentiment"]:
        if col in row.index and not pd.isna(row[col]):
            wsb_col = col
            break

    if wsb_col is None:
        return None

    score = float(row[wsb_col])

    if score > 0.2:
        signal = "positive"
        reason = f"Reddit/WSB community is bullish on this stock (sentiment: {score:+.3f})"
        weight = min(0.5 + score * 0.4, 0.95)
    elif score < -0.2:
        signal = "negative"
        reason = f"Reddit/WSB community is bearish on this stock (sentiment: {score:+.3f})"
        weight = max(0.5 + score * 0.4, 0.05)
    else:
        signal = "neutral"
        reason = f"Reddit/WSB sentiment is mixed or low volume (sentiment: {score:+.3f})"
        weight = 0.50

    return SentimentSignal("Reddit/WSB", score, signal, reason, weight)


# Finance-relevant subset of GoEmotions' 27 labels (proposal §6.5.2) — the
# only ones scored by score_wsb_emotion.py and merged into features_sentiment.csv.
EMOTION_COLUMNS = {
    "emo_fear"           : ("Fear",           "negative"),
    "emo_optimism"       : ("Optimism",       "positive"),
    "emo_anger"          : ("Anger",          "negative"),
    "emo_excitement"     : ("Excitement",     "positive"),
    "emo_confusion"      : ("Confusion",      "neutral"),
    "emo_disappointment" : ("Disappointment", "negative"),
}

# GoEmotions probabilities are small even for a genuinely-present emotion —
# it's a 28-way multi-label competition, not a single dominant score — so
# the bar is much lower than the sentiment module's +-0.2 cutoff.
EMOTION_THRESHOLD = 0.05


def analyse_wsb_emotion(row: pd.Series) -> list[SentimentSignal]:
    """
    Interpret WSB/Reddit multi-label emotion (GoEmotions taxonomy).
    Unlike sentiment (one polarity score), a post can carry several emotions
    at once — this returns one SentimentSignal per finance-relevant emotion
    that clears EMOTION_THRESHOLD, sorted by intensity (dominant first).
    Returns an empty list if no emotion data is available for this row
    (most rows: WSB coverage is ~3.5% of trading days, same as sentiment).
    """
    available = [c for c in EMOTION_COLUMNS if c in row.index and not pd.isna(row[c])]
    if not available or all(float(row[c]) == 0 for c in available):
        return []

    signals = []
    for col in available:
        score = float(row[col])
        if score < EMOTION_THRESHOLD:
            continue
        label, polarity = EMOTION_COLUMNS[col]
        reason = f"Reddit/WSB posts show {label.lower()} (intensity: {score:.3f})"
        # Scale the small raw probability into a more legible 0-1 weight for
        # the frontend's progress-bar display, same spirit as the sentiment
        # weight scaling above.
        weight = min(max(score * 3, 0.05), 0.95)
        signals.append(SentimentSignal(f"WSB Emotion: {label}", score, polarity, reason, weight))

    signals.sort(key=lambda s: s.score, reverse=True)
    return signals


# ── Agreement Analysis ─────────────────────────────────────────────────────────

def get_agreement_level(predictions: list[ModelPrediction]) -> str:
    """
    Classify how much the 4 model variants agree.
        4/4 same → "Strong"
        3/4 same → "Moderate"
        2/4 same → "Mixed"
    """
    buy_count = sum(1 for p in predictions if p.signal == 1)
    if buy_count == 4 or buy_count == 0:
        return "Strong"
    elif buy_count == 3 or buy_count == 1:
        return "Moderate"
    else:
        return "Mixed"


def majority_signal(predictions: list[ModelPrediction]) -> tuple[str, float]:
    """
    Returns (consensus_signal_label, average_confidence).

    A genuine 2-2 split between the 4 model variants means the models
    disagree — this is reported as "HOLD", not forced into a BUY, so
    the Overview/Detail pages accurately reflect model uncertainty
    (Decision Support Layer, proposal Section 6.5.2 — Consensus Engine).
    """
    buy_count  = sum(1 for p in predictions if p.signal == 1)
    sell_count = len(predictions) - buy_count
    avg_conf   = np.mean([p.confidence for p in predictions])

    if buy_count > sell_count:
        signal_label = "BUY"
    elif sell_count > buy_count:
        signal_label = "SELL"
    else:
        signal_label = "HOLD"

    return signal_label, round(avg_conf, 1)


# ── Summary Generator ──────────────────────────────────────────────────────────

def generate_summary(ticker: str, overall_signal: str, agreement: str,
                     indicator_signals: list[IndicatorSignal],
                     sentiment_signals: list[SentimentSignal],
                     models: list[ModelPrediction]) -> str:
    """Generate a 2–3 sentence plain English summary for the dashboard."""

    # Count bullish indicators
    bullish_indicators = [s for s in indicator_signals if s.signal == "bullish"]
    bearish_indicators = [s for s in indicator_signals if s.signal == "bearish"]

    # Sentiment summary
    positive_sentiment = [s for s in sentiment_signals if s.signal == "positive"]
    negative_sentiment = [s for s in sentiment_signals if s.signal == "negative"]

    # Models that agree with overall signal
    agreeing_models = [m for m in models if m.signal_label == overall_signal]

    # Build summary
    parts = []

    # Part 1 — technical
    if len(bullish_indicators) > len(bearish_indicators):
        ind_names = ", ".join(s.name for s in bullish_indicators[:2])
        parts.append(
            f"{ticker} shows bullish technical signals from {ind_names}, "
            f"suggesting upward price momentum."
        )
    elif len(bearish_indicators) > len(bullish_indicators):
        ind_names = ", ".join(s.name for s in bearish_indicators[:2])
        parts.append(
            f"{ticker} shows bearish technical signals from {ind_names}, "
            f"suggesting downward price pressure."
        )
    else:
        parts.append(
            f"{ticker} shows mixed technical signals with no clear directional bias."
        )

    # Part 2 — sentiment (only if available)
    if sentiment_signals:
        if len(positive_sentiment) > len(negative_sentiment):
            parts.append(
                f"Market sentiment from news and social media is broadly positive, "
                f"supporting the {overall_signal} signal."
            )
        elif len(negative_sentiment) > len(positive_sentiment):
            parts.append(
                f"Market sentiment from news and social media is broadly negative, "
                f"adding conviction to the {overall_signal} signal."
            )

    # Part 3 — model agreement
    if agreement == "Strong":
        parts.append(
            f"All 4 model variants agree on a {overall_signal} signal — "
            f"this is a high-confidence recommendation."
        )
    elif agreement == "Moderate":
        parts.append(
            f"{len(agreeing_models)} out of 4 model variants signal {overall_signal} — "
            f"moderate confidence, consider risk management."
        )
    else:
        parts.append(
            f"Model variants are divided — treat this signal with caution "
            f"and consider waiting for stronger confirmation."
        )

    return " ".join(parts)


# ── Main Engine Function ───────────────────────────────────────────────────────

def get_latest_features(ticker: str, use_sentiment: bool) -> Optional[pd.Series]:
    """Get the most recent feature row for a ticker from the feature CSV."""
    file_key = "sentiment" if use_sentiment else "finance"
    feature_file = FEATURE_FILES[file_key]

    if not feature_file.exists():
        logger.warning(f"Feature file not found: {feature_file}")
        return None

    try:
        df = _read_csv_cached(feature_file)
        df = df[df["ticker"] == ticker].sort_values("date")
        if df.empty:
            return None
        return df.iloc[-1]  # most recent row
    except Exception as e:
        logger.error(f"Error loading features for {ticker}: {e}")
        return None


def get_latest_prediction(ticker: str, model_key: str) -> Optional[dict]:
    """
    Get the most recent prediction for a ticker.
    Tries new 4-variant files first, falls back to old prediction files.
    Handles both old column names (xgb_pred/xgb_prob, lstm_pred/lstm_prob)
    and new column names (predicted_signal/confidence).
    """
    # Try new prediction file first
    pred_file = PREDICTION_FILES.get(model_key)
    if pred_file and pred_file.exists():
        try:
            df = _read_csv_cached(pred_file)
            df = df[df["ticker"] == ticker].sort_values("date")
            if not df.empty:
                return df.iloc[-1].to_dict()
        except Exception as e:
            logger.error(f"Error loading {pred_file.name}: {e}")

    # Fall back to old prediction files — normalise column names
    old_file = OLD_PREDICTION_FILES.get(model_key)
    if old_file and old_file.exists():
        try:
            df = _read_csv_cached(old_file).copy()
            df.columns = df.columns.str.strip()

            # Normalise column names to match what the rest of the code expects
            col_map = {}
            if "Ticker" in df.columns: col_map["Ticker"] = "ticker"
            if "Date"   in df.columns: col_map["Date"]   = "date"
            if "xgb_pred"  in df.columns: col_map["xgb_pred"]  = "predicted_signal"
            if "xgb_prob"  in df.columns: col_map["xgb_prob"]  = "confidence_raw"
            if "lstm_pred" in df.columns: col_map["lstm_pred"] = "predicted_signal"
            if "lstm_prob" in df.columns: col_map["lstm_prob"] = "confidence_raw"
            if "Target"    in df.columns: col_map["Target"]    = "actual_signal"
            df = df.rename(columns=col_map)

            # Convert probability (0-1) to confidence (0-100)
            if "confidence_raw" in df.columns:
                df["confidence"] = df["confidence_raw"] * 100
            if "predicted_signal" in df.columns:
                df["signal_label"] = df["predicted_signal"].map({1: "BUY", 0: "SELL"})

            df["ticker"] = df["ticker"].astype(str).str.upper()
            df = df[df["ticker"] == ticker.upper()].sort_values("date")
            if not df.empty:
                return df.iloc[-1].to_dict()
        except Exception as e:
            logger.error(f"Error loading fallback {old_file.name}: {e}")

    logger.warning(f"No prediction found for {ticker} ({model_key})")
    return None


def calculate_risk_level(row, model_predictions, overall_confidence: float) -> dict:
    """
    Computes an investment risk level (Low / Medium / High) for a ticker
    by combining market volatility, Average True Range (ATR), and the
    confidence margin from the model consensus.

    Logic:
        - High volatility / ATR relative to price  -> higher risk
        - Low consensus confidence (near 50%)       -> higher risk
        - Model disagreement (mixed signals)         -> higher risk

    Returns a dict with: level ("Low"/"Medium"/"High"), score (0-100,
    higher = riskier), and a short reason string for display.
    """
    reasons = []
    risk_points = 0  # 0-100 scale, higher = riskier

    # ── Volatility component (0-40 points) ──────────────────────────────────
    volatility_pct = None
    if row is not None:
        close = row.get("Close", np.nan)
        vol_20 = row.get("volatility_20", np.nan)
        if not pd.isna(vol_20) and not pd.isna(close) and close > 0:
            # Normalise volatility as a % of price
            volatility_pct = round((float(vol_20) / float(close)) * 100, 2)
            if volatility_pct >= 4.0:
                risk_points += 40
                reasons.append(f"high volatility ({volatility_pct}% of price)")
            elif volatility_pct >= 2.0:
                risk_points += 22
                reasons.append(f"moderate volatility ({volatility_pct}% of price)")
            else:
                risk_points += 8
                reasons.append(f"low volatility ({volatility_pct}% of price)")

    # ── ATR component (0-25 points) ─────────────────────────────────────────
    atr_pct = None
    if row is not None:
        close = row.get("Close", np.nan)
        atr   = row.get("atr", np.nan)
        if not pd.isna(atr) and not pd.isna(close) and close > 0:
            atr_pct = round((float(atr) / float(close)) * 100, 2)
            if atr_pct >= 3.0:
                risk_points += 25
                reasons.append(f"wide daily trading range (ATR {atr_pct}% of price)")
            elif atr_pct >= 1.5:
                risk_points += 14
            else:
                risk_points += 5

    # ── Confidence margin component (0-20 points) ───────────────────────────
    # Confidence near 50% (coin-flip) is riskier than confidence near 100%/0%
    confidence_margin = abs(overall_confidence - 50.0)  # 0 (uncertain) to 50 (certain)
    if confidence_margin <= 5:
        risk_points += 20
        reasons.append("very low prediction confidence margin")
    elif confidence_margin <= 15:
        risk_points += 10
    else:
        risk_points += 2

    # ── Model disagreement component (0-15 points) ──────────────────────────
    if model_predictions:
        buy_count  = sum(1 for p in model_predictions if p.signal == 1)
        sell_count = len(model_predictions) - buy_count
        if buy_count > 0 and sell_count > 0:
            # Models disagree
            disagreement_ratio = min(buy_count, sell_count) / len(model_predictions)
            if disagreement_ratio >= 0.5:
                risk_points += 15
                reasons.append("models are split between BUY and SELL")
            else:
                risk_points += 8
                reasons.append("one model disagrees with the majority")

    risk_points = min(risk_points, 100)

    if risk_points >= 60:
        level = "High"
    elif risk_points >= 32:
        level = "Medium"
    else:
        level = "Low"

    # Build a short, readable reason (top 2 contributing factors)
    reason_text = "; ".join(reasons[:2]) if reasons else "insufficient data for full risk assessment"

    return {
        "level"          : level,
        "score"          : round(risk_points, 1),
        "volatility_pct" : volatility_pct,
        "atr_pct"        : atr_pct,
        "reason"         : reason_text.capitalize() + ".",
    }


def _short_reason_for_row(row, signal_label: str) -> str:
    """
    Generates a brief, one-line reason for a historical day's signal,
    based on that day's RSI/MACD/Bollinger state. Used by the Decision
    Timeline to explain why the signal was what it was on a given day.
    """
    if row is None:
        return "Limited indicator data available for this day."

    rsi      = row.get("rsi", np.nan)
    macd     = row.get("macd", np.nan)
    macd_sig = row.get("macd_signal", np.nan)
    close    = row.get("Close", np.nan)
    bb_upper = row.get("bb_upper", np.nan)
    bb_lower = row.get("bb_lower", np.nan)

    parts = []

    if not pd.isna(rsi):
        if rsi < 30:
            parts.append(f"RSI oversold at {rsi:.0f}")
        elif rsi > 70:
            parts.append(f"RSI overbought at {rsi:.0f}")

    if not pd.isna(macd) and not pd.isna(macd_sig):
        if macd > macd_sig:
            parts.append("MACD bullish crossover")
        else:
            parts.append("MACD bearish crossover")

    if not pd.isna(close) and not pd.isna(bb_upper) and not pd.isna(bb_lower):
        if close >= bb_upper:
            parts.append("price above upper Bollinger Band")
        elif close <= bb_lower:
            parts.append("price below lower Bollinger Band")

    if not parts:
        return f"Indicators were neutral; model leaned {signal_label} on balance."

    joined = ", ".join(parts[:2])
    return f"{joined[0].upper()}{joined[1:]} \u2192 {signal_label}."


def get_prediction_history(ticker: str, model_key: str = "xgb_sentiment", days: int = 7) -> list[dict]:
    """
    Returns the last N days of predictions for a ticker from a single
    model variant, each entry enriched with a short plain-English reason
    derived from that day's technical indicators.

    Used by the Detail page's Decision Timeline component to show how
    a stock's signal has evolved over recent trading days, and why.

    Response: list of dicts, oldest first, each with:
        date, signal_label, confidence, reason
    """
    ticker = ticker.upper()

    pred_file = PREDICTION_FILES.get(model_key)
    pred_df = None
    if pred_file and pred_file.exists():
        try:
            pred_df = pd.read_csv(pred_file)
            pred_df = pred_df[pred_df["ticker"] == ticker].sort_values("date")
        except Exception as e:
            logger.warning(f"History: failed to load {pred_file.name}: {e}")

    if pred_df is None or pred_df.empty:
        # Fall back to old prediction files, same normalisation as get_latest_prediction
        old_file = OLD_PREDICTION_FILES.get(model_key)
        if old_file and old_file.exists():
            try:
                df = pd.read_csv(old_file)
                df.columns = df.columns.str.strip()
                col_map = {}
                if "Ticker" in df.columns: col_map["Ticker"] = "ticker"
                if "Date"   in df.columns: col_map["Date"]   = "date"
                if "xgb_pred"  in df.columns: col_map["xgb_pred"]  = "predicted_signal"
                if "xgb_prob"  in df.columns: col_map["xgb_prob"]  = "confidence_raw"
                if "lstm_pred" in df.columns: col_map["lstm_pred"] = "predicted_signal"
                if "lstm_prob" in df.columns: col_map["lstm_prob"] = "confidence_raw"
                df = df.rename(columns=col_map)
                if "confidence_raw" in df.columns:
                    df["confidence"] = df["confidence_raw"] * 100
                if "predicted_signal" in df.columns:
                    df["signal_label"] = df["predicted_signal"].map({1: "BUY", 0: "SELL"})
                df["ticker"] = df["ticker"].astype(str).str.upper()
                pred_df = df[df["ticker"] == ticker].sort_values("date")
            except Exception as e:
                logger.warning(f"History: failed to load fallback {old_file.name}: {e}")

    if pred_df is None or pred_df.empty:
        return []

    recent = pred_df.tail(days)

    # Load feature rows once, so we can look up each historical day's indicators
    use_sentiment = "sentiment" in model_key
    file_key = "sentiment" if use_sentiment else "finance"
    feature_file = FEATURE_FILES[file_key]
    feat_df = None
    if feature_file.exists():
        try:
            feat_df = pd.read_csv(feature_file)
            feat_df = feat_df[feat_df["ticker"] == ticker].set_index("date")
        except Exception as e:
            logger.warning(f"History: failed to load features for reasons: {e}")

    history = []
    for _, pred_row in recent.iterrows():
        date = str(pred_row.get("date", ""))[:10]
        signal_label = str(pred_row.get("signal_label", "N/A"))
        confidence = pred_row.get("confidence", None)
        confidence = round(float(confidence), 1) if confidence is not None and not pd.isna(confidence) else None

        feat_row = None
        if feat_df is not None and date in feat_df.index:
            feat_row = feat_df.loc[date]
            # If duplicate dates exist, take first
            if isinstance(feat_row, pd.DataFrame):
                feat_row = feat_row.iloc[0]

        reason = _short_reason_for_row(feat_row, signal_label)

        history.append({
            "date"         : date,
            "signal_label" : signal_label,
            "confidence"   : confidence,
            "reason"       : reason,
        })

    return history


def get_accuracy_track_record(ticker: str, recent_n: int = 20) -> dict:
    """
    For each of the 4 core model variants, how often that model's predicted
    signal actually matched what happened to THIS specific ticker — not the
    model's headline dataset-wide accuracy (that's /metrics), but its real,
    computed track record on this one stock. Joins each prediction against
    the ground-truth `signal` column in features_finance.csv by (ticker,
    date); nothing here is fabricated or estimated.

    Used by the Detail page's History tab track-record panel — the existing
    Decision Timeline shows what each model PREDICTED, this shows whether
    those predictions were actually RIGHT.

    Response: { ticker, recent_n, models: [ { model_key, model_label,
        uses_sentiment, overall: {n, correct, accuracy}, recent: {n, correct,
        accuracy} } ] } — "overall" covers every test-period prediction for
    this ticker, "recent" is just the last `recent_n` (most relevant to
    "should I trust today's signal").
    """
    ticker = ticker.upper()

    finance_file = FEATURE_FILES["finance"]
    if not finance_file.exists():
        return {"ticker": ticker, "recent_n": recent_n, "models": []}

    finance_df = _read_csv_cached(finance_file)
    truth_df = finance_df[finance_df["ticker"] == ticker][["date", "signal"]].copy()
    truth_df["date"] = truth_df["date"].astype(str).str[:10]
    truth_map = dict(zip(truth_df["date"], truth_df["signal"]))

    def _summarize(sub_df: pd.DataFrame) -> dict:
        n = len(sub_df)
        correct = int(sub_df["correct"].sum()) if n else 0
        return {
            "n"        : n,
            "correct"  : correct,
            "accuracy" : round(correct / n * 100, 1) if n else None,
        }

    results = []
    for model_key, model_label in MODEL_LABELS.items():
        pred_file = PREDICTION_FILES.get(model_key)
        if not pred_file or not pred_file.exists():
            continue
        try:
            pred_df = _read_csv_cached(pred_file)
        except Exception as e:
            logger.warning(f"Track record: failed to load {pred_file.name}: {e}")
            continue

        tdf = pred_df[pred_df["ticker"] == ticker].copy()
        if tdf.empty:
            continue

        tdf["date"] = tdf["date"].astype(str).str[:10]
        tdf = tdf.sort_values("date")
        tdf["ground_truth"] = tdf["date"].map(truth_map)
        tdf = tdf.dropna(subset=["ground_truth", "predicted_signal"])
        if tdf.empty:
            continue
        tdf["correct"] = tdf["predicted_signal"].astype(int) == tdf["ground_truth"].astype(int)

        results.append({
            "model_key"      : model_key,
            "model_label"    : model_label,
            "uses_sentiment" : USES_SENTIMENT[model_key],
            "overall"        : _summarize(tdf),
            "recent"         : _summarize(tdf.tail(recent_n)),
        })

    return {
        "ticker"   : ticker,
        "recent_n" : recent_n,
        "models"   : results,
    }


def explain(ticker: str) -> dict:
    """
    Main entry point — generates a full CopilotExplanation for a ticker.
    Called by FastAPI GET /explain/{ticker}

    Returns a dict (JSON-serialisable) with everything the frontend needs
    to render the Detail page.
    """
    ticker = ticker.upper()
    logger.info(f"Generating copilot explanation for {ticker}")

    # ── 1. Collect predictions from all 4 models ───────────────────────────────
    model_predictions = []
    for model_key, model_label in MODEL_LABELS.items():
        pred = get_latest_prediction(ticker, model_key)
        if pred:
            model_predictions.append(ModelPrediction(
                model_key      = model_key,
                model_label    = model_label,
                signal_label   = str(pred.get("signal_label", "SELL")),
                signal         = int(pred.get("predicted_signal", 0)),
                confidence     = float(pred.get("confidence", 50.0)),
                uses_sentiment = USES_SENTIMENT[model_key],
            ))

    if not model_predictions:
        return {
            "error"  : f"No predictions found for ticker '{ticker}'. "
                       "Ensure all training scripts have been run.",
            "ticker" : ticker,
        }

    # ── 2. Get latest features for indicator analysis ──────────────────────────
    # Use finance features for indicator analysis (same regardless of variant)
    latest_row = get_latest_features(ticker, use_sentiment=True)
    latest_finance_row = get_latest_features(ticker, use_sentiment=False)
    row = latest_row if latest_row is not None else latest_finance_row

    # ── 3. Analyse technical indicators ───────────────────────────────────────
    indicator_signals = []
    if row is not None:
        close    = row.get("Close",      np.nan)
        rsi      = row.get("rsi",        np.nan)
        macd     = row.get("macd",       np.nan)
        macd_sig = row.get("macd_signal", np.nan)
        bb_upper = row.get("bb_upper",   np.nan)
        bb_mid   = row.get("bb_mid",     np.nan)
        bb_lower = row.get("bb_lower",   np.nan)
        sma_10   = row.get("sma_10",     np.nan)
        sma_50   = row.get("sma_50",     np.nan)

        indicator_signals = [
            analyse_rsi(rsi),
            analyse_macd(macd, macd_sig),
            analyse_bollinger(close, bb_upper, bb_mid, bb_lower),
            analyse_sma(close, sma_10, sma_50),
        ]
    else:
        logger.warning(f"No feature row found for {ticker} — skipping indicator analysis")

    # ── 4. Analyse sentiment ───────────────────────────────────────────────────
    sentiment_signals = []
    if latest_row is not None:
        gdelt_compound = latest_row.get("gdelt_compound", np.nan)
        gdelt_count    = int(latest_row.get("gdelt_article_count", 0))
        sentiment_signals.append(analyse_gdelt_sentiment(gdelt_compound, gdelt_count))

        wsb_signal = analyse_wsb_sentiment(latest_row)
        if wsb_signal:
            sentiment_signals.append(wsb_signal)

    # ── 4b. Analyse emotion (multi-label, separate from polarity sentiment) ────
    emotion_signals = []
    if latest_row is not None:
        emotion_signals = analyse_wsb_emotion(latest_row)

    # ── 5. Compute scores ──────────────────────────────────────────────────────
    if indicator_signals:
        avg_ind_weight   = np.mean([s.weight for s in indicator_signals])
        technical_score  = round(avg_ind_weight * 100, 1)
    else:
        technical_score  = 50.0

    if sentiment_signals:
        avg_sent_weight  = np.mean([s.weight for s in sentiment_signals])
        sentiment_score  = round(avg_sent_weight * 100, 1)
    else:
        sentiment_score  = 50.0

    if emotion_signals:
        avg_emo_weight   = np.mean([s.weight for s in emotion_signals])
        emotion_score    = round(avg_emo_weight * 100, 1)
    else:
        emotion_score    = 50.0

    buy_confidences = [p.confidence for p in model_predictions if p.signal == 1]
    sell_confidences = [100 - p.confidence for p in model_predictions if p.signal == 0]
    all_buy_scores   = buy_confidences + [100 - c for c in sell_confidences]
    model_score      = round(np.mean(all_buy_scores), 1) if all_buy_scores else 50.0

    # ── 6. Overall signal & agreement ─────────────────────────────────────────
    overall_signal, overall_confidence = majority_signal(model_predictions)
    agreement_level = get_agreement_level(model_predictions)

    # ── 6b. Risk level ─────────────────────────────────────────────────────────
    risk_info = calculate_risk_level(row, model_predictions, overall_confidence)

    # ── 7. Generate plain English summary ─────────────────────────────────────
    summary = generate_summary(
        ticker, overall_signal, agreement_level,
        indicator_signals, sentiment_signals, model_predictions
    )

    # ── 8. Get date from latest prediction ────────────────────────────────────
    latest_date = "N/A"
    if model_predictions:
        pred_row = get_latest_prediction(ticker, model_predictions[0].model_key)
        if pred_row:
            latest_date = str(pred_row.get("date", "N/A"))

    # ── 8b. LSTM attention weights — the flagship model's own "which days mattered" ──
    lstm_attention = None
    lstm_sentiment_pred = get_latest_prediction(ticker, "lstm_sentiment")
    if lstm_sentiment_pred:
        lstm_date = str(lstm_sentiment_pred.get("date", ""))
        lstm_attention = load_lstm_attention(ticker, lstm_date, "lstm_sentiment")

    # ── 9. Serialise and return ────────────────────────────────────────────────
    return {
        "ticker"             : ticker,
        "date"               : latest_date,
        "overall_signal"     : overall_signal,
        "overall_confidence" : overall_confidence,
        "agreement_level"    : agreement_level,
        "risk_level"         : risk_info["level"],
        "risk_score"         : risk_info["score"],
        "risk_reason"        : risk_info["reason"],
        "risk_volatility_pct": risk_info["volatility_pct"],
        "risk_atr_pct"       : risk_info["atr_pct"],
        "summary"            : summary,
        "technical_score"    : technical_score,
        "sentiment_score"    : sentiment_score,
        "emotion_score"      : emotion_score,
        "model_score"        : model_score,
        "models"             : [
            {
                "model_key"      : p.model_key,
                "model_label"    : p.model_label,
                "signal_label"   : p.signal_label,
                "signal"         : p.signal,
                "confidence"     : p.confidence,
                "uses_sentiment" : p.uses_sentiment,
            }
            for p in model_predictions
        ],
        "indicator_signals"  : [
            {
                "name"   : s.name,
                "value"  : round(float(s.value), 4) if not pd.isna(s.value) else None,
                "signal" : s.signal,
                "reason" : s.reason,
                "weight" : round(s.weight, 3),
            }
            for s in indicator_signals
        ],
        "sentiment_signals"  : [
            {
                "source" : s.source,
                "score"  : round(float(s.score), 4),
                "signal" : s.signal,
                "reason" : s.reason,
                "weight" : round(s.weight, 3),
            }
            for s in sentiment_signals
        ],
        "emotion_signals"    : [
            {
                "source" : s.source,
                "score"  : round(float(s.score), 4),
                "signal" : s.signal,
                "reason" : s.reason,
                "weight" : round(s.weight, 3),
            }
            for s in emotion_signals
        ],
        "lstm_attention"     : lstm_attention,
    }


def get_all_model_metrics() -> dict:
    """
    Load model_metrics.json and return all 4 model performance metrics.
    Called by FastAPI GET /metrics
    """
    if not METRICS_FILE.exists():
        return {"error": "model_metrics.json not found — run training scripts first"}

    with open(METRICS_FILE, "r") as f:
        return json.load(f)


def get_ticker_comparison(ticker: str) -> dict:
    """
    Returns side-by-side predictions from all 4 models for one ticker.
    Called by FastAPI GET /compare/{ticker}
    Used by the Compare page in the dashboard.
    """
    ticker = ticker.upper()
    result = {
        "ticker"  : ticker,
        "models"  : [],
    }

    for model_key, model_label in MODEL_LABELS.items():
        pred = get_latest_prediction(ticker, model_key)
        if pred:
            result["models"].append({
                "model_key"     : model_key,
                "model_label"   : model_label,
                "signal_label"  : pred.get("signal_label", "N/A"),
                "confidence"    : pred.get("confidence", 0),
                "uses_sentiment": USES_SENTIMENT[model_key],
                "date"          : pred.get("date", "N/A"),
            })

    # Add agreement info
    if result["models"]:
        preds_obj = [
            ModelPrediction(
                model_key      = m["model_key"],
                model_label    = m["model_label"],
                signal_label   = m["signal_label"],
                signal         = 1 if m["signal_label"] == "BUY" else 0,
                confidence     = m["confidence"],
                uses_sentiment = m["uses_sentiment"],
            )
            for m in result["models"]
        ]
        result["overall_signal"], result["overall_confidence"] = majority_signal(preds_obj)
        result["agreement_level"] = get_agreement_level(preds_obj)

    return result
