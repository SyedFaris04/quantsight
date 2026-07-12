"""
backend/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI server — the bridge between your ML models and the React frontend.

ALL ENDPOINTS:
    GET  /                          health check
    GET  /tickers                   list of all available tickers
    GET  /overview                  all tickers with signals from all 4 models
    GET  /market-news               recent GDELT headlines across all tickers
    GET  /dashboard                 KPIs + top opportunities + news (Dashboard page)
    GET  /stock/{ticker}            OHLCV price history for one ticker
    GET  /explain/{ticker}          AI copilot explanation (all 4 models)
    GET  /history/{ticker}          recent day-by-day signal history (Decision Timeline)
    GET  /accuracy-history/{ticker} per-ticker track record — was each model actually right?
    GET  /compare/{ticker}          side-by-side 4-model comparison for ticker
    GET  /metrics                   all 4 model accuracy metrics (for Compare page)
    GET  /confidence-boost          avg confidence delta from adding sentiment (Compare page)
    GET  /market-sentiment          real-time VADER sentiment over current news (Dashboard)
    GET  /news/{ticker}             GDELT news headlines for one ticker
    GET  /live/{ticker}             live signal for one ticker (Yahoo Finance + XGBoost)
    GET  /live-overview             live signals for all tickers (Market page Live Mode)
    GET  /realtime/{ticker}         real-time price for one ticker (Portfolio page)
    GET  /game/question             random stock question for prediction game
    POST /game/answer               submit game answer, get result + score

HOW TO RUN LOCALLY:
    pip install fastapi uvicorn pandas numpy
    uvicorn main:app --reload --port 8000

HOW TO DEPLOY ON RENDER:
    Start command: uvicorn main:app --host 0.0.0.0 --port $PORT

CORS is enabled for all origins during development.
Restrict to your Vercel domain before final submission.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

# Real-time price via yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger_temp = logging.getLogger("nuroquant-api")
    logger_temp.warning("yfinance not installed — real-time prices unavailable. Run: pip install yfinance")

from copilot_engine import explain, get_all_model_metrics, get_ticker_comparison, calculate_risk_level, get_latest_features, get_prediction_history, get_accuracy_track_record
from live_signals import get_live_signal, get_live_signals_batch

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("nuroquant-api")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
NEWS_DIR        = BASE_DIR / "data" / "raw" / "news"
METRICS_FILE    = PREDICTIONS_DIR / "model_metrics.json"

PREDICTION_FILES = {
    "xgb_finance"    : PREDICTIONS_DIR / "xgb_finance_predictions.csv",
    "xgb_sentiment"  : PREDICTIONS_DIR / "xgb_sentiment_predictions.csv",
    "lstm_finance"   : PREDICTIONS_DIR / "lstm_finance_predictions.csv",
    "lstm_sentiment" : PREDICTIONS_DIR / "lstm_sentiment_predictions.csv",
}

FINANCE_CSV   = PROCESSED_DIR / "features_finance.csv"
SENTIMENT_CSV = PROCESSED_DIR / "features_sentiment.csv"

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "NuroQuant API",
    description = "Sentiment-enhanced stock signal prediction — FYP backend",
    version     = "1.0.0",
)

# CORS — allow React frontend to call this API
# Update the origins list to your Vercel URL before final submission
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # replace with ["https://nuroquant.vercel.app"] for production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── In-memory cache — loaded once at startup ───────────────────────────────────
_cache: dict = {}
_vader = SentimentIntensityAnalyzer()


def load_predictions() -> dict[str, pd.DataFrame]:
    """
    Load all prediction CSVs into memory at startup.
    Handles both new 4-variant files and old xgb_predictions/lstm_predictions files.
    """
    preds = {}

    new_files = {
        "xgb_finance"    : PREDICTIONS_DIR / "xgb_finance_predictions.csv",
        "xgb_sentiment"  : PREDICTIONS_DIR / "xgb_sentiment_predictions.csv",
        "lstm_finance"   : PREDICTIONS_DIR / "lstm_finance_predictions.csv",
        "lstm_sentiment" : PREDICTIONS_DIR / "lstm_sentiment_predictions.csv",
    }

    # Old files used as fallback before retraining is done
    old_files = {
        "xgb_finance"  : PREDICTIONS_DIR / "xgb_predictions.csv",
        "lstm_finance" : PREDICTIONS_DIR / "lstm_predictions.csv",
    }

    for key, path in new_files.items():
        if path.exists():
            try:
                df = pd.read_csv(path)
                df["date"] = pd.to_datetime(df["date"])
                df["ticker"] = df["ticker"].astype(str).str.upper()
                preds[key] = df
                logger.info(f"Loaded {key}: {len(df):,} rows")
            except Exception as e:
                logger.warning(f"Failed to load {path.name}: {e}")
        else:
            # Try old file as fallback
            old_path = old_files.get(key)
            if old_path and old_path.exists():
                try:
                    df = pd.read_csv(old_path)
                    df.columns = df.columns.str.strip()

                    # Normalise column names
                    col_map = {}
                    if "Ticker" in df.columns: col_map["Ticker"] = "ticker"
                    if "Date"   in df.columns: col_map["Date"]   = "date"
                    if "xgb_pred"  in df.columns: col_map["xgb_pred"]  = "predicted_signal"
                    if "xgb_prob"  in df.columns: col_map["xgb_prob"]  = "confidence_raw"
                    if "lstm_pred" in df.columns: col_map["lstm_pred"] = "predicted_signal"
                    if "lstm_prob" in df.columns: col_map["lstm_prob"] = "confidence_raw"
                    if "Target"    in df.columns: col_map["Target"]    = "actual_signal"
                    df = df.rename(columns=col_map)

                    if "confidence_raw" in df.columns:
                        df["confidence"] = (df["confidence_raw"] * 100).round(2)
                    if "predicted_signal" in df.columns:
                        df["signal_label"] = df["predicted_signal"].map({1: "BUY", 0: "SELL"})

                    df["date"] = pd.to_datetime(df["date"])
                    df["ticker"] = df["ticker"].astype(str).str.upper()
                    preds[key] = df
                    logger.info(f"Loaded {key} (fallback from {old_path.name}): {len(df):,} rows")
                except Exception as e:
                    logger.warning(f"Failed to load fallback {old_path.name}: {e}")
            else:
                logger.warning(f"No prediction file found for {key} — run train_xgboost.py / train_lstm.py")

    return preds


def load_features() -> pd.DataFrame:
    """
    Load feature data for OHLCV + indicators.
    Tries features_finance.csv first (generated by build_features.py),
    then falls back to features_stock.csv (from original notebooks).
    """
    for path in [FINANCE_CSV, PROCESSED_DIR / "features_stock.csv"]:
        if path.exists():
            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()

                # Normalise column names — features_stock.csv uses Title Case
                col_map = {}
                if "Date"   in df.columns: col_map["Date"]   = "date"
                if "Ticker" in df.columns: col_map["Ticker"] = "ticker"
                if "RSI"    in df.columns: col_map["RSI"]    = "rsi"
                if "MACD"   in df.columns: col_map["MACD"]   = "macd"
                if "MACD_signal" in df.columns: col_map["MACD_signal"] = "macd_signal"
                if "BB_upper"    in df.columns: col_map["BB_upper"]    = "bb_upper"
                if "BB_mid"      in df.columns: col_map["BB_mid"]      = "bb_mid"
                if "BB_lower"    in df.columns: col_map["BB_lower"]    = "bb_lower"
                if "SMA_10"      in df.columns: col_map["SMA_10"]      = "sma_10"
                if "SMA_20"      in df.columns: col_map["SMA_20"]      = "sma_50"
                if "Target"      in df.columns: col_map["Target"]      = "signal"
                df = df.rename(columns=col_map)

                df["date"] = pd.to_datetime(df["date"])
                df["ticker"] = df["ticker"].astype(str).str.upper()
                logger.info(f"Loaded features from {path.name}: {len(df):,} rows")
                return df
            except Exception as e:
                logger.warning(f"Failed to load {path.name}: {e}")

    logger.warning("No feature file found — run build_features.py")
    return pd.DataFrame()


def load_news() -> pd.DataFrame:
    """Load combined GDELT news parquet."""
    news_path = NEWS_DIR / "all_news.parquet"
    if news_path.exists():
        df = pd.read_parquet(news_path)
        logger.info(f"Loaded all_news.parquet: {len(df):,} rows")
        return df
    logger.warning("all_news.parquet not found — run fetch_news.py")
    return pd.DataFrame()


@app.on_event("startup")
async def startup_event():
    """Load all data files into memory when the server starts."""
    logger.info("Starting NuroQuant API ...")
    _cache["predictions"] = load_predictions()
    _cache["features"]    = load_features()
    _cache["news"]        = load_news()

    # Get list of available tickers from predictions
    all_tickers = set()
    for df in _cache["predictions"].values():
        all_tickers.update(df["ticker"].unique())
    _cache["tickers"] = sorted(all_tickers)

    logger.info(f"Ready — {len(_cache['tickers'])} tickers available")


# ── Pydantic Models (request/response shapes) ─────────────────────────────────

class GameAnswer(BaseModel):
    ticker      : str
    date        : str
    user_signal : str   # "BUY" or "SELL"
    model_key   : str   # which model variant the question was from


class GameResult(BaseModel):
    correct         : bool
    actual_signal   : str
    confidence      : float
    explanation     : str
    points_earned   : int


# ── Helper Functions ───────────────────────────────────────────────────────────

def get_latest_signals_for_ticker(ticker: str) -> dict:
    """Get the most recent prediction from each of the 4 models for a ticker."""
    ticker  = ticker.upper()
    signals = {}

    for model_key, df in _cache.get("predictions", {}).items():
        ticker_df = df[df["ticker"] == ticker].sort_values("date")
        if not ticker_df.empty:
            row = ticker_df.iloc[-1]

            # Handle both column naming conventions
            signal_label = str(row.get("signal_label", "N/A"))
            if signal_label == "N/A" and "predicted_signal" in row:
                signal_label = "BUY" if int(row["predicted_signal"]) == 1 else "SELL"

            confidence = 50.0
            if "confidence" in row and pd.notna(row["confidence"]):
                confidence = round(float(row["confidence"]), 1)

            signals[model_key] = {
                "signal_label" : signal_label,
                "confidence"   : confidence,
                "date"         : row["date"].strftime("%Y-%m-%d"),
            }

    return signals


def get_ohlcv_for_ticker(ticker: str, days: int = 90) -> list[dict]:
    """Get the last N days of OHLCV + indicator data for a ticker."""
    features = _cache.get("features", pd.DataFrame())
    if features.empty:
        return []

    ticker_df = (
        features[features["ticker"] == ticker.upper()]
        .sort_values("date")
        .tail(days)
    )

    if ticker_df.empty:
        return []

    # Flexible column selection — works with both features_finance and features_stock
    possible_cols = [
        "date", "Open", "High", "Low", "Close", "Volume",
        "rsi", "macd", "macd_signal", "bb_upper", "bb_mid", "bb_lower",
        "sma_10", "sma_50", "ema_12", "ema_26",
    ]
    cols = [c for c in possible_cols if c in ticker_df.columns]

    result = ticker_df[cols].copy()
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result = result.where(pd.notna(result), None)
    return result.to_dict(orient="records")


def get_news_for_ticker(ticker: str, limit: int = 10) -> list[dict]:
    """Get the most recent news headlines for a ticker from GDELT."""
    news = _cache.get("news", pd.DataFrame())
    if news.empty:
        return []

    ticker_news = (
        news[news["ticker"] == ticker.upper()]
        .sort_values("date", ascending=False)
        .head(limit)
    )

    if ticker_news.empty:
        return []

    return ticker_news[["date", "title", "url", "source"]].to_dict(orient="records")


def get_market_news(limit: int = 8) -> list[dict]:
    """
    Get the most recent news headlines across ALL tickers (market-wide feed).
    Used by the Dashboard home page. Each item includes its ticker so the
    UI can show which stock the headline is about. Real GDELT data only.
    """
    news = _cache.get("news", pd.DataFrame())
    if news.empty:
        return []

    recent = (
        news.sort_values("date", ascending=False)
        .drop_duplicates(subset=["title"])
        .head(limit)
    )
    if recent.empty:
        return []

    cols = ["date", "title", "url", "source", "ticker"]
    cols = [c for c in cols if c in recent.columns]
    return recent[cols].to_dict(orient="records")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Health check — Render pings this to keep the server alive."""
    return {
        "status"  : "ok",
        "service" : "NuroQuant API",
        "tickers" : len(_cache.get("tickers", [])),
        "models"  : list(PREDICTION_FILES.keys()),
    }


@app.get("/tickers")
def get_tickers():
    """
    Returns the list of all available tickers.
    Used by the Overview page ticker dropdown and table.
    """
    tickers = _cache.get("tickers", [])
    if not tickers:
        raise HTTPException(
            status_code=503,
            detail="No tickers available — ensure training scripts have been run"
        )
    return {"tickers": tickers, "count": len(tickers)}


@app.get("/overview")
def get_overview():
    """
    Returns all tickers with their latest signal from all 4 models.
    This is the main data source for the Overview page table.

    Response shape per ticker:
    {
        ticker: "AAPL",
        signals: {
            xgb_finance:    { signal_label, confidence, date },
            xgb_sentiment:  { signal_label, confidence, date },
            lstm_finance:   { signal_label, confidence, date },
            lstm_sentiment: { signal_label, confidence, date },
        },
        overall_signal: "BUY",       ← majority vote
        overall_confidence: 72.3,
        agreement_level: "Strong"    ← "Strong" | "Moderate" | "Mixed"
    }
    """
    tickers = _cache.get("tickers", [])
    if not tickers:
        raise HTTPException(status_code=503, detail="No prediction data loaded")

    class _PredShim:
        """Minimal stand-in so calculate_risk_level() can read .signal"""
        def __init__(self, signal_label):
            self.signal = 1 if signal_label == "BUY" else 0

    overview = []
    for ticker in tickers:
        signals = get_latest_signals_for_ticker(ticker)
        if not signals:
            continue

        # ── Consensus recommendation ─────────────────────────────────────────
        # Strong/Moderate agreement -> follow the majority direction.
        # A genuine 2-2 split means the models disagree -> HOLD, not a
        # forced BUY. This matches the Decision Support Layer described
        # in the proposal (Section 6.5.2 — Consensus Engine).
        buy_count = sum(
            1 for s in signals.values() if s["signal_label"] == "BUY"
        )
        n = len(signals)
        sell_count = n - buy_count

        if buy_count > sell_count:
            overall_signal = "BUY"
        elif sell_count > buy_count:
            overall_signal = "SELL"
        else:
            overall_signal = "HOLD"

        all_conf = [s["confidence"] for s in signals.values()]
        avg_conf = round(float(np.mean(all_conf)), 1) if all_conf else 50.0

        if buy_count == n or buy_count == 0:
            agreement = "Strong"
        elif buy_count == n - 1 or buy_count == 1:
            agreement = "Moderate"
        else:
            agreement = "Mixed"

        # ── Risk level — reuses the same logic as the Detail page ──────────
        try:
            row = get_latest_features(ticker, use_sentiment=True)
            pred_shims = [_PredShim(s["signal_label"]) for s in signals.values()]
            risk_info = calculate_risk_level(row, pred_shims, avg_conf)
            risk_level = risk_info["level"]
        except Exception as e:
            logger.warning(f"Risk calc failed for {ticker}: {e}")
            risk_level = "Medium"

        overview.append({
            "ticker"             : ticker,
            "signals"            : signals,
            "overall_signal"     : overall_signal,
            "overall_confidence" : avg_conf,
            "agreement_level"    : agreement,
            "risk_level"         : risk_level,
        })

    return {"data": overview, "count": len(overview)}


@app.get("/market-news")
def get_market_news_endpoint(limit: int = Query(default=8, ge=1, le=30)):
    """
    Market-wide recent news headlines across all tickers (real GDELT data).
    Powers the Dashboard home page news feed.
    """
    return {"articles": get_market_news(limit), "count": None}


@app.get("/dashboard")
def get_dashboard():
    """
    Aggregated summary for the Dashboard (home) page — computed from the
    same real data as /overview, returned in a single call so the home
    page loads fast.

    Returns:
        kpis: total_tickers, buy_signals, sell_signals, hold_signals,
              strong_agreement, avg_confidence
        top_opportunities: top 5 BUY tickers by confidence (Strong first)
        news: recent market-wide headlines
    """
    # Reuse the overview computation
    overview_resp = get_overview()
    rows = overview_resp["data"]

    if not rows:
        raise HTTPException(status_code=503, detail="No prediction data loaded")

    buy   = [r for r in rows if r["overall_signal"] == "BUY"]
    sell  = [r for r in rows if r["overall_signal"] == "SELL"]
    hold  = [r for r in rows if r["overall_signal"] == "HOLD"]
    strong = [r for r in rows if r["agreement_level"] == "Strong"]
    avg_conf = round(float(np.mean([r["overall_confidence"] for r in rows])), 1)

    # Top opportunities: BUY signals, Strong agreement ranked first, then by confidence
    def opp_sort_key(r):
        agreement_rank = {"Strong": 0, "Moderate": 1, "Mixed": 2}.get(r["agreement_level"], 3)
        return (agreement_rank, -r["overall_confidence"])

    top_opps = sorted(buy, key=opp_sort_key)[:5]
    top_opportunities = [
        {
            "ticker"          : r["ticker"],
            "overall_signal"  : r["overall_signal"],
            "confidence"      : r["overall_confidence"],
            "agreement_level" : r["agreement_level"],
            "risk_level"      : r["risk_level"],
        }
        for r in top_opps
    ]

    return {
        "kpis": {
            "total_tickers"    : len(rows),
            "buy_signals"      : len(buy),
            "sell_signals"     : len(sell),
            "hold_signals"     : len(hold),
            "strong_agreement" : len(strong),
            "avg_confidence"   : avg_conf,
        },
        "top_opportunities": top_opportunities,
        "news"             : get_market_news(6),
    }


@app.get("/stock/{ticker}")
def get_stock(
    ticker : str,
    days   : int = Query(default=90, ge=1, le=3650),
):
    """
    Returns OHLCV + technical indicators for one ticker.
    Used by the Detail page stock chart and indicator panel.

    Query param:
        days — number of trading days to return (default 90, max 3650 ≈ 10y,
        covers the full 2015-2024 dataset for the "5Y"/"All" chart periods)
    """
    ticker = ticker.upper()
    if ticker not in _cache.get("tickers", []):
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    ohlcv = get_ohlcv_for_ticker(ticker, days)
    if not ohlcv:
        raise HTTPException(
            status_code=404,
            detail=f"No OHLCV data available for {ticker}"
        )

    return {
        "ticker" : ticker,
        "days"   : len(ohlcv),
        "data"   : ohlcv,
    }


@app.get("/explain/{ticker}")
def get_explanation(ticker: str):
    """
    Returns full AI Copilot explanation for one ticker.
    Includes:
        - Signals from all 4 models
        - Technical indicator analysis (RSI, MACD, Bollinger, SMA)
        - Sentiment analysis (GDELT + WSB)
        - Plain English summary
        - Technical / Sentiment / Model scores for the contribution bar

    Used by the Detail page.
    """
    ticker = ticker.upper()
    if ticker not in _cache.get("tickers", []):
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    result = explain(ticker)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@app.get("/history/{ticker}")
def get_decision_timeline(ticker: str, days: int = Query(default=7, ge=2, le=30),
                          model_key: str = Query(default="xgb_sentiment")):
    """
    Returns the Decision Timeline — the last N days of predictions for
    a ticker from a single model variant, each with a short explanation
    of why the signal was what it was on that day.

    Used by the Detail page's Decision Timeline component so users can
    see how a stock's signal has evolved and what changed day to day.
    """
    ticker = ticker.upper()
    if ticker not in _cache.get("tickers", []):
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    history = get_prediction_history(ticker, model_key=model_key, days=days)

    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction history found for '{ticker}' with model '{model_key}'"
        )

    return {
        "ticker"     : ticker,
        "model_key"  : model_key,
        "days"       : len(history),
        "history"    : history,
    }


@app.get("/accuracy-history/{ticker}")
def get_accuracy_history(ticker: str, recent_n: int = Query(default=20, ge=5, le=100)):
    """
    Per-ticker track record: for each of the 4 core model variants, how
    often its predicted signal actually matched what happened to THIS
    ticker (not the model's dataset-wide accuracy — see /metrics for that).
    Complements /history/{ticker}'s Decision Timeline, which shows what was
    predicted but not whether it was right.
    """
    ticker = ticker.upper()
    if ticker not in _cache.get("tickers", []):
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    return get_accuracy_track_record(ticker, recent_n=recent_n)


@app.get("/compare/{ticker}")
def get_ticker_compare(ticker: str):
    """
    Returns side-by-side model predictions for one ticker.
    Lighter than /explain — just the 4 model signals + agreement.
    Used by the Compare page ticker detail panel.
    """
    ticker = ticker.upper()
    if ticker not in _cache.get("tickers", []):
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    return get_ticker_comparison(ticker)


@app.get("/metrics")
def get_metrics():
    """
    Returns accuracy metrics for all 4 model variants.
    Used by the Compare page accuracy table and bar chart.

    Response shape:
    {
        "XGBoost_Finance Only": {
            model, variant, accuracy, f1, precision, recall, auc_roc
        },
        "XGBoost_Finance + Sentiment": { ... },
        "LSTM+Transformer_Finance Only": { ... },
        "LSTM+Transformer_Finance + Sentiment": { ... },
    }
    """
    metrics = get_all_model_metrics()

    if "error" in metrics:
        raise HTTPException(
            status_code=503,
            detail="Model metrics not found — run train_xgboost.py and train_lstm.py first"
        )

    return metrics


@app.get("/confidence-boost")
def get_confidence_boost():
    """
    Real aggregate: mean confidence delta between each finance-only and
    finance+sentiment model pair, computed by matching (ticker, date) rows
    across the already-loaded prediction CSVs. Used by the Compare page's
    "Confidence Boost" KPI card — not a fabricated number, a genuine
    average over every matched test-set prediction.
    """
    preds = _cache.get("predictions", {})
    pairs = {
        "xgb"  : ("xgb_finance",  "xgb_sentiment"),
        "lstm" : ("lstm_finance", "lstm_sentiment"),
    }

    result = {}
    for key, (fin_key, sent_key) in pairs.items():
        fin_df  = preds.get(fin_key)
        sent_df = preds.get(sent_key)
        if fin_df is None or sent_df is None or fin_df.empty or sent_df.empty:
            continue

        merged = fin_df.merge(sent_df, on=["ticker", "date"], suffixes=("_fin", "_sent"))
        if merged.empty:
            continue

        delta = merged["confidence_sent"] - merged["confidence_fin"]
        result[key] = {
            "avg_confidence_boost"     : round(float(delta.mean()), 2),
            "avg_abs_confidence_boost" : round(float(delta.abs().mean()), 2),
            "n_matched"                : int(len(merged)),
        }

    if not result:
        raise HTTPException(
            status_code=503,
            detail="Prediction data not available for confidence boost calculation"
        )

    best_key = max(result, key=lambda k: result[k]["avg_confidence_boost"])
    result["best"] = best_key
    return result


@app.get("/market-sentiment")
def get_market_sentiment(days: int = Query(default=14, ge=1, le=90)):
    """
    Real-time VADER sentiment over the CURRENT GDELT news feed
    (all_news.parquet — genuinely up to date, distinct from the 2015-2024
    historical training data). Scores each headline on the fly (VADER is a
    lightweight rule-based scorer, no model loading required) and buckets
    into positive/neutral/negative. Used by the Dashboard's Market
    Sentiment panel.
    """
    news = _cache.get("news", pd.DataFrame())
    empty_response = {
        "positive_pct": None, "neutral_pct": None, "negative_pct": None,
        "article_count": 0, "days": days, "trend": [],
    }
    if news.empty:
        return empty_response

    df = news.copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    recent = df[df["date"] >= cutoff].drop_duplicates(subset=["title"]).copy()

    if recent.empty:
        return empty_response

    recent["compound"] = recent["title"].fillna("").astype(str).apply(
        lambda t: _vader.polarity_scores(t)["compound"]
    )
    recent["bucket"] = recent["compound"].apply(
        lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
    )

    counts = recent["bucket"].value_counts()
    total  = len(recent)

    trend_df = (
        recent.groupby(recent["date"].dt.date)["compound"]
        .mean()
        .reset_index()
        .sort_values("date")
    )
    trend = [
        {"date": str(row["date"]), "avg_compound": round(float(row["compound"]), 3)}
        for _, row in trend_df.iterrows()
    ]

    return {
        "positive_pct"  : round(100 * counts.get("positive", 0) / total, 1),
        "neutral_pct"   : round(100 * counts.get("neutral",  0) / total, 1),
        "negative_pct"  : round(100 * counts.get("negative", 0) / total, 1),
        "article_count" : int(total),
        "days"          : days,
        "trend"         : trend,
    }


@app.get("/news/{ticker}")
def get_news(
    ticker : str,
    limit  : int = Query(default=10, ge=1, le=50),
):
    """
    Returns the most recent GDELT news headlines for one ticker.
    Used by the Detail page news feed section.

    Query param:
        limit — number of headlines to return (default 10, max 50)
    """
    ticker = ticker.upper()
    news   = get_news_for_ticker(ticker, limit)

    return {
        "ticker"  : ticker,
        "count"   : len(news),
        "articles": news,
    }


@app.get("/live/{ticker}")
def get_live_ticker_signal(ticker: str):
    """
    Returns a fresh BUY/SELL signal for a ticker using TODAY's live market data.

    Fetches last 90 days of OHLCV from Yahoo Finance, computes fresh
    technical indicators, and runs the trained XGBoost model.

    This is different from /explain/{ticker} which uses cached Dec 2024 data.

    Response includes:
        signal_label, confidence, date, close_price,
        indicators (rsi, macd_direction, bb_position),
        disclaimer, source ("live" or "fallback")
    """
    ticker = ticker.upper()
    result = get_live_signal(ticker)
    return result


@app.get("/live-overview")
def get_live_overview():
    """
    Returns live BUY/SELL signals for ALL tickers using today's market data.

    Used by the Overview page when the user toggles Live Mode on.
    Each ticker gets a fresh signal from Yahoo Finance + XGBoost model.

    Note: This endpoint takes 10–30 seconds for all tickers.
    The frontend should show a loading state while waiting.
    """
    tickers = _cache.get("tickers", [])
    if not tickers:
        raise HTTPException(status_code=503, detail="No tickers loaded")

    results = get_live_signals_batch(tickers)
    return {
        "data"      : results,
        "count"     : len(results),
        "generated" : pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disclaimer": (
            "Signals generated from live market data using a model trained on "
            "2015–2023 patterns. For educational purposes — not financial advice."
        ),
    }


@app.get("/realtime/{ticker}")
def get_realtime_price(ticker: str):
    """
    Returns the real-time (or latest available) stock price from Yahoo Finance.
    Used by the Portfolio page to show current market price.

    Falls back gracefully if yfinance is unavailable or the request fails.

    Response:
    {
        ticker: "AAPL",
        price: 189.45,
        change: +1.23,
        change_pct: +0.65,
        currency: "USD",
        market_state: "REGULAR" | "CLOSED" | "PRE" | "POST",
        source: "live" | "fallback"
    }
    """
    ticker = ticker.upper()

    # ── Try yfinance for live price ────────────────────────────────────────────
    if YFINANCE_AVAILABLE:
        try:
            stock     = yf.Ticker(ticker)
            info      = stock.fast_info

            price     = getattr(info, "last_price",        None)
            prev_close = getattr(info, "previous_close",   None)
            currency  = getattr(info, "currency",          "USD")
            mkt_state = getattr(info, "market_state",      "UNKNOWN")

            if price and price > 0:
                change     = round(price - prev_close, 2) if prev_close else None
                change_pct = round((change / prev_close) * 100, 2) if prev_close and change else None
                return {
                    "ticker"      : ticker,
                    "price"       : round(float(price), 2),
                    "change"      : change,
                    "change_pct"  : change_pct,
                    "currency"    : currency,
                    "market_state": mkt_state,
                    "source"      : "live",
                }
        except Exception as e:
            logger.warning(f"yfinance failed for {ticker}: {e}")

    # ── Fallback: use latest price from our dataset ────────────────────────────
    features = _cache.get("features", pd.DataFrame())
    if not features.empty:
        ticker_df = features[features["ticker"] == ticker].sort_values("date")
        if not ticker_df.empty:
            last_row  = ticker_df.iloc[-1]
            price     = last_row.get("Close", None)
            prev_row  = ticker_df.iloc[-2] if len(ticker_df) > 1 else None
            prev_price = prev_row.get("Close", None) if prev_row is not None else None
            change     = round(float(price) - float(prev_price), 2) if prev_price else None
            change_pct = round((change / float(prev_price)) * 100, 2) if prev_price and change else None
            return {
                "ticker"      : ticker,
                "price"       : round(float(price), 2) if price else None,
                "change"      : change,
                "change_pct"  : change_pct,
                "currency"    : "USD",
                "market_state": "CLOSED",
                "source"      : "fallback",
                "note"        : f"Live price unavailable. Showing last dataset price ({str(last_row.get('date', ''))[:10]}).",
            }

    raise HTTPException(status_code=404, detail=f"No price data available for {ticker}")


@app.get("/game/question")
def get_game_question(
    difficulty : str  = Query(default="medium", pattern="^(easy|medium|hard)$"),
    model_key  : str  = Query(default="xgb_sentiment"),
):
    """
    Returns a random stock prediction question for the game page.
    Picks a random ticker + date from the prediction data.

    Query params:
        difficulty — "easy" (high confidence) | "medium" | "hard" (low confidence)
        model_key  — which model's predictions to use for the game

    Response:
    {
        ticker, date, close_price,
        hint_rsi, hint_macd_direction,
        model_key, difficulty
    }
    """
    preds = _cache.get("predictions", {})
    if model_key not in preds:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_key '{model_key}'. "
                   f"Valid options: {list(preds.keys())}"
        )

    df = preds[model_key].copy()

    # Filter by difficulty — confidence level determines difficulty
    if difficulty == "easy":
        # High confidence = easier to guess
        df = df[df["confidence"] >= 70]
    elif difficulty == "hard":
        # Confidence near 50% = hardest (model is uncertain)
        df = df[(df["confidence"] >= 45) & (df["confidence"] <= 55)]
    # medium = no filter, any confidence

    if df.empty:
        # Fallback to any row if filter too strict
        df = preds[model_key].copy()

    # Pick a random row
    row = df.sample(1).iloc[0]
    ticker = str(row["ticker"])
    date   = row["date"].strftime("%Y-%m-%d")

    # Get hints from features
    features = _cache.get("features", pd.DataFrame())
    hint_rsi  = None
    hint_macd = None
    close     = None

    if not features.empty:
        feat_row = features[
            (features["ticker"] == ticker) &
            (features["date"].dt.strftime("%Y-%m-%d") == date)
        ]
        if not feat_row.empty:
            r         = feat_row.iloc[0]
            hint_rsi  = round(float(r.get("rsi",  np.nan)), 1) if not pd.isna(r.get("rsi")) else None
            macd      = r.get("macd", np.nan)
            macd_sig  = r.get("macd_signal", np.nan)
            if not pd.isna(macd) and not pd.isna(macd_sig):
                hint_macd = "bullish" if macd > macd_sig else "bearish"
            close     = round(float(r.get("Close", np.nan)), 2) if not pd.isna(r.get("Close")) else None

    return {
        "ticker"           : ticker,
        "date"             : date,
        "close_price"      : close,
        "hint_rsi"         : hint_rsi,
        "hint_macd"        : hint_macd,
        "model_key"        : model_key,
        "difficulty"       : difficulty,
        # Don't reveal the answer — frontend sends it back to /game/answer
        "_answer_token"    : f"{ticker}_{date}_{model_key}",
    }


@app.post("/game/answer")
def submit_game_answer(answer: GameAnswer) -> GameResult:
    """
    Receives the user's BUY/SELL guess and returns the result.
    Called when the player clicks BUY or SELL in the game.

    Request body:
        ticker, date, user_signal ("BUY"/"SELL"), model_key

    Response:
        correct, actual_signal, confidence, explanation, points_earned
    """
    preds = _cache.get("predictions", {})
    model_key = answer.model_key

    if model_key not in preds:
        raise HTTPException(status_code=400, detail=f"Unknown model_key '{model_key}'")

    df = preds[model_key]
    ticker = answer.ticker.upper()

    # Find the matching prediction row
    match = df[
        (df["ticker"] == ticker) &
        (df["date"].dt.strftime("%Y-%m-%d") == answer.date)
    ]

    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction found for {ticker} on {answer.date}"
        )

    row           = match.iloc[0]
    actual_signal = str(row.get("signal_label", "SELL"))
    confidence    = float(row.get("confidence", 50))
    correct       = answer.user_signal.upper() == actual_signal

    # Points: base 10, bonus for hard questions (low model confidence)
    base_points    = 10 if correct else 0
    confidence_gap = abs(confidence - 50)  # 0 = hardest, 50 = easiest
    difficulty_bonus = max(0, round((50 - confidence_gap) / 10)) if correct else 0
    points_earned  = base_points + difficulty_bonus

    # Plain English explanation for the result
    if correct:
        explanation = (
            f"Correct! The model predicted {actual_signal} for {ticker} "
            f"with {confidence:.1f}% confidence."
        )
    else:
        explanation = (
            f"Not quite — the model predicted {actual_signal} for {ticker} "
            f"with {confidence:.1f}% confidence. "
            f"{'High confidence suggests a strong signal.' if confidence > 70 else 'Low confidence means this was a tough one.'}"
        )

    return GameResult(
        correct       = correct,
        actual_signal = actual_signal,
        confidence    = confidence,
        explanation   = explanation,
        points_earned = points_earned,
    )
