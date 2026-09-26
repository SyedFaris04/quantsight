"""Read-only data/indicator audit. Does not rebuild features or retrain models.
Run: python data_audit.py --output ../docs/data_audit.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from live_signals import build_features_from_ohlcv, load_xgb_model
from build_features import add_cross_sectional_features, add_sector_relative_features
from market_calendar import schedule

ROOT = Path(__file__).resolve().parent


def audit():
    processed = ROOT / "data/processed"
    finance = pd.read_csv(processed / "features_finance.csv")
    sentiment = pd.read_csv(processed / "features_sentiment.csv")
    stock = pd.read_csv(processed / "features_stock.csv")
    keys = ["ticker", "date"]
    result = {"training_rows": len(finance), "tickers": finance.ticker.nunique(),
              "date_start": finance.date.min(), "date_end": finance.date.max(),
              "duplicate_ticker_dates": int(finance.duplicated(keys).sum()),
              "invalid_close_rows": int((~np.isfinite(finance.Close) | (finance.Close <= 0)).sum()),
              "sources": {}, "coverage": {}, "feature_parity": {},
              "limitations": ["Historical price adjustment metadata is absent; corporate-action consistency is not proven.",
                              "The fixed surviving ticker universe is not point-in-time membership.",
                              "Legacy date-only news cannot establish availability before a session close.",
                              "Existing holdout has been reused for model decisions and is exploratory."]}
    for name in ["gdelt_sentiment", "wsb_sentiment", "wsb_emotion", "yahoo_news_sentiment"]:
        path = processed / (name + ".csv")
        source = pd.read_csv(path)
        matches = finance[keys].merge(source[keys].drop_duplicates(), on=keys)
        result["sources"][name] = {"rows": len(source), "start": source.date.min(), "end": source.date.max(),
                                  "matching_training_ticker_dates": len(matches),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for label, frame in [("all", sentiment), ("2023_2024", sentiment[sentiment.date >= "2023-01-01"])]:
        columns = [c for c in frame if c.startswith(("gdelt_", "wsb_", "emo_"))]
        result["coverage"][label] = {"rows": len(frame), "nonzero_rows": {c: int(frame[c].fillna(0).ne(0).sum()) for c in columns}}
    expected = schedule(finance.date.min(), finance.date.max()).index.strftime("%Y-%m-%d")
    result["missing_sessions_by_ticker"] = {}
    mapping = {"rsi": "RSI", "atr": "ATR", "macd": "MACD", "macd_signal": "MACD_signal",
               "macd_hist": "MACD_hist", "bb_upper": "BB_upper", "bb_lower": "BB_lower",
               "bb_width": "BB_width", "ema_12": "EMA_12", "ema_26": "EMA_26",
               "return_5": "Return_5", "return_10": "Return_10", "volatility_20": "Volatility_20"}
    # Saved feature_stock starts after its original warm-up. Allow another 300
    # observations for recursive indicators to converge; disclose this boundary.
    result["parity_warmup_rows"] = 300
    errors = {key: [] for key in mapping}
    live_frames = []
    for ticker, group in stock.groupby("Ticker"):
        group = group.sort_values("Date").set_index("Date")
        live = build_features_from_ohlcv(group[["Open", "High", "Low", "Close", "Volume"]])
        common = live.index.intersection(group.index[300:])
        live_frame = live.reset_index().rename(columns={"Date": "date"})
        live_frame["ticker"] = ticker
        live_frame["burned_in"] = live_frame.date.isin(group.index[300:])
        live_frames.append(live_frame)
        for col, saved in mapping.items():
            errors[col].append((live.loc[common, col] - group.loc[common, saved]).abs())
        dates = finance.loc[finance.ticker == ticker, "date"]
        applicable = expected[(expected >= dates.min()) & (expected <= dates.max())]
        missing = sorted(set(applicable) - set(dates))
        result["missing_sessions_by_ticker"][ticker] = {"count": len(missing), "first_ten": missing[:10]}
    for col, differences in errors.items():
        values = pd.concat(differences)
        result["feature_parity"][col] = {"rows": len(values), "max_abs_error": float(values.max()),
                                         "mean_abs_error": float(values.mean()),
                                         "within_1e_minus_5": bool((values <= 1e-5).all())}
    panel = pd.concat(live_frames, ignore_index=True)
    panel = add_sector_relative_features(add_cross_sectional_features(panel))
    paired = finance.merge(panel, on=keys, suffixes=("_saved", "_live"))
    # Every constituent must be warmed up before evaluating a market-relative
    # feature: a newly listed stock's RSI changes other stocks' market median.
    complete = panel.groupby("date").agg(warmed=("burned_in", "all"), n=("ticker", "nunique"))
    valid_dates = complete.index[complete.warmed & complete.n.eq(finance.ticker.nunique())]
    paired = paired[paired.date.isin(valid_dates)]
    result["model_parity_start"] = paired.date.min()
    result["model_parity_end"] = paired.date.max()
    result["parity_warmup_policy"] = "Require 300 prior observations for every constituent, including newly listed stocks, before checking market-relative features."

    features = load_xgb_model()["features"]
    result["model_feature_parity"] = {}
    for col in features:
        differences = (paired[col + "_saved"] - paired[col + "_live"]).abs()
        result["model_feature_parity"][col] = {"rows": len(differences), "max_abs_error": float(differences.max()),
            "mean_abs_error": float(differences.mean()), "within_1e_minus_5": bool((differences <= 1e-5).all())}
    result["parity_note"] = "Replays saved stock OHLCV and stored sector returns; does not independently verify Yahoo corporate-action adjustments or source ETF return construction."
    result["input_hashes"] = {name: hashlib.sha256((processed / name).read_bytes()).hexdigest()
                              for name in ["features_finance.csv", "features_sentiment.csv", "features_stock.csv", "sector_returns.csv"]}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT.parent / "docs/data_audit.json")
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Audit written to {args.output}; {result['training_rows']} rows, {result['tickers']} tickers")
    print("Indicator parity:", {k: v['within_1e_minus_5'] for k, v in result['feature_parity'].items()})
