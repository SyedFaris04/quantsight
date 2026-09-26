"""Reproducible evaluation artifacts. Never trains or overwrites model files."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score, brier_score_loss,
                             confusion_matrix, log_loss, average_precision_score)
from prediction_contract import LABEL_HORIZON_TRADING_DAYS, TARGET_DESCRIPTION
from backtesting.engine import (BacktestConfig, simulate, performance_metrics,
                               block_bootstrap_intervals, prepare_predictions, normalise_rows)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data"
OUTPUT = DATA / "backtests"
MODEL_NAMES = {
    "xgb_finance": "XGBoost - Finance", "xgb_sentiment": "XGBoost + Sentiment",
    "lstm_finance": "LSTM - Finance", "lstm_sentiment": "LSTM + Sentiment",
    "gru_finance": "GRU - Finance", "gru_sentiment": "GRU + Sentiment",
    "rf_finance": "Random Forest - Finance", "rf_sentiment": "Random Forest + Sentiment",
    "logreg_finance": "Logistic Regression - Finance", "logreg_sentiment": "Logistic Regression + Sentiment",
    "transformer_finance": "Transformer - Finance", "transformer_sentiment": "Transformer + Sentiment",
    "ensemble": "Four-model probability ensemble",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classification_metrics(predictions, labels):
    d = predictions.merge(labels[["date", "ticker", "signal"]], on=["date", "ticker"], validate="one_to_one")
    if len(d) != len(predictions):
        raise ValueError("Prediction labels are missing")
    y, pred, p = d.signal.astype(int), d.predicted_signal, d.p_up
    counts, edges = np.histogram(p, bins=np.linspace(0, 1, 11))
    calibration, ece = [], 0.0
    for i, count in enumerate(counts):
        mask = (p >= edges[i]) & ((p <= edges[i + 1]) if i == 9 else (p < edges[i + 1]))
        if not count:
            continue
        observed, probability = float(y[mask].mean()), float(p[mask].mean())
        ece += count / len(d) * abs(observed - probability)
        calibration.append({"count": int(count), "mean_p_up": probability, "observed_up_rate": observed})
    base = float(y.mean())
    return {"rows": len(d), "accuracy": float(accuracy_score(y, pred)),
            "always_up_accuracy": base, "always_down_accuracy": 1 - base,
            "majority_class_accuracy_reference": max(base, 1 - base),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(y, pred)),
            "roc_auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
            "average_precision_up": float(average_precision_score(y, p)) if y.nunique() == 2 else None,
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "ece_10_bins": float(ece), "buy_rate": float(pred.mean()),
            "confusion_matrix_sell_buy": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "calibration_bins": calibration}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def generate(config, random_runs=100, bootstrap_samples=1000, output=OUTPUT):
    if random_runs < 1:
        raise ValueError("At least one random baseline run required")
    files = [DATA / "processed/features_finance.csv", DATA / "processed/features_sentiment.csv",
             Path(__file__), Path(__file__).with_name("engine.py"), ROOT / "backend/prediction_contract.py"]
    features = normalise_rows(pd.read_csv(files[0]), ["date", "ticker", "signal", "Open", "Close"])
    if not features.signal.isin([0, 1]).all():
        raise ValueError("Unknown/nonbinary labels in evaluation data")
    ordered = features.sort_values(["ticker", "date"])
    future = ordered.groupby("ticker").Close.shift(-LABEL_HORIZON_TRADING_DAYS)
    known = future.notna()
    label_mismatches = int(((future > ordered.Close).astype(int)[known] != ordered.signal[known]).sum())
    if label_mismatches:
        raise ValueError(f"{label_mismatches} saved labels disagree with the five-session contract")
    predictions = {}
    for key in MODEL_NAMES:
        path = DATA / "predictions" / f"{key}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing required model predictions: {path.name}")
        predictions[key] = prepare_predictions(pd.read_csv(path))
        if "actual_signal" in predictions[key]:
            check = predictions[key].merge(features[["ticker", "date", "signal"]], on=["ticker", "date"], validate="one_to_one")
            if len(check) != len(predictions[key]) or not check.actual_signal.eq(check.signal).all():
                raise ValueError(f"Saved ground truth disagrees with features: {key}")
        files.append(path)
    # Same ticker/date pairs for every classification comparison.
    common = None
    for frame in predictions.values():
        ix = pd.MultiIndex.from_frame(frame[["date", "ticker"]])
        common = ix if common is None else common.intersection(ix)
    if len(common) == 0:
        raise ValueError("Models have no common evaluation rows")
    common_frame = common.to_frame(index=False)
    paired = {k: d.merge(common_frame, on=["date", "ticker"], validate="one_to_one") for k, d in predictions.items()}
    calendar = features.loc[features.ticker == "SPY", "date"].sort_values()
    first_signal = common_frame.date.min()
    end = min(common_frame.date.max(), calendar.max())
    sessions = calendar[(calendar > first_signal) & (calendar <= end)]
    if len(sessions) < 30:
        raise ValueError("At least 30 shared tradable sessions required")
    start = sessions.iloc[0]
    prices = features.loc[features.date.between(first_signal, end), ["date", "ticker", "Open", "Close"]]
    print(f"Shared sample: {len(common):,} rows; trading {start.date()} to {end.date()}", flush=True)
    warnings = [
        "Exploratory retrospective evaluation: saved models were developed using this historical period. This is not a fresh untouched test or a live return record.",
        "The fixed current universe may have survivorship/selection bias; delisted assets are not reconstructed.",
        "Stored OHLCV adjustment provenance has not been independently audited. Execution uses supplied Open/Close on the same basis; no extra dividends are added.",
        "Commission/slippage and a constant cash/risk-free rate are assumptions, not historical quotes or Treasury observations.",
        "The label is five-session close-to-close direction. The strategy is weekly long-only allocation at next open, with variable holding periods; these are separate outcomes.",
        "Bootstrap intervals are descriptive and do not correct for all research trials. Deflated Sharpe is unavailable because the complete trial count is unknown.",
    ]
    sentiment = normalise_rows(pd.read_csv(files[1]), ["date", "ticker"])
    sentiment = sentiment.merge(common_frame, on=["date", "ticker"], validate="one_to_one")
    text_cols = [c for c in sentiment if c.startswith(("wsb_", "gdelt_", "emo_"))]
    nonzero = {c: int(pd.to_numeric(sentiment[c], errors="coerce").fillna(0).ne(0).sum()) for c in text_cols}
    if nonzero and not any(nonzero.values()):
        warnings.append("All sentiment/emotion features are zero in the shared evaluation sample. These results cannot establish a benefit from current text coverage.")
    runs = {k: simulate(prices, None, config, start, end, mode=k) for k in ["spy", "universe"]}
    names = {"spy": "SPY buy and hold", "universe": "Equal-weight universe (weekly)", **MODEL_NAMES}
    for key, frame in paired.items():
        print(f"Backtesting {names[key]}", flush=True)
        runs[key] = simulate(prices, frame, config, start, end)
    benchmark = runs["spy"]["daily"]["return"].to_numpy()
    report = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
              "status": "exploratory", "config": asdict(config),
              "target": {"horizon_trading_sessions": LABEL_HORIZON_TRADING_DAYS, "description": TARGET_DESCRIPTION},
              "evaluation": {"first_signal_date": str(first_signal.date()), "start": str(start.date()),
                             "end": str(end.date()), "sessions": len(sessions), "paired_prediction_rows": len(common),
                             "universe": sorted(prices.ticker.unique().tolist()), "benchmark_excluded_from_selection": "SPY"},
              "warnings": warnings, "sentiment_nonzero_rows": nonzero, "strategies": [],
              "label_validation": {"checked_rows": int(known.sum()), "mismatches": label_mismatches,
                                   "unverifiable_tail_rows": int((~known).sum()),
                                   "note": "Last five rows per ticker retain saved labels; their future prices are outside this feature snapshot."},
              "methodology": {"execution": "Previous session close signal; next session open execution",
                              "rebalance": "First evaluation open, then first observed session of each week",
                              "selection": "Top P(up) among BUY predictions above minimum probability; ticker breaks ties",
                              "sizing": "Each model position gets min(1/top_n, position cap); unused slots remain cash",
                              "costs": "Commission plus adverse slippage per side on actual traded notional, including final liquidation",
                              "accounting": "Shares persist across sessions, including overnight gaps; final liquidation at last close",
                              "benchmarks": "SPY fully invested and held; other universe equally weighted weekly; same execution/cost window",
                              "risk_free": "Constant annual effective rate; cash after open accrues one trading-session equivalent",
                              "metrics": "Daily net returns, 252 sessions/year, sample volatility; Sortino uses downside RMS of excess returns"}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output / run_id
    run_dir.mkdir()
    report["run_id"] = run_id
    for key, result in runs.items():
        daily = result["daily"]
        if daily.date.tolist() != runs["spy"]["daily"].date.tolist():
            raise ValueError("Strategy and benchmark dates differ")
        metrics = performance_metrics(daily, config, benchmark)
        wealth = np.r_[config.initial_capital, daily.equity.to_numpy()]
        dd = wealth / np.maximum.accumulate(wealth) - 1
        curve = [{"date": str(first_signal.date()), "equity": config.initial_capital, "drawdown": 0.0}]
        curve += [{"date": row.date, "equity": row.equity, "drawdown": float(dd[i + 1])}
                  for i, row in enumerate(daily.itertuples())]
        strategy = {"key": key, "name": names[key], "metrics": metrics, "curve": curve,
                    "intervals": block_bootstrap_intervals(daily["return"], config, bootstrap_samples),
                    "rebalance_count": len(result["rebalances"]), "trade_count": len(result["trades"]),
                    "recent_trades": result["trades"][-30:],
                    "classification": classification_metrics(paired[key], features) if key in paired else None}
        if key in paired:
            strategy["mean_rebalance_prediction_coverage"] = float(np.mean([x["prediction_coverage"] for x in result["rebalances"]]))
            strategy["mean_target_exposure"] = float(np.mean([x["target_exposure"] for x in result["rebalances"]]))
        report["strategies"].append(strategy)
        daily.to_csv(run_dir / f"{key}_daily.csv", index=False)
        columns = ["date", "signal_date", "ticker", "side", "quantity", "reference_price", "notional", "commission", "slippage", "reason"]
        pd.DataFrame(result["trades"], columns=columns).to_csv(run_dir / f"{key}_trades.csv", index=False)
        write_json(run_dir / f"{key}_rebalances.json", result["rebalances"])
    reference = "ensemble"  # Fixed before this evaluation, not selected by performance.
    random_sharpes, random_returns = [], []
    print(f"Random null: {random_runs} portfolios matching ensemble slots/exposure", flush=True)
    for i in range(random_runs):
        result = simulate(prices, paired[reference], config, start, end, mode="random", random_seed=config.seed + i, record_trades=False)
        m = performance_metrics(result["daily"], config)
        if m["sharpe"] is not None:
            random_sharpes.append(m["sharpe"])
        random_returns.append(m["total_return"])
        if (i + 1) % 50 == 0:
            print(f"  Random portfolios {i + 1}/{random_runs}", flush=True)
    ref = next(s["metrics"] for s in report["strategies"] if s["key"] == reference)
    valid_null = bool(random_sharpes) and ref["sharpe"] is not None
    report["random_null"] = {"reference": reference, "runs": random_runs, "seed": config.seed,
                             "matching": "Same rebalance dates, position count, target exposure, costs, holding schedule; random tickers",
                             "sharpe_p05_p50_p95": np.quantile(random_sharpes, [.05, .5, .95]).tolist() if random_sharpes else None,
                             "total_return_p05_p50_p95": np.quantile(random_returns, [.05, .5, .95]).tolist(),
                             "reference_sharpe_percentile": float(np.mean(np.array(random_sharpes) <= ref["sharpe"]) * 100) if valid_null else None,
                             "one_sided_monte_carlo_p": float((1 + np.sum(np.array(random_sharpes) >= ref["sharpe"])) / (1 + len(random_sharpes))) if valid_null else None}
    report["cost_sensitivity"] = []
    for cost in [0, 5, 15, 30, 50]:
        cfg = BacktestConfig(**{**asdict(config), "commission_bps": float(cost), "slippage_bps": 0.0})
        run = simulate(prices, paired[reference], cfg, start, end)
        spy = simulate(prices, None, cfg, start, end, mode="spy")
        m = performance_metrics(run["daily"], cfg, spy["daily"]["return"])
        report["cost_sensitivity"].append({"one_way_cost_bps": cost, "total_return": m["total_return"], "sharpe": m["sharpe"], "excess_total_return": m["excess_total_return"]})
    report["sources"] = [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in files]
    report["exports"] = {"metrics": f"{run_id}/metrics.csv", "audit_directory": run_id}
    pd.DataFrame([{"strategy": s["name"], **s["metrics"]} for s in report["strategies"]]).to_csv(run_dir / "metrics.csv", index=False)
    write_json(run_dir / "report.json", report)
    write_json(output / "report.json", report)
    print(f"Report: {output / 'report.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-runs", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--commission-bps", type=float, default=10)
    parser.add_argument("--slippage-bps", type=float, default=5)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--min-probability", type=float, default=.5)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    generate(BacktestConfig(top_n=args.top_n, commission_bps=args.commission_bps,
                            slippage_bps=args.slippage_bps, annual_risk_free_rate=args.risk_free_rate,
                            min_probability=args.min_probability), args.random_runs, args.bootstrap_samples, args.output)


if __name__ == "__main__":
    main()
