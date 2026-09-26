"""Fixed, isolated historical news ablation; does not change serving artifacts."""
from collections import Counter
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research.news_dataset import ROOT, REVISION

PROTOCOL = Path(__file__).with_name("monday_pilot_protocol.json")


def chronological_segments(frame, protocol):
    """Purge by each row's actual label end, not shuffled rows or calendar offsets."""
    result = frame.sort_values(["ticker", "date"]).copy()
    result["label_end"] = result.groupby("ticker").date.shift(-protocol["horizon_sessions"])
    result = result[result.label_end.notna() & result.date.ge(protocol["fit_start"])]
    first, second, end = (protocol[k] for k in ["fit_end_exclusive", "calibration_end_exclusive", "evaluation_end_exclusive"])
    return {
        "fit": result[result.date.lt(first) & result.label_end.lt(first)],
        "calibration": result[result.date.ge(first) & result.date.lt(second) & result.label_end.lt(second)],
        "evaluation": result[result.date.ge(second) & result.date.lt(end) & result.label_end.lt(end)],
    }


def aggregate_news(frame):
    """Text is scored once per unique headline; no supplied outcome fields enter."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    scores = {text: analyzer.polarity_scores(text)["compound"] for text in frame.headline.unique()}
    data = frame.copy()
    data["polarity"] = data.headline.map(scores)
    data["positive"] = data.polarity.gt(.05).astype(float)
    data["negative"] = data.polarity.lt(-.05).astype(float)
    daily = data.groupby(["ticker", "date"]).agg(
        news_count=("article_id", "size"), news_polarity_mean=("polarity", "mean"),
        news_polarity_std=("polarity", "std"), news_positive_share=("positive", "mean"),
        news_negative_share=("negative", "mean")).reset_index()
    daily["news_observed"] = 1.
    daily["news_log_count"] = np.log1p(daily.pop("news_count"))
    daily["news_polarity_std"] = daily.news_polarity_std.fillna(0)
    return daily


def metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = probabilities >= .5
    return {"n": len(labels), "accuracy": float(accuracy_score(labels, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "mcc": float(matthews_corrcoef(labels, predicted)),
            "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
            "brier_score": float(np.mean((probabilities-labels)**2)),
            "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
            "buy_fraction": float(predicted.mean())}


def paired_brier(frame, baseline, candidate, block, repetitions, seed):
    # Resample whole dates jointly across all stocks to retain cross-stock
    # dependence. Contiguous date blocks address overlapping five-session labels.
    data = pd.DataFrame({"date": frame.date, "gain": (baseline-frame.signal.to_numpy())**2 - (candidate-frame.signal.to_numpy())**2})
    daily = data.groupby("date").gain.agg(["sum", "count"])
    values, weights = daily["sum"].to_numpy(), daily["count"].to_numpy()
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(daily), size=int(np.ceil(len(daily)/block)))
        indices = ((starts[:, None]+np.arange(block)) % len(daily)).ravel()[:len(daily)]
        estimates.append(float(values[indices].sum()/weights[indices].sum()))
    return {"brier_improvement": float(values.sum()/weights.sum()),
            "interval_95": np.quantile(estimates, [.025, .975]).tolist(),
            "interpretation": "Positive means lower Brier loss for sentiment; exploratory interval, not adjusted for multiple comparisons."}


def run():
    protocol_bytes = PROTOCOL.read_bytes()
    protocol = json.loads(protocol_bytes)
    if importlib.metadata.version("vaderSentiment") != "3.3.2":
        raise ValueError("Install vaderSentiment==3.3.2 to reproduce this protocol")
    news_path = ROOT / "data/research/drinkall" / REVISION / "normalized_news.jsonl"
    input_path = ROOT / "data/processed/features_finance.csv"
    feature_model = ROOT / "data/models/xgb_finance.pkl"
    input_hashes = {"finance": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    "news": hashlib.sha256(news_path.read_bytes()).hexdigest(),
                    "feature_list_model": hashlib.sha256(feature_model.read_bytes()).hexdigest()}
    run_id = hashlib.sha256(protocol_bytes + json.dumps(input_hashes, sort_keys=True).encode()).hexdigest()[:16]
    directory = ROOT / "data/research/pilots" / run_id
    if directory.exists():
        raise ValueError(f"Run {run_id} already exists; preserve its result instead of overwriting")
    directory.mkdir(parents=True)
    (directory / "protocol.json").write_bytes(protocol_bytes)
    news = pd.read_json(news_path, lines=True, convert_dates=False)
    # Read only already-trusted local artifact metadata; never download pickles.
    with feature_model.open("rb") as handle:
        finance_features = pickle.load(handle)["features"]
    daily = aggregate_news(news)
    finance = pd.read_csv(input_path)
    frame = finance.merge(daily, on=["ticker", "date"], how="left", validate="one_to_one")
    news_features = protocol["coverage_features"] + protocol["sentiment_features"]
    frame[news_features] = frame[news_features].fillna(0)
    segments = chronological_segments(frame, protocol)
    train, calibration, evaluation = (segments[key] for key in ["fit", "calibration", "evaluation"])
    output = {"run_id": run_id, "protocol": protocol, "input_sha256": input_hashes,
              "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
              "software": {k: importlib.metadata.version(k) for k in ["numpy", "pandas", "scikit-learn", "vaderSentiment"]},
              "segments": {name: {"rows": len(part), "date_start": part.date.min(), "date_end": part.date.max(),
                                    "last_label_end": part.label_end.max(), "news_observed_rows": int(part.news_observed.sum())}
                           for name, part in segments.items()}, "models": {}, "paired_sentiment_effect": {}}
    predictions = evaluation[["ticker", "date", "signal", "news_observed"]].copy()
    for family in protocol["model_families"]:
        for arm in protocol["arms"]:
            columns = finance_features.copy()
            if arm != "finance":
                columns += protocol["coverage_features"]
            if arm == "finance_coverage_sentiment":
                columns += protocol["sentiment_features"]
            if family == "logistic":
                model = make_pipeline(StandardScaler(), LogisticRegression(**protocol[family], random_state=protocol["seed"]))
            else:
                model = HistGradientBoostingClassifier(**protocol[family], random_state=protocol["seed"])
            with threadpool_limits(limits=4):
                model.fit(train[columns], train.signal)
                raw_calibration = model.predict_proba(calibration[columns])[:, 1]
                sigmoid = LogisticRegression(C=1e6, max_iter=1000, random_state=protocol["seed"])
                sigmoid.fit(raw_calibration.reshape(-1, 1), calibration.signal)
                raw = model.predict_proba(evaluation[columns])[:, 1]
                probability = sigmoid.predict_proba(raw.reshape(-1, 1))[:, 1]
            key = family + "_" + arm
            predictions[key] = probability
            news_mask = evaluation.news_observed.eq(1).to_numpy()
            output["models"][key] = {"all_rows": metrics(evaluation.signal, probability),
                                    "news_observed_rows": metrics(evaluation.signal.to_numpy()[news_mask], probability[news_mask]),
                                    "features": columns}
            print(key, json.dumps(output["models"][key]["all_rows"]), flush=True)
        output["paired_sentiment_effect"][family] = paired_brier(
            evaluation, predictions[family+"_finance_coverage"].to_numpy(),
            predictions[family+"_finance_coverage_sentiment"].to_numpy(),
            protocol["bootstrap_block_sessions"], protocol["bootstrap_repetitions"], protocol["seed"])
    output["baseline_always_up"] = {"accuracy": float(evaluation.signal.mean()), "buy_fraction": 1.0}
    output["baseline_training_prior"] = metrics(evaluation.signal, np.full(len(evaluation), train.signal.mean()))
    output["completed_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    predictions.to_csv(directory / "predictions.csv", index=False)
    (directory / "results.json").write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    report = ROOT.parent / "docs/research/news_pilot_results.json"
    report.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    print("Completed isolated pilot", run_id, flush=True)


if __name__ == "__main__":
    run()
