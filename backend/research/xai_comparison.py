"""Run the registered XAI study offline. No model fitting or external requests.

From backend with requirements-xai-research: python -m research.xai_comparison
Only repository-owned model bundles are deserialized.
"""
import hashlib
import importlib.metadata
import json
import pickle
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from captum.attr import IntegratedGradients
from lime.lime_base import LimeBase
from sklearn.linear_model import Ridge
from cv_utils import apply_calibrator
from train_lstm import LSTMAttention
from research.xai_metrics import top_overlap, removal_response, random_response, day_occlusion
from research.xai_report import source_digest, summarize, validate

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data"
PROTOCOL = ROOT / "backend/research/protocols/xai_comparison_v1.json"
OUTPUT = DATA / "research_reports/xai_comparison.json"


def verify_prediction(bundle, raw, saved, threshold):
    calibrated = float(apply_calibrator(bundle["calibrator"], [raw], bundle["calibration_method"])[0])
    if not np.isfinite(calibrated) or abs(calibrated - float(saved.confidence) / 100) > .00005001:
        raise ValueError("Saved calibrated probability does not reproduce.")
    if int(raw >= threshold) != int(saved.predicted_signal):
        raise ValueError("Saved model label does not reproduce.")
    return calibrated


def lime_attribution(original, reference, predict, seed):
    """LIME weighted ridge on observed historical rows, no synthetic columns."""
    centre = reference.mean(axis=0)
    spread = reference.std(axis=0)
    spread[spread < 1e-8] = 1
    rng = np.random.default_rng(seed)
    sample = reference[rng.choice(len(reference), 1024, replace=False)]
    points = np.vstack([original, sample])
    interpretable = (points - centre) / spread
    distances = np.linalg.norm(interpretable - interpretable[0], axis=1)
    width = .75 * np.sqrt(len(original))
    kernel = lambda d: np.sqrt(np.exp(-(d ** 2) / width ** 2))
    labels = predict(points).reshape(-1, 1)
    intercept, local, fidelity, local_prediction = LimeBase(kernel, random_state=seed).explain_instance_with_data(
        interpretable, labels, distances, 0, len(original), feature_selection="none",
        model_regressor=Ridge(alpha=1, fit_intercept=True),
    )
    coefficients = np.zeros(len(original))
    for index, weight in local:
        coefficients[index] = weight
    values = coefficients * interpretable[0]
    return values, {"weighted_r2": float(fidelity), "point_error": abs(float(local_prediction[0]) - float(labels[0, 0])),
                    "surrogate_intercept": float(intercept)}


def records_for_xgb(frame, saved):
    with (DATA / "models/xgb_finance.pkl").open("rb") as stream:
        bundle = pickle.load(stream)
    features = bundle["features"]
    raw_reference = frame.loc[frame.date < "2023-01-01", features].to_numpy(dtype=float)
    reference = bundle["scaler"].transform(raw_reference)
    model = bundle["model"]
    booster = model.get_booster()
    best = getattr(model, "best_iteration", None)
    iterations = (0, int(best) + 1) if best is not None else (0, 0)
    def predict(values):
        return booster.predict(xgb.DMatrix(values, feature_names=booster.feature_names), output_margin=True, iteration_range=iterations)
    baseline, alternative = reference.mean(axis=0), np.median(reference, axis=0)
    rows = []
    for pred in saved.itertuples(index=False):
        source = frame.loc[(frame.ticker == pred.ticker) & (frame.date == pred.date)]
        if len(source) != 1: raise ValueError("Missing or duplicate input.")
        original = bundle["scaler"].transform(source[features].to_numpy(dtype=float))[0]
        margin = float(predict(original[None])[0])
        raw = float(model.predict_proba(original[None])[0, 1])
        calibrated = verify_prediction(bundle, raw, pred, .5)
        start = perf_counter()
        shap = booster.predict(xgb.DMatrix(original[None], feature_names=booster.feature_names),
                               pred_contribs=True, approx_contribs=False, iteration_range=iterations)[0]
        runtime = (perf_counter() - start) * 1000
        error = abs(float(shap.sum()) - margin)
        if error > 1e-5: raise ValueError("TreeSHAP completeness failed.")
        random = random_response(predict, original, baseline)
        methods = [{"key": "native_treeshap", "name": "TreeSHAP", "values": shap[:-1].astype(float).tolist(),
                    "runtime_ms": runtime, "completeness_error": error, "baseline_logit": float(shap[-1]),
                    "removal_response": removal_response(predict, original, baseline, shap[:-1]),
                    "reference_top3_overlap": None, "seed_top3_overlap": 1.0,
                    "repeatability_definition": "Deterministic exact method on unchanged input; not perturbation stability."}]
        results, times = [], []
        for seed in [42, 43, 44]:
            start = perf_counter()
            values, diagnostics = lime_attribution(original, reference, predict, seed)
            times.append((perf_counter() - start) * 1000)
            results.append(values)
            if seed == 42: first = diagnostics
        methods.append({"key": "lime_historical_neighborhood", "name": "LIME (historical neighborhood)",
                        "values": results[0].tolist(), "runtime_ms": float(np.mean(times)),
                        "completeness_error": None, **first, "reference_top3_overlap": None,
                        "seed_top3_overlap": float(np.mean([top_overlap(a, b) for a, b in combinations(results, 2)])),
                        "removal_response": removal_response(predict, original, baseline, results[0])})
        # Reference sensitivity of the evaluation response, not a second explanation.
        for method in methods:
            method["alternative_removal_response"] = removal_response(predict, original, alternative, np.array(method["values"]))
        rows.append({"model": "xgb_finance", "ticker": pred.ticker, "date": pred.date,
                     "units": features, "raw_logit": margin, "raw_probability_up": raw,
                     "calibrated_probability_up": calibrated, "saved_probability_up": float(pred.confidence) / 100,
                     "direction": "BUY" if pred.predicted_signal else "SELL", "random_response": random,
                     "methods": methods})
        print(f"XGB {pred.ticker} verified", flush=True)
    return rows


def records_for_lstm(frame, saved):
    bundle = torch.load(DATA / "models/lstm_finance.pt", map_location="cpu", weights_only=False)
    params = bundle["hyperparams"]
    model = LSTMAttention(bundle["n_features"], params["lstm_hidden"], params["lstm_layers"], params["dropout"])
    model.load_state_dict(bundle["model_state"])
    model.eval()
    features = bundle["feature_cols"]
    reference = bundle["scaler"].transform(frame.loc[frame.date < "2023-01-01", features].to_numpy(dtype=float))
    length = params["seq_len"]
    baseline = np.repeat(reference.mean(axis=0)[None], length, axis=0).astype(np.float32)
    alternative = np.repeat(np.median(reference, axis=0)[None], length, axis=0).astype(np.float32)
    def forward(x): return model(x, return_logits=True)
    def predict(values):
        with torch.no_grad(): return forward(torch.tensor(values, dtype=torch.float32)).numpy()
    ig = IntegratedGradients(forward)
    rows = []
    for pred in saved.itertuples(index=False):
        history = frame.loc[(frame.ticker == pred.ticker) & (frame.date < pred.date)].sort_values("date").tail(length)
        if len(history) != length: raise ValueError("Incomplete sequence.")
        original = bundle["scaler"].transform(history[features].to_numpy(dtype=float)).astype(np.float32)
        margin = float(predict(original[None])[0])
        raw = float(torch.sigmoid(torch.tensor(margin)).item())
        calibrated = verify_prediction(bundle, raw, pred, bundle["threshold"])
        tensor = torch.tensor(original[None])
        start = perf_counter()
        attribution, delta = ig.attribute(tensor, baselines=torch.tensor(baseline[None]), n_steps=128,
                                          method="gausslegendre", return_convergence_delta=True)
        runtime = (perf_counter() - start) * 1000
        values = attribution.detach().numpy()[0].sum(axis=1)
        alt_attr, _ = ig.attribute(tensor, baselines=torch.tensor(alternative[None]), n_steps=128,
                                  method="gausslegendre", return_convergence_delta=True)
        alt_values = alt_attr.detach().numpy()[0].sum(axis=1)
        error = float(delta.abs().item())
        if not np.isfinite(error): raise ValueError("Invalid IG convergence.")
        random = random_response(predict, original, baseline, temporal=True)
        methods = [{"key": "integrated_gradients", "name": "Integrated Gradients", "values": values.astype(float).tolist(),
                    "runtime_ms": runtime, "completeness_error": error, "baseline_logit": float(predict(baseline[None])[0]),
                    "seed_top3_overlap": None, "reference_top3_overlap": top_overlap(values, alt_values),
                    "removal_response": removal_response(predict, original, baseline, values, temporal=True),
                    "alternative_removal_response": removal_response(predict, original, alternative, alt_values, temporal=True)}]
        start = perf_counter()
        occlusion = day_occlusion(predict, original, baseline)
        runtime = (perf_counter() - start) * 1000
        alt_occlusion = day_occlusion(predict, original, alternative)
        methods.append({"key": "day_window_occlusion", "name": "Day-window occlusion", "values": occlusion.astype(float).tolist(),
                        "runtime_ms": runtime, "completeness_error": None, "seed_top3_overlap": None,
                        "reference_top3_overlap": top_overlap(occlusion, alt_occlusion),
                        "removal_response": removal_response(predict, original, baseline, occlusion, temporal=True),
                        "alternative_removal_response": removal_response(predict, original, alternative, alt_occlusion, temporal=True)})
        rows.append({"model": "lstm_finance", "ticker": pred.ticker, "date": pred.date,
                     "window_start": str(history.date.iloc[0]), "window_end": str(history.date.iloc[-1]),
                     "units": history.date.tolist(), "raw_logit": margin, "raw_probability_up": raw,
                     "calibrated_probability_up": calibrated, "saved_probability_up": float(pred.confidence) / 100,
                     "decision_threshold": float(bundle["threshold"]),
                     "direction": "BUY" if pred.predicted_signal else "SELL", "random_response": random, "methods": methods})
        print(f"LSTM {pred.ticker} verified", flush=True)
    return rows


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    predictions = {key: pd.read_csv(DATA / f"predictions/{key}_predictions.csv", dtype={"date": str})
                   for key in protocol["model_variants"]}
    for df in predictions.values():
        df["date"] = df["date"].str[:10]
    date = min(str(df.date.max()) for df in predictions.values())
    samples = {key: df.loc[df.date == date].sort_values("ticker") for key, df in predictions.items()}
    if any(len(df) != 44 or df.ticker.nunique() != 44 for df in samples.values()):
        raise ValueError("Protocol requires 44 distinct matched instruments per model.")
    if set(samples["xgb_finance"].ticker) != set(samples["lstm_finance"].ticker):
        raise ValueError("Model samples do not match.")
    frame = pd.read_csv(DATA / "processed/features_finance.csv", dtype={"date": str})
    frame["date"] = frame["date"].str[:10]
    if frame.date.max() >= "2025-01-01": raise ValueError("Only the legacy pre-2025 development data are allowed.")
    rows = records_for_xgb(frame, samples["xgb_finance"]) + records_for_lstm(frame, samples["lstm_finance"])
    report = {"schema_version": 1, "status": "exploratory", "protocol": protocol,
              "generated_at": datetime.now(timezone.utc).isoformat(), "prediction_date": date,
              "holdout_files_opened": False, "serving_models_changed": False,
              "scale": "raw_log_odds", "rows": rows,
              "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "torch", "xgboost", "scikit-learn", "captum", "lime"]}}
    files = ["backend/research/protocols/xai_comparison_v1.json", "backend/research/xai_comparison.py",
             "backend/research/xai_metrics.py", "backend/train_lstm.py", "backend/cv_utils.py",
             "backend/data/models/xgb_finance.pkl", "backend/data/models/lstm_finance.pt",
             "backend/data/processed/features_finance.csv",
             "backend/data/predictions/xgb_finance_predictions.csv", "backend/data/predictions/lstm_finance_predictions.csv"]
    report["sources"] = [{"path": f, "sha256": source_digest(ROOT / f)} for f in files]
    report["summaries"] = summarize(rows)
    validate(report)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summaries": report["summaries"]}), flush=True)


if __name__ == "__main__":
    main()
