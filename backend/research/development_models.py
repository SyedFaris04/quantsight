"""Fixed development-only model study. Run from backend with -m research.development_models."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from market_calendar import schedule
from research.freeze_prices import ROOT, digest, load_development, write_json
from research.news_pilot import metrics
from research.validation import calendar_labels, purged_segments

PROTOCOL = Path(__file__).with_name('development_model_protocol.json')


def finance_features(frame, sessions):
    """Only backward-looking, scale-free inputs. Missing sessions remain missing."""
    data = frame.copy()
    data['date'] = pd.to_datetime(data.date)
    if data.date.duplicated().any() or not data.date.isin(sessions).all():
        raise ValueError('Invalid feature dates')
    data = data.set_index('date').sort_index().reindex(sessions)
    close = data['Adj Close']
    result = pd.DataFrame(index=sessions)
    for lag in [1, 5, 10, 20]:
        valid = close.notna().rolling(lag + 1).sum().eq(lag + 1)
        result[f'return_{lag}'] = close.pct_change(lag, fill_method=None).where(valid)
    result['volatility_20'] = result.return_1.rolling(20).std()
    result['drawdown_20'] = close / close.rolling(20).max() - 1
    result['ma20_distance'] = close / close.rolling(20).mean() - 1
    result['ma60_distance'] = close / close.rolling(60).mean() - 1
    result['intraday_range'] = (data.High - data.Low) / data.Close
    result['volume_relative_20'] = data.Volume / data.Volume.rolling(20).mean().replace(0, np.nan) - 1
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    # A simple rolling RSI variant, explicitly fixed here; constant prices => 0.5.
    result['rsi_14'] = (gain / (gain + loss).replace(0, np.nan)).where((gain + loss).ne(0), 0.5)
    result['Close'] = close
    result.index.name = 'date'
    return result.loc[data.Close.notna()].replace([np.inf, -np.inf], np.nan).reset_index()


def prepare_inputs(snapshot, protocol):
    request = json.loads((snapshot / 'request.json').read_text(encoding='utf-8'))
    source = request['protocol']
    if request['attempt_id'] != protocol['snapshot_id'] or source['development_end_exclusive'] != protocol['holdout_start']:
        raise ValueError('Snapshot does not match the registered protocol')
    sessions = schedule(source['download_start'], protocol['holdout_start']).index
    sessions = sessions[sessions < pd.Timestamp(protocol['holdout_start'])]
    parts = []
    for ticker in source['universe']:
        part = finance_features(load_development(snapshot, ticker), sessions)
        part['ticker'] = ticker
        parts.append(part)
    panel = pd.concat(parts, ignore_index=True)
    spy = panel[panel.ticker.eq('SPY')].set_index('date').return_5
    panel['spy_return_5'] = panel.date.map(spy)
    panel['excess_return_5'] = panel.return_5 - panel.spy_return_5
    labeled = calendar_labels(panel, sessions, protocol['horizon_sessions'])
    complete = labeled.dropna(subset=protocol['features']).copy()
    return complete, source['development_folds'], {'input_rows': len(labeled), 'feature_complete_rows': len(complete),
                                                'dropped_warmup_or_missing_rows': len(labeled) - len(complete)}


def raw_logit(probabilities, clip=1e-6):
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Invalid probability')
    p = np.clip(probabilities, clip, 1 - clip)
    return np.log(p / (1 - p)).reshape(-1, 1)


def reliability(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    bins = np.minimum((probabilities * 10).astype(int), 9)
    return [{'lower': index / 10, 'upper': (index + 1) / 10, 'n': int((bins == index).sum()),
             'mean_probability': float(probabilities[bins == index].mean()) if (bins == index).any() else None,
             'observed_up_rate': float(labels[bins == index].mean()) if (bins == index).any() else None}
            for index in range(10)]


def model_factory(name, parameters):
    if name == 'logistic':
        return make_pipeline(StandardScaler(), LogisticRegression(**parameters))
    if name == 'hist_boost':
        return HistGradientBoostingClassifier(**parameters)
    if name == 'xgboost':
        return XGBClassifier(**parameters)
    raise ValueError('Unregistered model')


def paired_interval(predictions, candidate, protocol):
    rng = np.random.default_rng(protocol['seed'])
    reps, block = protocol['repetitions'], protocol['block_sessions']
    yearly_samples, yearly_means = [], []
    for _, part in predictions.groupby('fold', sort=True):
        gain = (part.fit_prior - part.signal) ** 2 - (part[candidate] - part.signal) ** 2
        daily = part.assign(gain=gain).groupby('date').gain.agg(['sum', 'count'])
        values, counts = daily['sum'].to_numpy(), daily['count'].to_numpy()
        starts = rng.integers(0, len(daily), (reps, int(np.ceil(len(daily) / block))))
        indices = ((starts[:, :, None] + np.arange(block)) % len(daily)).reshape(reps, -1)[:, :len(daily)]
        yearly_samples.append(values[indices].sum(axis=1) / counts[indices].sum(axis=1))
        yearly_means.append(values.sum() / counts.sum())
    return {'brier_gain_vs_fit_prior': float(np.mean(yearly_means)),
            'interval_95': np.quantile(np.mean(yearly_samples, axis=0), [.025, .975]).tolist()}


def select_candidate(summaries):
    # Input order implements the predeclared exact-tie rule; baselines first.
    return min(summaries, key=lambda row: row['mean_metrics']['brier_score'])['candidate']


def run(snapshot, output_root):
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if protocol['fit_budget'] != {'predictors': 9, 'calibrators': 9, 'total': 18}:
        raise ValueError('Unexpected fit budget')
    code_files = [Path(__file__), Path(__file__).with_name('validation.py'),
                  Path(__file__).with_name('freeze_prices.py'), Path(__file__).with_name('news_pilot.py'),
                  ROOT / 'market_calendar.py']
    registration = {'registered_at': datetime.now(timezone.utc).isoformat(), 'protocol': protocol,
                    'protocol_sha256': digest(PROTOCOL), 'snapshot_manifest_sha256': digest(snapshot / 'manifest.json'),
                    'code_hashes': {p.name: digest(p) for p in code_files},
                    'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'pandas-market-calendars']}}
    identity = {k: v for k, v in registration.items() if k != 'registered_at'}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    directory = output_root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'registration.json', registration)  # before loading features or fitting
    panel, folds, coverage = prepare_inputs(snapshot, protocol)
    if len(folds) != 3:
        raise ValueError('Expected exactly three registered folds')
    results, frames, artifact_hashes = [], [], {}
    predictor_fits = calibrator_fits = 0
    features = protocol['features']
    for fold in folds:
        parts = purged_segments(panel, fold, protocol['holdout_start'])
        fit, calibration, validation = (parts[key] for key in ['fit', 'calibration', 'validation'])
        if any(part.empty or part.signal.nunique() != 2 for part in parts.values()):
            raise ValueError('Every segment requires observations of both classes')
        y = fit.signal.to_numpy(int)
        outputs = {key: np.full(len(validation), value) for key, value in [
            ('fit_prior', y.mean()), ('calibration_prior', calibration.signal.mean()), ('always_up', 1.0)]}
        fold_dir = directory / fold['name']
        fold_dir.mkdir()
        print(f"Fitting {fold['name']}: {len(fit)} fit / {len(calibration)} calibration / {len(validation)} validation", flush=True)
        for name, parameters in protocol['models'].items():
            model = model_factory(name, parameters)
            with threadpool_limits(limits=2):
                model.fit(fit[features], y)
                predictor_fits += 1
                raw_cal = model.predict_proba(calibration[features])[:, 1]
                raw_val = model.predict_proba(validation[features])[:, 1]
                config = protocol['calibration']
                calibrator = LogisticRegression(C=config['C'], max_iter=config['max_iter'])
                calibrator.fit(raw_logit(raw_cal, config['probability_clip']), calibration.signal.to_numpy(int))
                calibrator_fits += 1
                outputs[name + '_raw'] = raw_val
                outputs[name + '_sigmoid'] = calibrator.predict_proba(raw_logit(raw_val, config['probability_clip']))[:, 1]
            artifact = fold_dir / f'{name}.pkl'
            artifact.write_bytes(pickle.dumps({'model': model, 'calibrator': calibrator, 'features': features,
                                              'fold': fold, 'protocol_sha256': registration['protocol_sha256']}))
            artifact_hashes[f"{fold['name']}/{name}.pkl"] = digest(artifact)
        saved = validation[['ticker', 'date', 'label_end', 'signal']].copy()
        saved['fold'] = fold['name']
        for candidate, probabilities in outputs.items():
            saved[candidate] = probabilities
            results.append({'fold': fold['name'], 'year': int(validation.date.dt.year.iloc[0]), 'candidate': candidate,
                            'metrics': metrics(validation.signal.to_numpy(int), probabilities),
                            'reliability': reliability(validation.signal.to_numpy(int), probabilities)})
        saved.to_csv(fold_dir / 'predictions.csv', index=False)
        artifact_hashes[f"{fold['name']}/predictions.csv"] = digest(fold_dir / 'predictions.csv')
        frames.append(saved)
    predictions = pd.concat(frames, ignore_index=True)
    summaries = []
    for candidate in outputs:
        rows = [row for row in results if row['candidate'] == candidate]
        mean = {key: float(np.mean([row['metrics'][key] for row in rows])) for key in rows[0]['metrics'] if key != 'n'}
        summaries.append({'candidate': candidate, 'mean_metrics': mean,
                          'worst_year_brier': max(row['metrics']['brier_score'] for row in rows),
                          **paired_interval(predictions, candidate, protocol['uncertainty'])})
    report = {'schema_version': 1, 'status': 'development_only', 'run_id': run_id,
              'created_at': datetime.now(timezone.utc).isoformat(), 'registration': registration,
              'coverage': coverage, 'validation_rows': len(predictions), 'tickers': int(predictions.ticker.nunique()),
              'folds': folds, 'results': results, 'summaries': summaries, 'selected_candidate': select_candidate(summaries),
              'fits': {'predictors': predictor_fits, 'calibrators': calibrator_fits},
              'artifact_hashes': artifact_hashes, 'holdout_files_opened': False, 'holdout_scores_computed': False,
              'serving_models_changed': False}
    if predictor_fits != 9 or calibrator_fits != 9:
        raise ValueError('Fit budget mismatch')
    write_json(directory / 'report.json', report)
    write_json(ROOT / 'data/research_reports/development_models.json', report)
    print(json.dumps({'run_id': run_id, 'selected_candidate': report['selected_candidate'],
                      'summaries': summaries}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=ROOT / 'data/research/fresh_prices/20260927T141409402344Z')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'data/research/development_models')
    arguments = parser.parse_args()
    run(arguments.snapshot, arguments.output_root)
