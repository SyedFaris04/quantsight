"""Crisis-year walk-forward for two training starts. Run from backend with -m research.crisis_walkforward."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from market_calendar import schedule
from research.development_models import finance_features, model_factory, raw_logit, reliability
from research.freeze_prices import ROOT, digest, load_development, write_json
from research.news_pilot import metrics
from research.validation import calendar_labels, purged_segments

PROTOCOL = Path(__file__).with_name('crisis_walkforward_protocol.json')


def walkforward_folds(arms):
    return [{'name': f'{arm}_{year}', 'arm': arm, 'year': year, 'fit_start': spec['fit_start'],
             'fit_end_exclusive': f'{year - 1}-01-01', 'calibration_end_exclusive': f'{year}-01-01',
             'evaluation_end_exclusive': f'{year + 1}-01-01'}
            for arm, spec in arms.items() for year in spec['years']]


def check_budget(protocol):
    fits = len(walkforward_folds(protocol['arms'])) * len(protocol['models'])
    if protocol['fit_budget'] != {'predictors': fits, 'calibrators': fits}:
        raise ValueError('Registered fit budget does not match folds x models')
    return fits


def validation_rows(labeled, fold, holdout_start):
    rows = purged_segments(labeled, fold, holdout_start)['validation']
    return rows[['ticker', 'date']].sort_values(['date', 'ticker']).reset_index(drop=True)


def paired_difference(frame, a, b, metric, block=20, reps=1000, seed=42):
    """Mean of (a - b) per-row loss: brier (lower is better) or accuracy (higher is better)."""
    signal = frame.signal.to_numpy(float)
    if metric == 'brier':
        delta = (frame[a].to_numpy(float) - signal) ** 2 - (frame[b].to_numpy(float) - signal) ** 2
    elif metric == 'accuracy':
        delta = ((frame[a].to_numpy(float) >= .5) == signal).astype(float) - ((frame[b].to_numpy(float) >= .5) == signal)
    else:
        raise ValueError('Unknown metric')
    daily = pd.DataFrame({'date': frame.date.to_numpy(), 'delta': delta}).groupby('date').delta.agg(['sum', 'count'])
    values, counts = daily['sum'].to_numpy(), daily['count'].to_numpy()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(daily), (reps, int(np.ceil(len(daily) / block))))
    indices = ((starts[:, :, None] + np.arange(block)) % len(daily)).reshape(reps, -1)[:, :len(daily)]
    samples = values[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return {'difference': float(values.sum() / counts.sum()),
            'interval_95': [float(x) for x in np.quantile(samples, [.025, .975])]}


def episode_metrics(predictions, episodes, candidate):
    results = []
    for arm, part in predictions.groupby('arm', sort=True):
        for episode in episodes:
            rows = part[part.date.between(pd.Timestamp(episode['start']), pd.Timestamp(episode['end']))]
            if rows.empty:
                continue
            signal = rows.signal.to_numpy(int)
            results.append({'episode': episode['key'], 'label': episode['label'], 'arm': arm, 'candidate': candidate,
                            'n': len(rows), 'tickers': int(rows.ticker.nunique()) if 'ticker' in rows else None,
                            'metrics': metrics(signal, rows[candidate]),
                            'always_up_accuracy': float(signal.mean()),
                            'fit_prior_brier': float(np.mean((rows.fit_prior.to_numpy(float) - signal) ** 2))})
    return results


def prepare_panel(snapshot, protocol):
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
    return labeled.dropna(subset=protocol['features']).copy()


def run(snapshot, output_root):
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    budget = check_budget(protocol)
    code_files = [Path(__file__), Path(__file__).with_name('development_models.py'), Path(__file__).with_name('validation.py'),
                  Path(__file__).with_name('freeze_prices.py'), ROOT / 'market_calendar.py']
    registration = {'registered_at': datetime.now(timezone.utc).isoformat(), 'protocol': protocol,
                    'protocol_sha256': digest(PROTOCOL), 'snapshot_manifest_sha256': digest(snapshot / 'manifest.json'),
                    'code_hashes': {p.name: digest(p) for p in code_files},
                    'versions': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'xgboost']}}
    identity = {k: v for k, v in registration.items() if k != 'registered_at'}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    directory = output_root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'registration.json', registration)  # before loading features or fitting
    panel = prepare_panel(snapshot, protocol)
    features, config = protocol['features'], protocol['calibration']
    results, frames, hashes, fits = [], [], {}, 0
    for fold in walkforward_folds(protocol['arms']):
        parts = purged_segments(panel, fold, protocol['holdout_start'])
        fit, calibration, validation = parts['fit'], parts['calibration'], parts['validation']
        if any(part.empty or part.signal.nunique() != 2 for part in parts.values()):
            raise ValueError(f"Fold {fold['name']} lacks both classes in a segment")
        y = fit.signal.to_numpy(int)
        outputs = {'fit_prior': np.full(len(validation), y.mean()), 'always_up': np.ones(len(validation))}
        print(f"{fold['name']}: {len(fit)} fit / {len(calibration)} calibration / {len(validation)} validation", flush=True)
        artifact = {}
        for name, parameters in protocol['models'].items():
            model = model_factory(name, parameters)
            with threadpool_limits(limits=2):
                model.fit(fit[features], y)
                raw_cal = model.predict_proba(calibration[features])[:, 1]
                raw_val = model.predict_proba(validation[features])[:, 1]
                calibrator = LogisticRegression(C=config['C'], max_iter=config['max_iter'])
                calibrator.fit(raw_logit(raw_cal, config['probability_clip']), calibration.signal.to_numpy(int))
            fits += 1
            outputs[name + '_raw'] = raw_val
            outputs[name + '_sigmoid'] = calibrator.predict_proba(raw_logit(raw_val, config['probability_clip']))[:, 1]
            artifact[name] = {'model': model, 'calibrator': calibrator}
        path = directory / f"{fold['name']}.pkl"
        path.write_bytes(pickle.dumps({'fold': fold, 'features': features, 'models': artifact}))
        hashes[path.name] = digest(path)
        saved = validation[['ticker', 'date', 'label_end', 'signal']].copy()
        saved['arm'], saved['year'] = fold['arm'], fold['year']
        for candidate, probabilities in outputs.items():
            saved[candidate] = probabilities
            results.append({'fold': fold['name'], 'arm': fold['arm'], 'year': fold['year'], 'candidate': candidate,
                            'fit_rows': len(fit), 'tickers': int(validation.ticker.nunique()),
                            'metrics': metrics(validation.signal.to_numpy(int), probabilities),
                            'reliability': reliability(validation.signal.to_numpy(int), probabilities)})
        frames.append(saved)
    if fits != budget:
        raise ValueError('Fit budget mismatch')
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(directory / 'predictions.csv', index=False)
    hashes['predictions.csv'] = digest(directory / 'predictions.csv')
    first, second = protocol['arms']
    overlap = sorted(set(protocol['arms'][first]['years']) & set(protocol['arms'][second]['years']))
    by_arm = {arm: predictions[predictions.arm.eq(arm) & predictions.year.isin(overlap)]
              .sort_values(['date', 'ticker']).reset_index(drop=True) for arm in (first, second)}
    if not by_arm[first][['ticker', 'date']].equals(by_arm[second][['ticker', 'date']]):
        raise ValueError('Arms disagree on validation rows in overlapping years')
    candidates = [c for c in outputs if c not in ('fit_prior', 'always_up')]
    comparison = []
    for candidate in candidates:
        paired = by_arm[first][['date', 'signal']].assign(a=by_arm[first][candidate], b=by_arm[second][candidate])
        settings = protocol['uncertainty']
        comparison.append({'candidate': candidate, 'years': overlap, 'rows': len(paired),
                           **{metric: paired_difference(paired, 'a', 'b', metric, settings['block_sessions'],
                                                        settings['repetitions'], settings['seed'])
                              for metric in ['brier', 'accuracy']}})
    rows_by_year = panel.groupby(panel.date.dt.year).size()
    tickers_by_year = panel.groupby(panel.date.dt.year).ticker.nunique()
    report = {'schema_version': 1, 'status': 'development_only_exploratory', 'run_id': run_id,
              'created_at': datetime.now(timezone.utc).isoformat(), 'registration': registration,
              'comparison_direction': f'{first} minus {second}', 'results': results,
              'episodes': [row for c in candidates + ['fit_prior'] for row in episode_metrics(predictions, protocol['episodes'], c)],
              'arm_comparison': comparison,
              'rows_by_year': {str(k): int(v) for k, v in rows_by_year.items()},
              'tickers_by_year': {str(k): int(v) for k, v in tickers_by_year.items()},
              'fits': {'predictors': fits, 'calibrators': fits}, 'artifact_hashes': hashes,
              'holdout_files_opened': False, 'serving_models_changed': False}
    write_json(directory / 'report.json', report)
    target = ROOT / 'data/research_reports/crisis_walkforward.json'
    target.unlink(missing_ok=True)
    write_json(target, report)
    print(json.dumps({'run_id': run_id, 'fits': report['fits'],
                      'arm_comparison': [{k: c[k] for k in ('candidate', 'brier', 'accuracy')} for c in comparison]}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'data/research/crisis_walkforward')
    arguments = parser.parse_args()
    run(arguments.snapshot, arguments.output_root)
