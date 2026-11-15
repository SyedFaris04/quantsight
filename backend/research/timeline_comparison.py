"""Compare served models trained from 2015 and from 2005 on identical test rows.

Run from backend: python -m research.timeline_comparison
Writes data/research_reports/timeline_comparison.json, read by the API and frontend.
"""
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import xgboost as xgb

from backtesting.engine import BacktestConfig, performance_metrics, simulate
from research.crisis_walkforward import paired_difference
from research.freeze_prices import ROOT, digest
from timeline_split import fixed_split

PROTOCOL = Path(__file__).with_name('protocols') / 'timelines.json'
FEATURES_DIR = ROOT / 'data/research/timelines'
PREDICTIONS = ROOT / 'data/predictions/timelines'
SEEDS = ROOT / 'data/research/timelines/seeds'
MODELS_DIR = ROOT / 'data/models/timelines'
PRODUCTION = ROOT / 'data/predictions'
CRISIS_REPORT = ROOT / 'data/research_reports/crisis_walkforward.json'
CRISIS_PROTOCOL = Path(__file__).with_name('crisis_walkforward_protocol.json')
OUTPUT = ROOT / 'data/research_reports/timeline_comparison.json'
TIMELINES = ['2015', '2005']
LIMITATIONS = [
    'META, TSLA, ABBV, PYPL, SNAP, MRNA and UBER enter the 2005 timeline only after listing; about 37 instruments exist in 2008.',
    'All 44 instruments are 2026 survivors, so pre-2015 years flatter the universe.',
    'There is no text data before 2015: the 2005 sentiment models mostly learn more price history.',
    'Yahoo adjustments are retrieval-time, not point-in-time.',
    'The 2023-2024 test window has been examined many times; these comparisons are exploratory. The walk-forward study is the cleaner evidence.',
    'Differences use a paired 20-session block bootstrap and are not corrected for multiple comparisons.',
]


def load_predictions(path):
    frame = pd.read_csv(path)
    frame['date'] = pd.to_datetime(frame.date).dt.normalize()
    return frame[['ticker', 'date', 'predicted_signal', 'confidence']]


def common(frames):
    keys = None
    for frame in frames:
        index = pd.MultiIndex.from_frame(frame[['ticker', 'date']])
        keys = index if keys is None else keys.intersection(index)
    keys = keys.to_frame(index=False)
    return [keys.merge(frame, on=['ticker', 'date'], validate='one_to_one').sort_values(['date', 'ticker'])
            .reset_index(drop=True) for frame in frames]


def equal_count_reliability(signal, p, bins=10):
    """Reliability bins holding equal numbers of predictions; probabilities here cluster near 50%."""
    order = np.argsort(p, kind='stable')
    return [{'n': len(part), 'mean_probability': float(p[part].mean()), 'observed_up_rate': float(signal[part].mean())}
            for part in np.array_split(order, bins)]


def score(frame):
    signal, p = frame.signal.to_numpy(int), frame.confidence.to_numpy(float) / 100
    predicted = frame.predicted_signal.to_numpy(int)
    return {'rows': len(frame), 'accuracy': float((predicted == signal).mean()),
            'balanced_accuracy': float(balanced_accuracy_score(signal, predicted)),
            'auc': float(roc_auc_score(signal, p)), 'brier': float(np.mean((p - signal) ** 2)),
            'buy_fraction': float(predicted.mean()), 'reliability': equal_count_reliability(signal, p)}


def agreement(a, b, labels):
    data = a.merge(b, on=['ticker', 'date'], suffixes=('_a', '_b')).merge(labels, on=['ticker', 'date'])
    disagree = data[data.predicted_signal_a != data.predicted_signal_b]
    rate = lambda column: float((disagree[column] == disagree.signal).mean()) if len(disagree) else None
    return {'rows': len(data), 'agree_rate': float(1 - len(disagree) / len(data)), 'disagree_rows': len(disagree),
            'a_correct_when_disagree': rate('predicted_signal_a'), 'b_correct_when_disagree': rate('predicted_signal_b')}


def ensemble(frames):
    """Row-wise mean P(up) of the served models on their common rows; BUY at 50% or above."""
    aligned = common(list(frames.values()))
    result = aligned[0][['ticker', 'date']].copy()
    result['confidence'] = np.mean([frame.confidence.to_numpy(float) for frame in aligned], axis=0)
    result['predicted_signal'] = (result.confidence >= 50).astype(int)
    return result


def top_shap(model_path, features, split_date):
    saved = pickle.loads(Path(model_path).read_bytes())
    _, test = fixed_split(features, split_date)
    contributions = saved['model'].get_booster().predict(
        xgb.DMatrix(saved['scaler'].transform(test[saved['features']].to_numpy())), pred_contribs=True)[:, :-1]
    importance = pd.Series(np.abs(contributions).mean(axis=0), index=saved['features']).sort_values(ascending=False)
    return [{'feature': name, 'mean_abs': float(value)} for name, value in importance.items()]


def backtest(predictions, features):
    prices = features[['date', 'ticker', 'Open', 'Close']]
    calendar = prices.loc[prices.ticker == 'SPY', 'date'].sort_values()
    first_signal, end = predictions.date.min(), min(predictions.date.max(), calendar.max())
    sessions = calendar[(calendar > first_signal) & (calendar <= end)]
    start, config = sessions.iloc[0], BacktestConfig()
    prices = prices[prices.date.between(first_signal, end)]
    spy = simulate(prices, None, config, start, end, mode='spy', record_trades=False)
    model = simulate(prices, predictions, config, start, end, record_trades=False)
    benchmark = spy['daily']['return'].to_numpy()
    return {'evaluation': {'start': str(start.date()), 'end': str(end.date()), 'sessions': len(sessions)},
            'config': {'commission_bps': config.commission_bps, 'slippage_bps': config.slippage_bps,
                       'top_n': config.top_n, 'annual_risk_free_rate': config.annual_risk_free_rate},
            'ensemble': performance_metrics(model['daily'], config, benchmark),
            'spy': performance_metrics(spy['daily'], config)}


def build():
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    split, models = protocol['split_date'], protocol['models']
    inputs = [PROTOCOL, CRISIS_REPORT, CRISIS_PROTOCOL]
    features = {}
    for kind in ['finance', 'sentiment']:
        path = FEATURES_DIR / f'features_{kind}.csv'
        inputs.append(path)
        features[kind] = pd.read_csv(path)
        features[kind]['date'] = pd.to_datetime(features[kind].date)
    labels = features['finance'][['ticker', 'date', 'signal']]
    raw = {}
    for timeline in TIMELINES:
        for model in models:
            path = PREDICTIONS / timeline / f'{model}_predictions.csv'
            inputs.append(path)
            raw[timeline, model] = load_predictions(path)
    paired = {}
    for model in models:
        a, b = common([raw['2015', model], raw['2005', model]])
        paired[model] = {'2015': a.merge(labels, on=['ticker', 'date']), '2005': b.merge(labels, on=['ticker', 'date'])}
    episodes = json.loads(CRISIS_PROTOCOL.read_text(encoding='utf-8'))['episodes']
    report = {'schema_version': 1, 'status': 'exploratory', 'created_at': datetime.now(timezone.utc).isoformat(),
              'protocol': protocol, 'timelines': {}, 'differences': {}, 'agreement': {}, 'shap_rank_changes': {},
              'production_check': {}, 'limitations': LIMITATIONS}
    window = paired[models[0]]['2015']
    report['test_window'] = {'start': str(window.date.min().date()), 'end': str(window.date.max().date()),
                             'rows': len(window), 'rows_by_model': {m: len(paired[m]['2015']) for m in models}}
    report['baselines'] = {'always_up_accuracy': {m: float(paired[m]['2015'].signal.mean()) for m in models},
                           'train_prior_brier': {}}
    for timeline in TIMELINES:
        trainval, _ = fixed_split(features['finance'], split, protocol['starts'][timeline])
        prior = float(trainval.signal.mean())
        report['baselines']['train_prior_brier'][timeline] = {
            m: float(np.mean((prior - paired[m][timeline].signal) ** 2)) for m in models}
        train_start, train_end = trainval.date.min(), trainval.date.max()
        entry = {'training': {
            'start': str(train_start.date()), 'end': str(train_end.date()), 'rows': len(trainval), 'up_rate': prior,
            'tickers_by_year': {str(k): int(v) for k, v in trainval.groupby(trainval.date.dt.year).ticker.nunique().items()},
            'episodes_included': [e['key'] for e in episodes
                                  if pd.Timestamp(e['start']) >= train_start and pd.Timestamp(e['end']) <= train_end]},
            'models': {}, 'shap': {}}
        for model in models:
            entry['models'][model] = score(paired[model][timeline])
            if model.startswith('lstm'):
                runs = [paired[model][timeline]]
                for seed in protocol['lstm_seeds'][1:]:
                    path = SEEDS / timeline / f'seed{seed}' / f'{model}_predictions.csv'
                    inputs.append(path)
                    runs.append(load_predictions(path).merge(paired[model][timeline][['ticker', 'date', 'signal']],
                                                             on=['ticker', 'date'], validate='one_to_one'))
                scores = [score(run) for run in runs]
                entry['models'][model]['lstm_seeds'] = {
                    'seeds': protocol['lstm_seeds'], **{key: {'mean': float(np.mean([s[key] for s in scores])),
                                                             'min': float(min(s[key] for s in scores)),
                                                             'max': float(max(s[key] for s in scores))}
                                                       for key in ['accuracy', 'brier', 'auc']}}
        for model in [m for m in models if m.startswith('xgb')]:
            path = MODELS_DIR / timeline / f'{model}.pkl'
            inputs.append(path)
            kind = 'sentiment' if model.endswith('sentiment') else 'finance'
            entry['shap'][model] = top_shap(path, features[kind], split)[:10]
        report['timelines'][timeline] = entry
    blended = common([ensemble({m: raw[t, m] for m in models}) for t in TIMELINES])
    for timeline, frame in zip(TIMELINES, blended):
        report['timelines'][timeline]['backtest'] = backtest(frame, features['finance'])
        report['timelines'][timeline]['backtest']['rows'] = len(frame)
    for model in models:
        a, b = paired[model]['2005'], paired[model]['2015']
        frame = a[['date', 'signal']].assign(p_a=a.confidence / 100, p_b=b.confidence / 100,
                                             s_a=a.predicted_signal, s_b=b.predicted_signal)
        report['differences'][model] = {'direction': '2005 minus 2015',
                                        'accuracy': paired_difference(frame, 's_a', 's_b', 'accuracy'),
                                        'brier': paired_difference(frame, 'p_a', 'p_b', 'brier')}
        report['agreement'][model] = agreement(raw['2015', model], raw['2005', model], labels)
        production = PRODUCTION / f'{model}_predictions.csv'
        inputs.append(production)
        rebuilt, current = common([raw['2015', model], load_predictions(production)])
        rebuilt, current = rebuilt.merge(labels, on=['ticker', 'date']), current.merge(labels, on=['ticker', 'date'])
        report['production_check'][model] = {
            'rows': len(rebuilt), 'agree_rate': float((rebuilt.predicted_signal == current.predicted_signal).mean()),
            'accuracy_delta': float((rebuilt.predicted_signal == rebuilt.signal).mean()
                                    - (current.predicted_signal == current.signal).mean())}
    for model in [m for m in models if m.startswith('xgb')]:
        ranks = {t: {row['feature']: i + 1 for i, row in enumerate(
            top_shap(MODELS_DIR / t / f'{model}.pkl', features['sentiment' if model.endswith('sentiment') else 'finance'], split))}
            for t in TIMELINES}
        top = {row['feature'] for t in TIMELINES for row in report['timelines'][t]['shap'][model]}
        report['shap_rank_changes'][model] = sorted(
            [{'feature': f, 'rank_2015': ranks['2015'][f], 'rank_2005': ranks['2005'][f]} for f in top],
            key=lambda row: row['rank_2005'])
    crisis = json.loads(CRISIS_REPORT.read_text(encoding='utf-8'))
    keep = {'xgboost_sigmoid', 'fit_prior', 'always_up'}
    report['crisis'] = {
        'run_id': crisis['run_id'], 'direction': crisis['comparison_direction'], 'episodes_defined': episodes,
        'years': [{'arm': r['arm'], 'year': r['year'], 'candidate': r['candidate'], 'n': r['metrics']['n'],
                   'tickers': r['tickers'], 'fit_rows': r['fit_rows'], 'accuracy': r['metrics']['accuracy'],
                   'brier': r['metrics']['brier_score']} for r in crisis['results'] if r['candidate'] in keep],
        'episodes': [e for e in crisis['episodes'] if e['candidate'] in keep],
        'arm_comparison': crisis['arm_comparison'], 'tickers_by_year': crisis['tickers_by_year']}
    report['input_hashes'] = {Path(p).relative_to(ROOT).as_posix(): digest(Path(p)) for p in inputs}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    result = build()
    print(json.dumps({t: {m: {k: round(v[k], 4) for k in ['accuracy', 'auc', 'brier']}
                          for m, v in result['timelines'][t]['models'].items()} for t in TIMELINES}, indent=2))
    print(json.dumps({t: {k: result['timelines'][t]['backtest'][s][k] for s in ['ensemble', 'spy'] for k in ['total_return']}
                      for t in TIMELINES}))
