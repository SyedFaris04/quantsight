"""Offline fixed confidence/coverage audit of completed development predictions.

Run from backend: python -m research.confidence_audit [--check]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.confidence_report import ROOT, REPORT, PROTOCOL, checksum, normalized_hash, text_hashes, validate
from research.development_report import validate_report


def validate_predictions(frame, parent, year, candidates):
    expected = {'ticker', 'date', 'label_end', 'signal', 'fold', *candidates}
    if set(frame.columns) != expected or frame.empty or frame.isna().any().any():
        raise ValueError('Incomplete development predictions')
    dates, ends = pd.to_datetime(frame.date), pd.to_datetime(frame.label_end)
    if (not dates.dt.year.eq(year).all() or not ends.dt.year.eq(year).all() or
            not (ends > dates).all() or not frame.fold.eq(f'validate_{year}').all() or
            frame.duplicated(['ticker', 'date']).any() or not frame.signal.isin([0, 1]).all() or
            not np.isfinite(frame[candidates].to_numpy()).all() or
            not frame[candidates].ge(0).all().all() or not frame[candidates].le(1).all().all()):
        raise ValueError('Invalid dates, labels or probabilities; holdout rows are forbidden')
    n = next(r['metrics']['n'] for r in parent['results'] if r['year'] == year and r['candidate'] == 'fit_prior')
    if len(frame) != n or frame.ticker.nunique() != parent['tickers']:
        raise ValueError('Prediction coverage differs from the completed study')
    return frame


def threshold_rows(frame, candidate, thresholds):
    p, y, baseline = (frame[key].to_numpy(float) for key in [candidate, 'signal', 'fit_prior'])
    confidence = np.maximum(p, 1 - p)
    rows = []
    for threshold in thresholds:
        mask = confidence >= threshold
        count = int(mask.sum())
        dates = int(frame.loc[mask, 'date'].nunique())
        selected_p, selected_y, selected_baseline = p[mask], y[mask], baseline[mask]
        accuracy = float(np.mean((selected_p >= .5) == selected_y)) if count else None
        rows.append({'threshold': threshold, 'retained_rows': count, 'retained_dates': dates,
                     'coverage': count / len(frame), 'small_sample': bool(count and (count < 200 or dates < 20)),
                     'accuracy': accuracy, 'error_rate': 1 - accuracy if count else None,
                     'mean_confidence': float(confidence[mask].mean()) if count else None,
                     'brier_score': float(np.mean((selected_p - selected_y) ** 2)) if count else None,
                     'baseline_accuracy': float(np.mean((selected_baseline >= .5) == selected_y)) if count else None,
                     'baseline_brier': float(np.mean((selected_baseline - selected_y) ** 2)) if count else None,
                     'always_up_accuracy': float(selected_y.mean()) if count else None,
                     'predicts_up': float((selected_p >= .5).mean()) if count else None})
    return rows


def build():
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    parent = validate_report(json.loads((ROOT / 'data/research_reports/development_models.json').read_text(encoding='utf-8')))
    if protocol['fit_budget'] != 0 or protocol['parent_run_id'] != parent['run_id']:
        raise ValueError('Unsupported diagnostic protocol')
    frames, sources = [], {}
    for year in protocol['years']:
        relative = f"data/research_predictions/development_{parent['run_id']}/validate_{year}.csv"
        path = ROOT / relative
        source_hash = parent['artifact_hashes'][f'validate_{year}/predictions.csv']
        if source_hash not in text_hashes(path):
            raise ValueError('Prediction export does not match its original hash')
        sources[relative] = normalized_hash(path)
        frames.append(validate_predictions(pd.read_csv(path), parent, year, protocol['candidates']))
    periods = {}
    for name, frame in [('pooled', pd.concat(frames, ignore_index=True)), *zip(map(str, protocol['years']), frames)]:
        periods[name] = {'total_rows': len(frame), 'candidates': {
            candidate: threshold_rows(frame, candidate, protocol['thresholds']) for candidate in protocol['candidates']}}
    report = {'schema_version': 1, 'status': 'exploratory_diagnostic', 'parent_run_id': parent['run_id'],
              'protocol': protocol, 'parent_report_sha256': checksum(parent), 'sources': sources,
              'code_hashes': {name: normalized_hash(Path(__file__).with_name(name))
                             for name in ['confidence_audit.py', 'confidence_report.py']},
              'new_fits': 0, 'holdout_files_opened': False, 'serving_models_changed': False, 'periods': periods}
    report['content_sha256'] = checksum(report)
    return validate(report, parent, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = build()
    if args.check:
        if json.loads(REPORT.read_text(encoding='utf-8')) != report:
            raise SystemExit('Confidence audit differs from the completed predictions.')
        print('Verified 9 candidates, 7 thresholds and 4 period views against original prediction exports.')
    else:
        if REPORT.exists():
            raise SystemExit('Preserve the published diagnostic; it already exists.')
        temporary = REPORT.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
        temporary.replace(REPORT)
        print(f'Published {REPORT}')


if __name__ == '__main__':
    main()
