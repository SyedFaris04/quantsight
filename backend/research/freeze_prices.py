"""Freeze a candidate holdout without fitting models or reporting its returns."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from market_calendar import schedule

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name('fresh_data_protocol.json')
PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def audit_prices(frame, expected, holdout_start):
    """Only structural diagnostics; never compute holdout outcomes or returns."""
    missing_columns = sorted(set(PRICE_COLUMNS) - set(frame.columns))
    if missing_columns:
        return {'valid': False, 'missing_columns': missing_columns}
    dates = pd.to_datetime(frame.index, errors='coerce')
    dates = dates.tz_localize(None).normalize()
    values = frame[PRICE_COLUMNS].apply(pd.to_numeric, errors='coerce')
    array = values.to_numpy(float)
    bad_values = ~np.isfinite(array).all(axis=1) | (array[:, :5] <= 0).any(axis=1) | (array[:, 5] < 0)
    bad_range = ((values.High < values[['Open', 'Close', 'Low']].max(axis=1) - 1e-8)
                 | (values.Low > values[['Open', 'Close', 'High']].min(axis=1) + 1e-8))
    present = pd.DatetimeIndex(dates.dropna().unique()).sort_values()
    internal = expected[expected >= present.min()] if len(present) else expected
    missing = internal.difference(present)
    holdout_expected = expected[expected >= pd.Timestamp(holdout_start)]
    holdout_missing = holdout_expected.difference(present)
    unexpected = present.difference(expected)
    invalid_rows = int((bad_values | bad_range.to_numpy()).sum())
    duplicate_dates = int(dates.duplicated().sum())
    invalid_dates = int(dates.isna().sum())
    return {
        'valid': bool(len(frame) and not invalid_rows and not duplicate_dates and not invalid_dates
                      and not len(missing) and not len(unexpected) and not len(holdout_missing)),
        'rows': len(frame), 'start': present.min().date().isoformat() if len(present) else None,
        'end': present.max().date().isoformat() if len(present) else None,
        'invalid_value_rows': invalid_rows, 'duplicate_dates': duplicate_dates,
        'invalid_dates': invalid_dates, 'unexpected_dates': unexpected.strftime('%Y-%m-%d').tolist(),
        'missing_internal_sessions': missing.strftime('%Y-%m-%d').tolist(),
        'missing_holdout_sessions': holdout_missing.strftime('%Y-%m-%d').tolist(),
        'holdout_expected_sessions': len(holdout_expected),
        'leading_unavailable_sessions': int((expected < present.min()).sum()) if len(present) else len(expected),
    }


def freeze():
    import yfinance as yf
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    protocol_hash = digest(PROTOCOL)
    attempt_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = ROOT / 'data/research/fresh_prices' / attempt_id
    directory.mkdir(parents=True, exist_ok=False)
    for name in ['raw', 'development', 'candidate_holdout']:
        (directory / name).mkdir()
    tickers = sorted(set(protocol['universe'] + protocol['context_tickers']))
    request = {'attempt_id': attempt_id, 'created_at': datetime.now(timezone.utc).isoformat(),
               'protocol': protocol, 'protocol_sha256': protocol_hash,
               'code_sha256': digest(Path(__file__)),
               'software': {name: importlib.metadata.version(name) for name in ['yfinance', 'pandas', 'numpy', 'pandas-market-calendars']}}
    write_json(directory / 'request.json', request)
    print('Registered acquisition attempt ' + attempt_id, flush=True)
    raw = yf.download(tickers, start=protocol['download_start'], end=protocol['download_end_exclusive'],
                      **protocol['download_options'])
    expected = schedule(protocol['download_start'], protocol['download_end_exclusive']).index
    expected = expected[expected < pd.Timestamp(protocol['download_end_exclusive'])]
    manifest = {'attempt_id': attempt_id, 'protocol_sha256': protocol_hash,
                'prior_2025_use': protocol['prior_2025_use'], 'status': 'candidate_holdout_not_evaluated',
                'model_fits': 0, 'holdout_scores_computed': False, 'tickers': {}, 'files': {},
                'limitations': protocol['limitations']}
    for ticker in tickers:
        if raw is None or raw.empty or ticker not in raw.columns.get_level_values(0):
            manifest['tickers'][ticker] = {'valid': False, 'error': 'No data returned'}
            continue
        frame = raw[ticker].copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame.index.name = 'date'
        raw_path = directory / 'raw' / (ticker + '.csv')
        frame.to_csv(raw_path)
        manifest['files'][str(raw_path.relative_to(directory)).replace('\\', '/')] = digest(raw_path)
        # Batched download creates all-null rows before a ticker's listing.
        # Retain the raw file, but do not mistake those placeholders for bars.
        frame = frame.dropna(how='all', subset=[c for c in PRICE_COLUMNS if c in frame.columns])
        audit = audit_prices(frame, expected, protocol['holdout_start'])
        manifest['tickers'][ticker] = audit
        for part, mask in [('development', frame.index < protocol['holdout_start']),
                           ('candidate_holdout', frame.index >= protocol['holdout_start'])]:
            path = directory / part / (ticker + '.csv')
            frame.loc[mask].to_csv(path)
            manifest['files'][str(path.relative_to(directory)).replace('\\', '/')] = digest(path)
        print(json.dumps({'ticker': ticker, 'valid': audit['valid'], 'rows': audit.get('rows', 0),
                          'missing_holdout_sessions': len(audit.get('missing_holdout_sessions', []))}), flush=True)
    manifest['data_quality_passed'] = all(row['valid'] for row in manifest['tickers'].values())
    manifest['completed_at'] = datetime.now(timezone.utc).isoformat()
    write_json(directory / 'manifest.json', manifest)
    report = ROOT.parent / 'docs/research' / ('fresh_price_audit_' + attempt_id + '.json')
    write_json(report, manifest)
    print(json.dumps({'attempt_id': attempt_id, 'data_quality_passed': manifest['data_quality_passed'],
                      'report': str(report), 'holdout_scores_computed': False}), flush=True)
    return manifest


def load_development(directory, ticker):
    """Development access only; verify snapshot bytes before exposing any rows."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
    if not manifest['data_quality_passed']:
        raise ValueError('Snapshot has unresolved data quality failures')
    if ticker not in manifest['tickers'] or Path(ticker).name != ticker:
        raise ValueError('Unknown ticker')
    key = 'development/' + ticker + '.csv'
    path = directory / key
    if digest(path) != manifest['files'][key]:
        raise ValueError('Development artifact hash mismatch')
    frame = pd.read_csv(path, parse_dates=['date'])
    if not frame.date.lt(request['protocol']['development_end_exclusive']).all():
        raise ValueError('Holdout rows cannot enter development')
    return frame


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    freeze()
