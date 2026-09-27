"""Verify development folds from a frozen snapshot; never read holdout CSVs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd
from market_calendar import schedule
from research.freeze_prices import ROOT, digest, load_development, write_json
from research.validation import calendar_labels, purged_segments


def prepare(directory):
    directory = Path(directory)
    request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
    protocol = request['protocol']
    boundary = protocol['development_end_exclusive']
    sessions = schedule(protocol['download_start'], boundary).index
    sessions = sessions[sessions < pd.Timestamp(boundary)]
    frames = []
    for ticker in protocol['universe']:
        frame = load_development(directory, ticker)
        frame['ticker'] = ticker
        # Use a consistent adjusted close for this split demonstration. These
        # labels are not fed to a model here, and no outcome proportions are reported.
        frame['Close'] = frame['Adj Close']
        frames.append(frame[['ticker', 'date', 'Close']])
    labeled = calendar_labels(pd.concat(frames, ignore_index=True), sessions, protocol['horizon_sessions'])
    folds = {}
    for fold in protocol['development_folds']:
        parts = purged_segments(labeled, fold, protocol['holdout_start'])
        folds[fold['name']] = {name: {
            'rows': len(part), 'tickers': part.ticker.nunique(),
            'start': part.date.min().date().isoformat(), 'end': part.date.max().date().isoformat(),
            'last_label_end': part.label_end.max().date().isoformat(),
        } for name, part in parts.items()}
    result = {'snapshot_id': request['attempt_id'], 'created_at': datetime.now(timezone.utc).isoformat(),
              'status': 'development_splits_prepared', 'protocol_sha256': request['protocol_sha256'],
              'snapshot_manifest_sha256': digest(directory / 'manifest.json'),
              'preparation_code_sha256': digest(Path(__file__)),
              'split_code_sha256': digest(Path(__file__).with_name('validation.py')),
              'development_rows': len(labeled), 'unknown_tail_labels': int(labeled.signal.isna().sum()),
              'label_price_basis': 'Adj Close from the frozen vendor snapshot',
              'folds': folds, 'candidate_holdout_files_opened': False, 'model_fits': 0,
              'holdout_scores_computed': False, 'prior_2025_use': protocol['prior_2025_use']}
    write_json(directory / 'development_splits.json', result)
    write_json(ROOT.parent / 'docs/research/fresh_development_splits.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    prepare(parser.parse_args().snapshot)
