import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from research.freeze_prices import digest
from research.timeline_features import build, snapshot_ohlcv

BACKEND = Path(__file__).resolve().parents[1]
SESSIONS = pd.bdate_range('2010-01-04', periods=600)


def bars(seed, start=0, adj=1.0):
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(rng.normal(0, .01, len(SESSIONS) - start)))
    frame = pd.DataFrame({'date': SESSIONS[start:], 'Open': close * 1.001, 'High': close * 1.01,
                          'Low': close * .99, 'Close': close, 'Adj Close': close * adj,
                          'Volume': rng.integers(1_000, 5_000, len(close)).astype(float)})
    return frame


def fake_snapshot(root, frames, universe, context):
    (root / 'development').mkdir(parents=True)
    files = {}
    for ticker, frame in frames.items():
        path = root / 'development' / f'{ticker}.csv'
        frame.to_csv(path, index=False)
        files[f'development/{ticker}.csv'] = digest(path)
    (root / 'request.json').write_text(json.dumps({'protocol': {
        'universe': universe, 'context_tickers': context, 'development_end_exclusive': '2025-01-01'}}))
    (root / 'manifest.json').write_text(json.dumps({'data_quality_passed': True, 'files': files,
                                                    'tickers': {t: {} for t in frames}}))
    return root


class TimelineFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def snapshot(self, name, include_late=True):
        frames = {'SPY': bars(1), 'AAPL': bars(2), 'XLK': bars(3)}
        if include_late:
            frames['LATE'] = bars(4, start=300)
        universe = [t for t in frames if t != 'XLK']
        return fake_snapshot(self.root / name, frames, universe, ['XLK'])

    def test_adjusts_ohlc_by_adj_close_factor(self):
        frame = bars(5, adj=.5)
        snap = fake_snapshot(self.root / 'adj', {'AAA': frame}, ['AAA'], [])
        result = snapshot_ohlcv(snap, ['AAA'], '2025-01-01').set_index('Date')
        first = frame.iloc[0]
        self.assertAlmostEqual(result.iloc[0].Open, first.Open * .5)
        self.assertAlmostEqual(result.iloc[0].Close, first['Adj Close'])
        self.assertEqual(result.iloc[0].Volume, first.Volume)

    def test_drops_bars_on_or_after_end(self):
        snap = fake_snapshot(self.root / 'end', {'AAA': bars(6)}, ['AAA'], [])
        result = snapshot_ohlcv(snap, ['AAA'], '2011-01-03')
        self.assertLess(result.Date.max(), pd.Timestamp('2011-01-03'))

    def test_drops_nonpositive_volume_rows(self):
        frame = bars(7)
        frame.loc[10, 'Volume'] = 0
        snap = fake_snapshot(self.root / 'vol', {'AAA': frame}, ['AAA'], [])
        self.assertNotIn(SESSIONS[10], set(snapshot_ohlcv(snap, ['AAA'], '2025-01-01').Date))

    def test_late_listing_has_no_warmup_rows_and_does_not_enter_earlier_cross_sections(self):
        with_late = build(self.snapshot('with'), self.root / 'out_with', end_exclusive='2025-01-01')
        without = build(self.snapshot('without', include_late=False), self.root / 'out_without',
                        end_exclusive='2025-01-01')
        late_first = pd.Timestamp(with_late['first_date_by_ticker']['LATE'])
        self.assertGreaterEqual(late_first, SESSIONS[300 + 49])
        a = pd.read_csv(self.root / 'out_with/features_finance.csv', parse_dates=['date'])
        b = pd.read_csv(self.root / 'out_without/features_finance.csv', parse_dates=['date'])
        before = lambda f: f[(f.ticker == 'AAPL') & (f.date < late_first)].reset_index(drop=True)
        pd.testing.assert_series_equal(before(a).rsi_rel_market, before(b).rsi_rel_market)
        self.assertGreater(len(before(a)), 0)
        self.assertEqual(without['tickers'], 2)

    def test_finance_columns_match_production_header(self):
        result = build(self.snapshot('cols'), self.root / 'out_cols', end_exclusive='2025-01-01')
        production = pd.read_csv(BACKEND / 'data/processed/features_finance.csv', nrows=0).columns.tolist()
        self.assertEqual(result['columns'], production)
        saved = pd.read_csv(self.root / 'out_cols/features_finance.csv')
        self.assertEqual(saved.columns.tolist(), production)
        numeric = saved.drop(columns=['date', 'ticker']).to_numpy(float)
        self.assertTrue(np.isfinite(numeric).all())
        self.assertTrue((self.root / 'out_cols/features_sentiment.csv').exists())


if __name__ == '__main__':
    unittest.main()
