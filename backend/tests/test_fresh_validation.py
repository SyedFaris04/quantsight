import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from market_calendar import schedule
from research.freeze_prices import audit_prices, digest, load_development
from research.validation import calendar_labels, purged_segments


class FreshValidationTests(unittest.TestCase):
    def setUp(self):
        self.sessions = schedule('2024-11-01', '2025-01-31').index
        self.frame = pd.DataFrame({'ticker': 'AAA', 'date': self.sessions,
                                   'Close': range(100, 100 + len(self.sessions))})

    def test_fifth_session_uses_holiday_calendar_and_unknown_tail(self):
        result = calendar_labels(self.frame, self.sessions).set_index('date')
        self.assertEqual(result.loc['2024-12-20', 'label_end'], pd.Timestamp('2024-12-30'))
        self.assertEqual(result.loc['2024-12-20', 'signal'], 1)
        self.assertTrue(result.tail(5).signal.isna().all())

    def test_missing_middle_bar_cannot_shift_target_to_sixth_session(self):
        frame = self.frame[self.frame.date.ne('2024-12-24')]
        result = calendar_labels(frame, self.sessions).set_index('date')
        self.assertTrue(pd.isna(result.loc['2024-12-20', 'signal']))
        self.assertTrue(pd.isna(result.loc['2024-12-20', 'label_end']))
        self.assertEqual(result.loc['2024-12-26', 'signal'], 1)

    def test_ties_are_down_and_tickers_remain_independent(self):
        other = self.frame.assign(ticker='BBB', Close=10)
        result = calendar_labels(pd.concat([self.frame, other]), self.sessions)
        self.assertTrue(result[result.ticker.eq('BBB')].signal.dropna().eq(0).all())
        self.assertTrue(result[result.ticker.eq('AAA')].signal.dropna().eq(1).all())

    def test_duplicate_and_non_session_dates_fail(self):
        with self.assertRaises(ValueError):
            calendar_labels(pd.concat([self.frame, self.frame.head(1)]), self.sessions)
        bad = self.frame.copy()
        bad.loc[0, 'date'] = pd.Timestamp('2024-12-25')
        with self.assertRaises(ValueError):
            calendar_labels(bad, self.sessions)

    def test_purge_all_three_segments_and_keep_whole_dates_together(self):
        frame = pd.concat([self.frame, self.frame.assign(ticker='BBB')])
        labeled = calendar_labels(frame, self.sessions)
        fold = {'fit_start': '2024-11-01', 'fit_end_exclusive': '2024-12-01',
                'calibration_end_exclusive': '2025-01-01', 'evaluation_end_exclusive': '2025-02-01'}
        parts = purged_segments(labeled, fold, '2025-02-01')
        for (name, part), end in zip(parts.items(), ['2024-12-01', '2025-01-01', '2025-02-01']):
            self.assertGreater(len(part), 0, name)
            self.assertTrue(part.label_end.lt(end).all())
            self.assertTrue(part.groupby('date').size().eq(2).all())
        self.assertTrue(set(parts['fit'].date).isdisjoint(parts['calibration'].date))

    def test_holdout_rows_and_outcomes_are_rejected(self):
        labeled = calendar_labels(self.frame, self.sessions)
        fold = {'fit_start': '2024-11-01', 'fit_end_exclusive': '2024-11-15',
                'calibration_end_exclusive': '2024-12-01', 'evaluation_end_exclusive': '2025-01-01'}
        with self.assertRaisesRegex(ValueError, 'holdout'):
            purged_segments(labeled, fold, '2025-01-01')
        with self.assertRaisesRegex(ValueError, 'holdout'):
            purged_segments(labeled[labeled.date.lt('2025-01-01')], fold, '2025-01-01')

    def price_frame(self, dates):
        return pd.DataFrame({'Open': 10., 'High': 12., 'Low': 9., 'Close': 11.,
                             'Adj Close': 10.5, 'Volume': 100.}, index=dates)

    def test_quality_accepts_prelisting_gap_but_not_internal_or_holdout_gaps(self):
        dates = self.sessions
        frame = self.price_frame(dates[2:])
        good = audit_prices(frame, dates, '2025-01-01')
        self.assertTrue(good['valid'])
        self.assertEqual(good['leading_unavailable_sessions'], 2)
        bad = audit_prices(frame.drop(pd.Timestamp('2025-01-06')), dates, '2025-01-01')
        self.assertFalse(bad['valid'])
        self.assertEqual(bad['missing_holdout_sessions'], ['2025-01-06'])

    def test_quality_rejects_invalid_ranges_and_nonfinite_prices(self):
        frame = self.price_frame(self.sessions)
        frame.loc[self.sessions[0], 'High'] = 8
        frame.loc[self.sessions[1], 'Adj Close'] = float('nan')
        result = audit_prices(frame, self.sessions, '2025-01-01')
        self.assertFalse(result['valid'])
        self.assertEqual(result['invalid_value_rows'], 2)

    def test_development_loader_checks_hash_dates_and_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'development').mkdir()
            path = root / 'development/AAA.csv'
            pd.DataFrame({'date': ['2024-12-20'], 'Close': [10]}).to_csv(path, index=False)
            manifest = {'data_quality_passed': True, 'tickers': {'AAA': {}},
                        'files': {'development/AAA.csv': digest(path)}}
            (root / 'request.json').write_text(json.dumps({'protocol': {'development_end_exclusive': '2025-01-01'}}))
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            self.assertEqual(len(load_development(root, 'AAA')), 1)
            with path.open('a') as handle:
                handle.write('2025-01-02,11\n')
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_development(root, 'AAA')
            manifest['files']['development/AAA.csv'] = digest(path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Holdout'):
                load_development(root, 'AAA')
            manifest['data_quality_passed'] = False
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'quality'):
                load_development(root, 'AAA')


if __name__ == '__main__':
    unittest.main()
