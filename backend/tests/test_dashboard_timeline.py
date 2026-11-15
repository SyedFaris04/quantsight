import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import pandas as pd

import main
from research import routes

MODELS = ['xgb_finance', 'xgb_sentiment', 'lstm_finance', 'lstm_sentiment']


def latest(direction):
    """Latest-row tables in the startup-cache shape: one row per ticker per model."""
    frame = pd.DataFrame({'ticker': ['AAA', 'BBB'], 'date': pd.to_datetime(['2024-12-20'] * 2),
                          'predicted_signal': [direction, direction], 'confidence': [60.0, 55.0],
                          'signal_label': ['BUY' if direction else 'SELL'] * 2}).set_index('ticker', drop=False)
    return {model: frame for model in MODELS}


class DashboardTimelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.report = Path(self.temp.name) / 'timeline_comparison.json'
        backtest = lambda value: {'evaluation': {'start': '2023-02-07', 'end': '2024-12-20', 'sessions': 470},
                                  'ensemble': {'total_return': value}, 'spy': {'total_return': .5}}
        self.report.write_text(json.dumps({'schema_version': 1, 'timelines': {'2015': {'backtest': backtest(.1)},
                                                                              '2005': {'backtest': backtest(.2)}}}))
        cache = {'tickers': ['AAA', 'BBB'], 'latest_prediction_rows': latest(0), 'news': pd.DataFrame(),
                 'timeline_latest': {'2015': latest(1), '2005': latest(0)}}
        for target, name, value in [(main, '_cache', cache), (main, 'TIMELINE_REPORT', self.report),
                                    (routes, 'TIMELINE_REPORT', self.report)]:
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.cache = cache
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def test_default_equals_2015(self):
        default = self.client.get('/dashboard').json()
        explicit = self.client.get('/dashboard?timeline=2015').json()
        self.assertEqual(default, explicit)
        self.assertEqual(default['timeline'], '2015')
        self.assertEqual(default['kpis']['buy_signals'], 2)
        self.assertEqual(default['backtest']['ensemble']['total_return'], .1)

    def test_2005_reads_2005_rows(self):
        result = self.client.get('/dashboard?timeline=2005').json()
        self.assertEqual(result['timeline'], '2005')
        self.assertEqual(result['kpis']['sell_signals'], 2)
        self.assertEqual(result['backtest']['ensemble']['total_return'], .2)

    def test_unknown_timeline_422(self):
        self.assertEqual(self.client.get('/dashboard?timeline=1999').status_code, 422)

    def test_missing_timeline_503_and_overview_still_ok(self):
        del self.cache['timeline_latest']
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 503)
        self.assertIn('Timeline predictions are not available', response.json()['detail'])
        overview = self.client.get('/overview')
        self.assertEqual(overview.status_code, 200)
        self.assertEqual(overview.json()['data'][0]['overall_signal'], 'SELL')

    def test_missing_report_leaves_backtest_null(self):
        self.report.unlink()
        self.assertIsNone(self.client.get('/dashboard').json()['backtest'])

    def test_timeline_comparison_route_available_and_invalid(self):
        result = self.client.get('/research/timeline-comparison').json()
        self.assertTrue(result['available'])
        self.report.write_text(json.dumps({'schema_version': 1, 'timelines': {'2015': {}}}))
        self.assertEqual(self.client.get('/research/timeline-comparison').status_code, 503)
        self.report.unlink()
        self.assertEqual(self.client.get('/research/timeline-comparison').json(), {'available': False})


if __name__ == '__main__':
    unittest.main()
