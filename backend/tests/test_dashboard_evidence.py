import unittest
import pandas as pd
from dashboard_evidence import news_sentiment, snapshot_metadata


class Scorer:
    def polarity_scores(self, title):
        return {'compound': {'good': .5, 'bad': -.5, 'flat': 0}[title]}


class DashboardEvidenceTests(unittest.TestCase):
    def test_snapshot_reports_mixed_and_unknown_dates_without_calling_them_live(self):
        result = snapshot_metadata([
            {'signals': {'a': {'date': '2024-12-20'}, 'b': {'date': '2024-12-19'}}},
            {'signals': {'a': {'date': None}, 'b': {'date': 'invalid'}}},
        ])
        self.assertEqual(result['kind'], 'historical_predictions')
        self.assertEqual(result['earliest_signal_date'], '2024-12-19')
        self.assertEqual(result['latest_signal_date'], '2024-12-20')
        self.assertTrue(result['mixed_signal_dates'])
        self.assertEqual(result['undated_signals'], 2)
        self.assertEqual(result['model_count'], 2)

    def test_empty_snapshot_dates_are_unknown(self):
        result = snapshot_metadata([])
        self.assertIsNone(result['latest_signal_date'])
        self.assertFalse(result['mixed_signal_dates'])

    def test_archive_age_is_wall_clock_based_even_though_window_ends_at_saved_article(self):
        result = news_sentiment(pd.DataFrame([
            {'date': '2026-01-01', 'title': 'good'},
            {'date': '2026-01-20', 'title': 'bad'},
        ]), Scorer(), days=14, now='2026-02-20T00:00:00Z')
        self.assertEqual(result['source'], 'saved_news_archive')
        self.assertEqual(result['window_start'], '2026-01-06')
        self.assertEqual(result['window_end'], '2026-01-20')
        self.assertEqual(result['age_days'], 31)
        self.assertEqual(result['article_count'], 1)
        self.assertEqual(result['negative_pct'], 100)

    def test_invalid_future_and_blank_rows_cannot_anchor_window(self):
        result = news_sentiment(pd.DataFrame([
            {'date': 'invalid', 'title': 'good'},
            {'date': '2030-01-01', 'title': 'bad'},
            {'date': '2026-02-19', 'title': '   '},
            {'date': '2026-01-20', 'title': 'flat'},
        ]), Scorer(), now='2026-02-20')
        self.assertEqual(result['excluded_rows'], 3)
        self.assertEqual(result['window_end'], '2026-01-20')
        self.assertEqual(result['neutral_pct'], 100)

    def test_duplicate_headlines_count_once_and_use_latest_date(self):
        result = news_sentiment(pd.DataFrame([
            {'date': '2026-01-19', 'title': 'good'},
            {'date': '2026-01-20', 'title': ' good '},
            {'date': '2026-01-20', 'title': 'bad'},
            {'date': '2026-01-20', 'title': 'flat'},
        ]), Scorer(), now='2026-01-21')
        self.assertEqual(result['article_count'], 3)
        self.assertEqual(result['positive_pct'], 33.3)
        self.assertEqual(result['trend'], [{'date': '2026-01-20', 'avg_compound': 0.0}])

    def test_empty_news_is_unknown_not_neutral(self):
        for frame in [pd.DataFrame(), pd.DataFrame([{'date': None, 'title': 'good'}])]:
            result = news_sentiment(frame, Scorer(), now='2026-01-21')
            self.assertEqual(result['article_count'], 0)
            self.assertIsNone(result['neutral_pct'])
            self.assertIsNone(result['window_end'])

    def test_timezone_normalization_and_inclusive_window_boundary(self):
        result = news_sentiment(pd.DataFrame([
            {'date': '2026-01-20T08:00:00+08:00', 'title': 'good'},
            {'date': '2026-01-06T00:00:00Z', 'title': 'flat'},
            {'date': '2026-01-05T23:59:59Z', 'title': 'bad'},
        ]), Scorer(), now='2026-01-20T00:00:00Z')
        self.assertEqual(result['article_count'], 2)
        self.assertEqual(result['age_days'], 0)
        self.assertEqual(result['latest_article_at'], '2026-01-20T00:00:00+00:00')


if __name__ == '__main__':
    unittest.main()
