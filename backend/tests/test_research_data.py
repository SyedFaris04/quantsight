import io
import json
import unittest
import numpy as np
import pandas as pd

from research.news_dataset import normalize_article, assign_sessions, stream_json_array
from research.news_pilot import chronological_segments, paired_brier


class ResearchDataTests(unittest.TestCase):
    def article(self, **updates):
        row = {"language": "en", "title": "Company raises outlook", "url": "https://example.org/a",
               "date_publish": "2024-07-03 16:00:00", "date_download": "2024-07-03T16:10:00Z",
               "date_modify": None, "mentioned_companies": ["AAPL"],
               "related_companies": ["MSFT"], "next_day_price_AAPL": 99999,
               "sentiment": {"positive": 1}}
        return dict(row, **updates)

    def test_no_future_price_or_supplied_sentiment_enters(self):
        rows, reason = normalize_article(self.article(), {"AAPL", "MSFT"})
        self.assertIsNone(reason)
        self.assertEqual([r["ticker"] for r in rows], ["AAPL"])
        self.assertFalse(any("price" in key or "sentiment" in key for key in rows[0]))

    def test_article_not_assigned_before_archive_or_modification(self):
        rows, _ = normalize_article(self.article(date_modify="2024-07-03T18:00:00Z"), {"AAPL"})
        result = assign_sessions(pd.DataFrame(rows))
        self.assertEqual(result.date.iloc[0], "2024-07-05")
        self.assertEqual(rows[0]["available_at"], "2024-07-03T18:00:00+00:00")

    def test_ambiguous_download_and_impossible_order_rejected(self):
        for value, reason in [("2024-07-03 16:10", "invalid_or_ambiguous_timestamp"),
                              ("2024-07-03T15:59Z", "archive_precedes_publication"),
                              (None, "missing_publication_or_archive_time")]:
            rows, actual = normalize_article(self.article(date_download=value), {"AAPL"})
            self.assertEqual(actual, reason)
            self.assertEqual(rows, [])

    def test_date_only_publication_is_not_assumed_known_at_midnight(self):
        rows, _ = normalize_article(self.article(date_publish="2024-07-03 00:00:00"), {"AAPL"})
        self.assertEqual(assign_sessions(pd.DataFrame(rows)).date.iloc[0], "2024-07-05")

    def test_parser_handles_chunks_and_embedded_delimiters(self):
        rows = [{"title": 'Text with ], { and " quotes'}, {"unicode": "\u03b1", "nested": [1, 2]}]
        self.assertEqual(list(stream_json_array(io.StringIO(json.dumps(rows)), chunk_size=3)), rows)
        self.assertEqual(list(stream_json_array(io.StringIO("[]"), chunk_size=1)), [])

    def test_parser_rejects_truncation_and_trailing_content(self):
        for raw in ['[{"a": 2}', '[{},]', '[{}]bad', '{}', '[2]']:
            with self.assertRaises(ValueError):
                list(stream_json_array(io.StringIO(raw), chunk_size=2))

    def test_purge_uses_each_tickers_fifth_outcome_session(self):
        dates = pd.bdate_range("2021-12-20", "2023-01-20").strftime("%Y-%m-%d")
        frame = pd.DataFrame([(ticker, date) for ticker in ["A", "B"] for date in dates], columns=["ticker", "date"])
        protocol = {"horizon_sessions": 5, "fit_start": "2021-01-01", "fit_end_exclusive": "2022-01-01",
                    "calibration_end_exclusive": "2023-01-01", "evaluation_end_exclusive": "2024-01-01"}
        parts = chronological_segments(frame, protocol)
        self.assertTrue(parts["fit"].label_end.lt("2022-01-01").all())
        self.assertTrue(parts["calibration"].label_end.lt("2023-01-01").all())
        self.assertFalse(parts["fit"].date.eq("2021-12-31").any())
        self.assertEqual(parts["evaluation"].groupby("ticker").size().nunique(), 1)

    def test_paired_interval_identity_and_direction(self):
        frame = pd.DataFrame({"date": np.repeat(["2023-01-03", "2023-01-04"], 2), "signal": [1, 0, 0, 1]})
        prior = np.full(4, .5)
        identical = paired_brier(frame, prior, prior, 1, 30, 42)
        self.assertEqual(identical["interval_95"], [0., 0.])
        better = paired_brier(frame, prior, np.array([.9, .1, .1, .9]), 1, 30, 42)
        self.assertAlmostEqual(better["brier_improvement"], .24)


if __name__ == "__main__":
    unittest.main()
