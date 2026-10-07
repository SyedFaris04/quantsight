import unittest

import pandas as pd

from research.news_quality import build_report


class NewsQualityTests(unittest.TestCase):
    def fixture(self):
        news = pd.DataFrame([
            {"article_id": "one", "ticker": "AAPL", "headline": "Company   outlook",
             "url": "https://example.org/one", "published_at": "2022-12-30T18:00:00Z",
             "archive_at": "2022-12-30T19:00:00Z", "available_at": "2022-12-30T19:00:00Z",
             "date": "2022-12-30", "date_only_delayed": False},
            {"article_id": "two", "ticker": "AAPL", "headline": "COMPANY outlook",
             "url": "https://example.org/two", "published_at": "2023-01-03T18:00:00Z",
             "archive_at": "2023-01-03T19:00:00Z", "available_at": "2023-01-03T19:00:00Z",
             "date": "2023-01-03", "date_only_delayed": False},
        ])
        finance = pd.DataFrame({"ticker": ["AAPL"] * 4 + ["MSFT"] * 4,
                                "date": ["2022-12-30", "2023-01-03", "2023-01-04", "2023-01-05"] * 2})
        protocol = {"maximum_finance_date": "2024-12-31", "lag_sensitivity_sessions": [0, 1, 2],
                    "segments": {"calibration": ["2022-01-01", "2023-01-01"],
                                 "evaluation": ["2023-01-01", "2024-01-01"]}}
        return news, finance, protocol

    def test_exact_text_overlap_is_not_article_overlap_and_denominators_are_local(self):
        report = build_report(*self.fixture())
        self.assertEqual(report["cross_segment_overlap"][0]["headline_key"], 1)
        self.assertEqual(report["cross_segment_overlap"][0]["article_id"], 0)
        self.assertEqual(report["duplication"]["excess_article_ids_for_identical_headlines"], 1)
        self.assertEqual(report["missing_tickers"], ["MSFT"])
        self.assertEqual([x["coverage_pct"] for x in report["coverage"]["by_year"]], [50, 16.667])
        self.assertEqual([x["news_sessions"] for x in report["lag_coverage"]], [2, 2, 2])
        self.assertNotIn("headline", report)

    def test_syndication_same_session_and_url_versions_are_separate(self):
        news, finance, protocol = self.fixture()
        extra = news.iloc[1].copy()
        extra["article_id"] = "three"
        extra["url"] = "https://example.org/three"
        version = extra.copy()
        version["article_id"] = "four"
        version["headline"] = "Updated company outlook"
        news = pd.concat([news, pd.DataFrame([extra, version])], ignore_index=True)
        report = build_report(news, finance, protocol)
        self.assertEqual(report["duplication"]["urls_with_multiple_headlines"], 1)
        self.assertEqual(report["duplication"]["excess_links_within_same_ticker_session_headline"], 1)
        self.assertEqual(report["coverage"]["news_sessions"], 2)

    def test_multi_ticker_link_does_not_count_as_duplicate_article(self):
        news, finance, protocol = self.fixture()
        linked = news.iloc[0].copy()
        linked["ticker"] = "MSFT"
        report = build_report(pd.concat([news, pd.DataFrame([linked])]), finance, protocol)
        self.assertEqual(report["unique_articles"], 2)
        self.assertEqual(report["ticker_article_links"], 3)

    def test_reject_bad_timing_duplicates_identity_and_candidate_period(self):
        news, finance, protocol = self.fixture()
        for field, value in [("date", "2022-12-29"), ("archive_at", "2022-12-30T17:00:00Z"),
                             ("available_at", "2022-12-30T18:30:00Z")]:
            changed = news.copy()
            changed.loc[0, field] = value
            with self.assertRaises(ValueError):
                build_report(changed, finance, protocol)
        with self.assertRaises(ValueError):
            build_report(pd.concat([news, news.iloc[:1]]), finance, protocol)
        changed = finance.copy()
        changed.loc[0, "date"] = "2025-01-02"
        with self.assertRaises(ValueError):
            build_report(news, changed, protocol)
        changed = news.copy()
        changed.loc[1, "article_id"] = "one"
        changed.loc[1, "ticker"] = "MSFT"
        with self.assertRaises(ValueError):
            build_report(changed, finance, protocol)

    def test_after_early_close_moves_past_holiday(self):
        news, finance, protocol = self.fixture()
        for field in ["published_at", "archive_at", "available_at"]:
            news.loc[0, field] = "2023-07-03T18:00:00Z"
        news.loc[0, "date"] = "2023-07-05"
        self.assertEqual(build_report(news, finance, protocol)["integrity"], "passed")
        news.loc[0, "date"] = "2023-07-03"
        with self.assertRaises(ValueError):
            build_report(news, finance, protocol)


if __name__ == "__main__":
    unittest.main()
