import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import copilot_engine as copilot


class CopilotCacheTests(unittest.TestCase):
    def setUp(self):
        copilot.clear_data_caches()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(copilot.clear_data_caches)
        self.path = Path(self.temp.name) / "features.csv"
        self.frame = pd.DataFrame([
            {"ticker": "BBB", "date": "2024-02-01", "signal": 1, "Close": 12, "rsi": 40, "unused": 7},
            {"ticker": "AAA", "date": "2024-01-03", "signal": 0, "Close": 10, "rsi": 70, "unused": 4},
            {"ticker": "AAA", "date": "2024-01-02", "signal": 1, "Close": 9, "rsi": 50, "unused": 3},
        ])
        self.frame.to_csv(self.path, index=False)

    def test_concurrent_latest_requests_read_once_and_keep_full_latest_rows(self):
        with patch.dict(copilot.FEATURE_FILES, {"finance": self.path}), \
             patch.object(copilot.pd, "read_csv", wraps=pd.read_csv) as read:
            with ThreadPoolExecutor(max_workers=8) as pool:
                rows = list(pool.map(lambda _: copilot.get_latest_features("AAA", False), range(16)))
            self.assertEqual(read.call_count, 1)
            for row in rows:
                pd.testing.assert_series_equal(row, self.frame.iloc[1], check_names=False)
            self.assertEqual(len(copilot._LATEST_FEATURE_CACHE[str(self.path)]), 2)
            rows[0]["Close"] = 999
            self.assertEqual(copilot.get_latest_features("AAA", False)["Close"], 10)
            self.assertIsNone(copilot.get_latest_features("MISSING", False))

    def test_label_and_history_projections_preserve_required_data(self):
        with patch.dict(copilot.FEATURE_FILES, {"finance": self.path}):
            labels = copilot._read_csv_cached(self.path)
            self.assertEqual(set(labels.columns), {"ticker", "date", "signal"})
            self.assertEqual(labels.signal.tolist(), [1, 0, 1])
            history = copilot._history_features_cached(self.path)
            self.assertEqual(set(history.columns), {"ticker", "date", "Close", "rsi"})
            self.assertEqual(history.Close.tolist(), [12, 10, 9])

    def test_seed_keeps_original_dates_without_loading_another_csv(self):
        original = self.frame.copy()
        original.loc[1, "date"] = "2024-01-03T00:00:00.000000"
        seeded = original.copy()
        seeded.attrs["source_path"] = str(self.path)
        seeded.attrs["latest_raw_dates"] = {"AAA": original.iloc[1].date, "BBB": original.iloc[0].date}
        seeded["date"] = pd.to_datetime(seeded.date, format="mixed")
        with patch.dict(copilot.FEATURE_FILES, {"finance": self.path}), \
             patch.object(copilot.pd, "read_csv", side_effect=AssertionError("unexpected disk read")):
            copilot.seed_latest_finance_features(seeded)
            pd.testing.assert_series_equal(copilot.get_latest_features("AAA", False), original.iloc[1], check_names=False)
        self.assertEqual(str(seeded.iloc[1].date), "2024-01-03 00:00:00")

    def test_restart_clears_caches_and_reload_observes_new_data(self):
        copilot._latest_features_cached(self.path)
        self.frame.loc[1, "Close"] = 15
        self.frame.to_csv(self.path, index=False)
        copilot.clear_data_caches()
        self.assertEqual(copilot._latest_features_cached(self.path).loc["AAA", "Close"], 15)
