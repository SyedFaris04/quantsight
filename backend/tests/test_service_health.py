import os
import unittest
from unittest.mock import patch

import pandas as pd

from service_health import inspect_data, release_id


class ReadinessTests(unittest.TestCase):
    def cache(self):
        return {"tickers": ["AAPL"], "predictions": {"xgb": pd.DataFrame([
            {"date": pd.Timestamp("2024-12-20"), "ticker": "AAPL",
             "predicted_signal": 1, "actual_signal": 0, "confidence": 60}])},
            "features": pd.DataFrame([{"date": pd.Timestamp("2024-12-20"),
                "ticker": "AAPL", "Open": 10, "High": 11, "Low": 9, "Close": 10,
                "Volume": 100}])}

    def test_optional_news_does_not_block_saved_data(self):
        result = inspect_data(self.cache(), ["xgb"])
        self.assertTrue(result["ready"])
        self.assertFalse(result["datasets"]["news"]["available"])

    def test_missing_model_and_unstarted_cache_fail(self):
        self.assertFalse(inspect_data({}, ["xgb"])["ready"])
        result = inspect_data(self.cache(), ["xgb", "lstm"])
        self.assertEqual(result["problems"], ["predictions:lstm"])

    def test_lstm_export_can_omit_actual_label(self):
        cache = self.cache()
        cache["predictions"]["xgb"] = cache["predictions"]["xgb"].drop(columns=["actual_signal"])
        self.assertTrue(inspect_data(cache, ["xgb"])["ready"])

    def test_corrupt_prediction_contract_fails(self):
        for column, value in (("confidence", float("nan")), ("confidence", 101),
                              ("predicted_signal", 2), ("actual_signal", None),
                              ("ticker", None)):
            with self.subTest(column=column, value=value):
                cache = self.cache()
                cache["predictions"]["xgb"][column] = value
                self.assertFalse(inspect_data(cache, ["xgb"])["ready"])

    def test_missing_price_columns_duplicate_keys_and_missing_ticker_fail(self):
        for kind in ("missing", "duplicate", "ticker", "infinite"):
            cache = self.cache()
            if kind == "missing":
                cache["features"] = cache["features"].drop(columns=["Close"])
            elif kind == "duplicate":
                cache["features"] = pd.concat([cache["features"]] * 2)
            elif kind == "ticker":
                cache["tickers"] = ["MSFT"]
            else:
                cache["features"]["Close"] = float("inf")
            self.assertIn("features:finance", inspect_data(cache, ["xgb"])["problems"])

    def test_release_exposes_only_commit_hash(self):
        for value, expected in (("a" * 40, "a" * 40), ("secret/path", "unknown"), ("", "unknown")):
            with patch.dict(os.environ, {"RENDER_GIT_COMMIT": value}):
                self.assertEqual(release_id(), expected)
