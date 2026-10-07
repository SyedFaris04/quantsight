import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from fastapi import FastAPI

import prediction_tracker as tracker
from forward_metrics import build_evidence
from forward_routes import router
from market_calendar import schedule


def row(date="2024-07-03", ticker="AAPL", prediction="BUY", actual="BUY", p=.6, resolved=True):
    return {"ticker": ticker, "predicted_date": date, "resolved": resolved,
            "predicted_signal": prediction, "actual_signal": actual if resolved else None,
            "correct": prediction == actual if resolved else None, "probability_up": p,
            "model_version": "model-a", "feature_version": "features-a", "calibration_method": "isotonic",
            "target_close_at": "2024-07-11T20:00:00Z"}


class ForwardEvidenceTests(unittest.TestCase):
    def test_empty_is_unknown_and_json_safe(self):
        result = build_evidence([], "2024-07-12T00:00Z")
        self.assertIsNone(result["metrics"]["accuracy_pct"])
        self.assertEqual(result["resolved_forecast_dates"], 0)
        self.assertFalse(result["intervals"]["available"])
        json.dumps(result, allow_nan=False)

    def test_one_date_many_stocks_cannot_unlock_interval(self):
        rows = [row(ticker=f"T{i}", prediction="BUY" if i < 23 else "SELL",
                    actual="BUY" if i < 26 else "SELL") for i in range(44)]
        result = build_evidence(rows)
        self.assertEqual(result["resolved_forecast_dates"], 1)
        self.assertFalse(result["intervals"]["available"])
        self.assertEqual(result["cohorts"][0]["resolved"], 44)
        self.assertEqual(result["metrics"]["rows"], 44)

    def test_metrics_calibration_and_one_class_boundary(self):
        result = build_evidence([row(ticker="A", prediction="BUY", actual="BUY", p=.8),
                                 row(ticker="B", prediction="BUY", actual="SELL", p=.2)])
        metrics = result["metrics"]
        self.assertEqual(metrics["accuracy_pct"], 50)
        self.assertEqual(metrics["always_up_accuracy_pct"], 50)
        self.assertEqual(metrics["balanced_accuracy_pct"], 50)
        self.assertAlmostEqual(metrics["brier_score"], .04)
        self.assertAlmostEqual(metrics["brier_improvement_over_flat"], .21)
        # Recorded raw direction is intentionally distinct from calibrated P(UP).
        self.assertEqual(result["probability_bins"][1]["rows"], 1)
        self.assertEqual(result["probability_bins"][4]["rows"], 1)
        self.assertIsNone(build_evidence([row()])["metrics"]["balanced_accuracy_pct"])

    def test_probability_endpoints_not_lost_or_overlapped(self):
        rows = [row(ticker=f"T{i}", p=p) for i, p in enumerate([0, .2, .4, .6, .8, 1])]
        bins = build_evidence(rows)["probability_bins"]
        self.assertEqual([b["rows"] for b in bins], [1, 1, 1, 1, 2])

    def test_collection_gap_after_grace_respects_holiday_and_start(self):
        rows = [row()]
        early = build_evidence(rows, "2024-07-05T22:59:59Z")["collection"]
        due = build_evidence(rows, "2024-07-05T23:00:00Z")["collection"]
        self.assertEqual(early["expected_sessions"], 1)
        self.assertEqual(early["missing_sessions"], [])
        self.assertEqual(due["expected_sessions"], 2)
        self.assertEqual(due["missing_sessions"], ["2024-07-05"])
        self.assertEqual(due["recorded_sessions"], 1)
        self.assertEqual(due["through_session"], "2024-07-05")
        self.assertFalse(build_evidence([])["collection"]["available"])

    def test_partial_cohort_and_overdue_at_exact_delayed_close(self):
        rows = [row(ticker="A"), row(ticker="B", resolved=False)]
        early = build_evidence(rows, "2024-07-11T20:19:59Z")
        due = build_evidence(rows, "2024-07-11T20:20:00Z")
        self.assertEqual(early["overdue_pending"], 0)
        self.assertEqual(due["overdue_pending"], 1)
        self.assertEqual(due["partial_forecast_dates"], ["2024-07-03"])
        self.assertEqual(due["cohorts"][0]["status"], "partial")
        self.assertEqual(due["metrics"]["rows"], 1)

    def long_panel(self):
        dates = schedule("2024-01-01", "2024-06-01").index.strftime("%Y-%m-%d")
        # Missing calendar sessions must remain on the block grid.
        return [row(date=date, ticker=ticker, prediction="BUY" if i % 3 else "SELL",
                    actual="BUY" if i % 2 else "SELL", p=.65 if i % 3 else .35)
                for i, date in enumerate(dates) if i % 7 != 0 for ticker in ["A", "B"]]

    def test_date_blocks_reproducible_paired_and_calendar_exact(self):
        rows = self.long_panel()
        first, second = build_evidence(rows), build_evidence(list(reversed(rows)))
        ci = first["intervals"]
        self.assertTrue(ci["available"])
        self.assertEqual(ci, second["intervals"])
        self.assertGreater(ci["session_span"], first["resolved_forecast_dates"])
        # Independent mathematical replay of the fixed block sample.
        sessions = schedule(min(r["predicted_date"] for r in rows), max(r["predicted_date"] for r in rows)).index.strftime("%Y-%m-%d")
        n = len(sessions)
        starts = np.random.Generator(np.random.PCG64(42)).integers(0, n, (1000, int(np.ceil(n / 20))))
        indices = ((starts[:, :, None] + np.arange(20)) % n).reshape(1000, -1)[:, :n]
        by_date = {date: [r for r in rows if r["predicted_date"] == date] for date in sessions}
        count = np.array([len(by_date[d]) for d in sessions])
        gain = np.array([sum(int(r["correct"]) - int(r["actual_signal"] == "BUY") for r in by_date[d]) for d in sessions])
        expected = np.quantile(100 * gain[indices].sum(axis=1) / count[indices].sum(axis=1), [.025, .975])
        np.testing.assert_allclose(ci["bounds"]["accuracy_difference_pp"], expected)

    def test_all_up_predictor_has_identically_zero_paired_gain(self):
        rows = self.long_panel()
        for r in rows:
            r["predicted_signal"] = "BUY"
            r["correct"] = r["actual_signal"] == "BUY"
        result = build_evidence(rows)
        self.assertEqual(result["metrics"]["accuracy_difference_pp"], 0)
        self.assertEqual(result["intervals"]["bounds"]["accuracy_difference_pp"], [0, 0])

    def test_mixed_missing_versions_and_partial_dates_suppress_intervals(self):
        rows = self.long_panel()
        rows[0]["model_version"] = "model-b"
        result = build_evidence(rows)
        self.assertTrue(result["mixed_versions"])
        self.assertEqual(sum(g["metrics"]["rows"] for g in result["model_groups"]), len(rows))
        self.assertFalse(result["intervals"]["available"])
        rows[0]["model_version"] = ""
        self.assertFalse(build_evidence(rows)["intervals"]["available"])
        rows[0]["model_version"] = "model-a"
        rows.append(row(date=rows[0]["predicted_date"], ticker="C", resolved=False))
        self.assertFalse(build_evidence(rows)["intervals"]["available"])

    def test_integrity_fails_instead_of_reporting_false_success(self):
        for updates in [{"probability_up": float("nan")}, {"probability_up": 1.1},
                        {"actual_signal": None}, {"correct": False}, {"predicted_date": "2024-07-04"}]:
            with self.assertRaises(ValueError):
                build_evidence([dict(row(), **updates)])
        with self.assertRaises(ValueError):
            build_evidence([row(), row()])

    def test_window_query_validation_and_forwarding(self):
        app = FastAPI()
        app.include_router(router)
        with TestClient(app) as client, patch.object(tracker, "get_summary", return_value={"available": False}) as summary:
            self.assertEqual(client.get("/live-track-record?days=90").status_code, 200)
            summary.assert_called_once_with(days=90)
            for days in [0, 366, "invalid"]:
                self.assertEqual(client.get(f"/live-track-record?days={days}").status_code, 422)


if __name__ == "__main__":
    unittest.main()
