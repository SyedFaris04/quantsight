import copy
import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_health import build_health, finite_number, probability_summary, timestamp_problem
from model_health_reference import validate, PROTOCOL
import model_health_reference as reference_module
import model_health_routes as routes
from test_live_evaluation import Database


def reference(n=20):
    return {"schema_version": 1, "protocol_id": PROTOCOL["protocol_id"], "model_sha256": "a" * 64,
            "source_sha256": "b" * 64, "protocol_sha256": "c" * 64,
            "features": ["rsi"], "start": "2015-05-01", "end": "2022-12-30", "rows": 300 * n,
            "tickers": {f"T{i}": {"rows": 300, "start": "2015-05-01", "end": "2022-12-30",
                                   "features": {"rsi": {"n": 300, "quantiles": [0, 25, 50, 75, 100]}}} for i in range(n)}}


def records(n=20):
    return [{"id": str(i), "ticker": f"T{i}", "predicted_date": "2024-07-03",
             "protocol_version": routes.prediction_tracker.PROTOCOL,
             "model_version": "a" * 64, "feature_version": "features-v1", "calibration_method": "isotonic",
             "feature_values": {"rsi": 50.0}, "raw_probability_up": float(p), "probability_up": .55,
             "predicted_signal": "BUY" if p >= .5 else "SELL",
             "data_cutoff_at": "2024-07-03T17:00Z", "data_fetched_at": "2024-07-03T21:00Z",
             "generated_at": "2024-07-03T21:01Z", "created_at": "2024-07-03T21:02Z",
             "record_before": "2024-07-05T13:30Z"} for i, p in enumerate(np.linspace(.4, .6, n))]


NOW = "2024-07-05T22:00Z"


class MonitoringTests(unittest.TestCase):
    def test_calibration_compression_and_raw_directions(self):
        report = build_health(records(), reference(), NOW)
        group = report["groups"][0]
        self.assertEqual(group["probability_status"], "calibration_compression")
        self.assertEqual(group["calibrated_half_disagreements"], 10)
        self.assertEqual(group["quality"]["direction_mismatch_rows"], 0)
        self.assertEqual(group["raw_probability"]["n"], 20)
        self.assertEqual(group["calibrated_probability"]["distinct_values"], 1)
        self.assertEqual(group["calibrated_probability"]["largest_tie_count"], 20)
        self.assertEqual(group["range_review_count"], 0)
        self.assertEqual(group["buy_count"], 10)
        self.assertEqual(report["latest_session_coverage"], 0)
        self.assertTrue(report["recording_window_open"])

    def test_range_threshold_and_boundaries_are_fixed(self):
        rows = records()
        for i in range(4): rows[i]["feature_values"]["rsi"] = 101
        group = build_health(rows, reference(), NOW)["groups"][0]
        stats = group["input_ranges"][0]
        self.assertEqual(stats["outside"], 4)
        self.assertEqual(stats["outside_pct"], 20)
        self.assertTrue(stats["review"])
        rows[0]["feature_values"]["rsi"] = 100
        rows[1]["feature_values"]["rsi"] = 0
        stats = build_health(rows, reference(), NOW)["groups"][0]["input_ranges"][0]
        self.assertEqual(stats["outside"], 2)
        self.assertFalse(stats["review"])

    def test_each_ticker_uses_its_own_reference(self):
        baseline = reference()
        rows = records()
        baseline["tickers"]["T0"]["features"]["rsi"]["quantiles"] = [100, 125, 150, 175, 200]
        rows[0]["feature_values"]["rsi"] = 150
        self.assertEqual(build_health(rows, baseline, NOW)["groups"][0]["range_review_count"], 0)

    def test_missing_and_nonfinite_inputs_are_not_zero_filled(self):
        for value in (None, np.inf, np.nan, True, "50"):
            rows = records()
            rows[0]["feature_values"]["rsi"] = value
            group = build_health(rows, reference(), NOW)["groups"][0]
            self.assertEqual(group["quality"]["input_problem_rows"], 1)
            self.assertEqual(group["quality"]["missing_or_nonfinite_feature_cells"], 1)
            self.assertEqual(group["input_ranges"][0]["n"], 19)
            self.assertIsNone(group["input_ranges"][0]["review"])

    def test_invalid_probability_and_wrong_raw_direction_are_excluded(self):
        rows = records()
        rows[0]["probability_up"] = np.inf
        rows[1]["predicted_signal"] = "BUY"
        group = build_health(rows, reference(), NOW)["groups"][0]
        self.assertEqual(group["quality"]["invalid_probability_rows"], 1)
        self.assertEqual(group["quality"]["direction_mismatch_rows"], 1)
        self.assertEqual(group["raw_probability"]["n"], 18)
        self.assertEqual(group["probability_status"], "insufficient_sample")

    def test_bad_timestamps_and_late_records_are_flagged(self):
        for field, value in (("data_fetched_at", None), ("created_at", "2024-07-05T13:30Z"),
                             ("generated_at", "2024-07-03T21:03Z"), ("data_cutoff_at", "2024-07-03T16:00Z"),
                             ("record_before", "2024-07-08T13:30Z"), ("created_at", "2024-07-03T17:19Z"),
                             ("created_at", "2024-07-03 21:02")):
            row = dict(records()[0], **{field: value})
            self.assertTrue(timestamp_problem(row, pd.Timestamp(NOW)), (field, value))
            self.assertEqual(build_health([row], reference(), NOW)["groups"][0]["raw_probability"]["n"], 0)

    def test_unknown_models_withhold_input_reference(self):
        rows = records()
        for row in rows: row["model_version"] = "different-model"
        group = build_health(rows, reference(), NOW)["groups"][0]
        self.assertFalse(group["reference_supported"])
        self.assertEqual(group["input_ranges"], [])
        self.assertEqual(group["raw_probability"]["n"], 20)

    def test_versions_are_never_pooled_to_pass_minimum_sample(self):
        rows = records()
        for row in rows[:10]: row["feature_version"] = "features-v2"
        report = build_health(rows, reference(), NOW)
        self.assertEqual(len(report["groups"]), 2)
        self.assertTrue(all(g["probability_status"] == "insufficient_sample" for g in report["groups"]))
        self.assertTrue(all(not g["complete_cohort"] for g in report["groups"]))

    def test_empty_unknown_ticker_and_duplicate_keys(self):
        self.assertEqual(build_health([], reference(), NOW)["groups"], [])
        rows = records()
        rows[0]["ticker"] = "UNKNOWN"
        group = build_health(rows, reference(), NOW)["groups"][0]
        self.assertEqual(group["quality"]["unknown_ticker_rows"], 1)
        self.assertFalse(group["complete_cohort"])
        with self.assertRaises(ValueError): build_health(records() + [records()[0]], reference(), NOW)
        with self.assertRaises(ValueError): build_health([dict(records()[0], predicted_date="2024-07-04")], reference(), NOW)

    def test_probability_bins_include_zero_and_one_and_exact_flat_threshold(self):
        summary = probability_summary([0, 1] * 10)
        self.assertEqual(summary["bins"][0]["count"], 10)
        self.assertEqual(summary["bins"][-1]["count"], 10)
        self.assertFalse(summary["flat"])
        self.assertTrue(probability_summary([.5, .505] * 10)["flat"])
        self.assertFalse(finite_number(True))
        self.assertIsNone(probability_summary([.5])["flat"])

    def test_reference_validation_rejects_invalid_quantiles(self):
        validate(reference())
        for values in ([0, 25, float("nan"), 75, 100], [0, 75, 50, 25, 100], [0, 100]):
            ref = reference()
            ref["tickers"]["T0"]["features"]["rsi"]["quantiles"] = values
            with self.assertRaises(ValueError): validate(ref)


class MonitoringApiTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(routes.router)
        self.client = TestClient(app)

    def test_aggregate_projection_pagination_and_no_writes(self):
        rows = records()
        db = Database(rows)
        snapshots = copy.deepcopy(rows)
        with patch.object(routes, "load_reference", return_value=reference()), patch.object(routes, "utc_now", return_value=pd.Timestamp(NOW)), patch.object(routes.prediction_tracker, "get_admin_client", return_value=db):
            response = self.client.get("/model-health?days=30")
        self.assertEqual(response.status_code, 200)
        report = response.json()
        self.assertTrue(report["available"])
        self.assertEqual(report["records"], 20)
        self.assertEqual(rows, snapshots)
        self.assertNotIn("feature_values", json.dumps(report))
        self.assertNotIn("actual_signal", routes.FIELDS)
        self.assertIn("feature_values", routes.FIELDS)

    def test_more_than_one_page_is_reported(self):
        rows = records(1100)
        with patch.object(routes, "load_reference", return_value=reference(1100)), patch.object(routes, "utc_now", return_value=pd.Timestamp(NOW)), patch.object(routes.prediction_tracker, "get_admin_client", return_value=Database(rows)):
            report = self.client.get("/model-health?days=30").json()
        self.assertTrue(report["available"])
        self.assertEqual(report["records"], 1100)

    def test_query_bounds_empty_and_dependency_failures(self):
        self.assertEqual(self.client.get("/model-health?days=6").status_code, 422)
        self.assertEqual(self.client.get("/model-health?days=91").status_code, 422)
        with patch.object(routes, "load_reference", side_effect=ValueError("secret-text")):
            response = self.client.get("/model-health")
            self.assertFalse(response.json()["available"])
            self.assertNotIn("secret-text", response.text)

        with patch.object(routes, "load_reference", return_value=reference()), patch.object(routes.prediction_tracker, "get_admin_client", return_value=None):
            self.assertFalse(self.client.get("/model-health").json()["available"])
        db = Mock()
        db.table.side_effect = RuntimeError("secret-text")
        with patch.object(routes, "load_reference", return_value=reference()), patch.object(routes.prediction_tracker, "get_admin_client", return_value=db):
            response = self.client.get("/model-health")
            self.assertFalse(response.json()["available"])
            self.assertNotIn("secret-text", response.text)

    def test_bad_keys_return_an_unavailable_report(self):
        with patch.object(routes, "load_reference", return_value=reference()), patch.object(routes, "utc_now", return_value=pd.Timestamp(NOW)), patch.object(routes.prediction_tracker, "get_admin_client", return_value=Database(records() + [records()[0]])):
            result = self.client.get("/model-health?days=30").json()
        self.assertFalse(result["available"])


class ReferenceArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.paths = {name: root / name for name in ("REFERENCE_PATH", "SOURCE_PATH", "MODEL_PATH", "PROTOCOL_PATH")}
        self.paths["SOURCE_PATH"].write_bytes(b"date,rsi\n2015-05-01,50\n")
        self.paths["MODEL_PATH"].write_bytes(b"trusted-model-fixture")
        self.paths["PROTOCOL_PATH"].write_text(json.dumps(PROTOCOL), encoding="utf-8")
        data = reference()
        data["source_sha256"] = reference_module.digest(self.paths["SOURCE_PATH"])
        data["model_sha256"] = reference_module.digest(self.paths["MODEL_PATH"], binary=True)
        data["protocol_sha256"] = reference_module.digest(self.paths["PROTOCOL_PATH"])
        self.paths["REFERENCE_PATH"].write_text(json.dumps(data), encoding="utf-8")
        self.patch = patch.multiple(reference_module, **self.paths)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        reference_module._load.cache_clear()
        self.addCleanup(reference_module._load.cache_clear)

    def test_source_line_endings_are_portable(self):
        first = reference_module.load_reference()
        self.paths["SOURCE_PATH"].write_bytes(b"date,rsi\r\n2015-05-01,50\r\n")
        self.assertEqual(reference_module.load_reference(), first)

    def test_changed_source_model_or_rules_are_rejected(self):
        for name in ("SOURCE_PATH", "MODEL_PATH", "PROTOCOL_PATH"):
            path = self.paths[name]
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.assertRaises(ValueError): reference_module.load_reference()
            path.write_bytes(original)
        self.assertEqual(reference_module.load_reference()["rows"], 6000)


if __name__ == "__main__":
    unittest.main()
