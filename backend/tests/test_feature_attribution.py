import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

import feature_attribution as attribution
from attribution_snapshot import artifact_paths, load_attributions, write_attributions
from serving_snapshot import SnapshotUnavailable, sha256_file


class FixedCalibrator:
    def predict(self, probabilities):
        return np.full(len(probabilities), 0.612345)


class FeatureAttributionTests(unittest.TestCase):
    def setUp(self):
        attribution._latest_evidence.cache_clear()
        attribution._saved_attributions.cache_clear()
        self.addCleanup(attribution._latest_evidence.cache_clear)
        self.addCleanup(attribution._saved_attributions.cache_clear)

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(51)
        cls.X = rng.normal(size=(100, 3))
        cls.y = (cls.X[:, 0] - cls.X[:, 2] > 0).astype(int)
        scaler = StandardScaler().fit(cls.X)
        model = xgb.XGBClassifier(n_estimators=15, max_depth=2, n_jobs=1)
        model.fit(scaler.transform(cls.X), cls.y)
        cls.bundle = {"model": model, "scaler": scaler, "features": ["first", "second", "third"],
                      "calibrator": FixedCalibrator(), "calibration_method": "isotonic"}

    def evidence(self, bundle=None):
        bundle = bundle or self.bundle
        index = int(np.flatnonzero(self.y == 0)[0])
        row = dict(zip(bundle["features"], self.X[index]))
        raw = bundle["model"].predict_proba(bundle["scaler"].transform(self.X[[index]]))[0, 1]
        prediction = {"ticker": "TEST", "date": "2024-12-20", "confidence": 61.23,
                      "predicted_signal": int(raw >= .5)}
        return row, prediction

    def test_exact_contributions_reconstruct_real_tree_model_and_keep_calibration_separate(self):
        row, prediction = self.evidence()
        result = attribution.explain_row(self.bundle, row, prediction, "abc")
        total = result["baseline"] + sum(f["contribution"] for f in result["features"])
        self.assertAlmostEqual(total, result["raw_margin"], places=5)
        self.assertLess(result["raw_probability_up"], .5)
        self.assertGreater(result["calibrated_probability_up"], .5)
        self.assertEqual(result["direction"], "SELL")
        self.assertEqual(result["model_sha256"], "abc")
        self.assertEqual(len(result["input_sha256"]), 64)
        self.assertEqual([f["feature"] for f in result["features"]], self.bundle["features"])
        reversed_row = dict(reversed(list(row.items())))
        self.assertEqual(result, attribution.explain_row(self.bundle, reversed_row, prediction, "abc"))

    def test_best_iteration_uses_same_tree_range_as_classifier(self):
        model = xgb.XGBClassifier(n_estimators=15, max_depth=2, n_jobs=1)
        X = self.bundle["scaler"].transform(self.X)
        model.fit(X, self.y)
        # Native Booster otherwise uses all 15 trees, unlike sklearn predict_proba.
        model.get_booster().set_attr(best_iteration="2")
        bundle = {**self.bundle, "model": model}
        result = attribution.explain_row(bundle, *self.evidence(bundle), "abc")
        self.assertEqual(result["trees_used"], 3)
        self.assertLess(result["reconstruction_error"], 1e-5)

    def test_stale_saved_probability_or_direction_is_rejected(self):
        row, prediction = self.evidence()
        for change in [{"confidence": 61.24}, {"confidence": float("nan")},
                       {"predicted_signal": 1 - prediction["predicted_signal"]}]:
            with self.subTest(change=change), self.assertRaises(attribution.AttributionUnavailable):
                attribution.explain_row(self.bundle, row, {**prediction, **change}, "abc")

    def test_missing_and_nonfinite_inputs_are_rejected(self):
        row, prediction = self.evidence()
        for invalid in [{k: v for k, v in row.items() if k != "first"},
                        {**row, "first": float("nan")}, {**row, "first": float("inf")}]:
            with self.assertRaises(attribution.AttributionUnavailable):
                attribution.explain_row(self.bundle, invalid, prediction, "abc")

    def test_corrupt_contributions_are_rejected(self):
        row, prediction = self.evidence()
        booster = self.bundle["model"].get_booster()
        original = booster.predict
        def corrupted(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get("pred_contribs"):
                result[0, 0] += .1
            return result
        with patch.object(booster, "predict", side_effect=corrupted):
            with self.assertRaises(attribution.AttributionUnavailable):
                attribution.explain_row(self.bundle, row, prediction, "abc")

    def test_endpoint_success_and_failure_contract(self):
        app = FastAPI()
        app.include_router(attribution.router)
        client = TestClient(app)
        evidence = attribution.explain_row(self.bundle, *self.evidence(), "abc")
        with patch.object(attribution, "get_attribution", return_value=evidence):
            self.assertEqual(client.get("/feature-attribution/TEST").json(), evidence)
            self.assertEqual(client.get("/feature-attribution/TEST?model=lstm_finance").status_code, 422)
        with patch.object(attribution, "get_attribution", side_effect=KeyError("MISSING")):
            self.assertEqual(client.get("/feature-attribution/MISSING").status_code, 404)
        with patch.object(attribution, "get_attribution", side_effect=ValueError("private file path")):
            response = client.get("/feature-attribution/TEST")
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("private", response.text)

    def test_loading_uses_exact_latest_date_and_invalidates_changed_files(self):
        import pickle
        row, prediction = self.evidence()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ["models", "processed", "predictions"]:
                (root / directory).mkdir()
            (root / "models/xgb_finance.pkl").write_bytes(pickle.dumps(self.bundle))
            feature_file = root / "processed/features_finance.csv"
            pred_file = root / "predictions/xgb_finance_predictions.csv"
            pd.DataFrame([{**row, "ticker": "TEST", "date": "2024-12-19"}]).to_csv(feature_file, index=False)
            pd.DataFrame([prediction]).to_csv(pred_file, index=False)
            with patch.object(attribution, "DATA", root):
                with self.assertRaisesRegex(attribution.AttributionUnavailable, "exact feature row"):
                    attribution.get_attribution("TEST")
                pd.DataFrame([{**row, "ticker": "TEST", "date": prediction["date"]},
                              {**row, "ticker": "OTHER", "date": prediction["date"]}]).to_csv(feature_file, index=False)
                result = attribution.get_attribution("test")
                self.assertEqual(result["ticker"], "TEST")
                self.assertEqual(result["date"], prediction["date"])
                pd.DataFrame([prediction, prediction]).to_csv(pred_file, index=False)
                with self.assertRaisesRegex(attribution.AttributionUnavailable, "Duplicate saved"):
                    attribution.get_attribution("TEST")
            attribution._latest_evidence.cache_clear()

    def make_prepared_fixture(self, root):
        import pickle
        row, prediction = self.evidence()
        for directory in ["models", "processed", "predictions"]:
            (root / directory).mkdir()
        (root / "models/xgb_finance.pkl").write_bytes(pickle.dumps(self.bundle))
        # A newer feature row must not replace the exact historical model input.
        pd.DataFrame([{**row, "ticker": "TEST", "date": prediction["date"]},
                      {**{name: value + 10 for name, value in row.items()},
                       "ticker": "TEST", "date": "2024-12-21"}]).to_csv(
            root / "processed/features_finance.csv", index=False)
        pd.DataFrame([prediction]).to_csv(root / "predictions/xgb_finance_predictions.csv", index=False)
        result = attribution.get_native_attribution("TEST")
        sources = attribution._paths("xgb_finance")
        write_attributions("xgb_finance", sources, [result], root / "serving")
        return result, sources

    def test_prepared_result_preserves_native_evidence_without_model_or_csv_load(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(attribution, "DATA", Path(temporary)):
            expected, _ = self.make_prepared_fixture(Path(temporary))
            with patch.object(attribution.pickle, "loads", side_effect=AssertionError("model load")), \
                 patch.object(attribution.pd, "read_csv", side_effect=AssertionError("CSV parse")):
                result = attribution.get_attribution("test")
                self.assertEqual(result, expected)
                self.assertEqual(result["date"], "2024-12-20")
                result["features"][0]["value"] = 999
                self.assertEqual(attribution.get_attribution("TEST"), expected)
                with self.assertRaises(KeyError):
                    attribution.get_attribution("MISSING")

    def test_changed_prediction_rejects_prepared_result_and_native_fallback_rechecks_it(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(attribution, "DATA", Path(temporary)):
            _, sources = self.make_prepared_fixture(Path(temporary))
            self.assertTrue(attribution.get_attribution("TEST")["verified"])
            predictions = pd.read_csv(sources[2])
            predictions["confidence"] += .2
            predictions.to_csv(sources[2], index=False)
            with self.assertRaises(SnapshotUnavailable):
                load_attributions("xgb_finance", sources, Path(temporary) / "serving")
            with self.assertRaisesRegex(attribution.AttributionUnavailable, "reproduce"):
                attribution.get_attribution("TEST")

    def test_source_hashes_cover_model_and_feature_bytes(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(attribution, "DATA", Path(temporary)):
            _, sources = self.make_prepared_fixture(Path(temporary))
            for source in sources[:2]:
                original = source.read_bytes()
                source.write_bytes(original + b" ")
                with self.assertRaisesRegex(SnapshotUnavailable, "stale"):
                    load_attributions("xgb_finance", sources, Path(temporary) / "serving")
                source.write_bytes(original)

    def test_corrupt_artifact_uses_native_fallback(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(attribution, "DATA", Path(temporary)):
            expected, _ = self.make_prepared_fixture(Path(temporary))
            artifact, _ = artifact_paths("xgb_finance", Path(temporary) / "serving")
            artifact.write_text("corrupted", encoding="utf-8")
            self.assertEqual(attribution.get_attribution("TEST"), expected)

    def test_implementation_change_invalidates_prepared_result(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(attribution, "DATA", Path(temporary)):
            _, sources = self.make_prepared_fixture(Path(temporary))
            with patch("attribution_snapshot.implementation_hash", return_value="new implementation"):
                with self.assertRaisesRegex(SnapshotUnavailable, "implementation"):
                    load_attributions("xgb_finance", sources, Path(temporary) / "serving")

    def test_inconsistent_record_is_rejected_even_if_artifact_hash_is_updated(self):
        import json
        with tempfile.TemporaryDirectory() as temporary, patch.object(attribution, "DATA", Path(temporary)):
            _, sources = self.make_prepared_fixture(Path(temporary))
            artifact, manifest = artifact_paths("xgb_finance", Path(temporary) / "serving")
            rows = json.loads(artifact.read_text())
            rows[0]["verified"] = False
            artifact.write_text(json.dumps(rows), encoding="utf-8")
            metadata = json.loads(manifest.read_text())
            metadata["artifact_sha256"] = sha256_file(artifact)
            manifest.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(SnapshotUnavailable, "inconsistent"):
                load_attributions("xgb_finance", sources, Path(temporary) / "serving")

    def test_recomputation_check_rejects_changed_contributions_with_matching_hashes(self):
        import json
        from research.build_attribution_snapshots import run
        with tempfile.TemporaryDirectory() as temporary, patch.object(attribution, "DATA", Path(temporary)):
            self.make_prepared_fixture(Path(temporary))
            artifact, manifest = artifact_paths("xgb_finance", Path(temporary) / "serving")
            rows = json.loads(artifact.read_text())
            rows[0]["features"][0]["contribution"] += .01
            artifact.write_text(json.dumps(rows), encoding="utf-8")
            metadata = json.loads(manifest.read_text())
            metadata["artifact_sha256"] = sha256_file(artifact)
            manifest.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "native model computation"):
                run("xgb_finance", check=True)


if __name__ == "__main__":
    unittest.main()
