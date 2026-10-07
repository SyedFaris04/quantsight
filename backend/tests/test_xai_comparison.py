import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from research import routes, xai_report
from research.xai_metrics import day_occlusion, removal_response, top_overlap


class XaiEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(xai_report.REPORT.read_text(encoding="utf-8"))

    def test_real_prepared_report_and_source_files_are_verified(self):
        self.assertEqual(len(xai_report.load()["rows"]), 88)

    def test_modified_summary_prediction_and_attributions_are_rejected(self):
        for change in [
            lambda r: r["summaries"][0].update(removal_response=999),
            lambda r: r["rows"][0].update(calibrated_probability_up=.99),
            lambda r: r["rows"][0]["methods"][0]["values"].__setitem__(0, float("nan")),
            lambda r: r["rows"][0]["methods"][0]["values"].pop(),
            lambda r: r["rows"][0].update(direction="HOLD"),
            lambda r: r["rows"][-1].update(window_end=r["prediction_date"]),
            lambda r: r.update(holdout_files_opened=True),
            lambda r: r["sources"][0].update(path="../../private"),
        ]:
            with self.subTest(change=change):
                damaged = copy.deepcopy(self.report)
                change(damaged)
                with self.assertRaises(ValueError): xai_report.validate(damaged)

    def test_stale_sources_fail_closed_and_export_is_unavailable(self):
        app = FastAPI()
        app.include_router(routes.router)
        with TestClient(app) as client, patch.object(xai_report, "_cached_digest", return_value="0" * 64):
            self.assertEqual(client.get("/research/xai-comparison").status_code, 503)
            self.assertEqual(client.get("/research/xai-comparison/export").status_code, 503)

    def test_missing_report_is_unknown_not_an_empty_success(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(xai_report, "REPORT", Path(directory) / "missing"):
            self.assertIsNone(xai_report.load())

    def test_text_checkout_line_endings_are_equivalent_but_values_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_bytes(b"a,b\r\n1,2\r\n")
            original = xai_report.source_digest(path)
            path.write_bytes(b"a,b\n1,2\n")
            self.assertEqual(original, xai_report.source_digest(path))
            path.write_bytes(b"a,b\n1,3\n")
            self.assertNotEqual(original, xai_report.source_digest(path))

    def test_linear_sequence_occlusion_matches_analytical_contributions(self):
        weights = np.array([[1., 2.], [3., 4.], [5., 6.]])
        values = np.array([[2., 1.], [1., 3.], [2., -1.]])
        reference = np.zeros_like(values)
        predict = lambda batch: (batch * weights).sum(axis=(1, 2))
        expected = (values * weights).sum(axis=1)
        np.testing.assert_allclose(day_occlusion(predict, values, reference), expected)
        self.assertAlmostEqual(removal_response(predict, values, reference, expected, temporal=True),
                               np.mean(np.cumsum(np.abs(expected[np.argsort(-np.abs(expected))]))))
        self.assertEqual(top_overlap([1, 2, 3, 0], [-1, -2, -3, 0]), 1)


if __name__ == "__main__":
    unittest.main()
