import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from backtesting import routes


class BacktestApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "backtests"
        self.output.mkdir()
        for name, value in [("ROOT", self.root), ("OUTPUT", self.output), ("REPORT", self.output / "report.json")]:
            replacement = patch.object(routes, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        app = FastAPI()
        app.include_router(routes.router)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.source = self.root / "prices.csv"
        self.source.write_text("original", encoding="utf-8")
        self.report = {"schema_version": 1, "run_id": "test-run", "strategies": [{"key": "ensemble"}],
                       "sources": [{"path": "prices.csv", "sha256": hashlib.sha256(b"original").hexdigest()}],
                       "exports": {"metrics": "test-run/metrics.csv"}}

    def publish(self):
        routes.REPORT.write_text(json.dumps(self.report), encoding="utf-8")

    def test_missing_report_is_explicit_empty_state(self):
        response = self.client.get("/backtest")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["available"])

    def test_source_change_and_missing_file_mark_stale(self):
        self.publish()
        self.assertTrue(self.client.get("/backtest").json()["report"]["freshness"]["matches_current_files"])
        self.source.write_text("changed data", encoding="utf-8")
        report = self.client.get("/backtest").json()["report"]
        self.assertEqual(report["freshness"]["changed_sources"], ["prices.csv"])
        self.source.unlink()
        self.assertFalse(self.client.get("/backtest").json()["report"]["freshness"]["matches_current_files"])

    def test_corrupt_report_fails_explicitly(self):
        routes.REPORT.write_text("{", encoding="utf-8")
        self.assertEqual(self.client.get("/backtest").status_code, 503)

    def test_home_summary_cannot_present_stale_results_as_current_evidence(self):
        self.report.update(evaluation={"start": "2024-01-02"}, config={"commission_bps": 10})
        self.report["strategies"] = [
            {"key": "ensemble", "name": "Ensemble", "metrics": {"sharpe": None}, "curve": [1, 2]},
            {"key": "spy", "name": "SPY", "metrics": {"sharpe": 1.2}},
            {"key": "random", "name": "Random", "metrics": {"sharpe": .1}},
        ]
        self.publish()
        result = self.client.get("/backtest/summary").json()
        self.assertTrue(result["available"])
        self.assertEqual([s["key"] for s in result["strategies"]], ["ensemble", "spy"])
        self.assertIsNone(result["strategies"][0]["metrics"]["sharpe"])
        self.assertNotIn("curve", result["strategies"][0])
        self.source.write_text("changed", encoding="utf-8")
        result = self.client.get("/backtest/summary").json()
        self.assertFalse(result["available"])
        self.assertNotIn("strategies", result)

    def test_git_line_endings_do_not_make_identical_data_stale(self):
        raw = b"date,close\r\n2024-01-01,10\r\n"
        self.report["sources"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
        self.publish()
        self.source.write_bytes(raw.replace(b"\r\n", b"\n"))
        self.assertTrue(self.client.get("/backtest").json()["report"]["freshness"]["matches_current_files"])

    def test_home_summary_requires_a_paired_benchmark(self):
        self.publish()
        self.assertEqual(self.client.get("/backtest/summary").status_code, 503)

    def test_export_is_limited_to_report_artifacts(self):
        self.publish()
        folder = self.output / "test-run"
        folder.mkdir()
        (folder / "metrics.csv").write_text("strategy,sharpe\nensemble,0.1\n", encoding="utf-8")
        response = self.client.get("/backtest/export?kind=metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment", response.headers["content-disposition"])
        self.assertIn("ensemble,0.1", response.text)
        self.assertEqual(self.client.get("/backtest/export?strategy=../../secret").status_code, 400)
        self.assertEqual(self.client.get("/backtest/export?kind=other").status_code, 422)
        self.report["exports"]["metrics"] = "../prices.csv"
        self.publish()
        self.assertEqual(self.client.get("/backtest/export?kind=metrics").status_code, 404)
