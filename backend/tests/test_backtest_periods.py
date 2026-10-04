import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from backtesting import periods, routes


class PeriodReturnTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.run = self.output / "saved-run"
        self.run.mkdir()
        self.dates = ["2023-12-28", "2023-12-29", "2024-01-02", "2024-01-03"]
        self.returns = {"ensemble": [.10, -.10, .20, -.05], "spy": [.01, .01, .01, .01],
                        "universe": [0, 0, 0, 0]}
        self.report = {"schema_version": 1, "run_id": "saved-run", "generated_at": "2026-10-04T00:00:00Z",
                       "config": {"initial_capital": 100}, "sources": [],
                       "evaluation": {"sessions": 4, "start": self.dates[0], "end": self.dates[-1]},
                       "strategies": []}
        for key, returns in self.returns.items():
            equity = 100
            curve = [{"date": "2023-12-27", "equity": equity}]
            with (self.run / f"{key}_daily.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["date", "equity", "return"])
                writer.writeheader()
                for day, ret in zip(self.dates, returns):
                    equity *= 1 + ret
                    writer.writerow({"date": day, "equity": equity, "return": ret})
                    curve.append({"date": day, "equity": equity})
            self.report["strategies"].append({"key": key, "curve": curve,
                                              "metrics": {"total_return": equity / 100 - 1}})
        self.publish_report()

    def publish_report(self):
        (self.output / "report.json").write_text(json.dumps(self.report), encoding="utf-8")

    def publish_periods(self):
        result = periods.build(self.output)
        (self.output / periods.FILENAME).write_text(json.dumps(result), encoding="utf-8")
        return result

    def client(self):
        for name, value in [("ROOT", self.output), ("OUTPUT", self.output),
                            ("REPORT", self.output / "report.json")]:
            replacement = patch.object(routes, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        app = FastAPI()
        app.include_router(routes.router)
        client = TestClient(app)
        self.addCleanup(client.close)
        return client

    def test_compounds_partial_periods_and_keeps_cross_year_holdings(self):
        result = self.publish_periods()["strategies"]["ensemble"]
        years, months = result["years"], result["months"]
        self.assertAlmostEqual(years[0]["net_return"], -.01)
        self.assertAlmostEqual(years[1]["net_return"], .14)
        self.assertEqual(years[0]["start"], "2023-12-28")
        self.assertEqual(years[1]["end"], "2024-01-03")
        self.assertEqual(years[1]["sessions"], 2)
        self.assertAlmostEqual(years[1]["difference_vs_spy"], .14 - .0201)
        # The first January return uses December closing equity, with no reset.
        expected = math.prod(1 + r for r in self.returns["ensemble"]) - 1
        self.assertAlmostEqual(math.prod(1 + r["net_return"] for r in months) - 1, expected)
        self.assertEqual(result["summary"]["months_beating_spy"], 1)
        self.assertEqual(result["summary"]["best_month"], "2024-01")
        self.assertEqual(result["summary"]["worst_month"], "2023-12")

    def test_benchmark_self_comparison_is_tied(self):
        result = self.publish_periods()["strategies"]["spy"]
        self.assertEqual(result["summary"]["months_beating_spy"], 0)
        self.assertEqual(result["summary"]["months_tied_spy"], 2)
        self.assertTrue(all(r["difference_vs_spy"] == 0 for r in result["months"]))

    def test_rejects_corrupt_daily_accounting_and_nonfinite_returns(self):
        path = self.run / "ensemble_daily.csv"
        original = path.read_text()
        for value in ["0.2", "nan", "inf", "-1"]:
            path.write_text(original.replace("0.1", value, 1))
            with self.subTest(value=value), self.assertRaises(ValueError):
                periods.build(self.output)
        path.write_text(original)
        self.report["strategies"][0]["metrics"]["total_return"] += .01
        self.publish_report()
        with self.assertRaises(ValueError):
            periods.build(self.output)

    def test_rejects_missing_duplicate_and_mismatched_sessions(self):
        path = self.run / "ensemble_daily.csv"
        original = path.read_text()
        for text in [original.replace("2024-01-02", "2024-01-01"),
                     original.replace("2024-01-02", "2023-12-29"),
                     "\n".join(original.splitlines()[:-1])]:
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                periods.build(self.output)

    def test_unchanged_newlines_and_deterministic_regeneration(self):
        expected = self.publish_periods()
        for path in [self.output / "report.json", *self.run.glob("*_daily.csv")]:
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.assertTrue(periods.load(self.output, self.report)["available"])
        self.assertEqual(periods.build(self.output), expected)

    def test_missing_or_stale_sidecar_leaves_main_api_usable(self):
        client = self.client()
        self.assertEqual(client.get("/backtest").json()["report"]["period_analysis"]["reason"], "missing")
        self.publish_periods()
        self.assertTrue(client.get("/backtest").json()["report"]["period_analysis"]["available"])
        path = self.run / "ensemble_daily.csv"
        path.write_text(path.read_text().replace("0.1", "0.2", 1))
        response = client.get("/backtest")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["report"]["period_analysis"]["reason"], "stale")
        self.assertEqual(client.get("/backtest/export?kind=periods").status_code, 409)

    def test_exports_checked_sidecar_and_rejects_corruption(self):
        expected = self.publish_periods()
        client = self.client()
        response = client.get("/backtest/export?kind=periods")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        self.assertIn("application/json", response.headers["content-type"])
        path = self.output / periods.FILENAME
        for raw in ["{", "[]", json.dumps({**expected, "strategies": {}})]:
            path.write_text(raw)
            self.assertEqual(client.get("/backtest").status_code, 200)
            self.assertEqual(client.get("/backtest/export?kind=periods").status_code, 409)

    def test_changed_run_and_builder_are_not_mixed(self):
        self.publish_periods()
        changed = {**self.report, "run_id": "different-run"}
        self.assertFalse(periods.load(self.output, changed)["available"])
        original_digest = periods.text_digest
        with patch.object(periods, "text_digest", side_effect=lambda p:
                          "changed" if str(p) == periods.__file__ else original_digest(p)):
            self.assertEqual(periods.load(self.output, self.report)["reason"], "stale")

    def test_path_traversal_cannot_build_or_load(self):
        self.report["run_id"] = "../../outside"
        self.publish_report()
        with self.assertRaises(ValueError):
            periods.build(self.output)


if __name__ == "__main__":
    unittest.main()
