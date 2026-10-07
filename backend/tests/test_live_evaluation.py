import unittest
import sys
from unittest.mock import patch, Mock
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import numpy as np
import pandas as pd

import live_signals as live
import prediction_tracker as tracker
from market_calendar import forecast_window, eligible_to_record, latest_completed_session, news_session, schedule


class CalendarTests(unittest.TestCase):
    def test_holiday_weekend_and_five_sessions(self):
        self.assertEqual(forecast_window("2024-07-03")["target_date"], "2024-07-11")
        self.assertEqual(forecast_window("2024-11-27")["target_date"], "2024-12-05")

    def test_early_close_and_bar_delay(self):
        self.assertEqual(latest_completed_session("2024-07-03T17:19Z"), "2024-07-02")
        self.assertEqual(latest_completed_session("2024-07-03T17:20Z"), "2024-07-03")

    def test_no_backdating_after_next_open(self):
        self.assertTrue(eligible_to_record("2024-07-03", "2024-07-05T13:29Z"))
        self.assertFalse(eligible_to_record("2024-07-03", "2024-07-05T13:30Z"))
        self.assertFalse(eligible_to_record("2024-07-03", "2024-07-03T17:19Z"))

    def test_dst_and_naive_time(self):
        self.assertIn("21:00", forecast_window("2024-03-08")["data_cutoff_at"])
        self.assertIn("20:00", forecast_window("2024-03-11")["data_cutoff_at"])
        with self.assertRaises(ValueError):
            eligible_to_record("2024-03-11", "2024-03-11 22:00")

    def test_after_close_news_rolls_forward(self):
        self.assertEqual(news_session("2024-07-03T16:59Z"), "2024-07-03")
        self.assertEqual(news_session("2024-07-03T17:01Z"), "2024-07-05")


class InferenceTests(unittest.TestCase):
    def bundle(self, raw=.3, calibrated=.4):
        return {"model": SimpleNamespace(predict_proba=lambda x: np.array([[1-raw, raw]])),
                "scaler": SimpleNamespace(transform=lambda x: x), "features": ["rsi"],
                "calibrator": SimpleNamespace(predict=lambda x: np.array([calibrated])),
                "calibration_method": "isotonic", "artifact_sha256": "test-model"}

    def test_saved_calibrator_and_sell_confidence(self):
        with patch.object(live, "load_xgb_model", return_value=self.bundle()):
            result = live._predict_from_row("TEST", pd.Series({"rsi": 40, "Close": 100.12345}), "2024-07-03", "live")
        self.assertEqual(result["signal_label"], "SELL")
        self.assertEqual(result["probability_up"], .4)
        self.assertEqual(result["direction_confidence"], 60)
        self.assertEqual(result["close_price"], 100.12345)
        self.assertEqual(result["target_date"], "2024-07-11")

    def test_raw_decision_is_preserved_when_calibration_crosses_half(self):
        with patch.object(live, "load_xgb_model", return_value=self.bundle(.6, .4)):
            result = live._predict_from_row("TEST", pd.Series({"rsi": 40, "Close": 100}), "2024-07-03", "live")
        self.assertEqual(result["signal_label"], "BUY")
        self.assertEqual(result["direction_confidence"], 40)

    def test_missing_and_infinite_features_rejected(self):
        for value in [np.nan, np.inf]:
            with patch.object(live, "load_xgb_model", return_value=self.bundle()):
                result = live._predict_from_row("TEST", pd.Series({"rsi": value}), "2024-07-03", "live")
            self.assertIsNone(result["signal"])

    def test_expired_cache_is_not_served_on_failure(self):
        old = {"data": pd.DataFrame({"Close": [100]}), "fetched_at": datetime.now(timezone.utc) - timedelta(hours=1)}
        with patch.object(live, "_panel_cache", old), patch.object(live, "get_ticker_universe", return_value=["TEST"]), patch.object(live, "_fetch_market_panel", side_effect=RuntimeError("offline")) as fetch:
            self.assertTrue(live._get_cached_panel().empty)
            self.assertTrue(live._get_cached_panel().empty)
            self.assertEqual(fetch.call_count, 1)

    def test_stale_session_is_unavailable(self):
        panel = pd.DataFrame([{"ticker": "TEST", "date": "2024-07-02"}])
        with patch.object(live, "_get_cached_panel", return_value=panel), patch.object(live, "latest_completed_session", return_value="2024-07-03"):
            self.assertEqual(live.get_live_signal("TEST")["source"], "fallback")

    def test_slow_failed_fetch_cache_starts_when_fetch_finishes(self):
        start = datetime(2024, 7, 3, 22, tzinfo=timezone.utc)
        cache = {"data": None, "fetched_at": None}
        with patch.object(live, "_panel_cache", cache), patch.object(live, "get_ticker_universe", return_value=["TEST"]), patch.object(live, "datetime") as clock, patch.object(live, "_fetch_market_panel", side_effect=RuntimeError("offline")) as fetch:
            clock.now.side_effect = [start, start + timedelta(seconds=90), start + timedelta(seconds=91)]
            self.assertTrue(live._get_cached_panel().empty)
            self.assertTrue(live._get_cached_panel().empty)
            self.assertEqual(fetch.call_count, 1)

    def test_cache_refreshes_when_completed_session_changes(self):
        new = pd.DataFrame([{"ticker": "TEST", "date": "2024-07-03"}])
        cache = {"data": pd.DataFrame([{"ticker": "TEST", "date": "2024-07-02"}]), "fetched_at": datetime.now(timezone.utc)}
        with patch.object(live, "_panel_cache", cache), patch.object(live, "get_ticker_universe", return_value=["TEST"]), patch.object(live, "latest_completed_session", return_value="2024-07-03"), patch.object(live, "_fetch_market_panel", return_value=new) as fetch:
            self.assertEqual(live._get_cached_panel().date.max(), "2024-07-03")
            fetch.assert_called_once()

    def test_concurrent_requests_share_completed_fetch(self):
        entered, release, second_started = Event(), Event(), Event()
        panel = pd.DataFrame([{"ticker": "TEST", "date": "2024-07-03"}])
        def download(tickers):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("Fixture wait expired")
            return panel
        def second():
            second_started.set()
            return live._get_cached_panel()
        with patch.object(live, "_panel_cache", {"data": None, "fetched_at": None}), patch.object(live, "get_ticker_universe", return_value=["TEST"]), patch.object(live, "latest_completed_session", return_value="2024-07-03"), patch.object(live, "_fetch_market_panel", side_effect=download) as fetch, ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(live._get_cached_panel)
            self.assertTrue(entered.wait(2))
            other = pool.submit(second)
            self.assertTrue(second_started.wait(2))
            release.set()
            self.assertFalse(first.result(timeout=3).empty)
            self.assertFalse(other.result(timeout=3).empty)
            self.assertEqual(fetch.call_count, 1)

    def test_gap_validation(self):
        dates = schedule("2023-01-01", "2024-12-20").index
        frame = pd.DataFrame(100., index=dates, columns=["Open", "High", "Low", "Close", "Volume"])
        live.validate_daily_history(frame, "2024-12-20")
        with self.assertRaises(ValueError):
            live.validate_daily_history(frame.drop(dates[-5]), "2024-12-20")

    def test_rsi_all_up_and_flat(self):
        self.assertEqual(live.compute_rsi(pd.Series(range(30), dtype=float)).iloc[-1], 100)
        self.assertEqual(live.compute_rsi(pd.Series([100.] * 30)).iloc[-1], 50)

    def test_partial_universe_and_intraday_candle(self):
        dates = schedule("2023-01-01", "2024-12-23").index
        values = 100 + np.sin(np.arange(len(dates))) * 3 + np.arange(len(dates)) * .01
        frame = pd.DataFrame({"Open": values, "High": values+2, "Low": values-2,
                              "Close": values, "Volume": 1000.}, index=dates)
        raw = pd.concat({"AAPL": frame, "MSFT": frame, "XLK": frame}, axis=1)
        yf = SimpleNamespace(download=Mock(return_value=raw))
        with patch.dict(sys.modules, {"yfinance": yf}), patch.object(live, "SECTOR_MAP", {"AAPL": "XLK", "MSFT": "XLK"}), patch.object(live, "latest_completed_session", return_value="2024-12-20"):
            complete = live._fetch_market_panel(["AAPL", "MSFT"])
            self.assertEqual(complete.date.max(), "2024-12-20")
            self.assertEqual(set(complete.ticker), {"AAPL", "MSFT"})
            self.assertEqual(yf.download.call_args.kwargs["threads"], 8)
            self.assertEqual(yf.download.call_args.kwargs["timeout"], 15)
            yf.download.return_value = raw.drop(columns="XLK", level=0)
            self.assertTrue(live._fetch_market_panel(["AAPL", "MSFT"]).empty)
            yf.download.return_value = raw.drop(columns="MSFT", level=0)
            self.assertTrue(live._fetch_market_panel(["AAPL", "MSFT"]).empty)


class Query:
    """Small in-memory repository double; exercises pagination/filter semantics."""
    def __init__(self, db):
        self.db, self.filters, self.start, self.end = db, [], 0, 499
        self.action, self.payload = "select", None
        self.orders = []
    def select(self, *args): return self
    def eq(self, key, val): self.filters.append(lambda r: r.get(key) == val); return self
    def lte(self, key, val): self.filters.append(lambda r: r.get(key, "") <= val); return self
    def gte(self, key, val): self.filters.append(lambda r: r.get(key, "") >= val); return self
    def order(self, key, desc=False): self.orders.append((key, desc)); return self
    def range(self, start, end): self.start, self.end = start, end; return self
    def update(self, payload): self.action, self.payload = "update", payload; return self
    def upsert(self, payload, **kwargs): self.action, self.payload = "upsert", payload; return self
    def execute(self):
        rows = [r for r in self.db.rows if all(f(r) for f in self.filters)]
        if self.action == "upsert":
            p = self.payload
            if any(all(r.get(k) == p[k] for k in ["ticker", "predicted_date", "protocol_version"]) for r in self.db.rows):
                rows = []
            else:
                row = dict(p, id=str(len(self.db.rows)), resolved=False)
                self.db.rows.append(row); rows = [row]
        elif self.action == "update":
            for row in rows: row.update(self.payload)
        else:
            for key, desc in reversed(self.orders): rows.sort(key=lambda r: r[key], reverse=desc)
            rows = rows[self.start:self.end + 1]
        return SimpleNamespace(data=rows)


class Database:
    def __init__(self, rows): self.rows = rows
    def table(self, name):
        assert name == tracker.TABLE, "Legacy records must not be scored by v2"
        return Query(self)


class TrackerTests(unittest.TestCase):
    def row(self):
        return {"id": "1", "ticker": "TEST", "predicted_date": "2024-07-03", "target_date": "2024-07-11", "predicted_signal": "BUY", "resolved": False}

    def test_missed_job_uses_exact_target_not_latest(self):
        prices = pd.Series({"2024-07-03": 100., "2024-07-11": 105., "2024-07-12": 80.})
        result = tracker.outcome_for(self.row(), prices)
        self.assertTrue(result["correct"])
        self.assertEqual(result["actual_close"], 105)
        self.assertEqual(result["resolved_date"], "2024-07-11")

    def test_missing_target_remains_pending_and_tie_is_sell(self):
        self.assertIsNone(tracker.outcome_for(self.row(), pd.Series({"2024-07-03": 100, "2024-07-12": 110})))
        result = tracker.outcome_for(self.row(), pd.Series({"2024-07-03": 100, "2024-07-11": 100}))
        self.assertEqual(result["actual_signal"], "SELL")

    def test_outcome_rebases_both_prices_together(self):
        row = dict(self.row(), price_at_prediction=200)
        result = tracker.outcome_for(row, pd.Series({"2024-07-03": 100, "2024-07-11": 105}))
        self.assertAlmostEqual(result["actual_return"], .05)

    def test_resolution_survives_unavailable_inference(self):
        db = Database([self.row()])
        with patch.object(tracker, "get_admin_client", return_value=db), patch.object(tracker, "fetch_outcome_prices", return_value=pd.Series({"2024-07-03": 100, "2024-07-11": 105})), patch.object(tracker, "get_live_signal", return_value={"source": "fallback"}):
            result = tracker.run_daily_job(["TEST"], now="2024-07-12T22:00Z")
        self.assertEqual(result["resolved"], 1)
        self.assertEqual(result["logged"], 0)
        self.assertEqual(result["status"], "partial_failure")

    def test_idempotent_insert_and_horizon(self):
        db = Database([])
        sig = {"source": "live", "date": "2024-07-03", "signal_label": "BUY", "model_version": "hash",
               "feature_version": "features", "feature_values": {"rsi": 50}, "generated_at": "2024-07-03T22:00Z", "data_fetched_at": "2024-07-03T21:59Z",
               "direction_confidence": 60., "probability_up": .6, "raw_probability_up": .7, "calibration_method": "isotonic", "close_price": 100.}
        with patch.object(tracker, "get_admin_client", return_value=db), patch.object(tracker, "get_live_signal", return_value=sig):
            first = tracker.run_daily_job(["TEST"], now="2024-07-03T22:00Z")
            second = tracker.run_daily_job(["TEST"], now="2024-07-03T22:01Z")
        self.assertEqual(first["logged"], 1)
        self.assertEqual(second["logged"], 0)
        self.assertEqual(second["duplicates"], 1)
        self.assertEqual(db.rows[0]["target_date"], "2024-07-11")
        self.assertEqual(first["cohort_logged"], 1)
        self.assertEqual(second["missing_tickers"], [])
        # A recovery attempt must succeed even if inference is now unavailable.
        with patch.object(tracker, "get_admin_client", return_value=db), patch.object(tracker, "get_live_signal") as infer:
            recovery = tracker.run_daily_job(["TEST"], now="2024-07-05T14:00Z")
        infer.assert_not_called()
        self.assertEqual(recovery["duplicates"], 1)
        self.assertEqual(recovery["cohort_logged"], 1)
        self.assertEqual(recovery["status"], "skipped")

    def test_known_inference_failure_has_safe_stage_code(self):
        with patch.object(tracker, "get_admin_client", return_value=Database([])), patch.object(tracker, "get_live_signal", return_value={"source": "fallback", "error": "Complete market panel unavailable"}):
            result = tracker.run_daily_job(["TEST"], now="2024-07-03T22:00Z")
        self.assertEqual(result["cohort_logged"], 0)
        self.assertEqual(result["missing_tickers"], ["TEST"])
        self.assertEqual(result["error_details"], [{"ticker": "TEST", "stage": "inference", "code": "market_panel_unavailable"}])
        self.assertEqual(tracker._signal_failure_code({"error": "secret-provider-text"}), "signal_unavailable")

    def test_final_coverage_read_failure_is_not_success(self):
        db = Database([])
        original = db.table
        db.table = Mock(side_effect=[original(tracker.TABLE), original(tracker.TABLE), RuntimeError("offline")])
        with patch.object(tracker, "get_admin_client", return_value=db):
            result = tracker.run_daily_job(["TEST"], now="2024-07-05T14:00Z")
        self.assertEqual(result["status"], "partial_failure")
        self.assertIsNone(result["cohort_logged"])
        self.assertEqual(result["error_details"][0]["code"], "database_read_failed")

    def test_recording_deadline_is_rechecked_after_inference(self):
        db = Database([])
        sig = {"source": "live", "date": "2024-07-03", "signal_label": "BUY"}
        with patch.object(tracker, "get_admin_client", return_value=db), patch.object(tracker, "utc_now", side_effect=[pd.Timestamp("2024-07-05T13:29Z"), pd.Timestamp("2024-07-05T13:30Z")]), patch.object(tracker, "get_live_signal", return_value=sig):
            result = tracker.run_daily_job(["TEST"])
        self.assertEqual(db.rows, [])
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["missing_tickers"], ["TEST"])

    def test_empty_universe_is_an_explicit_failure(self):
        with self.assertRaisesRegex(RuntimeError, "no configured ticker"):
            tracker.run_daily_job([])

    def test_summary_reads_beyond_500_rows(self):
        rows = [{"id": str(i), "ticker": f"TEST{i}", "predicted_date": "2024-07-03", "protocol_version": tracker.PROTOCOL,
                 "resolved": True, "correct": i < 600, "predicted_signal": "BUY" if i < 600 else "SELL",
                 "actual_signal": "BUY", "probability_up": .6} for i in range(1100)]
        with patch.object(tracker, "get_admin_client", return_value=Database(rows)), patch.object(tracker, "utc_now", return_value=pd.Timestamp("2024-07-20T22:00Z")):
            summary = tracker.get_summary()
        self.assertEqual(summary["total_logged"], 1100)
        self.assertEqual(summary["accuracy_pct"], 54.5)
        self.assertEqual(summary["always_up_accuracy_pct"], 100.)
        self.assertEqual(summary["brier_score"], .16)
        self.assertEqual(len(summary["recent"]), 20)
        self.assertEqual(summary["evidence"]["resolved_forecast_dates"], 1)
        self.assertFalse(summary["evidence"]["intervals"]["available"])

    def test_outside_window_does_not_create_backdated_forecast(self):
        with patch.object(tracker, "get_admin_client", return_value=Database([])), patch.object(tracker, "get_live_signal") as infer:
            result = tracker.run_daily_job(["TEST"], now="2024-07-05T14:00Z")
        self.assertEqual(result["logged"], 0)
        self.assertEqual(result["skipped"], 1)
        infer.assert_not_called()

    def test_schema_failure_is_explicit_and_does_not_use_legacy(self):
        db = Mock()
        db.table.side_effect = RuntimeError("table unavailable")
        with patch.object(tracker, "get_admin_client", return_value=db):
            self.assertFalse(tracker.get_summary()["available"])
            with self.assertRaises(RuntimeError):
                tracker.run_daily_job(["TEST"], now="2024-07-03T22:00Z")
        self.assertTrue(all(call.args == (tracker.TABLE,) for call in db.table.call_args_list))


if __name__ == "__main__":
    unittest.main()
