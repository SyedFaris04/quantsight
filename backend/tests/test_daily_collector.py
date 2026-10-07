import http.client
import json
import unittest
from io import BytesIO
from unittest.mock import Mock, patch

from research import run_daily_collector as runner


def result(status="ok", coverage=2, code=None):
    body = {"protocol_version": runner.PROTOCOL, "status": status,
            "logged": coverage, "resolved": 0, "duplicates": 0, "skipped": 2 - coverage,
            "tickers_processed": 2, "cohort_logged": coverage,
            "missing_tickers": ["AAPL", "MSFT"][coverage:],
            "forecast_date": "2024-07-03", "record_before": "2024-07-05T13:30:00+00:00",
            "errors": [], "error_details": []}
    if code:
        body["errors"] = ["raw provider text containing secret-value"]
        body["error_details"] = [{"ticker": "MSFT", "stage": "inference", "code": code}]
    return body


class RunnerTests(unittest.TestCase):
    def run_collector(self, responses):
        transport = Mock(side_effect=responses)
        sleep = Mock()
        report = runner.collect(transport, sleep=sleep)
        return report, transport, sleep

    def ready(self):
        return (200, {"ready": True, "release": "a" * 40})

    def test_cold_start_then_complete_coverage(self):
        report, transport, sleep = self.run_collector([(503, None), (200, {"ready": False}), self.ready(), (200, result())])
        self.assertTrue(report["success"])
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [15, 15])
        self.assertEqual(transport.call_args_list[-1].kwargs, {"method": "POST", "timeout": 240})

    def test_partial_provider_failure_retries_beyond_failure_cache(self):
        report, _, sleep = self.run_collector([self.ready(), (503, result("partial_failure", 1, "market_panel_unavailable")), (200, result())])
        self.assertTrue(report["success"])
        sleep.assert_called_once_with(75)
        self.assertNotIn("secret-value", json.dumps(report) + runner.summary(report))
        self.assertIn("market_panel_unavailable", runner.summary(report))

    def test_timeout_retries_and_exhaustion_is_failure(self):
        report, _, sleep = self.run_collector([self.ready(), (0, None), (502, None), (503, {"detail": "secret-value"})])
        self.assertFalse(report["success"])
        self.assertEqual(len(report["attempts"]), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertNotIn("secret-value", json.dumps(report))

    def test_authentication_and_redirect_errors_are_not_retried(self):
        for status in (403, 301, 404):
            report, transport, sleep = self.run_collector([self.ready(), (status, {"detail": "secret-value"})])
            self.assertFalse(report["success"])
            self.assertEqual(transport.call_count, 2)
            sleep.assert_not_called()

    def test_skipped_outside_window_requires_existing_complete_cohort(self):
        for coverage in (0, 1, 2):
            report, _, sleep = self.run_collector([self.ready(), (200, result("skipped", coverage))])
            self.assertEqual(report["success"], coverage == 2)
            sleep.assert_not_called()
            if coverage < 2:
                self.assertEqual(report["failure"], "missed_window")

    def test_http_200_incomplete_is_not_green(self):
        report, _, _ = self.run_collector([self.ready()] + [(200, result(coverage=1))] * 3)
        self.assertFalse(report["success"])

    def test_resolution_failure_still_fails_with_complete_forecasts(self):
        partial = result("partial_failure", 2, "exact_prices_missing")
        partial["error_details"][0]["stage"] = "resolution"
        report, _, _ = self.run_collector([self.ready()] + [(503, partial)] * 3)
        self.assertFalse(report["success"])

    def test_missing_model_does_not_retry_but_missing_features_does(self):
        report, _, sleep = self.run_collector([self.ready(), (503, result("partial_failure", 1, "model_unavailable"))])
        self.assertFalse(report["success"])
        sleep.assert_not_called()
        report, _, sleep = self.run_collector([self.ready(), (503, result("partial_failure", 1, "required_features_missing")), (200, result())])
        self.assertTrue(report["success"])
        sleep.assert_called_once_with(75)

    def test_readiness_exhaustion_never_posts(self):
        report, transport, _ = self.run_collector([(0, None)] * 4)
        self.assertFalse(report["success"])
        self.assertTrue(all(c.args == ("/health/ready",) for c in transport.call_args_list))

    def test_malformed_success_response_is_rejected_without_raw_output(self):
        malformed = result()
        malformed["cohort_logged"] = True
        report, _, sleep = self.run_collector([self.ready(), (200, malformed)])
        self.assertFalse(report["success"])
        sleep.assert_not_called()
        for change in ({"tickers_processed": 0}, {"missing_tickers": ["bad\nvalue"]},
                       {"errors": ["oops"]}, {"forecast_date": "secret-value"},
                       {"cohort_logged": None, "missing_tickers": None}):
            with self.assertRaises(ValueError):
                runner.safe_result(dict(result(), **change))

    def test_unverified_database_coverage_is_reported_and_retried(self):
        partial = result("partial_failure", 0, "database_read_failed")
        partial.update(cohort_logged=None, missing_tickers=None)
        partial["error_details"][0]["stage"] = "coverage"
        report, _, sleep = self.run_collector([self.ready(), (503, partial), (200, result())])
        self.assertTrue(report["success"])
        self.assertIn("database_read_failed", runner.summary(report))
        sleep.assert_called_once_with(75)

    def test_transport_requires_https_and_refuses_redirects(self):
        for base in ("http://backend.example", "https://user:secret@backend.example", "https://backend.example?key=secret"):
            with self.assertRaises(ValueError):
                runner.HTTPTransport(base, "secret")
        self.assertIsNone(runner.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.example"))

    def test_transport_does_not_send_admin_header_on_readiness(self):
        transport = runner.HTTPTransport("https://backend.example", "secret")
        def response(request, timeout):
            self.assertEqual(request.get_header("X-admin-key"), "secret" if request.method == "POST" else None)
            data = BytesIO(b'{"ready":true}')
            data.code = 200
            return data
        with patch.object(transport.opener, "open", side_effect=response):
            self.assertEqual(transport("/health/ready")[0], 200)
            self.assertEqual(transport("/admin/run-daily-predictions", "POST")[0], 200)

    def test_broken_http_connection_is_a_safe_retryable_result(self):
        transport = runner.HTTPTransport("https://backend.example", "secret")
        with patch.object(transport.opener, "open", side_effect=http.client.RemoteDisconnected("secret-value")):
            self.assertEqual(transport("/health/ready"), (0, None))


if __name__ == "__main__":
    unittest.main()
