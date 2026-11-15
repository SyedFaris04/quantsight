import importlib.util
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.datastructures import Headers

# main.py needs the serving stack; CI runs this file in its serving job instead.
SERVING = all(importlib.util.find_spec(name) for name in ['dotenv', 'vaderSentiment', 'yfinance'])
main = __import__('main') if SERVING else None


def request(headers, host='10.0.0.1'):
    return SimpleNamespace(headers=Headers(headers), client=SimpleNamespace(host=host))


@unittest.skipUnless(SERVING, 'serving dependencies not installed')
class ClientKeyTests(unittest.TestCase):
    def test_cloudflare_header_identifies_visitor_behind_render_proxy(self):
        self.assertEqual(main.client_key(request({'CF-Connecting-IP': '203.0.113.7'})), '203.0.113.7')

    def test_spoofable_forwarded_for_is_ignored(self):
        self.assertEqual(main.client_key(request({'X-Forwarded-For': '1.2.3.4'})), '10.0.0.1')

    def test_direct_connection_falls_back_to_socket_address(self):
        self.assertEqual(main.client_key(request({})), '10.0.0.1')
        self.assertEqual(main.client_key(SimpleNamespace(headers=Headers({}), client=None)), 'unknown')

    def test_two_visitors_get_separate_limits(self):
        with patch.object(main, '_chat_rate_limit', {}):
            for _ in range(main.CHAT_RATE_LIMIT_PER_MIN):
                main._check_rate_limit('203.0.113.7')
            main._check_rate_limit('198.51.100.9')


@unittest.skipUnless(SERVING, 'serving dependencies not installed')
class AdminKeyTests(unittest.TestCase):
    def setUp(self):
        patchers = [patch.dict(os.environ, {'ADMIN_JOB_SECRET': 'right-key'}),
                    patch.object(main.prediction_tracker, 'run_daily_job', return_value={'inserted': 0})]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def post(self, headers):
        return self.client.post('/admin/run-daily-predictions', headers=headers)

    def test_right_key_runs_job(self):
        self.assertEqual(self.post({'X-Admin-Key': 'right-key'}).status_code, 200)

    def test_wrong_or_missing_key_is_forbidden(self):
        for headers in [{'X-Admin-Key': 'wrong-key'}, {}]:
            self.assertEqual(self.post(headers).status_code, 403)


if __name__ == '__main__':
    unittest.main()
