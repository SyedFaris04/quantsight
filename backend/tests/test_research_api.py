import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from research import routes


class ResearchApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'report.json'
        patched = patch.object(routes, 'REPORT', self.path)
        patched.start()
        self.addCleanup(patched.stop)
        app = FastAPI()
        app.include_router(routes.router)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_missing_report_is_explicitly_unavailable(self):
        response = self.client.get('/research/news-comparison')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'available': False})

    def test_only_valid_exploratory_report_is_served(self):
        report = {'schema_version': 1, 'status': 'exploratory', 'models': [{'key': 'example'}]}
        self.path.write_text(json.dumps(report), encoding='utf-8')
        self.assertEqual(self.client.get('/research/news-comparison').json(), {'available': True, 'report': report})
        report['status'] = 'confirmed'
        self.path.write_text(json.dumps(report), encoding='utf-8')
        self.assertEqual(self.client.get('/research/news-comparison').status_code, 503)

    def test_corrupt_report_does_not_return_partial_evidence(self):
        self.path.write_text('{broken', encoding='utf-8')
        response = self.client.get('/research/news-comparison')
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('report', response.json())


if __name__ == '__main__':
    unittest.main()
