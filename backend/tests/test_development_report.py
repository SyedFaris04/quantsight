import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from research import routes
from research.development_report import validate_report


class DevelopmentReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((Path(__file__).resolve().parents[1] / 'data/research_reports/development_models.json').read_text())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'report.json'
        patcher = patch.object(routes, 'DEVELOPMENT_REPORT', self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        app = FastAPI()
        app.include_router(routes.router)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_published_report_reconciles_and_export_matches_api(self):
        self.path.write_text(json.dumps(self.report))
        response = self.client.get('/research/development-models')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['report'], self.report)
        exported = self.client.get('/research/development-models/export')
        self.assertEqual(exported.json(), response.json()['report'])
        self.assertIn('attachment;', exported.headers['content-disposition'])

    def test_missing_and_corrupt_evidence_is_explicit(self):
        self.assertEqual(self.client.get('/research/development-models').json(), {'available': False})
        self.assertEqual(self.client.get('/research/development-models/export').status_code, 404)
        self.path.write_text('{broken')
        self.assertEqual(self.client.get('/research/development-models').status_code, 503)
        self.assertEqual(self.client.get('/research/development-models/export').status_code, 503)

    def test_final_test_claims_and_wrong_selection_are_rejected(self):
        for key, value in [('status', 'final_test'), ('holdout_files_opened', True),
                           ('holdout_scores_computed', True), ('serving_models_changed', True),
                           ('selected_candidate', 'xgboost_raw')]:
            report = copy.deepcopy(self.report)
            report[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_report(report)

    def test_inconsistent_means_nonfinite_metrics_and_duplicate_rows_are_rejected(self):
        variants = []
        report = copy.deepcopy(self.report)
        report['summaries'][0]['mean_metrics']['accuracy'] += .1
        variants.append(report)
        report = copy.deepcopy(self.report)
        report['results'][0]['metrics']['brier_score'] = float('nan')
        variants.append(report)
        report = copy.deepcopy(self.report)
        report['results'][1] = report['results'][0]
        variants.append(report)
        report = copy.deepcopy(self.report)
        report['results'][0]['reliability'][0]['n'] += 1
        variants.append(report)
        for report in variants:
            with self.assertRaises(ValueError):
                validate_report(report)


if __name__ == '__main__':
    unittest.main()
