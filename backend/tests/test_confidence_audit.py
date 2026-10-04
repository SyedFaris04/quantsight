import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from research import confidence_report as evidence, routes
from research.confidence_audit import threshold_rows, validate_predictions


class ConfidenceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(evidence.REPORT.read_text(encoding='utf-8'))
        cls.parent = json.loads(routes.DEVELOPMENT_REPORT.read_text(encoding='utf-8'))
        cls.protocol = json.loads(evidence.PROTOCOL.read_text(encoding='utf-8'))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'confidence.json'
        patcher = patch.object(routes, 'CONFIDENCE_REPORT', self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        app = FastAPI()
        app.include_router(routes.router)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.frame = pd.DataFrame({'date': ['2022-01-03', '2022-01-03', '2022-01-04', '2022-01-04'],
                                   'ticker': ['A', 'B', 'A', 'B'], 'signal': [1, 1, 0, 0],
                                   'label_end': ['2022-01-10', '2022-01-10', '2022-01-11', '2022-01-11'],
                                   'fold': ['validate_2022'] * 4})
        for candidate in self.protocol['candidates']:
            self.frame[candidate] = [.5, .6, .2, .95] if candidate == 'xgboost_raw' else .55

    def test_exact_thresholds_symmetric_direction_and_matching_baseline(self):
        rows = threshold_rows(self.frame, 'xgboost_raw', [.5, .6, .99])
        self.assertEqual(rows[0]['retained_rows'], 4)
        self.assertAlmostEqual(rows[0]['accuracy'], .75)
        self.assertEqual(rows[1]['retained_rows'], 3)
        self.assertEqual(rows[1]['retained_dates'], 2)
        self.assertAlmostEqual(rows[1]['coverage'], .75)
        self.assertAlmostEqual(rows[1]['accuracy'], 2 / 3)
        self.assertAlmostEqual(rows[1]['baseline_accuracy'], 1 / 3)
        self.assertAlmostEqual(rows[1]['brier_score'], (.16 + .04 + .9025) / 3)
        self.assertAlmostEqual(rows[1]['baseline_brier'], (.2025 + .3025 + .3025) / 3)
        self.assertTrue(rows[1]['small_sample'])
        self.assertEqual(rows[2]['coverage'], 0)
        self.assertFalse(rows[2]['small_sample'])
        for key in ['accuracy', 'error_rate', 'mean_confidence', 'brier_score', 'baseline_accuracy', 'baseline_brier']:
            self.assertIsNone(rows[2][key])

    def test_strong_confidence_can_be_wrong(self):
        row = threshold_rows(self.frame, 'xgboost_raw', [.9])[0]
        self.assertEqual(row['retained_rows'], 1)
        self.assertEqual(row['accuracy'], 0)
        self.assertAlmostEqual(row['mean_confidence'], .95)

    def test_holdout_dates_duplicate_keys_and_invalid_probabilities_rejected(self):
        parent = {'tickers': 2, 'results': [{'candidate': 'fit_prior', 'year': 2022, 'metrics': {'n': 4}}]}
        validate_predictions(self.frame, parent, 2022, self.protocol['candidates'])
        for column, value in [('date', '2025-01-03'), ('label_end', '2025-01-10'),
                              ('xgboost_raw', np.nan), ('xgboost_raw', np.inf), ('xgboost_raw', 1.1),
                              ('signal', 2), ('fold', 'validate_2025')]:
            frame = self.frame.copy()
            frame.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                validate_predictions(frame, parent, 2022, self.protocol['candidates'])
        frame = self.frame.copy()
        frame.loc[1, 'ticker'] = 'A'
        with self.assertRaises(ValueError):
            validate_predictions(frame, parent, 2022, self.protocol['candidates'])

    def test_published_evidence_and_export_reconcile(self):
        evidence.validate(self.report, self.parent, self.protocol)
        self.path.write_text(json.dumps(self.report), encoding='utf-8')
        response = self.client.get('/research/development-confidence')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['report'], self.report)
        exported = self.client.get('/research/development-confidence/export')
        self.assertEqual(exported.json(), self.report)
        self.assertIn('attachment;', exported.headers['content-disposition'])

    def test_missing_invalid_and_modified_sources_do_not_hide_parent_report(self):
        self.assertEqual(self.client.get('/research/development-confidence').json(), {'available': False})
        self.assertEqual(self.client.get('/research/development-confidence/export').status_code, 404)
        self.path.write_text('{broken', encoding='utf-8')
        self.assertEqual(self.client.get('/research/development-confidence').status_code, 503)
        self.assertEqual(self.client.get('/research/development-models').status_code, 200)
        self.path.write_text(json.dumps(self.report), encoding='utf-8')
        with patch.object(evidence, 'text_hashes', return_value={'changed'}):
            self.assertEqual(self.client.get('/research/development-confidence').status_code, 503)
            self.assertEqual(self.client.get('/research/development-confidence/export').status_code, 503)

    def test_rejects_inconsistent_counts_means_empty_scores_and_final_claims(self):
        variants = []
        for key, value in [('holdout_files_opened', True), ('serving_models_changed', True),
                           ('new_fits', 1), ('status', 'final_test')]:
            report = copy.deepcopy(self.report)
            report[key] = value
            variants.append(report)
        for field, value in [('coverage', .999), ('retained_rows', 1), ('mean_confidence', .2)]:
            report = copy.deepcopy(self.report)
            report['periods']['pooled']['candidates']['xgboost_raw'][0][field] = value
            variants.append(report)
        report = copy.deepcopy(self.report)
        report['periods']['pooled']['candidates']['xgboost_raw'][-1]['accuracy'] = 1
        variants.append(report)
        report = copy.deepcopy(self.report)
        report['periods']['pooled']['candidates']['xgboost_raw'][1]['accuracy'] += .01
        report['periods']['pooled']['candidates']['xgboost_raw'][1]['error_rate'] -= .01
        variants.append(report)
        for report in variants:
            report['content_sha256'] = evidence.checksum({k: v for k, v in report.items() if k != 'content_sha256'})
            with self.assertRaises(ValueError):
                evidence.validate(report, self.parent, self.protocol)

    def test_line_endings_do_not_change_source_identity(self):
        path = Path(self.temp.name) / 'predictions.csv'
        path.write_bytes(b'date,signal\n2022-01-03,1\n')
        expected = evidence.text_hashes(path)
        path.write_bytes(b'date,signal\r\n2022-01-03,1\r\n')
        self.assertEqual(evidence.text_hashes(path), expected)

    def test_confidence_fetch_imports_no_training_libraries(self):
        # A separate interpreter avoids imports made by other test modules.
        import subprocess
        import sys
        result = subprocess.run([sys.executable, '-c',
            "import sys; from research.confidence_report import load; "
            "assert not any(n.split('.')[0] in {'numpy','pandas','sklearn','xgboost','torch'} for n in sys.modules)"],
            cwd=evidence.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
