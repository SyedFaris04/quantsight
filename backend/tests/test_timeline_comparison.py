import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from research import timeline_comparison as tc

MODELS = ['xgb_finance', 'xgb_sentiment', 'lstm_finance', 'lstm_sentiment']


def frame(rows):
    return pd.DataFrame(rows, columns=['ticker', 'date', 'predicted_signal', 'confidence']).assign(
        date=lambda f: pd.to_datetime(f.date))


class TimelineComparisonUnitTests(unittest.TestCase):
    def test_agreement_counts(self):
        dates = pd.bdate_range('2023-03-01', periods=6)
        a = pd.DataFrame({'ticker': 'A', 'date': dates, 'predicted_signal': [1, 1, 0, 0, 1, 0]})
        b = pd.DataFrame({'ticker': 'A', 'date': dates, 'predicted_signal': [1, 0, 0, 1, 0, 0]})
        labels = pd.DataFrame({'ticker': 'A', 'date': dates, 'signal': [1, 1, 0, 0, 0, 0]})
        result = tc.agreement(a, b, labels)
        self.assertEqual(result['rows'], 6)
        self.assertAlmostEqual(result['agree_rate'], 3 / 6)
        self.assertEqual(result['disagree_rows'], 3)
        self.assertAlmostEqual(result['a_correct_when_disagree'], 2 / 3)
        self.assertAlmostEqual(result['b_correct_when_disagree'], 1 / 3)

    def test_ensemble_common_rows_only(self):
        one = frame([('A', '2023-03-01', 1, 60), ('A', '2023-03-02', 0, 40)])
        two = frame([('A', '2023-03-01', 0, 40)])
        result = tc.ensemble({'one': one, 'two': two})
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].confidence, 50)
        self.assertEqual(result.iloc[0].predicted_signal, 1)


class TimelineComparisonBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        dates = pd.bdate_range('2022-06-01', '2023-06-30')
        rows = []
        for ticker in ['SPY', 'AAA', 'BBB']:
            close = 100 * np.exp(rng.normal(0, .01, len(dates)).cumsum())
            rows.append(pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'ticker': ticker, 'Open': close, 'Close': close,
                                      'f1': rng.normal(size=len(dates)), 'f2': rng.normal(size=len(dates))}))
        features = pd.concat(rows, ignore_index=True)
        features['signal'] = features.groupby('ticker').Close.shift(-5).gt(features.Close).astype(int)
        features = features[features.groupby('ticker').Close.shift(-5).notna()]
        features_dir = root / 'features'
        features_dir.mkdir()
        for name in ['features_finance.csv', 'features_sentiment.csv']:
            features.to_csv(features_dir / name, index=False)
        split = '2023-01-31'
        protocol = {'split_date': split, 'starts': {'2015': '2022-09-01', '2005': '2022-06-01'},
                    'lstm_seeds': [42, 43], 'models': MODELS}
        (root / 'timelines.json').write_text(json.dumps(protocol))
        test = features[features.date > split]
        train = features[features.date <= split]
        for timeline in ['2015', '2005']:
            for directory in [root / 'pred' / timeline, root / 'models' / timeline, root / 'seeds' / timeline / 'seed43']:
                directory.mkdir(parents=True)
            for model in MODELS:
                p = rng.uniform(.3, .7, len(test))
                out = test[['ticker', 'date']].assign(predicted_signal=(p >= .5).astype(int), confidence=np.round(p * 100, 2))
                out.to_csv(root / 'pred' / timeline / f'{model}_predictions.csv', index=False)
                if model.startswith('lstm'):
                    out.to_csv(root / 'seeds' / timeline / 'seed43' / f'{model}_predictions.csv', index=False)
            for model in ['xgb_finance', 'xgb_sentiment']:
                scaler = StandardScaler().fit(train[['f1', 'f2']])
                booster = xgb.XGBClassifier(n_estimators=5, max_depth=2).fit(scaler.transform(train[['f1', 'f2']]), train.signal)
                (root / 'models' / timeline / f'{model}.pkl').write_bytes(
                    pickle.dumps({'model': booster, 'scaler': scaler, 'features': ['f1', 'f2']}))
        production = root / 'production'
        production.mkdir()
        for model in MODELS:
            (root / 'pred' / '2015' / f'{model}_predictions.csv').read_bytes()
            (production / f'{model}_predictions.csv').write_bytes((root / 'pred' / '2015' / f'{model}_predictions.csv').read_bytes())
        crisis = {'run_id': 'x', 'comparison_direction': '2005 minus 2015', 'arm_comparison': [], 'episodes': [],
                  'tickers_by_year': {}, 'results': [{'arm': '2005', 'year': 2022, 'candidate': c, 'fit_rows': 1, 'tickers': 2,
                                                      'metrics': {'n': 1, 'accuracy': .5, 'brier_score': .25}}
                                                     for c in ['xgboost_sigmoid', 'fit_prior', 'always_up']]}
        (root / 'crisis.json').write_text(json.dumps(crisis))
        (root / 'crisis_protocol.json').write_text(json.dumps({'episodes': [
            {'key': 'e', 'label': 'E', 'start': '2022-07-01', 'end': '2022-07-15'}]}))
        self.output = root / 'out.json'
        for name, value in [('PROTOCOL', root / 'timelines.json'), ('FEATURES_DIR', features_dir), ('PREDICTIONS', root / 'pred'),
                            ('SEEDS', root / 'seeds'), ('MODELS_DIR', root / 'models'), ('PRODUCTION', production),
                            ('CRISIS_REPORT', root / 'crisis.json'), ('CRISIS_PROTOCOL', root / 'crisis_protocol.json'),
                            ('OUTPUT', self.output), ('ROOT', root)]:
            patcher = patch.object(tc, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_schema_keys(self):
        report = tc.build()
        for key in ['schema_version', 'status', 'created_at', 'input_hashes', 'test_window', 'timelines', 'baselines',
                    'differences', 'agreement', 'shap_rank_changes', 'production_check', 'crisis', 'limitations']:
            self.assertIn(key, report)
        for timeline in ['2015', '2005']:
            entry = report['timelines'][timeline]
            self.assertEqual(set(entry['models']), set(MODELS))
            self.assertIn('ensemble', entry['backtest'])
            self.assertIn('spy', entry['backtest'])
            self.assertEqual(len(entry['shap']['xgb_finance']), 2)
            self.assertIn('lstm_seeds', entry['models']['lstm_finance'])
        self.assertEqual(report['timelines']['2005']['training']['episodes_included'], ['e'])
        self.assertEqual(report['timelines']['2015']['training']['episodes_included'], [])
        self.assertEqual(report['production_check']['xgb_finance']['agree_rate'], 1.0)
        json.dumps(json.loads(self.output.read_text()), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
