import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research.development_models import finance_features, raw_logit, reliability, paired_interval, select_candidate, model_factory


class DevelopmentModelTests(unittest.TestCase):
    def setUp(self):
        self.sessions = pd.bdate_range('2020-01-01', periods=120)
        close = 100 + np.arange(120) * .1 + np.sin(np.arange(120))
        self.frame = pd.DataFrame({'date': self.sessions, 'Adj Close': close, 'Close': close,
                                   'High': close + 1, 'Low': close - 1, 'Volume': np.arange(120) + 1000})

    def test_features_do_not_change_when_future_prices_change(self):
        before = finance_features(self.frame, self.sessions)
        changed = self.frame.copy()
        changed.loc[90:, ['Adj Close', 'Close', 'High', 'Low', 'Volume']] *= 10
        after = finance_features(changed, self.sessions)
        pd.testing.assert_frame_equal(before.iloc[:90], after.iloc[:90])
        self.assertAlmostEqual(before.iloc[80].return_5, self.frame.iloc[80]['Adj Close'] / self.frame.iloc[75]['Adj Close'] - 1)

    def test_price_and_volume_units_do_not_change_ratio_features(self):
        before = finance_features(self.frame, self.sessions).drop(columns=['Close'])
        changed = self.frame.copy()
        changed[['Adj Close', 'Close', 'High', 'Low']] *= 7
        changed.Volume *= 100
        after = finance_features(changed, self.sessions).drop(columns=['Close'])
        pd.testing.assert_frame_equal(before, after, atol=1e-10, rtol=1e-10)

    def test_missing_sessions_are_not_filled_or_treated_as_next_observation(self):
        result = finance_features(self.frame.drop(index=75), self.sessions).set_index('date')
        self.assertNotIn(self.sessions[75], result.index)
        self.assertTrue(pd.isna(result.loc[self.sessions[79], 'return_5']))
        self.assertTrue(pd.isna(result.loc[self.sessions[90], 'ma60_distance']))

    def test_constant_prices_rsi_and_zero_volume(self):
        frame = self.frame.copy()
        frame[['Adj Close', 'Close', 'High', 'Low']] = 100.
        frame.Volume = 0
        result = finance_features(frame, self.sessions)
        self.assertEqual(result.iloc[-1].rsi_14, .5)
        self.assertTrue(pd.isna(result.iloc[-1].volume_relative_20))

    def test_logit_is_finite_at_endpoints_and_rejects_bad_probabilities(self):
        self.assertTrue(np.isfinite(raw_logit([0, .5, 1])).all())
        self.assertEqual(raw_logit([.5])[0, 0], 0)
        for invalid in [[float('nan')], [-.1], [1.1]]:
            with self.assertRaises(ValueError):
                raw_logit(invalid)

    def test_reliability_bins_preserve_all_observations_including_one(self):
        bins = reliability([0, 1, 1], [0, .55, 1])
        self.assertEqual(sum(item['n'] for item in bins), 3)
        self.assertEqual(bins[9]['observed_up_rate'], 1)
        self.assertIsNone(bins[1]['mean_probability'])

    def test_block_bootstrap_identity_zero_and_perfect_predictions_positive(self):
        frame = pd.DataFrame({'fold': ['2022'] * 40 + ['2023'] * 40,
                              'date': list(pd.bdate_range('2022-01-01', periods=40)) + list(pd.bdate_range('2023-01-01', periods=40)),
                              'signal': [0, 1] * 40, 'fit_prior': .5, 'perfect': [0, 1] * 40})
        protocol = {'seed': 42, 'repetitions': 100, 'block_sessions': 5}
        same = paired_interval(frame, 'fit_prior', protocol)
        perfect = paired_interval(frame, 'perfect', protocol)
        self.assertEqual(same['interval_95'], [0., 0.])
        self.assertEqual(perfect['interval_95'], [.25, .25])
        self.assertEqual(perfect['brier_gain_vs_fit_prior'], .25)

    def test_candidate_selection_includes_baselines_and_stable_tie_order(self):
        rows = [{'candidate': name, 'mean_metrics': {'brier_score': value}}
                for name, value in [('fit_prior', .2), ('logistic_raw', .2), ('xgboost_raw', .3)]]
        self.assertEqual(select_candidate(rows), 'fit_prior')
        rows[-1]['mean_metrics']['brier_score'] = .19
        self.assertEqual(select_candidate(rows), 'xgboost_raw')

    def test_scaler_is_fit_only_on_training_rows(self):
        X = np.arange(120).reshape(40, 3)
        model = model_factory('logistic', {'max_iter': 1000})
        model.fit(X, np.arange(40) % 2)
        center = model[0].mean_.copy()
        model.predict_proba(X + 10000)
        np.testing.assert_equal(model[0].mean_, center)
        np.testing.assert_equal(center, X.mean(axis=0))


if __name__ == '__main__':
    unittest.main()
