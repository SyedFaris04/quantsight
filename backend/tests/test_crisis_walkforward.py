import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from research.crisis_walkforward import (PROTOCOL, check_budget, episode_metrics, paired_difference,
                                          validation_rows, walkforward_folds)
from research.validation import calendar_labels

ARMS = json.loads(PROTOCOL.read_text(encoding='utf-8'))['arms']


def predictions(dates, a, b, signal):
    return pd.DataFrame({'date': dates, 'signal': signal, 'a': a, 'b': b})


class CrisisWalkforwardTests(unittest.TestCase):
    def test_folds(self):
        folds = walkforward_folds(ARMS)
        self.assertEqual(len(folds), 25)
        first = next(f for f in folds if f['name'] == '2005_2008')
        self.assertEqual((first['fit_start'], first['fit_end_exclusive'], first['calibration_end_exclusive'],
                          first['evaluation_end_exclusive']), ('2005-01-01', '2007-01-01', '2008-01-01', '2009-01-01'))
        self.assertEqual(next(f for f in folds if f['name'] == '2015_2017')['fit_start'], '2015-01-01')

    def test_overlapping_years_have_identical_validation_rows(self):
        sessions = pd.bdate_range('2004-06-01', '2019-12-31')
        rng = np.random.default_rng(0)
        frame = pd.concat([pd.DataFrame({'ticker': t, 'date': sessions,
                                         'Close': 100 * np.exp(rng.normal(0, .01, len(sessions)).cumsum())})
                           for t in 'AB'])
        labeled = calendar_labels(frame, sessions)
        folds = {f['name']: f for f in walkforward_folds(ARMS)}
        a = validation_rows(labeled, folds['2005_2018'], '2025-01-01')
        b = validation_rows(labeled, folds['2015_2018'], '2025-01-01')
        self.assertEqual(list(zip(a.ticker, a.date)), list(zip(b.ticker, b.date)))
        self.assertGreater(len(a), 400)

    def test_paired_difference_identical_is_zero(self):
        dates = np.repeat(pd.bdate_range('2020-01-01', periods=100), 2)
        p = np.linspace(.3, .7, 200)
        frame = predictions(dates, p, p, np.tile([0, 1], 100))
        for metric in ['brier', 'accuracy']:
            result = paired_difference(frame, 'a', 'b', metric, reps=200)
            self.assertEqual(result['difference'], 0)
            self.assertEqual(result['interval_95'], [0, 0])

    def test_paired_difference_sign(self):
        dates = np.repeat(pd.bdate_range('2020-01-01', periods=100), 2)
        signal = np.tile([0, 1], 100)
        frame = predictions(dates, np.where(signal == 1, .8, .2), np.full(200, .5), signal)
        brier = paired_difference(frame, 'a', 'b', 'brier', reps=200)
        self.assertAlmostEqual(brier['difference'], .04 - .25)
        self.assertLess(brier['interval_95'][1], 0)
        self.assertGreater(paired_difference(frame, 'a', 'b', 'accuracy', reps=200)['difference'], 0)

    def test_episode_spans_two_folds(self):
        dates = pd.to_datetime(['2008-08-29', '2008-10-01', '2009-02-02', '2009-04-01'])
        frame = pd.DataFrame({'arm': '2005', 'date': dates, 'signal': [1, 0, 1, 0],
                              'xgboost_sigmoid': [.6, .4, .6, .4], 'fit_prior': .52, 'always_up': 1.0})
        episodes = [{'key': 'gfc', 'label': 'GFC', 'start': '2008-09-01', 'end': '2009-03-31'}]
        result = episode_metrics(frame, episodes, 'xgboost_sigmoid')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['n'], 2)
        self.assertEqual(result[0]['metrics']['accuracy'], 1.0)
        self.assertEqual(result[0]['always_up_accuracy'], .5)

    def test_budget_asserted(self):
        protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
        check_budget(protocol)
        protocol['fit_budget']['predictors'] = 74
        with self.assertRaises(ValueError):
            check_budget(protocol)


if __name__ == '__main__':
    unittest.main()
