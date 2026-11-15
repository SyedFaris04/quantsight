import unittest

import numpy as np
import pandas as pd

from timeline_split import calib_split_at, fixed_split


def panel(start):
    dates = pd.bdate_range(start, '2024-12-20')
    return pd.DataFrame({'date': np.repeat(dates, 2), 'ticker': ['A', 'B'] * len(dates), 'x': 1.0})


class TimelineSplitTests(unittest.TestCase):
    def test_older_history_does_not_change_test_rows(self):
        _, short = fixed_split(panel('2015-05-01'), '2023-02-03')
        _, long = fixed_split(panel('2010-01-04'), '2023-02-03')
        pd.testing.assert_frame_equal(short.reset_index(drop=True), long.reset_index(drop=True))
        self.assertEqual(short.date.min(), pd.Timestamp('2023-02-06'))

    def test_start_filters_training_rows_only(self):
        trainval, test = fixed_split(panel('2010-01-04'), '2023-02-03', start='2015-05-01')
        self.assertEqual(trainval.date.min(), pd.Timestamp('2015-05-01'))
        self.assertEqual(test.date.min(), pd.Timestamp('2023-02-06'))

    def test_purge_gap_is_six_sessions(self):
        frame = panel('2015-05-01')
        trainval, _ = fixed_split(frame, '2023-02-03')
        dates = np.sort(frame.date.unique())
        position = int(np.searchsorted(dates, np.datetime64('2023-02-03')))
        self.assertEqual(trainval.date.max(), pd.Timestamp(dates[position - 6]))

    def test_calib_split_override(self):
        dates = panel('2015-05-01').date
        train, calib = calib_split_at(dates, '2021-11-01')
        self.assertTrue(dates[calib].min() > pd.Timestamp('2021-11-01'))
        self.assertTrue(dates[train].max() < pd.Timestamp('2021-11-01'))
        self.assertFalse((train & calib).any())
        _, calib_long = calib_split_at(panel('2010-01-04').date, '2021-11-01')
        self.assertEqual(int(calib.sum()), int(calib_long.sum()))


if __name__ == '__main__':
    unittest.main()
