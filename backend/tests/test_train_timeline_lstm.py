from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from cv_utils import purge_train_test_boundary
from research.train_timeline_lstm import check_quantile, configure, keep_test_window
from timeline_split import calib_split_at

DATES = pd.Series(pd.bdate_range('2005-01-03', '2024-12-20'))


class TimelineLstmDriverTests(unittest.TestCase):
    def test_configure_pins_purge_and_calibration_boundaries(self):
        module = SimpleNamespace()
        configure(module, '2023-02-03', '2021-12-09')
        self.assertEqual(module.purge_train_test_boundary(DATES, pd.Timestamp('2021-04-27')),
                         purge_train_test_boundary(DATES, pd.Timestamp('2023-02-03')))
        for got, want in zip(module.make_train_calib_split(DATES, calib_frac=0.15), calib_split_at(DATES, '2021-12-09')):
            np.testing.assert_array_equal(got, want)

    def test_keep_test_window_drops_earlier_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'p.csv'
            pd.DataFrame({'ticker': 'A', 'date': ['2023-02-03', '2023-02-06'], 'confidence': [50, 60]}).to_csv(path, index=False)
            keep_test_window(path, '2023-02-03')
            self.assertEqual(pd.read_csv(path).date.tolist(), ['2023-02-06'])

    def test_quantile_after_split_fails_closed(self):
        late = pd.DataFrame({'date': pd.bdate_range('2023-01-02', '2024-12-20')})
        with self.assertRaises(ValueError):
            check_quantile(late, '2023-02-03')
        check_quantile(pd.DataFrame({'date': DATES}), '2023-02-03')


if __name__ == '__main__':
    unittest.main()
