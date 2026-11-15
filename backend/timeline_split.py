"""Fixed-date splits for the two-timeline study.

Kept out of cv_utils.py on purpose: that file's hash guards the served TreeSHAP
snapshots, so editing it would disable them.
"""
import numpy as np
import pandas as pd

from cv_utils import DEFAULT_EMBARGO_TRADING_DAYS, _shift_trading_days, purge_train_test_boundary


def fixed_split(df, split_date, start=None):
    """(trainval, test) at a pinned date, so adding older history cannot move the test window."""
    split_date = pd.Timestamp(split_date)
    cutoff = purge_train_test_boundary(df["date"], split_date)
    trainval = df[df["date"] <= cutoff]
    if start is not None:
        trainval = trainval[trainval["date"] >= pd.Timestamp(start)]
    return trainval, df[df["date"] > split_date]


def calib_split_at(dates, calib_split, embargo_days=DEFAULT_EMBARGO_TRADING_DAYS):
    """make_train_calib_split with a pinned boundary instead of a quantile."""
    dates = pd.to_datetime(dates).reset_index(drop=True)
    unique_dates = np.sort(dates.unique())
    anchor = pd.Timestamp(calib_split)
    train_mask = (dates <= _shift_trading_days(unique_dates, anchor, -embargo_days)).values
    calib_mask = (dates > _shift_trading_days(unique_dates, anchor, +embargo_days)).values
    return train_mask, calib_mask


def training_arguments(lstm=False):
    """Shared CLI for train_xgboost.py / train_lstm.py. No options = the original production run."""
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--start", help="first training date, e.g. 2005-01-03")
    parser.add_argument("--split-date", help="pinned test boundary; default is the 80%% date quantile")
    parser.add_argument("--out-models", type=Path)
    parser.add_argument("--out-predictions", type=Path)
    if lstm:
        parser.add_argument("--seed", type=int)
        parser.add_argument("--calib-split", help="pinned calibration boundary; default is the 85%% quantile")
    return parser
