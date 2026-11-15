"""Train the served LSTM variants for one timeline without editing train_lstm.py.

train_lstm.py is a hashed source of the published XAI comparison, so this driver
redirects its split helpers instead: the train/test purge uses the pinned split date
and early stopping/calibration use the pinned calibration boundary. Its own test set
starts at its 80% date quantile; saved predictions are trimmed to the pinned window.
Run from backend: python -m research.train_timeline_lstm --timeline 2005 --seed 42 --out-models ... --out-predictions ...
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from cv_utils import purge_train_test_boundary
from research.freeze_prices import ROOT
from timeline_split import calib_split_at

PROTOCOL = Path(__file__).with_name('protocols') / 'timelines.json'
OUTPUTS = ['lstm_finance_predictions.csv', 'lstm_sentiment_predictions.csv']


def configure(module, split_date, calib_split):
    pinned = pd.Timestamp(split_date)
    module.purge_train_test_boundary = lambda dates, _quantile: purge_train_test_boundary(dates, pinned)
    module.make_train_calib_split = lambda dates, calib_frac=0.15: calib_split_at(dates, calib_split)


def check_quantile(frame, split_date):
    # A quantile after the pinned split would leave test days unpredicted.
    if pd.to_datetime(frame.date).quantile(0.80) > pd.Timestamp(split_date):
        raise ValueError('80% date quantile falls after the pinned split; test window would be incomplete')


def keep_test_window(path, split_date):
    frame = pd.read_csv(path)
    frame[pd.to_datetime(frame.date) > pd.Timestamp(split_date)].to_csv(path, index=False)


def run(timeline, seed, out_models, out_predictions):
    import train_lstm
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    source, work = ROOT / protocol['features_dir'], ROOT / protocol['features_dir'] / f'start_{timeline}'
    work.mkdir(parents=True, exist_ok=True)
    for name in ['features_finance.csv', 'features_sentiment.csv']:
        frame = pd.read_csv(source / name)
        frame = frame[frame.date >= protocol['starts'][timeline]]
        check_quantile(frame, protocol['split_date'])
        frame.to_csv(work / name, index=False)
    out_models.mkdir(parents=True, exist_ok=True)
    out_predictions.mkdir(parents=True, exist_ok=True)
    configure(train_lstm, protocol['split_date'], protocol['lstm_calib_split'])
    train_lstm.FINANCE_CSV, train_lstm.SENTIMENT_CSV = work / 'features_finance.csv', work / 'features_sentiment.csv'
    train_lstm.MODELS_DIR, train_lstm.PREDICTIONS_DIR = out_models, out_predictions
    train_lstm.METRICS_FILE = out_predictions / 'model_metrics.json'
    train_lstm.SEED = seed
    train_lstm.main()
    for name in OUTPUTS:
        keep_test_window(out_predictions / name, protocol['split_date'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeline', choices=['2015', '2005'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-models', type=Path, required=True)
    parser.add_argument('--out-predictions', type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.timeline, arguments.seed, arguments.out_models, arguments.out_predictions)
