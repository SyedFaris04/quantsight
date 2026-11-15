"""Build served-model features from a frozen price snapshot for the two-timeline study.

Runs build_features.main() unchanged, only redirecting its inputs and outputs, so the
timeline features follow the production path exactly. build_features.py itself is not
edited: its hash is part of every live forecast's feature_version.
Run from backend: python -m research.timeline_features --snapshot <dir> --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_features
from research.freeze_prices import ROOT, load_development

PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
# build_features.main() renames these features_stock.csv columns back to the lowercase names.
STOCK_COLUMNS = {
    'sma_5': 'SMA_5', 'sma_10': 'SMA_10', 'sma_20': 'SMA_20', 'ema_12': 'EMA_12', 'ema_26': 'EMA_26',
    'macd': 'MACD', 'macd_signal': 'MACD_signal', 'macd_hist': 'MACD_hist', 'rsi': 'RSI',
    'return_5': 'Return_5', 'return_10': 'Return_10', 'bb_upper': 'BB_upper', 'bb_lower': 'BB_lower',
    'bb_mid': 'BB_mid', 'bb_width': 'BB_width', 'volatility_20': 'Volatility_20', 'atr': 'ATR',
    'volume_change': 'Volume_change', 'volume_sma_10': 'Volume_MA_10',
}


def snapshot_ohlcv(snapshot, tickers, end_exclusive):
    """Adjusted daily bars (OHLC scaled by Adj Close / Close), like yfinance auto_adjust."""
    frames = []
    for ticker in tickers:
        raw = load_development(snapshot, ticker)
        raw = raw[raw.date < pd.Timestamp(end_exclusive)].dropna(subset=PRICE_COLUMNS + ['Adj Close'])
        factor = raw['Adj Close'] / raw.Close
        frame = pd.DataFrame({'Date': raw.date, 'Volume': raw.Volume, 'Ticker': ticker})
        for column in ['Open', 'High', 'Low', 'Close']:
            frame[column] = raw[column] * factor
        # Same cleaning as notebook 14: nonpositive bars are dropped, never repaired.
        frame = frame[(frame[PRICE_COLUMNS] > 0).all(axis=1)]
        frames.append(frame[['Date'] + PRICE_COLUMNS + ['Ticker']])
    return pd.concat(frames, ignore_index=True).sort_values(['Ticker', 'Date']).reset_index(drop=True)


def sector_returns(snapshot, tickers, end_exclusive):
    frames = []
    for etf in tickers:
        raw = load_development(snapshot, etf)
        close = raw[raw.date < pd.Timestamp(end_exclusive)].set_index('date')['Adj Close'].dropna()
        frames.append(pd.DataFrame({'date': close.index.strftime('%Y-%m-%d'), 'sector_etf': etf,
                                    'sret_5': close.pct_change(5).values, 'sret_20': close.pct_change(20).values}))
    return pd.concat(frames, ignore_index=True).dropna(subset=['sret_5', 'sret_20'])


def stock_features(ohlcv):
    """features_stock.csv format, using the indicator primitives shared with live inference."""
    parts = []
    for _, group in ohlcv.groupby('Ticker', sort=True):
        enriched = build_features.add_indicators(group)
        enriched['Forward_return'] = enriched.Close.shift(-5) / enriched.Close - 1
        enriched['Target'] = enriched.pop('signal')
        parts.append(enriched)
    data = pd.concat(parts, ignore_index=True).rename(columns=STOCK_COLUMNS)
    columns = ['Date'] + PRICE_COLUMNS + ['Ticker'] + list(STOCK_COLUMNS.values()) + ['Forward_return', 'Target']
    data = data[columns].replace([np.inf, -np.inf], np.nan).dropna()
    data['Target'] = data.Target.astype(int)
    data['Date'] = data.Date.dt.strftime('%Y-%m-%d')
    return data


def build(snapshot, out_dir, end_exclusive='2024-12-31'):
    snapshot, out_dir = Path(snapshot), Path(out_dir)
    protocol = json.loads((snapshot / 'request.json').read_text(encoding='utf-8'))['protocol']
    sectors = sorted(set(build_features.SECTOR_MAP.values()) & set(protocol['context_tickers']))
    out_dir.mkdir(parents=True, exist_ok=True)
    stock_features(snapshot_ohlcv(snapshot, protocol['universe'], end_exclusive)).to_csv(
        out_dir / 'features_stock.csv', index=False)
    if sectors:
        sector_returns(snapshot, sectors, end_exclusive).to_csv(out_dir / 'sector_returns.csv', index=False)
    redirected = {'FEATURES_STOCK': out_dir / 'features_stock.csv', 'SECTOR_FILE': out_dir / 'sector_returns.csv',
                  'FINANCE_OUT': out_dir / 'features_finance.csv', 'SENTIMENT_OUT': out_dir / 'features_sentiment.csv'}
    original = {name: getattr(build_features, name) for name in redirected}
    try:
        for name, value in redirected.items():
            setattr(build_features, name, value)
        build_features.main()
    finally:
        for name, value in original.items():
            setattr(build_features, name, value)
    finance = pd.read_csv(out_dir / 'features_finance.csv')
    return {'rows': len(finance), 'tickers': int(finance.ticker.nunique()),
            'first_date_by_ticker': finance.groupby('ticker').date.min().to_dict(),
            'columns': finance.columns.tolist(), 'date_range': [finance.date.min(), finance.date.max()]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/research/timelines')
    arguments = parser.parse_args()
    summary = build(arguments.snapshot, arguments.out)
    (arguments.out / 'build_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ['rows', 'tickers', 'date_range']}))
