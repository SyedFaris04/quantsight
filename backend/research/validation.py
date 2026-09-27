"""Calendar-exact labels and purged development segments, isolated from serving."""
import numpy as np
import pandas as pd


def calendar_labels(frame, sessions, horizon=5):
    """Return labels only when all bars in the next five-session window exist."""
    if horizon < 1:
        raise ValueError('Horizon must be positive')
    sessions = pd.DatetimeIndex(sessions).normalize()
    if sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError('Sessions must be unique and increasing')
    data = frame.copy()
    data['date'] = pd.to_datetime(data.date, errors='raise')
    if data.date.isna().any() or data.duplicated(['ticker', 'date']).any():
        raise ValueError('Missing or duplicate ticker/date keys')
    if not data.date.isin(sessions).all():
        raise ValueError('A price date is outside the trading calendar')
    if not np.isfinite(data.Close.to_numpy(float)).all() or data.Close.le(0).any():
        raise ValueError('Close prices must be finite and positive')
    parts = []
    for _, part in data.groupby('ticker', sort=True):
        indexed = part.set_index('date').sort_index()
        close = indexed.Close.reindex(sessions)
        # Shift the backward rolling count so the window covers t through t+h.
        complete = close.notna().rolling(horizon + 1).sum().shift(-horizon).eq(horizon + 1)
        target = close.shift(-horizon)
        label_end = pd.Series(sessions, index=sessions).shift(-horizon).where(complete)
        indexed['label_end'] = label_end.reindex(indexed.index)
        signal = target.gt(close).astype('Int64').where(complete)
        indexed['signal'] = signal.reindex(indexed.index)
        parts.append(indexed.reset_index())
    if not parts:
        return data.assign(label_end=pd.NaT, signal=pd.Series(dtype='Int64'))
    return pd.concat(parts, ignore_index=True).sort_values(['ticker', 'date']).reset_index(drop=True)


def purged_segments(frame, fold, holdout_start):
    """Whole dates stay together; no label may cross its segment's right boundary."""
    data = frame.copy()
    data['date'] = pd.to_datetime(data.date, errors='raise')
    data['label_end'] = pd.to_datetime(data.label_end, errors='raise')
    if data.date.isna().any() or data.duplicated(['ticker', 'date']).any():
        raise ValueError('Missing or duplicate ticker/date keys')
    holdout = pd.Timestamp(holdout_start)
    if data.date.ge(holdout).any() or data.label_end.ge(holdout).any():
        raise ValueError('Candidate holdout rows or outcomes cannot enter development')
    known = data.label_end.notna() & data.signal.notna()
    if ((data.label_end <= data.date) & known).any() or not data.loc[known, 'signal'].isin([0, 1]).all():
        raise ValueError('Invalid label end or direction')
    boundaries = [pd.Timestamp(fold[key]) for key in ['fit_start', 'fit_end_exclusive',
                  'calibration_end_exclusive', 'evaluation_end_exclusive']]
    if not all(a < b for a, b in zip(boundaries, boundaries[1:])) or boundaries[-1] > holdout:
        raise ValueError('Fold boundaries must increase and remain within development')
    return {name: data[known & data.date.ge(left) & data.date.lt(right) & data.label_end.lt(right)].copy()
            for name, left, right in zip(['fit', 'calibration', 'validation'], boundaries, boundaries[1:])}
