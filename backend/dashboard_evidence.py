"""Date provenance for saved dashboard inputs; no live-data claims or fetching."""
import pandas as pd


def snapshot_metadata(rows):
    values = [signal.get('date') for row in rows for signal in row.get('signals', {}).values()]
    dates = pd.to_datetime(pd.Series(values, dtype='object'), errors='coerce', utc=True, format='mixed')
    valid = dates.dropna()
    return {
        'kind': 'historical_predictions', 'horizon_sessions': 5,
        'earliest_signal_date': valid.min().date().isoformat() if len(valid) else None,
        'latest_signal_date': valid.max().date().isoformat() if len(valid) else None,
        'mixed_signal_dates': valid.dt.date.nunique() > 1,
        'undated_signals': int(dates.isna().sum()),
        'model_count': len({key for row in rows for key in row.get('signals', {})}),
    }


def news_sentiment(news, scorer, days=14, now=None):
    """Score a window ending at the latest saved article, explicitly dated.

    This preserves the archive use case even when news collection is inactive.
    Invalid/future dates and blank titles cannot anchor or enter the window.
    """
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    now = now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    result = {
        'positive_pct': None, 'neutral_pct': None, 'negative_pct': None,
        'article_count': 0, 'days': days, 'trend': [], 'source': 'saved_news_archive',
        'window_start': None, 'window_end': None, 'latest_article_at': None,
        'age_days': None, 'excluded_rows': 0,
    }
    if news.empty or not {'date', 'title'}.issubset(news.columns):
        return result
    df = news.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True, format='mixed')
    df['title'] = df['title'].fillna('').astype(str).str.strip()
    valid = df['date'].notna() & (df['date'] <= now) & df['title'].ne('')
    result['excluded_rows'] = int((~valid).sum())
    df = df[valid].sort_values('date')
    if df.empty:
        return result
    end = df['date'].max()
    start = end - pd.Timedelta(days=days)
    recent = df[df['date'] >= start].drop_duplicates('title', keep='last').copy()
    recent['compound'] = recent['title'].apply(lambda title: scorer.polarity_scores(title)['compound'])
    counts = {'positive': int((recent['compound'] >= .05).sum()),
              'negative': int((recent['compound'] <= -.05).sum())}
    total = len(recent)
    counts['neutral'] = total - counts['positive'] - counts['negative']
    for key, count in counts.items():
        result[key + '_pct'] = round(100 * count / total, 1)
    trend = recent.groupby(recent['date'].dt.date)['compound'].mean()
    result.update(article_count=total, window_start=start.date().isoformat(),
                  window_end=end.date().isoformat(), latest_article_at=end.isoformat(),
                  age_days=int((now - end).total_seconds() // 86400),
                  trend=[{'date': str(date), 'avg_compound': round(float(value), 3)}
                         for date, value in trend.items()])
    return result
