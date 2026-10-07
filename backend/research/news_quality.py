"""Offline news integrity/coverage audit. No model fits, labels or text publication."""
import hashlib
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from market_calendar import schedule

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocols") / "news_quality_v1.json"


def digest(path):
    with path.open("rb") as handle:
        result = hashlib.file_digest(handle, "sha256")
    return result.hexdigest()


def headline_key(text):
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def integrity(news, finance, maximum_date):
    required = {"article_id", "ticker", "headline", "url", "published_at",
                "archive_at", "available_at", "date", "date_only_delayed"}
    if not required.issubset(news.columns) or news.empty or finance.empty:
        raise ValueError("Missing required data")
    if news[list(required)].isna().any().any():
        raise ValueError("Missing news values")
    if not news.headline.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Invalid headline")
    if finance.isna().any().any() or finance.duplicated(["ticker", "date"]).any():
        raise ValueError("Invalid finance keys")
    if finance.date.max() > maximum_date:
        raise ValueError("Finance input extends beyond the registered development period")
    if news.duplicated(["ticker", "article_id"]).any():
        raise ValueError("Duplicate ticker/article keys")
    if not news.ticker.isin(finance.ticker.unique()).all():
        raise ValueError("News links outside the finance universe")
    for field in ["headline", "url", "published_at", "archive_at", "available_at", "date"]:
        if news.groupby("article_id")[field].nunique().gt(1).any():
            raise ValueError(f"Inconsistent article identity: {field}")
    published, archive, available = (pd.to_datetime(news[c], utc=True, errors="raise")
                                      for c in ["published_at", "archive_at", "available_at"])
    if archive.lt(published).any() or available.lt(archive).any() or available.lt(published).any():
        raise ValueError("Impossible news timestamp order")
    date_only = published.eq(published.dt.normalize())
    if not news.date_only_delayed.eq(date_only).all():
        raise ValueError("Incorrect date-only flag")
    if available[date_only].lt(published[date_only] + pd.Timedelta(days=1)).any():
        raise ValueError("Date-only publication not conservatively delayed")
    sessions = schedule(str((available.min() - pd.Timedelta(days=7)).date()),
                        str((max(available.max(), pd.Timestamp(finance.date.max(), tz="UTC"))
                             + pd.Timedelta(days=21)).date()))
    positions = pd.DatetimeIndex(sessions.market_close).searchsorted(available, side="left")
    expected = sessions.index[positions].strftime("%Y-%m-%d")
    if not (news.date.to_numpy() == expected).all():
        raise ValueError("News is not assigned to the first eligible NYSE close")
    return sessions, positions


def coverage(news, finance, dates):
    keys = pd.DataFrame({"ticker": news.ticker.to_numpy(), "date": dates}).drop_duplicates()
    paired = finance.merge(keys, on=["ticker", "date"], validate="one_to_one")
    panel = finance.assign(year=finance.date.str[:4])
    paired = paired.assign(year=paired.date.str[:4])
    years = []
    for year, part in panel.groupby("year"):
        observed = paired[paired.year.eq(year)]
        by_ticker = []
        counts = observed.groupby("ticker").size()
        for ticker, total in part.groupby("ticker").size().items():
            count = int(counts.get(ticker, 0))
            by_ticker.append({"ticker": ticker, "finance_sessions": int(total),
                              "news_sessions": count, "coverage_pct": round(100 * count / total, 3)})
        years.append({"year": year, "finance_sessions": len(part),
                      "news_sessions": len(observed),
                      "coverage_pct": round(100 * len(observed) / len(part), 3),
                      "by_ticker": by_ticker})
    return {"finance_sessions": len(finance), "news_sessions": len(paired),
            "coverage_pct": round(100 * len(paired) / len(finance), 3), "by_year": years}


def build_report(news, finance, protocol):
    sessions, positions = integrity(news, finance, protocol["maximum_finance_date"])
    data = news.copy()
    data["headline_key"] = data.headline.map(headline_key)
    articles = data.drop_duplicates("article_id")
    headline_counts = articles.groupby("headline_key").size()
    versions = articles.groupby("url").headline_key.nunique()
    repeated = data.groupby(["ticker", "date", "headline_key"]).size()
    identities = ["headline_key", "url", "article_id"]
    segment_sets = {}
    segment_counts = {}
    for name, (start, end) in protocol["segments"].items():
        part = articles[articles.date.ge(start) & articles.date.lt(end)]
        segment_sets[name] = {key: set(part[key]) for key in identities}
        segment_counts[name] = {"unique_articles": len(part), "unique_headlines": part.headline_key.nunique()}
    overlap = []
    for left, right in combinations(segment_sets, 2):
        overlap.append({"segments": [left, right],
                        **{key: len(segment_sets[left][key] & segment_sets[right][key]) for key in identities}})
    delay = (pd.to_datetime(articles.available_at, utc=True)
             - pd.to_datetime(articles.published_at, utc=True)).dt.total_seconds() / 86400
    baseline = coverage(data, finance, data.date.to_numpy())
    lag_coverage = []
    for lag in protocol["lag_sensitivity_sessions"]:
        dates = sessions.index[positions + lag].strftime("%Y-%m-%d")
        counts = coverage(data, finance, dates)
        lag_coverage.append({"additional_sessions": lag,
                             **{k: v for k, v in counts.items() if k != "by_year"},
                             "by_year": [{k: v for k, v in row.items() if k != "by_ticker"}
                                         for row in counts["by_year"]]})
    return {
        "integrity": "passed",
        "ticker_article_links": len(data), "unique_articles": len(articles),
        "unique_casefolded_headlines": int(articles.headline_key.nunique()),
        "covered_tickers": sorted(data.ticker.unique()),
        "missing_tickers": sorted(set(finance.ticker) - set(data.ticker)),
        "duplication": {
            "headlines_with_multiple_article_ids": int(headline_counts.gt(1).sum()),
            "excess_article_ids_for_identical_headlines": int((headline_counts - 1).clip(lower=0).sum()),
            "urls_with_multiple_headlines": int(versions.gt(1).sum()),
            "ticker_session_headline_groups_with_repeats": int(repeated.gt(1).sum()),
            "excess_links_within_same_ticker_session_headline": int((repeated - 1).clip(lower=0).sum())},
        "timing": {"date_only_articles": int(articles.date_only_delayed.sum()),
                   "availability_delay_days_median": float(delay.median()),
                   "availability_delay_days_p95": float(delay.quantile(.95)),
                   "availability_delay_over_one_day_articles": int(delay.gt(1).sum())},
        "segment_unique_counts": segment_counts, "cross_segment_overlap": overlap,
        "coverage": baseline, "lag_coverage": lag_coverage,
        "interpretation": "Coverage/duplication audit only. Overlap does not prove outcome leakage; article versions remain unverified. No model performance measured.",
    }


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    source_path = REPO / protocol["source_audit"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    news_path = REPO / source["local_text_path"]
    if source["revision"] != protocol["source_revision"] or digest(news_path) != source["normalized_sha256"]:
        raise ValueError("Pinned normalized news provenance mismatch")
    finance_path = REPO / protocol["finance_input"]
    finance = pd.read_csv(finance_path, usecols=protocol["finance_columns"], dtype=str)
    news = pd.read_json(news_path, lines=True, convert_dates=False)
    report = build_report(news, finance, protocol)
    report.update({"audit_id": protocol["audit_id"], "audited_at": pd.Timestamp.now(tz="UTC").isoformat(),
                   "protocol": protocol, "source": source["source"],
                   "input_sha256": {"news": digest(news_path), "finance": digest(finance_path),
                                    "source_audit": digest(source_path), "protocol": digest(PROTOCOL),
                                    "audit_code": digest(Path(__file__))}})
    output = REPO / "docs/research/news_quality_audit_2026-10-07.json"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing["input_sha256"] != report["input_sha256"]:
            raise ValueError("Existing audit has different inputs; use a new versioned audit")
        for key in ["audited_at"]:
            report[key] = existing[key]
        if report != existing:
            raise ValueError("Audit replay differs; preserve existing output")
        print("Existing audit replayed successfully")
    else:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"Saved {output.name}")
    print(json.dumps({key: report[key] for key in ["unique_articles", "duplication", "timing", "cross_segment_overlap"]}, indent=2))


if __name__ == "__main__":
    main()
