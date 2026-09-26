"""Acquire and audit a pinned, manageable historical news research candidate.

Run from backend: python -m research.news_dataset
Downloads data only, not executable repository code. Raw/derived text stays in
ignored data/research; the versioned report contains metadata and aggregate counts.
"""
import argparse
from collections import Counter
import hashlib
import json
import lzma
from pathlib import Path
import urllib.request

import pandas as pd
from market_calendar import schedule

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "FelixDrinkall/financial-news-dataset"
REVISION = "3d87f0a8140d6517619d56ad9014f7f6c68abdcb"
SOURCE_URL = f"https://github.com/{REPOSITORY}/tree/{REVISION}"
LICENSE = "CC-BY-NC-SA-4.0"
MAX_FILE_BYTES = 48 * 1024 * 1024
MAX_EXPANDED_CHARACTERS = 2 * 1024 * 1024 * 1024


def utc_timestamp(value, allow_naive=False):
    if value is None or value == "":
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        if not allow_naive:
            raise ValueError("timestamp_missing_timezone")
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def normalize_article(row, universe):
    """Whitelist inputs. Ignore supplied sentiment, emotions and ALL price fields.

    The provider documents publication time as UTC even where serialized without
    an offset. Download time must include an offset. Use the latest known source
    timestamp conservatively; this is archive-availability, not proof of original
    publication-time trading access. Only explicit mentions become ticker links.
    """
    if row.get("language") != "en":
        return [], "not_english"
    title, url = row.get("title"), row.get("url")
    if not isinstance(title, str) or not title.strip() or not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return [], "missing_headline_or_url"
    try:
        published = utc_timestamp(row.get("date_publish"), allow_naive=True)
        downloaded = utc_timestamp(row.get("date_download"))
        modified = utc_timestamp(row.get("date_modify"))
    except (TypeError, ValueError, OverflowError):
        return [], "invalid_or_ambiguous_timestamp"
    if published is None or downloaded is None:
        return [], "missing_publication_or_archive_time"
    if downloaded < published:
        return [], "archive_precedes_publication"
    # A midnight timestamp may represent date-only precision; delay it until
    # the end of that UTC day rather than assuming morning availability.
    date_only = published == published.normalize()
    publication_bound = published + pd.Timedelta(days=1) if date_only else published
    available = max(t for t in [publication_bound, downloaded, modified] if t is not None)
    mentions = row.get("mentioned_companies")
    if not isinstance(mentions, list):
        return [], "missing_explicit_mentions"
    tickers = sorted({t for t in mentions if isinstance(t, str)} & set(universe))
    if not tickers:
        return [], "outside_universe"
    normalized_title = " ".join(title.split())
    identity = hashlib.sha256((url + "\n" + normalized_title).encode()).hexdigest()
    return [{"article_id": identity, "ticker": ticker, "headline": normalized_title,
             "url": url, "published_at": published.isoformat(),
             "available_at": available.isoformat(), "archive_at": downloaded.isoformat(),
             "availability_basis": "max_source_publication_archive_modification",
             "date_only_delayed": date_only,
             "publication_timezone_policy": "UTC_as_documented_by_source",
             "source_dataset": REPOSITORY, "source_revision": REVISION}
            for ticker in tickers], None


def assign_sessions(frame):
    if frame.empty:
        return frame.assign(date=pd.Series(dtype=str))
    available = pd.to_datetime(frame.available_at, utc=True)
    first = str((available.min() - pd.Timedelta(days=1)).date())
    last = str((available.max() + pd.Timedelta(days=14)).date())
    sessions = schedule(first, last)
    close_times = pd.DatetimeIndex(sessions.market_close)
    positions = close_times.searchsorted(available, side="left")
    result = frame.copy()
    result["date"] = sessions.index[positions].strftime("%Y-%m-%d")
    return result


def fetch_bytes(url, cap):
    request = urllib.request.Request(url, headers={"User-Agent": "QuantSight-academic-dataset-audit/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        chunks, size = [], 0
        while chunk := response.read(min(65536, cap + 1 - size)):
            chunks.append(chunk)
            size += len(chunk)
            if size > cap:
                raise ValueError("Download exceeds the declared size limit")
        return b"".join(chunks)


def acquire(output, years):
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(fetch_bytes(f"https://api.github.com/repos/{REPOSITORY}/contents/data?ref={REVISION}", 1024 * 1024))
    selected = [item for item in metadata if item["name"] in {f"{year}_processed.json.xz" for year in years}]
    if len(selected) != len(set(years)) or sum(item["size"] for item in selected) > 128 * 1024 * 1024:
        raise ValueError("Missing expected files or dataset exceeds download budget")
    manifest = []
    for item in selected:
        if item["size"] > MAX_FILE_BYTES:
            raise ValueError("Source file exceeds download budget")
        path = output / item["name"]
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/data/{item['name']}"
        content = path.read_bytes() if path.exists() else fetch_bytes(url, MAX_FILE_BYTES)
        git_hash = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if len(content) != item["size"] or git_hash != item["sha"]:
            raise ValueError(f"Source checksum mismatch: {item['name']}")
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(content)
        manifest.append({"file": path.name, "url": url, "bytes": len(content),
                         "sha256": hashlib.sha256(content).hexdigest(), "git_blob_sha1": git_hash})
        print(f"Verified {path.name}: {len(content):,} bytes", flush=True)
    license_path = output / "SOURCE_LICENSE.txt"
    if not license_path.exists():
        license_path.write_bytes(fetch_bytes(f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/LICENSE", 100000))
    return manifest


def stream_json_array(handle, chunk_size=65536):
    """Read large source arrays without loading their price-enriched bodies into RAM."""
    decoder, buffer, total, eof = json.JSONDecoder(), "", 0, False
    state = "start"
    while True:
        buffer = buffer.lstrip()
        if state == "start" and buffer:
            if buffer[0] != "[":
                raise ValueError("Expected a JSON array")
            buffer, state = buffer[1:], "value_or_end"
            continue
        if state in ("value_or_end", "value") and buffer:
            if buffer[0] == "]" and state == "value_or_end":
                buffer, state = buffer[1:], "done"
                continue
            try:
                row, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError("Truncated or invalid source JSON")
            else:
                if not isinstance(row, dict):
                    raise ValueError("Expected article objects")
                buffer, state = buffer[end:], "separator"
                yield row
                continue
        if state == "separator" and buffer:
            if buffer[0] == ",":
                buffer, state = buffer[1:], "value"
                continue
            if buffer[0] == "]":
                buffer, state = buffer[1:], "done"
                continue
            raise ValueError("Expected array delimiter")
        if state == "done":
            if buffer.strip():
                raise ValueError("Unexpected trailing JSON data")
            if eof:
                return
        elif eof:
            raise ValueError("Truncated source array")
        if len(buffer) > 16 * 1024 * 1024:
            raise ValueError("Single source record exceeds size limit")
        chunk = handle.read(chunk_size)
        total += len(chunk)
        if total > MAX_EXPANDED_CHARACTERS:
            raise ValueError("Expanded data exceeds size limit")
        eof = not chunk
        buffer += chunk


def audit(output, manifest, report_path):
    finance = pd.read_csv(ROOT / "data/processed/features_finance.csv", usecols=["ticker", "date"])
    universe = set(finance.ticker)
    records, rejected, years = [], Counter(), {}
    for item in manifest:
        count = 0
        with lzma.open(output / item["file"], "rt", encoding="utf-8") as handle:
            for row in stream_json_array(handle):
                count += 1
                linked, reason = normalize_article(row, universe)
                records.extend(linked)
                if reason:
                    rejected[reason] += 1
        years[item["file"][:4]] = count
        print(f"Audited {item['file']}: {count:,} articles", flush=True)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No usable ticker-linked timestamped news")
    before = len(frame)
    frame = frame.sort_values("available_at").drop_duplicates(["ticker", "article_id"], keep="first")
    frame = assign_sessions(frame)
    frame.to_json(output / "normalized_news.jsonl", orient="records", lines=True, force_ascii=False)
    keys = frame[["ticker", "date"]].drop_duplicates()
    paired = finance.merge(keys, on=["ticker", "date"])
    delay = (pd.to_datetime(frame.available_at, utc=True) - pd.to_datetime(frame.published_at, utc=True)).dt.total_seconds() / 86400
    report = {"source": SOURCE_URL, "revision": REVISION, "license_declared": LICENSE,
              "audited_at": pd.Timestamp.now(tz="UTC").isoformat(), "source_files": manifest,
              "source_rows_by_file_year": years, "source_rows": sum(years.values()),
              "rejected_source_rows": dict(rejected), "retained_ticker_article_links": len(frame),
              "duplicate_links_removed": before-len(frame), "covered_tickers": sorted(frame.ticker.unique()),
              "missing_universe_tickers": sorted(universe-set(frame.ticker)),
              "first_available_session": frame.date.min(), "last_available_session": frame.date.max(),
              "matching_finance_ticker_sessions": len(paired),
              "finance_ticker_sessions": len(finance), "coverage_pct": round(len(paired)/len(finance)*100, 3),
              "matched_sessions_by_year": paired.groupby(paired.date.str[:4]).size().to_dict(),
              "matched_sessions_by_ticker": paired.groupby("ticker").size().to_dict(),
              "archive_delay_days_median": float(delay.median()), "archive_delay_days_p95": float(delay.quantile(.95)),
              "normalized_sha256": hashlib.sha256((output / "normalized_news.jsonl").read_bytes()).hexdigest(),
              "future_price_fields_imported": False, "precomputed_sentiment_imported": False,
              "assessment": "Candidate for exploratory historical ablation, not a fresh final test. Archive timing is conservative and still needs source review.",
              "local_text_path": str((output / "normalized_news.jsonl").relative_to(ROOT.parent)).replace("\\", "/")}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2017, 2024)))
    args = parser.parse_args()
    output = ROOT / "data/research/drinkall" / REVISION
    manifest = acquire(output, args.years)
    report = audit(output, manifest, ROOT.parent / "docs/research/news_candidate_audit.json")
    print(json.dumps({k: report[k] for k in ["source_rows", "retained_ticker_article_links", "matching_finance_ticker_sessions", "coverage_pct", "archive_delay_days_median"]}, indent=2))


if __name__ == "__main__":
    main()
