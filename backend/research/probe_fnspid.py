"""Inspect at most 2 MiB of the pinned FNSPID CSV; save metadata, not article text."""
import csv
import hashlib
import io
import json
import urllib.request
from collections import Counter
from pathlib import Path

REVISION = "bf9189c41527198897d1af3e17b1a0095279fc45"
URL = f"https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/{REVISION}/Stock_news/nasdaq_exteral_data.csv"
CAP = 2 * 1024 * 1024


def main():
    request = urllib.request.Request(URL, headers={"Range": f"bytes=0-{CAP-1}"})
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read(CAP)
        status, content_range = response.status, response.headers.get("Content-Range")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    # A prefix can end mid-record. Exclude the final parsed row unconditionally.
    rows = list(reader)[:-1]
    report = {"url": URL, "revision": REVISION, "http_status": status, "content_range": content_range,
              "bytes_read": len(raw), "prefix_sha256": hashlib.sha256(raw).hexdigest(),
              "columns": reader.fieldnames, "parsed_rows_excluding_last": len(rows),
              "symbol_counts": dict(Counter(row.get("Stock_symbol") for row in rows)),
              "midnight_timestamp_rows": sum("00:00:00" in row.get("Date", "") for row in rows),
              "scope": "Non-random file prefix, useful only for schema and access verification; not a coverage or performance estimate."}
    output = Path(__file__).resolve().parents[2] / "docs/research/fnspid_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
