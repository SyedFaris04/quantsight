"""Build or validate compact serving data; does not train or change predictions.

From backend:
    python -m research.build_serving_snapshots
    python -m research.build_serving_snapshots --check
"""
import argparse
from pathlib import Path

from serving_snapshot import build_snapshot, check_snapshot

SOURCE = Path(__file__).resolve().parents[1] / "data" / "processed" / "features_sentiment.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the committed snapshot is stale or differs.")
    args = parser.parse_args()
    if args.check:
        rows = check_snapshot(SOURCE)
        print(f"Verified sentiment snapshot: {rows} exact latest rows")
    else:
        result = build_snapshot(SOURCE)
        print(f"Built sentiment snapshot: {result['rows']} rows from {result['source_rows']} source rows")


if __name__ == "__main__":
    main()
