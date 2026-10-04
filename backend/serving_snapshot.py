"""Verified compact views of fixed, repository-owned CSV feature data."""
import hashlib
import json
from pathlib import Path

import pandas as pd

SERVING_DIR = Path(__file__).resolve().parent / "data" / "serving"
SCHEMA_VERSION = 1


class SnapshotUnavailable(ValueError):
    """A snapshot is absent, stale or inconsistent; use the original CSV."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_paths(source, directory=SERVING_DIR):
    stem = Path(source).stem + "_latest"
    return Path(directory) / (stem + ".parquet"), Path(directory) / (stem + ".json")


def latest_rows(frame):
    if not {"ticker", "date"}.issubset(frame.columns) or frame.empty:
        raise SnapshotUnavailable("Feature keys are missing.")
    if (frame[["ticker", "date"]].isna().any().any()
            or frame.duplicated(["ticker", "date"]).any()):
        raise SnapshotUnavailable("Feature keys are invalid or duplicated.")
    # Same selection as the existing copilot, retaining original CSV strings.
    return frame.sort_values("date").groupby("ticker", sort=False).tail(1).reset_index(drop=True)


def build_snapshot(source, directory=SERVING_DIR):
    source = Path(source)
    digest = sha256_file(source)
    frame = pd.read_csv(source)
    latest = latest_rows(frame)
    if sha256_file(source) != digest:
        raise SnapshotUnavailable("Source changed while building the snapshot.")
    artifact, manifest = snapshot_paths(source, directory)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    # Publish manifest last. A mixed generation is rejected by its artifact hash.
    temporary = artifact.with_suffix(".parquet.tmp")
    latest.to_parquet(temporary, index=False)
    artifact_hash = sha256_file(temporary)
    temporary.replace(artifact)
    metadata = {"schema_version": SCHEMA_VERSION, "source_filename": source.name,
                "source_sha256": digest, "source_rows": len(frame),
                "artifact_sha256": artifact_hash, "rows": len(latest),
                "columns": list(latest.columns),
                "dtypes": [str(dtype) for dtype in latest.dtypes],
                "selection": "latest_row_by_csv_date_per_ticker"}
    temporary_manifest = manifest.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest)
    return metadata


def load_snapshot(source, directory=SERVING_DIR):
    """Hash the source bytes, but do not parse its complete feature table."""
    source = Path(source)
    artifact, manifest = snapshot_paths(source, directory)
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if (metadata["schema_version"] != SCHEMA_VERSION
                or metadata["source_filename"] != source.name
                or metadata["selection"] != "latest_row_by_csv_date_per_ticker"):
            raise SnapshotUnavailable("Snapshot format does not match this loader.")
        if sha256_file(source) != metadata["source_sha256"]:
            raise SnapshotUnavailable("Snapshot source hash is stale.")
        if sha256_file(artifact) != metadata["artifact_sha256"]:
            raise SnapshotUnavailable("Snapshot artifact hash does not match.")
        latest = pd.read_parquet(artifact)
        if (len(latest) != metadata["rows"] or list(latest.columns) != metadata["columns"]
                or [str(dtype) for dtype in latest.dtypes] != metadata["dtypes"]
                or latest.empty or not {"ticker", "date"}.issubset(latest.columns)
                or latest[["ticker", "date"]].isna().any().any()
                or latest.ticker.duplicated().any()):
            raise SnapshotUnavailable("Snapshot rows or schema are invalid.")
        return latest
    except SnapshotUnavailable:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SnapshotUnavailable("Snapshot could not be loaded.") from exc


def check_snapshot(source, directory=SERVING_DIR):
    """Offline/CI semantic verification against the entire original CSV."""
    latest = load_snapshot(source, directory)
    expected = latest_rows(pd.read_csv(source))
    pd.testing.assert_frame_equal(latest, expected, check_exact=True)
    return len(latest)
