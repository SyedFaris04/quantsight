"""Integrity checks for repository-owned, precomputed native TreeSHAP results."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from serving_snapshot import SERVING_DIR, SnapshotUnavailable, sha256_file

SCHEMA_VERSION = 1
BACKEND = Path(__file__).resolve().parent


def implementation_hash():
    # Code line endings differ between Git platforms; normalize them as text.
    digest = hashlib.sha256()
    for name in ("feature_attribution.py", "cv_utils.py"):
        digest.update(name.encode())
        digest.update((BACKEND / name).read_text(encoding="utf-8").encode("utf-8"))
    return digest.hexdigest()


def artifact_paths(variant, directory=SERVING_DIR):
    return (Path(directory) / f"{variant}_attributions.json",
            Path(directory) / f"{variant}_attributions_manifest.json")


def file_signatures(paths):
    result = []
    for path in paths:
        try:
            stat = path.stat()
            result.append((str(path), stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            result.append((str(path), None, None))
    return tuple(result)


def load_attributions(variant, sources, directory=SERVING_DIR):
    artifact, manifest = artifact_paths(variant, directory)
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if (metadata["schema_version"] != SCHEMA_VERSION or metadata["variant"] != variant
                or metadata["implementation_sha256"] != implementation_hash()
                or metadata["xgboost_version"] != version("xgboost")):
            raise SnapshotUnavailable("Attribution implementation or version is stale.")
        hashes = {path.name: sha256_file(path) for path in sources}
        if hashes != metadata["source_sha256"]:
            raise SnapshotUnavailable("Attribution model, inputs or predictions are stale.")
        if sha256_file(artifact) != metadata["artifact_sha256"]:
            raise SnapshotUnavailable("Attribution artifact hash does not match.")
        rows = json.loads(artifact.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows or len(rows) != metadata["rows"]:
            raise SnapshotUnavailable("Attribution row count is invalid.")
        records = {}
        for row in rows:
            if (row["model"] != variant or row["verified"] is not True
                    or row["method"] != "XGBoost native TreeSHAP"
                    or row["source"] != "historical_saved_prediction"
                    or row["scale"] != "raw_log_odds"
                    or row["model_sha256"] != hashes[sources[0].name]
                    or row["xgboost_version"] != metadata["xgboost_version"]
                    or row["ticker"] in records):
                raise SnapshotUnavailable("Attribution record is inconsistent.")
            records[row["ticker"]] = row
        if sorted(records) != metadata["tickers"]:
            raise SnapshotUnavailable("Attribution ticker set is invalid.")
        return records
    except SnapshotUnavailable:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SnapshotUnavailable("Attribution snapshot could not be loaded.") from exc


def write_attributions(variant, sources, rows, directory=SERVING_DIR):
    artifact, manifest = artifact_paths(variant, directory)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(rows, separators=(",", ":"), allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    artifact_hash = sha256_file(temporary)
    temporary.replace(artifact)
    metadata = {"schema_version": SCHEMA_VERSION, "variant": variant,
                "implementation_sha256": implementation_hash(),
                "source_sha256": {path.name: sha256_file(path) for path in sources},
                "artifact_sha256": artifact_hash, "rows": len(rows),
                "tickers": sorted(row["ticker"] for row in rows),
                "xgboost_version": version("xgboost")}
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(manifest)
    return metadata
