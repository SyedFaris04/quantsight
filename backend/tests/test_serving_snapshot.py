import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import copilot_engine as copilot
from serving_snapshot import (SnapshotUnavailable, build_snapshot, check_snapshot,
                              load_snapshot, sha256_file, snapshot_paths)


class ServingSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(copilot.clear_data_caches)
        copilot.clear_data_caches()
        self.source = Path(self.temp.name) / "features.csv"
        self.directory = Path(self.temp.name) / "serving"
        self.frame = pd.DataFrame([
            {"ticker": "AAA", "date": "2024-01-03", "Close": 11.123456789012345, "optional": None},
            {"ticker": "BBB", "date": "2024-02-01", "Close": 15.3, "optional": 0.1},
            {"ticker": "AAA", "date": "2024-01-02", "Close": 10., "optional": 0.2},
        ])
        self.frame.to_csv(self.source, index=False)

    def build(self):
        return build_snapshot(self.source, self.directory)

    def test_latest_rows_preserve_exact_csv_precision_nulls_and_original_dates(self):
        metadata = self.build()
        self.assertEqual(metadata["rows"], 2)
        self.assertEqual(check_snapshot(self.source, self.directory), 2)
        with patch("serving_snapshot.pd.read_csv", side_effect=AssertionError("full CSV parse")):
            latest = load_snapshot(self.source, self.directory)
        self.assertEqual(latest.date.tolist(), ["2024-01-03", "2024-02-01"])

    def test_source_change_same_size_is_rejected(self):
        self.build()
        content = self.source.read_bytes()
        self.source.write_bytes(content.replace(b"15.3", b"15.4"))
        self.assertEqual(len(content), self.source.stat().st_size)
        with self.assertRaisesRegex(SnapshotUnavailable, "stale"):
            load_snapshot(self.source, self.directory)

    def test_corrupt_artifact_and_manifest_are_rejected(self):
        self.build()
        artifact, manifest = snapshot_paths(self.source, self.directory)
        artifact.write_bytes(b"invalid parquet")
        with self.assertRaisesRegex(SnapshotUnavailable, "artifact hash"):
            load_snapshot(self.source, self.directory)
        manifest.write_text("{}", encoding="utf-8")
        with self.assertRaises(SnapshotUnavailable):
            load_snapshot(self.source, self.directory)

    def test_offline_check_detects_wrong_selection_even_with_matching_hashes(self):
        self.build()
        artifact, manifest = snapshot_paths(self.source, self.directory)
        wrong = pd.read_parquet(artifact)
        wrong.loc[0, "date"] = "2024-01-02"
        wrong.to_parquet(artifact, index=False)
        metadata = json.loads(manifest.read_text())
        metadata["artifact_sha256"] = sha256_file(artifact)
        manifest.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(AssertionError):
            check_snapshot(self.source, self.directory)

    def test_duplicate_source_keys_fail_build(self):
        pd.concat([self.frame, self.frame.iloc[:1]]).to_csv(self.source, index=False)
        with self.assertRaisesRegex(SnapshotUnavailable, "duplicated"):
            self.build()

    def test_missing_or_stale_snapshot_falls_back_to_current_csv(self):
        with patch.object(copilot, "load_snapshot", side_effect=SnapshotUnavailable("stale")):
            latest = copilot._latest_features_cached(self.source)
        expected = pd.read_csv(self.source).sort_values("date").groupby("ticker").tail(1).set_index("ticker", drop=False)
        pd.testing.assert_frame_equal(latest, expected)

    def test_valid_snapshot_is_used_without_csv_parse(self):
        self.build()
        with patch.object(copilot, "load_snapshot", side_effect=lambda path: load_snapshot(path, self.directory)), \
             patch.object(copilot.pd, "read_csv", side_effect=AssertionError("full CSV parse")):
            latest = copilot._latest_features_cached(self.source)
        self.assertEqual(len(latest), 2)
