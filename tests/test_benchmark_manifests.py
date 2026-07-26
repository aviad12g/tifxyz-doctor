from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(relative_path: str) -> tuple[dict, bytes]:
    payload = (PROJECT_ROOT / relative_path).read_bytes()
    return json.loads(payload), payload


class BenchmarkManifestTests(unittest.TestCase):
    def test_realdata_snapshot_matches_manifest_identity(self) -> None:
        manifest, manifest_bytes = _read_json("benchmarks/realdata-smoke.json")
        snapshot, _ = _read_json("benchmarks/realdata-results-v0.1.0.json")

        self.assertEqual(
            snapshot["benchmark_manifest_sha256"],
            hashlib.sha256(manifest_bytes).hexdigest(),
        )
        self.assertEqual(
            [item["id"] for item in snapshot["observations"]],
            [item["id"] for item in manifest["cases"]],
        )

    def test_public_regression_snapshot_matches_manifest_identity(self) -> None:
        manifest, manifest_bytes = _read_json(
            "benchmarks/public-empty-regressions.json"
        )
        snapshot, _ = _read_json("benchmarks/public-empty-results-v0.1.0.json")

        self.assertTrue(snapshot["all_expectations_pass"])
        self.assertEqual(
            snapshot["benchmark_manifest_sha256"],
            hashlib.sha256(manifest_bytes).hexdigest(),
        )
        self.assertEqual(
            [item["id"] for item in snapshot["observations"]],
            [item["id"] for item in manifest["cases"]],
        )

    def test_download_specs_are_https_and_content_addressed(self) -> None:
        for relative_path in (
            "benchmarks/realdata-smoke.json",
            "benchmarks/public-empty-regressions.json",
        ):
            manifest, _ = _read_json(relative_path)
            for case in manifest["cases"]:
                self.assertTrue(case["base_url"].startswith("https://"))
                for specification in case["files"].values():
                    self.assertGreater(int(specification["bytes"]), 0)
                    self.assertRegex(specification["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
