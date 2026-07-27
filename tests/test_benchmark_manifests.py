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

    def test_public_empty_resolution_is_complete_and_passing(self) -> None:
        historical, _ = _read_json("benchmarks/public-empty-regressions.json")
        resolution, _ = _read_json(
            "benchmarks/public-empty-resolution-2026-07-27.json"
        )

        self.assertTrue(resolution["all_current_audits_pass"])
        upstream_response = resolution["upstream_response"]
        expected_messages = {
            "diagnosis": (
                "1531206054682165309",
                "2026-07-27T07:46:11.869Z",
            ),
            "remediation": (
                "1531220162190245909",
                "2026-07-27T08:42:15.361Z",
            ),
        }
        for event, (message_id, timestamp) in expected_messages.items():
            self.assertEqual(upstream_response[f"{event}_message_id"], message_id)
            self.assertEqual(
                upstream_response[f"{event}_message_url"],
                "https://discord.com/channels/"
                "1079907749569237093/1243576621722767412/"
                f"{message_id}",
            )
            self.assertEqual(
                upstream_response[f"{event}_message_timestamp"], timestamp
            )
        registry = resolution["current_registry"]
        self.assertEqual(
            registry["url"],
            "https://vesuvius-challenge-open-data.s3.amazonaws.com/metadata.min.json",
        )
        self.assertEqual(registry["observed_last_modified"], "2026-07-27T08:41:44Z")
        self.assertEqual(registry["etag"], "9f439dd6226a8a2de2a318e351506333")
        self.assertEqual(registry["content_encoding"], "gzip")
        self.assertEqual(registry["compressed_bytes"], 57193)
        self.assertEqual(
            registry["compressed_sha256"],
            "68e2e81d993184265774433e659280b96415ff4421d3d07358424ed4522747fd",
        )
        self.assertEqual(registry["decoded_bytes"], 1135989)
        self.assertEqual(
            registry["decoded_sha256"],
            "329bd1ee34ee2c3ecad6e42169e14f0d1a8be36f1e1ed29af942e638960ce14d",
        )

        self.assertEqual(
            {case["historical_case_id"] for case in resolution["resolved_cases"]},
            {case["id"] for case in historical["cases"]},
        )
        self.assertEqual(len(resolution["resolved_cases"]), 3)
        expected = {
            "pherc0332-20240711124827-normalized-empty": {
                "registration": (
                    "tifxyz",
                    "PHerc0332/segments/20240711124827-20240618142020/"
                    "mesh/intermediate/tifxyz_original/",
                ),
                "shape_hw": [126, 1286],
                "vertices": 132462,
                "faces": 130952,
            },
            "pherc0332-20240828190516-normalized-empty": {
                "registration": (
                    "tifxyz",
                    "PHerc0332/segments/20240828190516-20240716140050/"
                    "mesh/intermediate/tifxyz_original/",
                ),
                "shape_hw": [138, 1266],
                "vertices": 134880,
                "faces": 133373,
            },
            "pherc0500p2-20250716055236-normalized-empty": {
                "registration": (
                    "tifxyz-normalized",
                    "PHerc0500P2/segments/"
                    "20250716055236-z_dbg_gen_00356_inp_hr/"
                    "mesh/intermediate/tifxyz_normalized/",
                ),
                "shape_hw": [583, 339],
                "vertices": 113448,
                "faces": 111973,
            },
        }
        for case in resolution["resolved_cases"]:
            expected_case = expected[case["historical_case_id"]]
            registration = case["current_registration"]
            self.assertEqual(
                (registration["type"], registration["prefix"]),
                expected_case["registration"],
            )
            audit = case["current_audit"]
            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["shape_hw"], expected_case["shape_hw"])
            self.assertEqual(
                audit["portable_valid_vertices"], expected_case["vertices"]
            )
            self.assertEqual(audit["portable_valid_faces"], expected_case["faces"])
            self.assertEqual(audit["finding_codes"], [])
            self.assertEqual(
                set(case["files"]), {"meta.json", "x.tif", "y.tif", "z.tif"}
            )
            for specification in case["files"].values():
                self.assertTrue(specification["url"].startswith("https://"))
                self.assertIsInstance(specification["bytes"], int)
                self.assertGreater(specification["bytes"], 0)
                self.assertRegex(specification["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
