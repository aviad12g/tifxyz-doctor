from __future__ import annotations

import json
import unittest
from pathlib import Path


class ReaderDifferentialSnapshotTests(unittest.TestCase):
    def test_recorded_reader_differential_has_controls_and_three_disagreements(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        result = json.loads(
            (root / "verification" / "reader-differential-results-v1.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(
            result["summary"],
            {
                "case_count": 5,
                "failed": 0,
                "passed": 5,
                "exporter_probe_count": 8,
                "exporter_probe_failed": 0,
                "exporter_probe_passed": 8,
            },
        )
        self.assertTrue(
            result["villa_cpp"]["load_quad_from_tifxyz_symbol_present"]
        )
        self.assertEqual(
            result["villa_cpp"]["binary_sha256"],
            "71a32235924499dee93bd24944794c930dffdee8e95450220e2665bd2020e118",
        )
        self.assertEqual(
            result["villa_cpp"]["release_asset_sha256"],
            "d5fca566c8bc843f2350ab60e86e0def25ab88240e3f0ffcbd04662ba522c9a8",
        )
        self.assertEqual(
            result["villa_cpp"]["release_source"]["converter"]["blob_sha"],
            "eff32a4d0984bd6614308a4484f8dd70cb699d1a",
        )
        self.assertEqual(
            result["villa_cpp"]["release_source"]["geometry"]["blob_sha"],
            "edfec27becdda604281ad3d2feae26630d4ea62a",
        )
        self.assertIn(
            "separate read-only repository-history check",
            result["villa_cpp"]["release_source"]["verification_scope"],
        )
        cases = {item["name"]: item for item in result["cases"]}
        self.assertEqual(
            set(cases),
            {
                "no-mask-all-positive",
                "no-mask-nonpositive-z",
                "exact-mask-low-nonzero",
                "exact-mask-revalidates-nonpositive-z",
                "integer-multiple-mask",
            },
        )

        all_valid = cases["no-mask-all-positive"]["actual"]
        known_invalid = cases["no-mask-nonpositive-z"]["actual"]
        self.assertEqual(
            (all_valid["stable_cpp_obj_vertices"], all_valid["stable_cpp_obj_faces"]),
            (16, 18),
        )
        self.assertEqual(all_valid["stable_cpp_loaded_grid_shape_hw"], [5, 5])
        self.assertEqual(
            (
                known_invalid["stable_cpp_obj_vertices"],
                known_invalid["stable_cpp_obj_faces"],
            ),
            (12, 10),
        )

        for name in (
            "exact-mask-low-nonzero",
            "exact-mask-revalidates-nonpositive-z",
            "integer-multiple-mask",
        ):
            actual = cases[name]["actual"]
            self.assertEqual(actual["stable_cpp_loaded_grid_shape_hw"], [5, 5])
            self.assertEqual(actual["pinned_python_reader_valid_vertices"], 25)
            self.assertEqual(actual["doctor_modeled_python_valid_vertices"], 25)
            self.assertEqual(actual["doctor_modeled_cpp_finite_vertices"], 24)
            self.assertEqual(
                (
                    actual["stable_cpp_obj_vertices"],
                    actual["stable_cpp_obj_faces"],
                ),
                (12, 10),
            )

        probe = result["exporter_boundary_probe"]
        self.assertEqual(probe["status"], "pass")
        observations = probe["observations"]
        self.assertEqual(
            probe["domain"],
            "all-valid grids with height >= 3 and width >= 3",
        )
        self.assertEqual(
            [item["input_grid_shape_hw"] for item in observations],
            [
                [3, 3],
                [4, 4],
                [5, 5],
                [6, 6],
                [7, 7],
                [3, 4],
                [4, 7],
                [7, 4],
            ],
        )
        for item in observations:
            self.assertTrue(item["passed"])
            self.assertEqual(item["actual"], item["expected"])


if __name__ == "__main__":
    unittest.main()
