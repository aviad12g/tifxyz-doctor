from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from tifxyz_doctor.audit import (
    AuditConfig,
    _project_cell_adjacency_flags,
    _project_edge_flags_to_cells,
    audit_mesh,
    public_report,
)
from tifxyz_doctor.io import TifxyzData


def plane(
    height: int = 6,
    width: int = 7,
    *,
    valid: np.ndarray | None = None,
    explicit_mask: bool = False,
) -> TifxyzData:
    rows, cols = np.indices((height, width), dtype=np.float64)
    coordinates = np.stack([cols, rows, np.full_like(rows, 10.0)], axis=-1)
    effective_valid = np.ones((height, width), dtype=bool) if valid is None else valid
    return TifxyzData(
        path=Path("/synthetic/plane"),
        coordinates=coordinates,
        valid=effective_valid,
        metadata={"uuid": "synthetic-plane", "scale": [1.0, 1.0]},
        explicit_mask=effective_valid.copy() if explicit_mask else None,
    )


class AuditTests(unittest.TestCase):
    def test_clean_plane_has_expected_geometry(self) -> None:
        report = audit_mesh(plane())

        self.assertEqual(report["integrity"]["valid_vertex_count"], 42)
        self.assertEqual(report["topology"]["valid_quad_count"], 30)
        self.assertEqual(report["geometry"]["surface_area_voxel2"], 30.0)
        self.assertEqual(report["geometry"]["folded_quads"], 0)
        self.assertEqual(report["geometry"]["long_edges"], 0)
        self.assertEqual(report["geometry"]["short_edges"], 0)
        self.assertEqual(report["nonlocal_proximity"]["pair_count"], 0)
        self.assertEqual(report["findings"], [])
        self.assertTrue(np.isfinite(report["_arrays"]["review_score"]).all())
        self.assertFalse(report["_arrays"]["review_cue_mask"].any())

    def test_edge_and_adjacency_flags_project_to_both_incident_cells(self) -> None:
        target = np.zeros((3, 4), dtype=bool)
        horizontal_edges = np.zeros((4, 4), dtype=bool)
        vertical_edges = np.zeros((3, 5), dtype=bool)
        horizontal_edges[2, 1] = True
        vertical_edges[1, 3] = True

        _project_edge_flags_to_cells(target, horizontal_edges, vertical_edges)

        expected = np.zeros_like(target)
        expected[1, 1] = True
        expected[2, 1] = True
        expected[1, 2] = True
        expected[1, 3] = True
        np.testing.assert_array_equal(target, expected)

        target[:] = False
        horizontal_neighbors = np.zeros((3, 3), dtype=bool)
        vertical_neighbors = np.zeros((2, 4), dtype=bool)
        horizontal_neighbors[1, 1] = True
        vertical_neighbors[0, 3] = True

        _project_cell_adjacency_flags(
            target,
            horizontal_neighbors,
            vertical_neighbors,
        )

        expected[:] = False
        expected[1, 1] = True
        expected[1, 2] = True
        expected[0, 3] = True
        expected[1, 3] = True
        np.testing.assert_array_equal(target, expected)

    def test_enclosed_invalid_region_is_localized(self) -> None:
        valid = np.ones((7, 7), dtype=bool)
        valid[3, 3] = False
        data = plane(7, 7, valid=valid)
        data.coordinates[3, 3] = -1.0

        report = audit_mesh(data)

        self.assertEqual(report["topology"]["enclosed_invalid_regions"], 1)
        self.assertEqual(report["topology"]["enclosed_invalid_region_sizes"], [4])
        self.assertIn("enclosed-gaps", {finding["code"] for finding in report["findings"]})
        self.assertFalse(
            (
                report["_arrays"]["review_cue_mask"]
                & (report["_arrays"]["hole_labels"] > 0)
            ).any()
        )

    def test_folded_quad_is_detected(self) -> None:
        data = plane(2, 2)
        data.coordinates[1, 1] = (-1.0, 1.0, 10.0)

        report = audit_mesh(data)

        self.assertEqual(report["geometry"]["folded_quads"], 1)
        self.assertIn("folded-quads", {finding["code"] for finding in report["findings"]})
        self.assertEqual(float(report["_arrays"]["review_score"][0, 0]), 1.0)
        self.assertTrue(report["_arrays"]["review_cue_mask"][0, 0])

    def test_distant_grid_points_that_touch_in_3d_are_detected(self) -> None:
        data = plane(12, 12)
        data.coordinates[11, 11] = data.coordinates[0, 0] + (0.1, 0.0, 0.0)
        config = AuditConfig(
            nonlocal_uv_exclusion=8,
            max_proximity_points=10_000,
            max_proximity_pairs=0,
        )

        report = audit_mesh(data, config)

        self.assertGreaterEqual(report["nonlocal_proximity"]["pair_count"], 1)
        self.assertEqual(report["nonlocal_proximity"]["pairs"], [])
        self.assertTrue(report["nonlocal_proximity"]["pairs_truncated"])
        self.assertIn("nonlocal-proximity", {finding["code"] for finding in report["findings"]})
        self.assertTrue(report["_arrays"]["review_cue_mask"][0, 0])
        self.assertTrue(report["_arrays"]["review_cue_mask"][-1, -1])

    def test_sparse_stride_respects_proximity_point_limit(self) -> None:
        data = plane(20, 20)
        data.valid[:] = False
        data.valid[::2, ::2] = True
        config = AuditConfig(
            expected_spacing_x=1.0,
            expected_spacing_y=1.0,
            max_proximity_points=25,
        )

        report = audit_mesh(data, config)

        self.assertEqual(report["nonlocal_proximity"]["sampled_points"], 25)
        self.assertTrue(report["nonlocal_proximity"]["sampled_points_capped"])

    def test_integrity_errors_are_separate_from_review_cues(self) -> None:
        data = plane(4, 4, explicit_mask=True)
        data.coordinates[1, 1, 0] = -1.0
        data.coordinates[2, 2] = np.nan

        report = audit_mesh(data)
        findings = {finding["code"]: finding["level"] for finding in report["findings"]}

        self.assertEqual(report["integrity"]["partial_sentinel_vertices"], 1)
        self.assertEqual(report["integrity"]["nonfinite_valid_vertices"], 1)
        self.assertEqual(findings["partial-sentinel"], "error")
        self.assertEqual(findings["nonfinite-valid"], "error")

    def test_empty_surface_remains_strict_json(self) -> None:
        data = plane(1, 1, valid=np.zeros((1, 1), dtype=bool))
        data.coordinates[:] = -1.0
        data.metadata.pop("scale")

        report = audit_mesh(data)
        encoded = json.dumps(public_report(report), allow_nan=False)

        self.assertIn('"valid_vertex_count": 0', encoded)
        self.assertIsNone(report["geometry"]["observed_spacing_x"])
        self.assertEqual(report["nonlocal_proximity"]["sampled_points"], 0)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "long_edge_ratio"):
            audit_mesh(plane(), AuditConfig(long_edge_ratio=0.2, short_edge_ratio=0.35))

    def test_explicit_target_exposes_global_scale_and_anisotropy(self) -> None:
        data = plane(8, 8)
        data.coordinates[..., 0] *= 1.4
        data.coordinates[..., 1] *= 1.3
        report = audit_mesh(
            data,
            AuditConfig(expected_spacing_x=1.0, expected_spacing_y=1.0),
        )

        target = report["geometry"]["target_spacing"]
        self.assertAlmostEqual(target["isotropic_factor"], np.sqrt(1.4 * 1.3))
        self.assertAlmostEqual(target["anisotropy_factor"], 1.4 / 1.3)
        codes = {finding["code"] for finding in report["findings"]}
        self.assertIn("global-spacing-drift", codes)
        self.assertNotIn("global-spacing-anisotropy", codes)

    def test_target_relative_shear_is_detected(self) -> None:
        data = plane(8, 8)
        rows, cols = np.indices((8, 8), dtype=np.float64)
        data.coordinates[..., 0] = cols + 3.0 * rows
        data.coordinates[..., 1] = rows
        report = audit_mesh(
            data,
            AuditConfig(expected_spacing_x=1.0, expected_spacing_y=1.0),
        )

        self.assertGreater(report["geometry"]["condition_number"]["p50"], 10.0)
        self.assertGreater(report["geometry"]["high_shear_cells"], 0)
        self.assertIn("high-shear", {finding["code"] for finding in report["findings"]})

    def test_collapsed_edge_is_localized_and_json_safe(self) -> None:
        data = plane(4, 4)
        data.coordinates[1, 1] = data.coordinates[1, 0]

        report = audit_mesh(data)
        encoded = json.dumps(public_report(report), allow_nan=False)

        self.assertGreaterEqual(report["geometry"]["short_edges"], 1)
        self.assertGreaterEqual(report["geometry"]["degenerate_triangles"], 1)
        self.assertIn('"nonfinite": "nan"', encoded)

    def test_global_reflection_does_not_create_local_flip_cues(self) -> None:
        data = plane(8, 8)
        data.coordinates[..., 0] *= -1.0

        report = audit_mesh(data)

        self.assertEqual(report["geometry"]["folded_quads"], 0)
        self.assertEqual(report["geometry"]["normal_jumps"]["orientation_flips"], 0)

    def test_gradual_cylinder_does_not_trigger_jump_threshold(self) -> None:
        height, width = 10, 73
        rows, cols = np.indices((height, width), dtype=np.float64)
        angle = np.deg2rad(cols * 5.0)
        radius = 1.0 / (2.0 * np.sin(np.deg2rad(2.5)))
        coordinates = np.stack(
            [
                radius * np.sin(angle),
                rows,
                100.0 + radius * np.cos(angle),
            ],
            axis=-1,
        )
        data = TifxyzData(
            path=Path("/synthetic/cylinder"),
            coordinates=coordinates,
            valid=np.ones((height, width), dtype=bool),
            metadata={"uuid": "cylinder"},
            explicit_mask=None,
        )

        report = audit_mesh(data)

        self.assertEqual(report["geometry"]["normal_jumps"]["jumps_above_threshold"], 0)
        self.assertEqual(report["geometry"]["folded_quads"], 0)
        # The closed seam intentionally triggers nonlocal proximity, but its
        # ordinary five-degree normal variation must stay outside the cue mask.
        scores = report["_arrays"]["review_score"]
        cues = report["_arrays"]["review_cue_mask"]
        self.assertFalse((cues & (scores < 1.0)).any())


if __name__ == "__main__":
    unittest.main()
