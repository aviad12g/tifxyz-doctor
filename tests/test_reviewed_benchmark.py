from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tifxyz_doctor.audit import audit_mesh
from tifxyz_doctor.io import TifxyzData
from tifxyz_doctor.reviewed_benchmark import (
    averaged_vertex_normals,
    dilate_cells,
    inject_normal_offset_switch,
    same_wrap_corridor,
)


def plane(height: int = 24, width: int = 28) -> TifxyzData:
    rows, cols = np.indices((height, width), dtype=np.float32)
    coordinates = np.stack([cols, rows, np.full_like(rows, 20.0)], axis=-1)
    return TifxyzData(
        path=Path("/synthetic/reviewed-plane"),
        coordinates=coordinates,
        valid=np.ones((height, width), dtype=bool),
        metadata={"uuid": "reviewed-plane", "scale": [1.0, 1.0]},
        explicit_mask=None,
    )


class ReviewedBenchmarkTests(unittest.TestCase):
    def test_dilation_uses_chebyshev_radius(self) -> None:
        seed = np.zeros((7, 8), dtype=bool)
        seed[3, 4] = True

        dilated = dilate_cells(seed, 2)

        self.assertEqual(int(dilated.sum()), 25)
        self.assertTrue(dilated[1, 2])
        self.assertTrue(dilated[5, 6])
        self.assertFalse(dilated[0, 4])

    def test_corridor_maps_fractional_vertex_locations_to_cells(self) -> None:
        results = {
            "points_list": [
                {
                    "valid": True,
                    "model_locations": [{"h": 4.25, "w": 5.75}],
                },
                {
                    "valid": False,
                    "model_locations": [{"h": 1.0, "w": 1.0}],
                },
            ]
        }

        corridor = same_wrap_corridor(results, (10, 12), radius=0)

        expected = np.zeros((9, 11), dtype=bool)
        expected[3:5, 4:6] = True
        np.testing.assert_array_equal(corridor, expected)

    def test_plane_vertex_normals_are_finite_and_consistent(self) -> None:
        data = plane()

        normals = averaged_vertex_normals(data.coordinates, data.valid)

        self.assertTrue(np.isfinite(normals).all())
        self.assertTrue(np.allclose(np.abs(normals[..., 2]), 1.0))

    def test_zero_offset_is_an_exact_geometry_null(self) -> None:
        data = plane()

        synthetic = inject_normal_offset_switch(
            data,
            offset_voxels=0.0,
            transition_width_cells=4,
        )

        np.testing.assert_array_equal(synthetic.data.coordinates, data.coordinates)
        self.assertGreater(int(synthetic.seam_cells.sum()), 0)
        baseline = audit_mesh(data)
        control = audit_mesh(synthetic.data)
        np.testing.assert_array_equal(
            baseline["_arrays"]["review_cue_mask"],
            control["_arrays"]["review_cue_mask"],
        )
        self.assertEqual(baseline["findings"], control["findings"])

    def test_abrupt_full_winding_proxy_is_detected_at_the_seam(self) -> None:
        data = plane()

        synthetic = inject_normal_offset_switch(
            data,
            offset_voxels=16.0,
            transition_width_cells=1,
        )
        report = audit_mesh(synthetic.data)

        intersection = (
            report["_arrays"]["review_cue_mask"] & synthetic.evaluation_cells
        )
        self.assertTrue(intersection.any())
        self.assertIn(
            "long-edges",
            {finding["code"] for finding in report["findings"]},
        )
        self.assertIn(
            "coherent-normal-step",
            {finding["code"] for finding in report["findings"]},
        )
        self.assertGreater(
            report["geometry"]["coherent_normal_steps"]["coherent_cells"],
            0,
        )

    def test_isolated_normal_outlier_is_not_a_coherent_step(self) -> None:
        data = plane()
        data.coordinates[10, 12, 2] += 8.0

        report = audit_mesh(data)

        self.assertNotIn(
            "coherent-normal-step",
            {finding["code"] for finding in report["findings"]},
        )
        self.assertEqual(
            report["geometry"]["coherent_normal_steps"]["coherent_components"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
