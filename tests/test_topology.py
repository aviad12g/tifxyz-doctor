from __future__ import annotations

import unittest

import numpy as np

from tifxyz_doctor.topology import enclosed_invalid_regions, label_components_4, valid_quad_mask


class TopologyTests(unittest.TestCase):
    def test_diagonal_pixels_are_distinct_components(self) -> None:
        mask = np.array([[True, False], [False, True]])
        labels, sizes = label_components_4(mask)

        self.assertEqual(sorted(sizes), [1, 1])
        self.assertNotEqual(labels[0, 0], labels[1, 1])

    def test_only_interior_invalid_components_are_holes(self) -> None:
        valid = np.ones((6, 7), dtype=bool)
        valid[0, 1] = False
        valid[1, 1] = False
        valid[3, 3:5] = False

        labels, sizes = enclosed_invalid_regions(valid)

        self.assertEqual(sizes, [2])
        self.assertEqual(labels[0, 1], 0)
        self.assertGreater(labels[3, 3], 0)

    def test_valid_quad_requires_all_four_vertices(self) -> None:
        valid = np.ones((3, 3), dtype=bool)
        valid[1, 1] = False

        quads = valid_quad_mask(valid)

        self.assertEqual(quads.shape, (2, 2))
        self.assertFalse(quads.any())


if __name__ == "__main__":
    unittest.main()
