from __future__ import annotations

import unittest

import numpy as np

from tifxyz_doctor.report import _overlay_image


class ReportTests(unittest.TestCase):
    def test_overlay_colors_only_exact_cue_mask(self) -> None:
        scores = np.array([[1.0, 0.49, 0.5, 0.0]], dtype=np.float32)
        report = {
            "_arrays": {
                "review_score": scores,
                "review_cue_mask": np.array([[False, False, True, True]]),
                "valid_cells": np.ones_like(scores, dtype=bool),
                "hole_labels": np.zeros_like(scores, dtype=np.int32),
            },
        }

        pixels = np.asarray(_overlay_image(report))

        np.testing.assert_array_equal(pixels[0, 0], (30, 38, 50))
        np.testing.assert_array_equal(pixels[0, 1], (30, 38, 50))
        np.testing.assert_array_equal(pixels[0, 2], (255, 205, 45))
        np.testing.assert_array_equal(pixels[0, 3], (40, 145, 235))

    def test_invalid_and_hole_colors_override_score_ramp(self) -> None:
        scores = np.ones((1, 2), dtype=np.float32)
        report = {
            "_arrays": {
                "review_score": scores,
                "review_cue_mask": np.ones_like(scores, dtype=bool),
                "valid_cells": np.array([[False, True]]),
                "hole_labels": np.array([[0, 1]], dtype=np.int32),
            }
        }

        pixels = np.asarray(_overlay_image(report))

        np.testing.assert_array_equal(pixels[0, 0], (16, 19, 25))
        np.testing.assert_array_equal(pixels[0, 1], (220, 40, 220))


if __name__ == "__main__":
    unittest.main()
