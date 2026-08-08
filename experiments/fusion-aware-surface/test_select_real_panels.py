import numpy as np

from select_real_panels import geometry_normals_at, rank_records, representative_center


def test_panel_ranking_uses_frozen_tiebreaks() -> None:
    records = [
        {"image": "b.tif", "compressed_fraction_le_8": 0.2, "median_spacing": 7.0},
        {"image": "a.tif", "compressed_fraction_le_8": 0.2, "median_spacing": 7.0},
        {"image": "c.tif", "compressed_fraction_le_8": 0.3, "median_spacing": 9.0},
    ]
    assert [record["image"] for record in rank_records(records)] == [
        "c.tif",
        "a.tif",
        "b.tif",
    ]


def test_representative_center_is_a_real_compressed_site() -> None:
    points = np.array([[0, 0, 0], [5, 5, 5], [6, 5, 5], [20, 20, 20]])
    spacing = np.array([40.0, 6.0, 7.0, 40.0])
    assert representative_center(points, spacing) in ([5, 5, 5], [6, 5, 5])


def test_sampled_normals_have_expected_orientation_and_unit_length() -> None:
    gt = np.zeros((21, 21, 21), dtype=bool)
    gt[9:12] = True
    points = np.array([[9, 10, 10], [11, 10, 10]])
    normals = geometry_normals_at(gt, points)
    assert normals.shape == (2, 3)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-4)
    assert normals[0, 0] < 0
    assert normals[1, 0] > 0
