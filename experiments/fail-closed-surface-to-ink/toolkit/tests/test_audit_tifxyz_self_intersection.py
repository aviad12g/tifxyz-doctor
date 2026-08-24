import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_tifxyz_self_intersection import triangle_distance, triangles_intersect


def test_disjoint_parallel_triangles():
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    b = a + np.array([0, 0, 2], dtype=float)
    assert not triangles_intersect(a, b)
    assert np.isclose(triangle_distance(a, b), 2.0)


def test_transverse_intersection():
    a = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=float)
    b = np.array([[0.5, 0.5, -1], [0.5, 0.5, 1], [1.5, 0.5, 0]], dtype=float)
    assert triangles_intersect(a, b)
    assert triangle_distance(a, b) == 0.0


def test_coplanar_overlap_and_separation():
    a = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=float)
    overlap = np.array([[0.2, 0.2, 0], [0.8, 0.2, 0], [0.2, 0.8, 0]], dtype=float)
    separate = overlap + np.array([4, 0, 0], dtype=float)
    assert triangles_intersect(a, overlap)
    assert not triangles_intersect(a, separate)
