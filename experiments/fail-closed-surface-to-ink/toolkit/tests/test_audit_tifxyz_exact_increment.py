from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from audit_tifxyz_exact_increment import audit


def _write_surface(directory: Path, x_shift: float, mask: np.ndarray) -> None:
    directory.mkdir()
    rows, columns = np.indices(mask.shape, dtype=np.float32)
    tifffile.imwrite(directory / "x.tif", columns + np.float32(x_shift))
    tifffile.imwrite(directory / "y.tif", rows)
    tifffile.imwrite(directory / "z.tif", np.full(mask.shape, 10, dtype=np.float32))
    tifffile.imwrite(directory / "mask.tif", mask.astype(np.uint8) * 255)
    (directory / "meta.json").write_text(
        json.dumps({"format": "tifxyz", "type": "seg", "scale": [1, 1]}) + "\n"
    )


def test_exact_increment_separates_shared_and_new_quads(tmp_path: Path) -> None:
    old_mask = np.ones((3, 3), dtype=bool)
    new_mask = old_mask.copy()
    new_mask[0, 0] = False
    old = tmp_path / "old"
    new = tmp_path / "new"
    _write_surface(old, 0.0, old_mask)
    _write_surface(new, 0.0, new_mask)

    report = audit(old, new, voxel_um=10.0)

    assert report["inputs"]["old"]["active_quad_count"] == 4
    assert report["inputs"]["new"]["active_quad_count"] == 3
    assert report["result"]["shared_exact_quad_count"] == 3
    assert report["result"]["new_only_exact_quad_count"] == 0
    assert report["result"]["old_only_exact_quad_count"] == 1


def test_exact_increment_does_not_infer_tolerance_equivalence(tmp_path: Path) -> None:
    mask = np.ones((2, 2), dtype=bool)
    old = tmp_path / "old"
    shifted = tmp_path / "shifted"
    _write_surface(old, 0.0, mask)
    _write_surface(shifted, 1e-4, mask)

    report = audit(old, shifted, voxel_um=10.0)

    assert report["result"]["shared_exact_quad_count"] == 0
    assert report["result"]["new_only_exact_quad_count"] == 1
    assert report["result"]["old_only_exact_quad_count"] == 1
