from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from native_surface_sampler import SurfaceGrid, estimate_surface_normals, sample_trilinear_zyx  # noqa: E402
from sweep_public_m7_uniform_offsets import (  # noqa: E402
    evaluate_shifted_surface,
    materialize_corrected_copy,
    shift_rank_key,
    shifted_surface_float32,
)
from tifxyz_render_pipeline import load_tifxyz_asset  # noqa: E402


class MemorySampler:
    def __init__(self, volume: np.ndarray) -> None:
        self.volume = volume

    def sample(self, coordinates_xyz: np.ndarray, *, point_valid: np.ndarray):
        result = sample_trilinear_zyx(
            self.volume, coordinates_xyz, point_valid=point_valid
        )
        return result.values, result.valid


def plane_surface(z_value: float = 5.0) -> SurfaceGrid:
    rows, columns = np.mgrid[:8, :8]
    return SurfaceGrid.from_tifxyz(
        columns.astype(float) + 6,
        rows.astype(float) + 6,
        np.full((8, 8), z_value),
        mask=np.ones((8, 8), dtype=bool),
    )


def test_uniform_shift_salvages_synthetic_parallel_plane() -> None:
    volume = np.zeros((40, 40, 40), dtype=np.uint8)
    volume[20, :, :] = 255
    original = plane_surface(15.0)
    normals = estimate_surface_normals(original).positive_xyz
    generations = np.ones((8, 8), dtype=np.uint16)
    unshifted = shifted_surface_float32(original, normals, 0)
    shifted = shifted_surface_float32(original, normals, 5)
    before, _, _ = evaluate_shifted_surface(
        unshifted,
        MemorySampler(volume),
        shift_voxels=0,
        volume_shape_zyx=volume.shape,
        voxel_um=1.0,
        generations=generations,
        metadata_seed_xyz=(9.5, 9.5, 20),
        rng_seed=1447,
        minimum_support=0.9,
        minimum_seed_support=1.0,
    )
    after, _, _ = evaluate_shifted_surface(
        shifted,
        MemorySampler(volume),
        shift_voxels=5,
        volume_shape_zyx=volume.shape,
        voxel_um=1.0,
        generations=generations,
        metadata_seed_xyz=(9.5, 9.5, 20),
        rng_seed=1447,
        minimum_support=0.9,
        minimum_seed_support=1.0,
    )
    assert not before["pass_all_measured_gates"]
    assert after["pass_all_measured_gates"]
    assert after["m7_support"]["support_fraction"] == 1.0
    assert after["exhaustive_transects"]["all_clean"]


def test_shifted_surface_uses_positive_original_normal_and_float32() -> None:
    surface = plane_surface(5.0)
    normals = estimate_surface_normals(surface).positive_xyz
    shifted = shifted_surface_float32(surface, normals, 3)
    np.testing.assert_array_equal(shifted.z, np.full((8, 8), 8, dtype=np.float32))
    assert shifted_surface_float32(surface, normals, -2).z[0, 0] == 3.0


def test_rank_tie_prefers_smaller_absolute_then_signed_shift() -> None:
    def record(shift: int):
        return {
            "shift_voxels": shift,
            "pass_all_measured_gates": True,
            "m7_support": {"support_fraction": 1.0, "offset0_support_fraction": 1.0},
            "generation_support": {"seed_core": {"support": {"support_fraction": 1.0}}},
            "deterministic_50_transects": {"pass_count": 50},
            "exhaustive_transects": {"clean_fraction": 1.0},
            "metadata_seed_surface_alignment": {"evaluated": True, "distance_voxels": 0.0},
            "gate_results": [
                {"name": "support", "value": 1.0, "threshold": 0.9, "pass": True},
                {"name": "transects", "value": 50, "threshold": 50, "pass": True},
            ],
        }
    assert max((record(-4), record(3)), key=shift_rank_key)["shift_voxels"] == 3
    assert max((record(-3), record(3)), key=shift_rank_key)["shift_voxels"] == -3


def test_materialization_is_separate_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    surface = plane_surface(5.0)
    for component, array in (("x", surface.x), ("y", surface.y), ("z", surface.z)):
        tifffile.imwrite(source / f"{component}.tif", array.astype(np.float32))
    tifffile.imwrite(source / "generations.tif", np.ones((8, 8), dtype=np.uint16))
    (source / "meta.json").write_text(
        json.dumps({"uuid": "source-id", "scale": [1, 1], "seed": [9.5, 9.5, 5]}),
        encoding="utf-8",
    )
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    destination, provenance = materialize_corrected_copy(
        source,
        tmp_path / "out",
        surface,
        shift_voxels=5,
        voxel_um=1.0,
        original_normals_sha256="abc",
    )
    assert destination != source
    assert provenance["source_unchanged"]
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    loaded = load_tifxyz_asset(destination, resolution="stored")
    np.testing.assert_allclose(loaded.surface.z, surface.z)
