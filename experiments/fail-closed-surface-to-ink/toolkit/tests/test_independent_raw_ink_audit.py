from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import independent_raw_ink_audit as audit  # noqa: E402


def _write_exact_stack_manifest(tmp_path: Path, voxel_um: float) -> tuple[Path, Path, Path]:
    stack_dir = tmp_path / "positive"
    stack_dir.mkdir()
    mask_path = tmp_path / "mask.tif"
    mask = np.ones((4, 5), dtype=np.uint8)
    tifffile.imwrite(mask_path, mask)
    tifffile.imwrite(stack_dir / "valid-all.tif", mask)
    records = []
    for index, offset in enumerate(audit.EXACT_OFFSETS_VOXELS):
        path = stack_dir / f"{index:02d}.tif"
        tifffile.imwrite(path, np.full(mask.shape, index, dtype=np.uint8))
        records.append(
            {
                "frame": index,
                "offset_voxels": offset,
                "path": str(path.resolve()),
                "sha256": audit.sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "status": "complete",
        "candidate_id": "synthetic",
        "raw_stack": {
            "layer_count": audit.STACK_DEPTH,
            "shape_frame_row_column": [audit.STACK_DEPTH, 4, 5],
            "dtype": "uint8",
            "directory": str(stack_dir.resolve()),
            "offsets_voxels": list(audit.EXACT_OFFSETS_VOXELS),
            "offsets_micrometers": [
                offset * voxel_um for offset in audit.EXACT_OFFSETS_VOXELS
            ],
            "negative_stack_policy": "not stored; exact layer reversal 92..0",
            "layers": records,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return stack_dir, mask_path, manifest_path


def test_exact_manifest_verifies_all_layers_mapping_hashes_and_mask(tmp_path: Path) -> None:
    voxel_um = 9.362
    stack_dir, mask_path, manifest_path = _write_exact_stack_manifest(
        tmp_path, voxel_um
    )
    stack, hashes = audit.load_exact_stack(stack_dir)
    mask = audit.load_mask(mask_path, stack.shape[1:])
    result = audit.validate_exact_93_manifest(
        manifest_path,
        stack_dir,
        hashes,
        stack.shape,
        mask_path,
        mask,
        voxel_um=voxel_um,
    )
    assert result["status"] == "pass"
    assert result["center_index"] == 46
    assert result["layer_record_hashes_verified"] == 93

    changed = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed["raw_stack"]["offsets_voxels"][46] = 1
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(audit.IndependentInkAuditError, match="contract failed"):
        audit.validate_exact_93_manifest(
            manifest_path,
            stack_dir,
            hashes,
            stack.shape,
            mask_path,
            mask,
            voxel_um=voxel_um,
        )


def test_mask_normalized_dark_residual_ignores_hole_but_finds_dark_mark() -> None:
    shape = (160, 160)
    image = np.full(shape, 120, dtype=np.uint8)
    mask = np.ones(shape, dtype=bool)
    mask[20:50, 20:50] = False
    image[~mask] = 0
    image[90:98, 75:105] = 70
    interior = mask.copy()
    residual, statistics = audit.robust_dark_residual(
        image, mask, interior, background_size=41
    )
    assert statistics["robust_sigma"] >= 0.25
    assert float(np.mean(residual[91:97, 78:102])) > 4.0
    assert np.all(residual[~mask] == 0)
    # Mask normalization prevents the zero-valued hole from creating a broad
    # false dark halo in otherwise constant material.
    assert float(np.max(residual[15:55, 15:55][mask[15:55, 15:55]])) < 0.1


def test_depth_aggregation_rewards_adjacent_near_blocks_and_penalizes_far() -> None:
    shape = (20, 20)
    near = np.zeros((len(audit.NEAR_INDICES), *shape), dtype=np.float32)
    far = np.zeros((len(audit.FAR_INDICES), *shape), dtype=np.float32)
    near[:, 5, 5] = 3.0
    near[:, 10, 10] = 3.0
    far[:, 10, 10] = 3.0
    result = audit.aggregate_depth_evidence(near, far)
    assert result["adjacent_persistence"][5, 5] == 3.0
    assert result["near_support"][5, 5] == 1.0
    assert result["raw_depth_score"][5, 5] > result["raw_depth_score"][10, 10]
    # A feature present in only one seven-layer block cannot pass the adjacent
    # block persistence operation.
    near[:] = 0
    near[:7, 2, 2] = 5.0
    result = audit.aggregate_depth_evidence(near, far * 0)
    assert result["adjacent_persistence"][2, 2] == 0.0


def test_structure_tensor_marks_long_fiber_more_than_crossing() -> None:
    shape = (256, 256)
    score = np.zeros(shape, dtype=np.float32)
    score[60:65, 20:236] = 1.0
    score[174:179, 30:226] = 1.0
    score[80:250, 126:131] = 1.0
    valid = np.ones(shape, dtype=bool)
    evidence = audit.fiber_orientation_evidence(score, valid, tensor_sigma=18)
    straight = float(evidence["fiber_likelihood"][62, 120])
    crossing = float(evidence["fiber_likelihood"][176, 128])
    assert straight > 0.35
    assert straight > crossing
    assert float(evidence["ridge_fiber_likelihood"][62, 120]) > 0.0


def test_component_features_flags_only_long_thin_proxy() -> None:
    shape = (220, 260)
    binary = np.zeros(shape, dtype=bool)
    binary[30:35, 20:230] = True
    binary[100:150, 80:130] = True
    score = binary.astype(np.float32)
    fiber = np.zeros(shape, dtype=np.float32)
    components = audit.component_features(binary, score, fiber)
    assert len(components) == 2
    assert sum(bool(component["long_thin_fiber_proxy"]) for component in components) == 1


def test_row_topology_can_nominate_but_never_claim_ten_letters() -> None:
    shape = (300, 900)
    score = np.zeros(shape, dtype=np.float32)
    fiber = np.zeros(shape, dtype=np.float32)
    orientation = np.zeros(shape, dtype=np.float32)
    interior = np.ones(shape, dtype=bool)
    components = []
    for index in range(10):
        x0 = 80 + index * 70
        y0 = 120
        score[y0 : y0 + 55, x0 : x0 + 28] = 0.9
        # Diverse stroke orientations keep this synthetic row from looking
        # like one uniformly aligned fiber bundle.
        orientation[y0 : y0 + 55, x0 : x0 + 28] = (index % 6) * np.pi / 6
        components.append(
            {
                "x": x0 + 14.0,
                "y": y0 + 27.5,
                "x0": x0,
                "y0": y0,
                "x1": x0 + 28,
                "y1": y0 + 55,
                "area": 28 * 55,
                "elongation": 2.0,
                "principal_angle_degrees": 90.0,
                "mean_score": 0.9,
                "mean_fiber_likelihood": 0.0,
                "long_thin_fiber_proxy": False,
            }
        )
    candidates = audit.rank_row_hypotheses(
        components,
        score,
        fiber,
        orientation,
        interior,
        voxel_um=8.64,
        top=3,
    )
    assert candidates
    assert candidates[0]["glyph_proxy_count"] >= 10
    assert candidates[0]["automated_morphology_gate"] is True
    assert candidates[0]["defensible_text_candidate"] is False


def test_row_search_covers_rotated_baselines_outside_old_horizontal_band() -> None:
    shape = (1000, 1000)
    score = np.zeros(shape, dtype=np.float32)
    fiber = np.zeros(shape, dtype=np.float32)
    orientation = np.zeros(shape, dtype=np.float32)
    interior = np.ones(shape, dtype=bool)
    components = []
    angle = np.deg2rad(60.0)
    for index in range(10):
        cx = 180.0 + index * 80.0 * np.cos(angle)
        cy = 160.0 + index * 80.0 * np.sin(angle)
        x0, y0 = int(cx - 10), int(cy - 15)
        score[y0 : y0 + 30, x0 : x0 + 20] = 0.9
        orientation[y0 : y0 + 30, x0 : x0 + 20] = (index % 6) * np.pi / 6
        components.append(
            {
                "x": cx,
                "y": cy,
                "x0": x0,
                "y0": y0,
                "x1": x0 + 20,
                "y1": y0 + 30,
                "area": 600,
                "elongation": 1.6,
                "principal_angle_degrees": float((index * 23) % 180),
                "mean_score": 0.9,
                "mean_fiber_likelihood": 0.0,
                "long_thin_fiber_proxy": False,
            }
        )
    candidates = audit.rank_row_hypotheses(
        components,
        score,
        fiber,
        orientation,
        interior,
        voxel_um=8.64,
        top=20,
    )
    assert any(
        candidate["glyph_proxy_count"] >= 10
        and abs(candidate["angle_degrees"] - 60) <= 6
        for candidate in candidates
    )
