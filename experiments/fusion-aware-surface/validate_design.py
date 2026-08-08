#!/usr/bin/env python3
"""Mechanical pre-training checks against the pinned finite-thickness painter."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

from gap_supervision import inter_sheet_gap_mask
from normalization import EXPECTED_CT_PROPERTIES, validate_ct_contract


HERE = Path(__file__).resolve().parent
PAINTER_ROOT = HERE.parent / "contrast-painter-source"
PAINTER_FILE = PAINTER_ROOT / "scripts" / "contrast_phantom.py"
EXPECTED_SHA256 = "41f2a097b819bc486819d6702ed3949d0bdfe722c2405b1fe86828062da1d9b8"
DIAGNOSTIC_LOADER = HERE.parent / "surface-geometry-diagnostic" / "scripts" / "loader059.py"
EXPECTED_LOADER_SHA256 = "49a1d4ea3ee611236d53b1291ca2e8cd6e1450f2fc032223ff89a5b53d9ec903"
CHECKPOINT = HERE.parent / "models" / "surface_recto_059_redo" / "Model_epoch499.pth"
EXPECTED_CHECKPOINT_SHA256 = "f1990a02ac91889c1f989522ae0e45421a91cb666320448aaf579d42b081636f"
sys.path.insert(0, str(PAINTER_FILE.parent))

import contrast_phantom  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_cell(papyrus: int, pitch: float, seed: int, *, kollesis: bool):
    return contrast_phantom.emit_cell(
        12,
        papyrus,
        pitch,
        6.0,
        seed=seed,
        z_window_mm=0.06,
        voxel_um=30.0,
        sheet_um=150.0,
        arm="physical",
        kollesis=kollesis,
    )


def main() -> int:
    checks: list[tuple[str, bool]] = []
    checks.append(("painter file hash", file_sha256(PAINTER_FILE) == EXPECTED_SHA256))
    checks.append((
        "diagnostic loader hash",
        file_sha256(DIAGNOSTIC_LOADER) == EXPECTED_LOADER_SHA256,
    ))
    checks.append((
        "source checkpoint hash",
        file_sha256(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256,
    ))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    try:
        checkpoint_properties = validate_ct_contract(
            checkpoint.get("normalization_scheme"),
            checkpoint.get("intensity_properties"),
        )
    except (TypeError, ValueError) as error:
        print(f"FAIL: checkpoint normalization contract: {error}")
        return 1
    checks.append((
        "checkpoint normalization scheme",
        checkpoint.get("normalization_scheme") == "ct",
    ))
    checks.append((
        "checkpoint normalization properties",
        checkpoint_properties == EXPECTED_CT_PROPERTIES,
    ))

    gap_counts = {}
    for pitch in (170.0, 200.0, 230.0, 260.0):
        volume, surface, _, meta = make_cell(65, pitch, 100, kollesis=True)
        turn = np.asarray(meta["turn_id"], dtype=np.int16)
        join = np.asarray(meta["kollesis_mask"], dtype=bool)
        surface_2d = np.asarray(surface[0]) > 0
        gap = inter_sheet_gap_mask(turn, surface_2d, radius=3)
        gap_counts[int(pitch)] = int(gap.sum())
        checks.append((f"pitch {pitch:g}: gap supervision present", bool(gap.any())))
        checks.append((f"pitch {pitch:g}: gap is background", not np.any(gap & surface_2d)))
        checks.append((f"pitch {pitch:g}: turn subset surface", not np.any((turn > 0) & ~surface_2d)))
        checks.append((
            f"pitch {pitch:g}: join-only surface contract",
            np.array_equal(surface_2d & ~(turn > 0), join & ~(turn > 0)),
        ))
        checks.append((f"pitch {pitch:g}: nonempty generated volume", bool(volume.any())))

    low = make_cell(35, 200.0, 101, kollesis=True)
    high = make_cell(90, 200.0, 101, kollesis=True)
    low_gap = inter_sheet_gap_mask(low[3]["turn_id"], low[1][0], radius=3)
    high_gap = inter_sheet_gap_mask(high[3]["turn_id"], high[1][0], radius=3)
    checks.append(("gap labels are contrast invariant", np.array_equal(low_gap, high_gap)))
    checks.append(("turn labels are contrast invariant", np.array_equal(low[3]["turn_id"], high[3]["turn_id"])))

    _, control_surface, _, control_meta = make_cell(65, 700.0, 100, kollesis=False)
    control_turn = np.asarray(control_meta["turn_id"], dtype=np.int16)
    control_gap = inter_sheet_gap_mask(control_turn, control_surface[0], radius=3)
    checks.append(("700 um control has no gap-supervision voxels", not control_gap.any()))
    checks.append(("700 um control has no kollesis", not np.asarray(control_meta["kollesis_mask"]).any()))

    failures = [name for name, passed in checks if not passed]
    print("gap voxels by physical pitch:", gap_counts)
    print("checks:", sum(passed for _, passed in checks), "/", len(checks))
    if failures:
        print("FAIL:", "; ".join(failures))
        return 1
    print("FUSION-AWARE DESIGN CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
