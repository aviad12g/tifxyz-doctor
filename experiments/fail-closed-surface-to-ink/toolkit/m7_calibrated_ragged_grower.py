#!/usr/bin/env python3
"""Topology-safe calibrated ragged continuation of PHerc1447 v3.1.

Whole-edge growth stops when one weak vertex vetoes an otherwise viable row.
This prototype projects every unique missing frontier vertex from the same
cycle-start surface, keeps only quads whose new vertices have a calibrated
center lock, and greedily forms an induced batch that remains a single disk.
The batch is committed only after full masked native geometry, strict neighbor
coherence, and published-control-calibrated m7 support pass.  Multiple distant
runs remain diagnostics.  No raw CT or paid compute is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import tifffile

from m7_calibrated_directional_grower import (
    CalibrationGate,
    DEFAULT_CALIBRATION,
    project_target_calibrated,
)
from m7_locked_directional_grower import DEFAULT_M7_ROOT, load_public_sampler
from m7_locked_ragged_grower import (
    coherence_report,
    frontier_quads,
    grid_from_points,
    induced_quads,
    masked_area_cm2,
    quad_vertices,
    ragged_geometry_pass,
    topology_audit,
    vertices_for_quads,
)
from m7_locked_surface_grower import (
    GrowConfig,
    PublicM7BinaryAccessor,
    _array_sha256,
    _json_safe,
    transported_basis_zyx,
)
from native_surface_sampler import SurfaceGrid, estimate_surface_normals, validate_surface_geometry
from preflight_debug_m7 import (
    DEFAULT_SUPPORT_OFFSETS,
    DEFAULT_TRANSECT_OFFSETS,
    DEFAULT_VOLUME_SHAPE_ZYX,
    DEFAULT_VOXEL_UM,
    sample_offsets,
)
from sample_m7_seed_chunk import DEFAULT_BLOSC
from sample_raw_ct_seed_cube import sha256_file
from select_m7_seed_chunk import contiguous_true_runs
from tifxyz_render_pipeline import load_tifxyz_asset
from validate_public_m7_patch import (
    PublicZarrChunkSampler,
    evaluate_all_vertex_transects,
    support_summary,
)


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "m7-calibrated-ragged-frontier-v3.3"
DEFAULT_BASE = Path(
    "outputs/first-letters-geometry/m7-locked-grower/seed-x3525-y4191-z14071/"
    "prototype-v3p2-calibrated-ragged-0p5cm2"
)
DEFAULT_OUTPUT = Path(
    "outputs/first-letters-geometry/m7-locked-grower/seed-x3525-y4191-z14071/"
    "prototype-v3p3-calibrated-ragged-0p5cm2"
)


@dataclass(frozen=True)
class RaggedGrowthConfig:
    maximum_area_cm2: float = 0.5
    maximum_cycles: int = 64


@dataclass(frozen=True)
class StateEvaluation:
    passed: bool
    area_cm2: float
    native_reports: Mapping[str, Any]
    support: Mapping[str, Any]
    run_diagnostic: Mapping[str, Any]
    centered_run_diagnostic: Mapping[str, Any]
    coherence: Mapping[str, Any]
    topology: Mapping[str, Any]
    normals_zyx: Mapping[tuple[int, int], tuple[float, float, float]]
    frames_sha256: str
    sample_valid_sha256: str


@dataclass(frozen=True)
class RaggedState:
    points_zyx: Mapping[tuple[int, int], tuple[float, float, float]]
    generations: Mapping[tuple[int, int], int]
    active_quads: frozenset[tuple[int, int]]
    evaluation: StateEvaluation


def _read_manifest(directory: Path) -> Mapping[str, str]:
    hashes: dict[str, str] = {}
    for line in (directory / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"base manifest mismatch: {name}")
        hashes[name] = digest
    return hashes


def load_rectangle_base(directory: Path) -> tuple[dict, dict, frozenset, Mapping[str, Any]]:
    directory = directory.resolve()
    hashes = _read_manifest(directory)
    provenance = json.loads((directory / "growth-provenance.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    meta = json.loads((directory / "meta.json").read_text())
    if provenance.get("algorithm_version") not in {
        "m7-calibrated-directional-edges-v3.1",
        "m7-calibrated-ragged-frontier-v3.2",
        "m7-calibrated-ragged-frontier-v3.3",
    }:
        raise ValueError("base is not a pinned calibrated v3.1/v3.2/v3.3 artifact")
    if not validation.get("pass"):
        raise ValueError("base calibrated validation does not pass")
    asset = load_tifxyz_asset(directory, resolution="stored")
    generations_grid = np.asarray(tifffile.imread(directory / "generations.tif"), dtype=np.uint16)
    bounds = tuple(int(value) for value in meta["logical_bounds"])
    r0, r1, c0, c1 = bounds
    points: dict[tuple[int, int], tuple[float, float, float]] = {}
    generations: dict[tuple[int, int], int] = {}
    for row in range(r0, r1 + 1):
        for column in range(c0, c1 + 1):
            stored = (row - r0, column - c0)
            if not bool(asset.surface.valid[stored]):
                continue
            points[(row, column)] = tuple(
                float(value) for value in asset.surface.points_xyz[stored][::-1]
            )
            generations[(row, column)] = int(generations_grid[stored])
    quads = induced_quads(points)
    return points, generations, quads, {
        "directory": str(directory),
        "manifest_sha256": sha256_file(directory / "MANIFEST.sha256"),
        "member_sha256": hashes,
        "validation_sha256": sha256_file(directory / "validation.json"),
        "stored_asset_hashes": dict(asset.input_sha256),
    }


def _centered_run_diagnostic(
    frames: NDArray[np.uint8], mask: NDArray[np.bool_]
) -> Mapping[str, Any]:
    gaps: list[int] = []
    centers: list[float] = []
    absent = 0
    for row, column in np.argwhere(mask):
        raw = contiguous_true_runs(frames[:, row, column] > 0)
        runs = [[a - 15, b - 1 - 15] for a, b in raw]
        selected = [run for run in runs if run[0] <= 0 <= run[1]]
        if len(selected) != 1:
            absent += 1
            continue
        chosen = selected[0]
        centers.append(0.5 * (chosen[0] + chosen[1]))
        competitors = [
            chosen[0] - run[1] - 1 if run[1] < chosen[0] else run[0] - chosen[1] - 1
            for run in runs
            if run != chosen
        ]
        if competitors:
            gaps.append(min(competitors))
    return {
        "no_unique_offset0_crossing_run_count": absent,
        "selected_center_min": min(centers) if centers else None,
        "selected_center_max": max(centers) if centers else None,
        "competitor_gap_count": len(gaps),
        "competitor_gap_min": min(gaps) if gaps else None,
        "competitor_gap_median": float(np.median(gaps)) if gaps else None,
        "competitor_gap_max": max(gaps) if gaps else None,
    }


def evaluate_state(
    points: Mapping[tuple[int, int], Sequence[float]],
    quads: Iterable[tuple[int, int]],
    sampler: PublicZarrChunkSampler,
    grow_config: GrowConfig,
    gate: CalibrationGate,
    *,
    voxel_um: float,
) -> StateEvaluation:
    topology = topology_audit(quads, seed=(0, 0))
    grid, mask, bounds = grid_from_points(points, quads)
    xyz = grid[..., ::-1]
    surface = SurfaceGrid.from_tifxyz(xyz[..., 0], xyz[..., 1], xyz[..., 2], mask=mask)
    normals = estimate_surface_normals(surface, voxel_size_zyx_um=(voxel_um,) * 3)
    offsets_um = [value * voxel_um for value in DEFAULT_TRANSECT_OFFSETS]
    reports = {
        label: validate_surface_geometry(
            surface,
            volume_shape_zyx=sampler.shape_zyx,
            voxel_size_zyx_um=(voxel_um,) * 3,
            normal_offsets_um=offsets_um,
            normal_sign=sign,
        ).to_dict()
        for label, sign in (("positive", 1), ("negative", -1))
    }
    geometry_pass = all(
        ragged_geometry_pass(report, zero_distortion=True) for report in reports.values()
    )
    frames, sample_valid = sample_offsets(
        sampler, surface, normals.positive_xyz, surface.valid & normals.valid,
        DEFAULT_TRANSECT_OFFSETS,
    )
    requested = np.broadcast_to(surface.valid, sample_valid.shape)
    sampling_complete = bool(np.all(sample_valid[requested]))
    support_indices = [DEFAULT_TRANSECT_OFFSETS.index(value) for value in DEFAULT_SUPPORT_OFFSETS]
    support = support_summary(frames[support_indices], DEFAULT_SUPPORT_OFFSETS, surface.valid)
    runs = evaluate_all_vertex_transects(frames, surface.valid)
    centered = _centered_run_diagnostic(frames, surface.valid)
    coherence = coherence_report(
        points, quads, surface, normals.positive_xyz, bounds, grow_config
    )
    normals_map: dict[tuple[int, int], tuple[float, float, float]] = {}
    r0, _, c0, _ = bounds
    for key in points:
        stored = (key[0] - r0, key[1] - c0)
        normals_map[key] = tuple(float(value) for value in normals.positive_xyz[stored][::-1])
    passed = bool(
        topology["pass"]
        and geometry_pass
        and sampling_complete
        and coherence["pass"]
        and float(support["support_fraction"]) >= gate.minimum_support_fraction_pm2
        and float(support["offset0_support_fraction"]) >= gate.minimum_offset0_fraction
    )
    return StateEvaluation(
        passed=passed,
        area_cm2=masked_area_cm2(points, quads, voxel_um),
        native_reports=reports,
        support=support,
        run_diagnostic=runs,
        centered_run_diagnostic=centered,
        coherence=coherence,
        topology=topology,
        normals_zyx=normals_map,
        frames_sha256=_array_sha256(frames),
        sample_valid_sha256=_array_sha256(sample_valid),
    )


def _reference_basis(normal: Sequence[float]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    normal_array = np.asarray(normal, dtype=np.float64)
    normal_array /= np.linalg.norm(normal_array)
    reference = np.eye(3)[int(np.argmin(np.abs(normal_array)))]
    column = np.cross(normal_array, reference)
    column /= np.linalg.norm(column)
    if column[int(np.argmax(np.abs(column)))] < 0:
        column = -column
    row = np.cross(normal_array, column)
    row /= np.linalg.norm(row)
    return row, column


def _propose_vertex(
    key: tuple[int, int],
    state: RaggedState,
    reference_row: Sequence[float],
    reference_column: Sequence[float],
    grow_config: GrowConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, int]:
    neighbors = [
        candidate
        for candidate in sorted(state.points_zyx)
        if max(abs(candidate[0] - key[0]), abs(candidate[1] - key[1])) <= 1
    ]
    if not neighbors:
        raise ValueError("frontier vertex has no inward neighbor")
    proposals, normals = [], []
    for neighbor in neighbors:
        normal = np.asarray(state.evaluation.normals_zyx[neighbor])
        local_row, local_column = transported_basis_zyx(
            normal, reference_row, reference_column
        )
        proposals.append(
            np.asarray(state.points_zyx[neighbor])
            + grow_config.spacing_voxels
            * ((key[0] - neighbor[0]) * local_row + (key[1] - neighbor[1]) * local_column)
        )
        normals.append(normal)
    array = np.stack(proposals)
    proposal = np.mean(array, axis=0)
    spread = float(np.max(np.linalg.norm(array - proposal, axis=1)))
    prior = np.sum(normals, axis=0)
    prior /= np.linalg.norm(prior)
    return proposal, prior, spread, len(neighbors)


def project_frontier_vertices(
    state: RaggedState,
    frontier: Sequence[tuple[int, int]],
    accessor: PublicM7BinaryAccessor,
    grow_config: GrowConfig,
    gate: CalibrationGate,
) -> tuple[Mapping[tuple[int, int], tuple[float, float, float]], Mapping[str, Any]]:
    missing = sorted(vertices_for_quads(frontier) - set(state.points_zyx))
    reference_row, reference_column = _reference_basis(
        state.evaluation.normals_zyx[(0, 0)]
    )
    points: dict[tuple[int, int], tuple[float, float, float]] = {}
    records: dict[str, Any] = {}
    for key in missing:
        proposal, prior, spread, count = _propose_vertex(
            key, state, reference_row, reference_column, grow_config
        )
        record: dict[str, Any] = {
            "logical_vertex": list(key),
            "proposal_zyx": proposal.tolist(),
            "proposal_spread": spread,
            "inward_neighbor_count": count,
        }
        if spread > grow_config.maximum_proposal_spread:
            record.update({"pass": False, "reason": "inward_proposals_incoherent"})
            records[str(key)] = record
            continue
        projected = project_target_calibrated(
            accessor, proposal, prior, grow_config, gate, inward_neighbor_count=count
        )
        record.update(
            {
                "pass": projected.passed,
                "reason": projected.reason,
                "foreground_candidate_count": projected.foreground_candidate_count,
                "evaluated_candidate_count": projected.evaluated_candidate_count,
                "passing_candidate_count": projected.passing_candidate_count,
                "diagnostic": projected.ambiguity,
            }
        )
        if projected.passed and projected.node is not None:
            point = tuple(float(value) for value in projected.node.point_zyx)
            points[key] = point
            record["point_zyx"] = list(point)
        records[str(key)] = record
    return points, records


def greedy_topology_batch(
    state: RaggedState,
    frontier: Sequence[tuple[int, int]],
    projected_points: Mapping[tuple[int, int], Sequence[float]],
) -> tuple[tuple[tuple[int, int], ...], Mapping[str, Any]]:
    chosen: list[tuple[int, int]] = []
    quads = set(state.active_quads)
    points = dict(state.points_zyx)
    rejections: dict[str, str] = {}
    for quad in sorted(frontier):
        missing = quad_vertices(quad) - set(points)
        if not missing.issubset(projected_points):
            rejections[str(quad)] = "missing_vertex_projection_failed"
            continue
        tentative_quads = frozenset((*quads, quad))
        tentative_points = {**points, **{key: projected_points[key] for key in missing}}
        if induced_quads(tentative_points) != tentative_quads:
            rejections[str(quad)] = "unintended_induced_quad"
            continue
        topology = topology_audit(tentative_quads, seed=(0, 0))
        if not topology["pass"]:
            rejections[str(quad)] = "topology_gate_failed"
            continue
        quads.add(quad)
        points.update({key: projected_points[key] for key in missing})
        chosen.append(quad)
    return tuple(chosen), {"rejections": rejections}


def _state_for_batch(
    base: RaggedState,
    additions: Sequence[tuple[int, int]],
    projected_points: Mapping[tuple[int, int], Sequence[float]],
    sampler: PublicZarrChunkSampler,
    grow_config: GrowConfig,
    gate: CalibrationGate,
    *,
    voxel_um: float,
) -> RaggedState | None:
    quads = frozenset((*base.active_quads, *additions))
    active = vertices_for_quads(quads)
    new_keys = active - set(base.points_zyx)
    if not new_keys.issubset(projected_points):
        return None
    points = {**base.points_zyx, **{key: tuple(projected_points[key]) for key in new_keys}}
    if induced_quads(points) != quads:
        return None
    evaluation = evaluate_state(
        points, quads, sampler, grow_config, gate, voxel_um=voxel_um
    )
    if not evaluation.passed:
        return None
    generation = max(base.generations.values()) + 1
    generations = {**base.generations, **{key: generation for key in new_keys}}
    return RaggedState(points, generations, quads, evaluation)


def grow_ragged_batches(
    state: RaggedState,
    accessor: PublicM7BinaryAccessor,
    sampler: PublicZarrChunkSampler,
    grow_config: GrowConfig,
    gate: CalibrationGate,
    config: RaggedGrowthConfig,
    *,
    voxel_um: float,
) -> tuple[RaggedState, tuple[Mapping[str, Any], ...], str]:
    records: list[Mapping[str, Any]] = []
    stop_reason = "maximum_cycles_reached_fail_closed"
    for cycle in range(1, config.maximum_cycles + 1):
        if state.evaluation.area_cm2 + 1e-15 >= config.maximum_area_cm2:
            stop_reason = "maximum_area_reached"
            break
        frontier = frontier_quads(state.active_quads)
        projected, projection_records = project_frontier_vertices(
            state, frontier, accessor, grow_config, gate
        )
        batch, topology_record = greedy_topology_batch(state, frontier, projected)
        attempts: list[Mapping[str, Any]] = []
        accepted_state: RaggedState | None = None
        accepted_count = 0
        # Deterministic largest-prefix search.  Prefix topology was certified
        # incrementally above; every attempted state is fully revalidated.
        for count in range(len(batch), 0, -1):
            candidate = _state_for_batch(
                state, batch[:count], projected, sampler, grow_config, gate,
                voxel_um=voxel_um,
            )
            attempts.append({"quad_count": count, "pass": candidate is not None})
            if candidate is not None:
                accepted_state = candidate
                accepted_count = count
                break
        individual_fallback_attempts: list[Mapping[str, Any]] = []
        if accepted_state is None:
            for index, quad in enumerate(batch):
                candidate = _state_for_batch(
                    state, (quad,), projected, sampler, grow_config, gate,
                    voxel_um=voxel_um,
                )
                individual_fallback_attempts.append(
                    {"quad": list(quad), "pass": candidate is not None}
                )
                if candidate is not None:
                    accepted_state = candidate
                    accepted_count = 1
                    # Rewrite the accepted prefix representation so provenance
                    # and state agree even when the winner is later in order.
                    batch = (quad,)
                    break
        records.append(
            {
                "cycle": cycle,
                "starting_area_cm2": state.evaluation.area_cm2,
                "frontier_quad_count": len(frontier),
                "projected_new_vertex_count": len(projected),
                "greedy_batch_quad_count": len(batch),
                "accepted_batch_quad_count": accepted_count,
                "accepted_quads": [list(quad) for quad in batch[:accepted_count]],
                "projection_records": projection_records,
                "topology_build": topology_record,
                "full_surface_prefix_attempts": attempts,
                "exhaustive_single_quad_fallback_attempts": individual_fallback_attempts,
            }
        )
        if accepted_state is None:
            stop_reason = f"cycle_{cycle}_no_progress"
            break
        state = accepted_state
    return state, tuple(records), stop_reason


def write_artifact(
    output: Path,
    state: RaggedState,
    validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    voxel_um: float,
    target_volume: str,
    seed_xyz: Sequence[float],
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.mkdir()
    try:
        grid, mask, bounds = grid_from_points(state.points_zyx, state.active_quads)
        xyz = grid[..., ::-1]
        for index, name in enumerate("xyz"):
            tifffile.imwrite(temporary / f"{name}.tif", xyz[..., index].astype(np.float32))
        tifffile.imwrite(temporary / "mask.tif", mask.astype(np.uint8) * 255)
        generations = np.zeros(mask.shape, dtype=np.uint16)
        r0, _, c0, _ = bounds
        for key, value in state.generations.items():
            generations[key[0] - r0, key[1] - c0] = int(value)
        tifffile.imwrite(temporary / "generations.tif", generations)
        active_xyz = xyz[mask]
        meta = {
            "format": "tifxyz",
            "type": "seg",
            "uuid": output.name,
            "source": ALGORITHM_VERSION,
            "target_volume": target_volume,
            "seed": [float(value) for value in seed_xyz],
            "scale": [0.05, 0.05],
            "bbox": [np.min(active_xyz, axis=0).tolist(), np.max(active_xyz, axis=0).tolist()],
            "area_cm2": state.evaluation.area_cm2,
            "voxel_size_um": voxel_um,
            "stored_shape": list(mask.shape),
            "logical_bounds": list(bounds),
            "active_vertex_count": len(state.points_zyx),
            "active_quad_count": len(state.active_quads),
            "mask_semantics": "vertices incident to one induced connected disk quad complex",
        }
        for name, document in (
            ("meta.json", meta),
            ("growth-provenance.json", provenance),
            ("validation.json", validation),
        ):
            (temporary / name).write_text(
                json.dumps(_json_safe(document), indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        lines = [f"{sha256_file(path)}  {path.name}" for path in sorted(temporary.iterdir())]
        (temporary / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-area-cm2", type=float, default=0.5)
    parser.add_argument("--maximum-cycles", type=int, default=64)
    parser.add_argument("--m7-root", default=DEFAULT_M7_ROOT)
    parser.add_argument("--m7-array-path", default="0")
    parser.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    parser.add_argument("--voxel-um", type=float, default=DEFAULT_VOXEL_UM)
    parser.add_argument("--target-volume", default="PHerc1447")
    parser.add_argument(
        "--seed-xyz",
        type=float,
        nargs=3,
        default=(3525.0, 4191.0, 14071.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--volume-shape-zyx",
        type=int,
        nargs=3,
        default=DEFAULT_VOLUME_SHAPE_ZYX,
        metavar=("Z", "Y", "X"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    started = time.monotonic()
    try:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
        calibration_path = Path(args.calibration)
        calibration = json.loads(calibration_path.read_text())
        conclusion = calibration["conclusion"]
        gate = CalibrationGate(
            minimum_support_fraction_pm2=float(
                conclusion["calibrated_minimum_plus_minus_2_support_fraction"]
            ),
            minimum_offset0_fraction=float(
                conclusion["calibrated_minimum_offset0_support_fraction"]
            ),
        )
        config = RaggedGrowthConfig(
            maximum_area_cm2=float(args.maximum_area_cm2),
            maximum_cycles=int(args.maximum_cycles),
        )
        grow_config = GrowConfig()
        points, generations, quads, base_evidence = load_rectangle_base(Path(args.base))
        sampler, m7_input = load_public_sampler(
            str(args.m7_root),
            str(args.m7_array_path),
            Path(args.blosc),
            args.volume_shape_zyx,
        )
        accessor = PublicM7BinaryAccessor(sampler)
        initial_evaluation = evaluate_state(
            points, quads, sampler, grow_config, gate, voxel_um=float(args.voxel_um)
        )
        if not initial_evaluation.passed:
            raise RuntimeError("reloaded v3.1 base fails calibrated masked evaluation")
        initial = RaggedState(points, generations, quads, initial_evaluation)
        state, cycles, stop_reason = grow_ragged_batches(
            initial,
            accessor,
            sampler,
            grow_config,
            gate,
            config,
            voxel_um=float(args.voxel_um),
        )
        reached_target = state.evaluation.area_cm2 + 1e-15 >= config.maximum_area_cm2
        validation = {
            "pass": state.evaluation.passed,
            "area_cm2": state.evaluation.area_cm2,
            "target_area_cm2": config.maximum_area_cm2,
            "target_area_reached": reached_target,
            "active_vertex_count": len(state.points_zyx),
            "active_quad_count": len(state.active_quads),
            "topology": state.evaluation.topology,
            "native_reports_by_orientation": state.evaluation.native_reports,
            "coherence": state.evaluation.coherence,
            "support": state.evaluation.support,
            "full_pm15_run_diagnostic": state.evaluation.run_diagnostic,
            "centered_selected_run_diagnostic": state.evaluation.centered_run_diagnostic,
            "frames_sha256": state.evaluation.frames_sha256,
            "sample_valid_sha256": state.evaluation.sample_valid_sha256,
            "run_count_gate": "none; distant run multiplicity is diagnostic only",
            "identity_gate": asdict(gate),
            "self_intersection_status": "not localized; unknown; never assumed zero",
        }
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "status": "complete",
            "verdict": "pass_calibrated_geometry" if state.evaluation.passed else "no_go",
            "verdict_scope": "public-m7 geometry only; raw fiber QC required before detector",
            "stop_reason": stop_reason,
            "elapsed_seconds": time.monotonic() - started,
            "paid_compute_cost_usd": 0.0,
            "target_volume": str(args.target_volume),
            "seed_xyz": [float(value) for value in args.seed_xyz],
            "inputs": {
                "base": base_evidence,
                "calibration_summary": {
                    "path": str(calibration_path.resolve()),
                    "sha256": sha256_file(calibration_path),
                },
                "m7": m7_input,
            },
            "calibration_gate": asdict(gate),
            "growth_config": asdict(config),
            "vertex_grow_config": asdict(grow_config),
            "growth": {
                "initial_area_cm2": initial.evaluation.area_cm2,
                "initial_vertex_count": len(initial.points_zyx),
                "initial_quad_count": len(initial.active_quads),
                "cycle_count": len(cycles),
                "cycles": list(cycles),
                "final_area_cm2": state.evaluation.area_cm2,
                "final_vertex_count": len(state.points_zyx),
                "final_quad_count": len(state.active_quads),
            },
            "public_m7_sampling": sampler.manifest(),
            "accessor_roi_read_count": accessor.roi_read_count,
            "limitations": [
                "Run multiplicity across +/-15 is diagnostic, calibrated against a clean published surface.",
                "Self-intersection is not localized and is never assumed zero.",
                "No raw CT, ink inference, detector, semantic judgment, paid GPU, or RunPod was used.",
            ],
        }
        write_artifact(
            output,
            state,
            validation,
            provenance,
            voxel_um=float(args.voxel_um),
            target_volume=str(args.target_volume),
            seed_xyz=args.seed_xyz,
        )
    except Exception as error:
        error_path = output.with_name(output.name + ".error.json")
        if error_path.exists():
            raise
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(
            json.dumps(
                {"schema_version": SCHEMA_VERSION, "algorithm_version": ALGORITHM_VERSION,
                 "status": "error", "error_type": type(error).__name__, "error": str(error),
                 "elapsed_seconds": time.monotonic() - started},
                indent=2, sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"FAIL CLOSED: {type(error).__name__}: {error}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {"output": str(output), "pass": state.evaluation.passed,
             "target_area_reached": reached_target, "area_cm2": state.evaluation.area_cm2,
             "active_vertices": len(state.points_zyx), "active_quads": len(state.active_quads),
             "stop_reason": stop_reason},
            sort_keys=True,
        )
    )
    return 0 if state.evaluation.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
