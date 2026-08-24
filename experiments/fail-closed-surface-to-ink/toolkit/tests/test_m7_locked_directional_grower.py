from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import m7_locked_directional_grower as directional  # noqa: E402
from m7_locked_directional_grower import (  # noqa: E402
    BaseState,
    DirectionTrial,
    DirectionalConfig,
    compact_rectangle,
    direction_rank_key,
    edge_cells,
    evaluate_direction,
    grow_directional_edges,
)
from m7_locked_surface_grower import GrowConfig, Node, ProjectionResult  # noqa: E402


class ShapeOnlyAccessor:
    shape_zyx = (200, 200, 200)


class ArrayAccessor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array, dtype=bool)
        self.shape_zyx = self.array.shape

    def read_roi(self, lower_zyx, upper_zyx):
        lower = np.asarray(lower_zyx, dtype=int)
        upper = np.asarray(upper_zyx, dtype=int)
        return self.array[
            tuple(slice(int(left), int(right)) for left, right in zip(lower, upper))
        ].copy()


def node_at(row: int, column: int, spacing: float = 10.0) -> Node:
    point = (80.0, 80.0 + spacing * row, 80.0 - spacing * column)
    return Node(
        point_zyx=point,
        normal_zyx=(1.0, 0.0, 0.0),
        proposal_zyx=point,
        candidate_voxel_zyx=tuple(int(value) for value in point),
        initial_run_offsets=(-1, 1),
        initial_run_center_offset=0.0,
        projected_run_offsets=(-1, 1),
        projected_run_center_offset=0.0,
        proposal_distance=0.0,
        pca_normal_ratio=0.01,
        pca_tangent_ratio=1.0,
        inward_neighbor_count=1,
    )


def planar_base(radius: int = 1) -> BaseState:
    nodes = {
        (row, column): node_at(row, column)
        for row in range(-radius, radius + 1)
        for column in range(-radius, radius + 1)
    }
    generations = {
        key: max(abs(key[0]), abs(key[1])) + 1 for key in nodes
    }
    return BaseState(
        nodes=nodes,
        generations=generations,
        bounds=(-radius, radius, -radius, radius),
        reference_row_zyx=(0.0, 1.0, 0.0),
        reference_column_zyx=(0.0, 0.0, -1.0),
        evidence={},
    )


def passing_projection(target, inward_neighbor_count: int) -> ProjectionResult:
    point = tuple(float(value) for value in target)
    node = replace(
        node_at(0, 0),
        point_zyx=point,
        proposal_zyx=point,
        candidate_voxel_zyx=tuple(int(round(value)) for value in point),
        inward_neighbor_count=inward_neighbor_count,
    )
    return ProjectionResult(True, "pass", node, 1, 1, 1, None)


def test_edge_cells_expand_exactly_one_rectangle_side() -> None:
    bounds = (-2, 3, -4, 5)
    north, north_bounds = edge_cells(bounds, "north")
    east, east_bounds = edge_cells(bounds, "east")
    assert north_bounds == (-3, 3, -4, 5)
    assert north[0] == (-3, -4) and north[-1] == (-3, 5)
    assert east_bounds == (-2, 3, -4, 6)
    assert east[0] == (-2, 6) and east[-1] == (3, 6)


def test_canonical_trilinear_transect_rejects_competing_sheet() -> None:
    volume = np.zeros((61, 21, 21), dtype=bool)
    volume[29:32, :, :] = True
    clean = directional.canonical_trilinear_transect(
        ArrayAccessor(volume), (30.0, 10.25, 10.25), (1.0, 0.0, 0.0), 12
    )
    assert clean["pass"]
    assert clean["runs"] == [[-1, 1]]
    volume[38:40, :, :] = True
    ambiguous = directional.canonical_trilinear_transect(
        ArrayAccessor(volume), (30.0, 10.25, 10.25), (1.0, 0.0, 0.0), 12
    )
    assert not ambiguous["pass"]
    assert ambiguous["reason"] == "competing_foreground_run"


def test_failed_edge_is_atomic(monkeypatch) -> None:
    calls = 0

    def fake_project(accessor, target, prior, config, *, inward_neighbor_count):
        nonlocal calls
        calls += 1
        if calls == 2:
            return ProjectionResult(False, "synthetic_ambiguity", None, 1, 1, 0, None)
        return passing_projection(target, inward_neighbor_count)

    monkeypatch.setattr(directional, "project_target_to_m7", fake_project)
    monkeypatch.setattr(
        directional, "canonical_trilinear_transect",
        lambda *args, **kwargs: {"pass": True, "reason": "pass", "run_center_offset": 0.0},
    )
    base = planar_base()
    trial = evaluate_direction(
        ShapeOnlyAccessor(), base.nodes, base.generations, base.bounds, "north",
        base.reference_row_zyx, base.reference_column_zyx,
        GrowConfig(spacing_voxels=10.0), voxel_um=1.0,
    )
    assert not trial.passed
    assert trial.reason == "synthetic_ambiguity"
    assert trial.trial_nodes == {}
    assert len(trial.cells) == 2


def test_passing_edge_forms_complete_rectangle(monkeypatch) -> None:
    monkeypatch.setattr(
        directional,
        "project_target_to_m7",
        lambda accessor, target, prior, config, *, inward_neighbor_count:
            passing_projection(target, inward_neighbor_count),
    )
    monkeypatch.setattr(
        directional, "canonical_trilinear_transect",
        lambda *args, **kwargs: {"pass": True, "reason": "pass", "run_center_offset": 0.0},
    )
    base = planar_base()
    trial = evaluate_direction(
        ShapeOnlyAccessor(), base.nodes, base.generations, base.bounds, "east",
        base.reference_row_zyx, base.reference_column_zyx,
        GrowConfig(spacing_voxels=10.0), voxel_um=1.0,
    )
    assert trial.passed
    assert len(trial.trial_nodes) == 3
    combined = {**base.nodes, **trial.trial_nodes}
    generations = {**base.generations, **{key: 3 for key in trial.trial_nodes}}
    points, generation_grid = compact_rectangle(combined, generations, trial.new_bounds)
    assert points.shape == (3, 4, 3)
    assert generation_grid.shape == (3, 4)


def test_rank_prefers_geometric_margin_then_fixed_direction() -> None:
    metrics = {
        "minimum_neighbor_normal_abs_dot": 0.99,
        "maximum_neighbor_distance_relative_error": 0.1,
        "maximum_neighbor_normal_displacement_fraction": 0.05,
        "mean_new_vertex_proposal_distance": 1.0,
    }
    north = DirectionTrial("north", True, "pass", (), {}, (-1, 1, -1, 1), 0.1, {}, {}, metrics)
    east = DirectionTrial("east", True, "pass", (), {}, (-1, 1, -1, 1), 0.1, {}, {}, metrics)
    better = DirectionTrial(
        "south", True, "pass", (), {}, (-1, 1, -1, 1), 0.1, {}, {},
        {**metrics, "minimum_neighbor_normal_abs_dot": 0.995},
    )
    assert min((north, east), key=direction_rank_key).direction == "north"
    assert min((north, better), key=direction_rank_key).direction == "south"


def test_growth_accepts_only_one_ranked_edge_per_cycle(monkeypatch) -> None:
    monkeypatch.setattr(
        directional,
        "project_target_to_m7",
        lambda accessor, target, prior, config, *, inward_neighbor_count:
            passing_projection(target, inward_neighbor_count),
    )
    monkeypatch.setattr(
        directional, "canonical_trilinear_transect",
        lambda *args, **kwargs: {"pass": True, "reason": "pass", "run_center_offset": 0.0},
    )
    base = planar_base()
    first = grow_directional_edges(
        ShapeOnlyAccessor(), base, GrowConfig(spacing_voxels=10.0),
        DirectionalConfig(maximum_area_cm2=1.0, maximum_cycles=1), voxel_um=1.0,
    )
    second = grow_directional_edges(
        ShapeOnlyAccessor(), base, GrowConfig(spacing_voxels=10.0),
        DirectionalConfig(maximum_area_cm2=1.0, maximum_cycles=1), voxel_um=1.0,
    )
    assert len(first.nodes) == 12
    assert first.points_zyx.shape in ((3, 4, 3), (4, 3, 3))
    assert first.bounds == second.bounds
    np.testing.assert_allclose(first.points_zyx, second.points_zyx, atol=0, rtol=0)
    assert len(first.cycle_records) == 1
    assert sum(trial["passed"] for trial in first.cycle_records[0]["trials"]) == 4
