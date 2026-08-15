from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import audit_pherc0800_published_segments as audit_module  # noqa: E402
from audit_pherc0800_published_segments import (  # noqa: E402
    DiskPayloadFetcher,
    connected_components_summary,
    connected_quad_components_summary,
    declared_seed_alignment,
    fetch_public_bytes,
    largest_clean_rectangle,
    largest_seed_containing_rectangle,
    rectangle_physical_area,
    rectangle_region_mask,
    run_metrics_summary,
    selected_run_metrics,
    selected_run_spatial_continuity,
    transect_category_codes,
)


def test_public_fetch_retries_connection_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"recovered"

    def fake_urlopen(_request, timeout):
        nonlocal attempts
        assert timeout == 1
        attempts += 1
        if attempts == 1:
            raise ConnectionResetError("transient reset")
        return Response()

    monkeypatch.setattr(audit_module.urllib.request, "urlopen", fake_urlopen)
    assert fetch_public_bytes(
        "https://example.invalid/object",
        timeout=1,
        attempts=2,
        backoff_seconds=0,
    ) == b"recovered"
    assert attempts == 2


def paint_run(
    frames: np.ndarray, row: int, column: int, lower: int, upper: int
) -> None:
    frames[lower + 15 : upper + 16, row, column] = 255


def test_centered_selected_run_accepts_far_secondary_run() -> None:
    frames = np.zeros((31, 2, 3), dtype=np.uint8)
    eligible = np.ones((2, 3), dtype=bool)
    paint_run(frames, 0, 0, -1, 1)
    paint_run(frames, 0, 0, 10, 12)
    paint_run(frames, 0, 1, -4, -2)
    paint_run(frames, 0, 1, 2, 4)
    paint_run(frames, 0, 2, -2, 2)
    paint_run(frames, 1, 0, 5, 8)
    paint_run(frames, 1, 1, -1, 1)
    paint_run(frames, 1, 2, -2, 0)

    codes, counts = transect_category_codes(frames, eligible)
    metrics = selected_run_metrics(frames, eligible)

    assert codes[0, 0] == 3
    assert counts["multi_run"] == 2
    assert metrics["contains_zero"][0, 0]
    assert metrics["run_count"][0, 0] == 2
    assert metrics["selected_lower_offset"][0, 0] == -1
    assert metrics["selected_upper_offset"][0, 0] == 1
    assert metrics["nearest_competitor_empty_gap"][0, 0] == 8
    assert not metrics["contains_zero"][0, 1]
    assert metrics["selected_lower_offset"][0, 1] == -4
    assert metrics["selected_upper_offset"][0, 1] == -2
    assert metrics["nearest_competitor_empty_gap"][0, 2] == 15
    summary = run_metrics_summary(metrics, eligible)
    assert summary["centered_selected_run_vertex_count"] == 4
    assert summary["multi_run_vertex_count"] == 2


def test_spatial_continuity_reports_interval_coherence_without_veto() -> None:
    frames = np.zeros((31, 2, 2), dtype=np.uint8)
    eligible = np.ones((2, 2), dtype=bool)
    paint_run(frames, 0, 0, -2, 1)
    paint_run(frames, 0, 1, -1, 2)
    paint_run(frames, 1, 0, -2, 0)
    paint_run(frames, 1, 1, 0, 3)
    metrics = selected_run_metrics(frames, eligible)
    report = selected_run_spatial_continuity(metrics, eligible)
    assert report["edge_count"] == 4
    assert report["interval_overlap_fraction"] == 1.0
    assert report["zero_empty_gap_fraction"] == 1.0
    assert report["selected_center_absolute_jump"]["quantiles"]["maximum"] == 2.5


def brute_largest_rectangle(
    mask: np.ndarray, quad_area: np.ndarray
) -> tuple[float, tuple[int, int, int, int] | None]:
    prefix = np.pad(np.cumsum(np.cumsum(quad_area, axis=0), axis=1), ((1, 0), (1, 0)))
    best_area = -1.0
    best_window = None
    rows, columns = mask.shape
    for row0, row1, column0, column1 in itertools.product(
        range(rows), range(1, rows + 1), range(columns), range(1, columns + 1)
    ):
        if row1 - row0 < 2 or column1 - column0 < 2:
            continue
        if row1 <= row0 or column1 <= column0:
            continue
        if not np.all(mask[row0:row1, column0:column1]):
            continue
        window = (row0, row1, column0, column1)
        area = rectangle_physical_area(prefix, window)
        if area > best_area:
            best_area, best_window = area, window
    return max(0.0, best_area), best_window


def test_weighted_largest_rectangle_matches_brute_force() -> None:
    rng = np.random.default_rng(800)
    for _ in range(30):
        mask = rng.random((5, 6)) > 0.22
        quad_area = rng.uniform(0.01, 3.0, size=(4, 5))
        expected_area, expected_window = brute_largest_rectangle(mask, quad_area)
        actual = largest_clean_rectangle(mask, quad_area)
        if expected_window is None:
            assert actual is None
        else:
            assert actual is not None
            assert np.isclose(actual["area_cm2"], expected_area)
            window = tuple(actual["window_half_open_r0_r1_c0_c1"])
            assert np.all(mask[window[0] : window[1], window[2] : window[3]])


def test_seed_rectangle_and_components_do_not_bridge_false_vertex() -> None:
    mask = np.ones((5, 6), dtype=bool)
    mask[:, 3] = False
    quad_area = np.ones((4, 5), dtype=float)
    seed_rectangle = largest_seed_containing_rectangle(mask, quad_area, (2, 1))
    assert seed_rectangle is not None
    assert seed_rectangle["window_half_open_r0_r1_c0_c1"] == [0, 5, 0, 3]
    summary, labels = connected_components_summary(mask, quad_area)
    assert summary["component_count"] == 2
    assert labels[2, 1] != labels[2, 5]
    region = rectangle_region_mask(mask.shape, seed_rectangle)
    assert np.count_nonzero(region) == 15


def test_quad_components_cannot_cross_vertex_only_bridge() -> None:
    mask = np.zeros((5, 7), dtype=bool)
    mask[0:3, 0:3] = True
    mask[2:5, 4:7] = True
    mask[2, 3] = True  # vertex path joins vertex components but carries no valid quad
    quad_area = np.ones((4, 6), dtype=float)
    vertex_summary, _ = connected_components_summary(mask, quad_area)
    quad_summary, labels = connected_quad_components_summary(mask, quad_area)
    assert vertex_summary["component_count"] == 1
    assert quad_summary["component_count"] == 2
    assert quad_summary["largest_component"]["area_cm2"] == 4.0
    assert np.count_nonzero(labels >= 0) == 8


def test_zero_seed_placeholder_is_not_treated_as_declared_seed() -> None:
    points = np.full((2, 2, 3), 100.0)
    valid = np.ones((2, 2), dtype=bool)
    report = declared_seed_alignment({"seed": [0, 0, 0]}, points, valid)
    assert not report["declared"]
    assert "placeholder" in report["reason"]


def test_disk_payload_fetcher_fetches_once_and_hashes(tmp_path: Path) -> None:
    calls: list[str] = []

    def upstream(url: str) -> bytes:
        calls.append(url)
        return b"compressed-public-m7"

    fetcher = DiskPayloadFetcher(tmp_path / "cache", upstream=upstream)
    assert fetcher("https://example.test/chunk") == b"compressed-public-m7"
    assert fetcher("https://example.test/chunk") == b"compressed-public-m7"
    manifest = fetcher.manifest()
    assert calls == ["https://example.test/chunk"]
    assert manifest["actual_network_fetch_count"] == 1
    assert manifest["payload_reads"] == 2
    assert manifest["records"][0]["sha256"]
