#!/usr/bin/env python3
"""Build the four-panel visual summary for the frozen v0.2 benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tifxyz_doctor.audit import AuditConfig, audit_mesh  # noqa: E402
from tifxyz_doctor.io import load_tifxyz  # noqa: E402
from tifxyz_doctor.reviewed_benchmark import inject_normal_offset_switch  # noqa: E402


DEFAULT_PATCH_ID = "same_wrap000148_lasagna"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "assets" / "reviewed-benchmark-four-panel.png"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data",
        type=Path,
        help="Root containing the official reviewed same_wrap patch directories",
    )
    parser.add_argument("--patch-id", default=DEFAULT_PATCH_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _style_axis(axis: plt.Axes, tag: str, title: str, subtitle: str) -> None:
    axis.text(
        0,
        1.085,
        title,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color="#f8fafc",
        fontsize=15,
        fontweight="bold",
    )
    axis.text(
        0,
        1.025,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color="#a9b4c5",
        fontsize=9.5,
    )
    axis.text(
        -0.025,
        1.105,
        tag,
        transform=axis.transAxes,
        ha="right",
        va="center",
        color="#07111f",
        fontsize=11,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "#76e4c4",
            "edgecolor": "none",
        },
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#344155")
        spine.set_linewidth(1.0)


def _seam_annotation(axis: plt.Axes, orientation: str, index: int, *, band: int = 0) -> None:
    if orientation == "horizontal":
        if band:
            axis.axhspan(index - band / 2, index + band / 2, color="#76e4c4", alpha=0.14)
        axis.axhline(index - 0.5, color="#76e4c4", linewidth=1.3, linestyle=(0, (4, 3)))
    else:
        if band:
            axis.axvspan(index - band / 2, index + band / 2, color="#76e4c4", alpha=0.14)
        axis.axvline(index - 0.5, color="#76e4c4", linewidth=1.3, linestyle=(0, (4, 3)))


def _incremental_mask(case_report: dict, base_report: dict) -> np.ndarray:
    case = np.asarray(case_report["_arrays"]["coherent_normal_step_cells"], dtype=bool)
    base = np.asarray(base_report["_arrays"]["coherent_normal_step_cells"], dtype=bool)
    return case & ~base


def _zoom_bounds(mask: np.ndarray, shape: tuple[int, int], margin: int = 11) -> tuple[int, int, int, int]:
    locations = np.argwhere(mask)
    if locations.size == 0:
        return 0, shape[0], 0, shape[1]
    row_min, col_min = locations.min(axis=0)
    row_max, col_max = locations.max(axis=0)
    return (
        max(0, int(row_min) - margin),
        min(shape[0], int(row_max) + margin + 1),
        max(0, int(col_min) - margin),
        min(shape[1], int(col_max) + margin + 1),
    )


def _show_crop(
    axis: plt.Axes,
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> None:
    row0, row1, col0, col1 = bounds
    axis.imshow(
        image[row0:row1, col0:col1],
        interpolation="nearest",
        extent=(col0, col1, row1, row0),
    )


def _cue_image(valid: np.ndarray, cue_mask: np.ndarray) -> np.ndarray:
    rgb = np.empty((*valid.shape, 3), dtype=np.uint8)
    rgb[:] = (31, 41, 55)
    rgb[~valid] = (15, 19, 26)
    rgb[valid & cue_mask] = (248, 55, 54)
    return rgb


def main() -> int:
    args = _parser().parse_args()
    patch_path = args.data / args.patch_id
    data = load_tifxyz(patch_path)
    config = AuditConfig()

    base_report = audit_mesh(data, config)
    abrupt = inject_normal_offset_switch(
        data,
        offset_voxels=16.0,
        transition_width_cells=1,
    )
    gradual = inject_normal_offset_switch(
        data,
        offset_voxels=16.0,
        transition_width_cells=12,
    )
    abrupt_report = audit_mesh(abrupt.data, config)
    gradual_report = audit_mesh(gradual.data, config)

    base_mask = np.asarray(
        base_report["_arrays"]["coherent_normal_step_cells"],
        dtype=bool,
    )
    abrupt_mask = _incremental_mask(abrupt_report, base_report)
    gradual_mask = _incremental_mask(gradual_report, base_report)
    if base_mask.any():
        raise RuntimeError(f"{args.patch_id}: expected a cue-free baseline")
    if not abrupt_mask.any():
        raise RuntimeError(f"{args.patch_id}: abrupt proxy was not localized")
    if gradual_mask.any():
        raise RuntimeError(f"{args.patch_id}: gradual proxy unexpectedly localized")

    valid_cells = np.asarray(base_report["_arrays"]["valid_cells"], dtype=bool)
    base_image = _cue_image(valid_cells, base_mask)
    abrupt_image = _cue_image(valid_cells, abrupt_mask)
    gradual_image = _cue_image(valid_cells, gradual_mask)
    surface_bounds = _zoom_bounds(valid_cells, valid_cells.shape, margin=8)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#101722",
            "figure.facecolor": "#09111d",
            "savefig.facecolor": "#09111d",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), constrained_layout=False)
    figure.subplots_adjust(left=0.055, right=0.975, top=0.855, bottom=0.105, wspace=0.13, hspace=0.31)

    figure.text(
        0.055,
        0.953,
        "What the frozen v0.2 cue sees—and what it misses",
        color="#f8fafc",
        fontsize=24,
        fontweight="bold",
        ha="left",
    )
    figure.text(
        0.055,
        0.915,
        "One overlap-isolated holdout surface · unchanged thresholds · exact cell-space output",
        color="#a9b4c5",
        fontsize=12,
        ha="left",
    )

    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    _show_crop(ax_a, base_image, surface_bounds)
    _style_axis(
        ax_a,
        "A",
        "Reviewed same-wrap surface",
        f"{args.patch_id} · 0 coherent-step cells",
    )

    _show_crop(ax_b, abrupt_image, surface_bounds)
    _seam_annotation(ax_b, abrupt.orientation, abrupt.seam_index)
    _style_axis(
        ax_b,
        "B",
        "Abrupt 16-voxel proxy: detected",
        f"{int(abrupt_mask.sum())} incremental cells · cyan line is the imposed seam",
    )

    _show_crop(ax_c, gradual_image, surface_bounds)
    _seam_annotation(
        ax_c,
        gradual.orientation,
        gradual.seam_index,
        band=gradual.transition_width_cells,
    )
    _style_axis(
        ax_c,
        "C",
        "Gradual 12-cell proxy: missed",
        "0 incremental cells · shaded corridor is the smooth transition",
    )

    row0, row1, col0, col1 = _zoom_bounds(abrupt_mask, abrupt_mask.shape)
    _show_crop(ax_d, abrupt_image, (row0, row1, col0, col1))
    ax_d.add_patch(
        Rectangle(
            (col0, row0),
            col1 - col0,
            row1 - row0,
            fill=False,
            edgecolor="#76e4c4",
            linewidth=1.2,
        )
    )
    score = np.asarray(abrupt_report["_arrays"]["review_score"], dtype=float)
    locations = np.argwhere(abrupt_mask)
    ranked = sorted(
        (tuple(map(int, rc)) for rc in locations),
        key=lambda rc: (-float(score[rc]), rc),
    )
    selected: list[tuple[int, int]] = []
    for row, col in ranked:
        if all(abs(row - old_row) + abs(col - old_col) >= 5 for old_row, old_col in selected):
            selected.append((row, col))
        if len(selected) == 5:
            break
    for index, (row, col) in enumerate(selected, start=1):
        ax_d.scatter(
            [col + 0.5],
            [row + 0.5],
            s=62,
            facecolor="#f8fafc",
            edgecolor="#07111f",
            linewidth=1.0,
            zorder=4,
        )
        ax_d.text(
            col + 0.5,
            row + 0.5,
            str(index),
            ha="center",
            va="center",
            color="#07111f",
            fontsize=7.5,
            fontweight="bold",
            zorder=5,
        )
    coordinate_text = "  ".join(
        f"{index}  [{row}, {col}]"
        for index, (row, col) in enumerate(selected, start=1)
    )
    _style_axis(
        ax_d,
        "D",
        "Exact sparse localization",
        coordinate_text,
    )

    figure.text(
        0.055,
        0.044,
        (
            "Controlled normal-offset proxies on real reviewed geometry; they are not labels for naturally occurring "
            "sheet switches.\nExternal natural-data behavior is reported separately; no real-switch recall is claimed."
        ),
        color="#8d99aa",
        fontsize=8.8,
        ha="left",
    )
    figure.text(
        0.975,
        0.044,
        "TIFXYZ Doctor v0.2 · cell coordinates are [row, col]",
        color="#8d99aa",
        fontsize=9.5,
        ha="right",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
