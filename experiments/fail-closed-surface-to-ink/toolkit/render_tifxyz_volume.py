#!/usr/bin/env python3
"""Render model-ready surface stacks from tifxyz and a native OME-Zarr volume.

Example for PHerc1447 (the array path/axes are stated explicitly):

    python render_tifxyz_volume.py \
      --tifxyz /data/segment/mesh/intermediate/tifxyz_original \
      --volume https://example/PHerc1447/raw.ome.zarr \
      --volume-array-path 0 --volume-axes zyx \
      --voxel-size-um 8.64 --frames 26 --spacing-um 8.64 \
      --normal-sign both --tile-size 512,512 --output /data/rendered

The output contains ``positive/00.tif ...`` and/or ``negative/00.tif ...``
directly, matching the layer-directory contract used by the existing ink-model
screeners.  ``manifest.json`` records coordinate conventions and QC provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tifxyz_render_pipeline import (
    CalibrationThresholds,
    RenderOptions,
    compare_published_stack,
    load_tifxyz_asset,
    open_volume_zyx,
    render_surface_to_directory,
)


def comma_floats(value: str, *, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated numbers") from exc
    if not result or not np.isfinite(result).all():
        raise argparse.ArgumentTypeError(f"{name} must be non-empty and finite")
    return result


def positive_pair(value: str, *, name: str) -> tuple[int, int]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be HEIGHT,WIDTH integers") from exc
    if len(result) != 2 or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(f"{name} must be two positive integers")
    return result  # type: ignore[return-value]


def parse_voxel_size(value: str) -> tuple[float, float, float]:
    values = comma_floats(value, name="--voxel-size-um")
    if len(values) == 1:
        values = values * 3
    if len(values) != 3 or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "--voxel-size-um accepts one isotropic value or z,y,x positive values"
        )
    return values  # type: ignore[return-value]


def parse_key_values(items: Sequence[str], *, name: str) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{name} entry must be KEY=VALUE, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key or key in result:
            raise ValueError(f"invalid or duplicate {name} key: {key!r}")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        result[key] = value
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render tifxyz surfaces from a local/remote OME-Zarr with one compact "
            "zyx ROI fetch per surface tile."
        )
    )
    parser.add_argument("--tifxyz", required=True, help="Local directory or remote URI")
    parser.add_argument("--volume", required=True, help="Local/remote OME-Zarr or local .npy")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--volume-array-path",
        help="Zarr array path; normally 0 for the base OME-Zarr scale",
    )
    parser.add_argument(
        "--volume-axes",
        default="auto",
        help="Source axis string such as zyx/tczyx, or auto for OME metadata",
    )
    parser.add_argument(
        "--axis-index",
        action="append",
        default=[],
        metavar="AXIS=INDEX",
        help="Fix a non-spatial axis, e.g. --axis-index t=0 --axis-index c=0",
    )
    parser.add_argument(
        "--storage-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="fsspec option; values accept JSON (public S3 often uses anon=true)",
    )
    parser.add_argument(
        "--voxel-size-um",
        required=True,
        type=parse_voxel_size,
        metavar="UM or Z,Y,X",
        help="Physical voxel calibration in explicit z,y,x order",
    )
    parser.add_argument("--offsets-um", help="Exact comma-separated physical offsets")
    parser.add_argument("--frames", type=int, default=26)
    parser.add_argument(
        "--spacing-um",
        type=float,
        help="Physical frame spacing; defaults to the z voxel size",
    )
    parser.add_argument("--center-um", type=float, default=0.0)
    parser.add_argument(
        "--normal-sign",
        choices=("positive", "negative", "both"),
        default="both",
    )
    parser.add_argument(
        "--tile-size",
        default=(512, 512),
        type=lambda value: positive_pair(value, name="--tile-size"),
        metavar="HEIGHT,WIDTH",
    )
    parser.add_argument(
        "--surface-resolution",
        choices=("stored", "full"),
        default="full",
        help=(
            "Metadata-scale full-resolution raster (default) or the sparse "
            "stored tifxyz control grid; use stored only for geometry/debug checks"
        ),
    )
    parser.add_argument(
        "--surface-output-shape",
        type=lambda value: positive_pair(value, name="--surface-output-shape"),
        metavar="HEIGHT,WIDTH",
        help="Explicit output shape, overriding the stored/full derived shape",
    )
    parser.add_argument(
        "--surface-interpolation", choices=("linear", "cubic"), default="linear"
    )
    parser.add_argument(
        "--max-surface-pixels",
        type=int,
        default=250_000_000,
        help="Safety ceiling applied before allocating coordinate rasters",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("source", "uint8", "uint16", "float32"),
        default="source",
    )
    parser.add_argument("--fill-value", type=float, default=0.0)
    parser.add_argument("--png-mode", choices=("none", "preview", "all"), default="preview")
    parser.add_argument("--hash-outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--published-stack", type=Path)
    parser.add_argument("--calibration-iou", type=float, default=0.995)
    parser.add_argument("--calibration-correlation", type=float, default=0.995)
    parser.add_argument("--calibration-mad", type=float, default=2.0)
    parser.add_argument("--calibration-erosion", type=int, default=2)
    parser.add_argument(
        "--expected-calibration-orientation",
        choices=(
            "any",
            "positive-direct",
            "positive-reversed",
            "negative-direct",
            "negative-reversed",
        ),
        default="any",
    )
    parser.add_argument(
        "--require-calibration-pass",
        action="store_true",
        help="Return exit code 2 if published-stack calibration misses any gate",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def offsets_from_args(args: argparse.Namespace) -> tuple[float, ...]:
    if args.offsets_um:
        return comma_floats(args.offsets_um, name="--offsets-um")
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    spacing = args.spacing_um
    if spacing is None:
        spacing = args.voxel_size_um[0]
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("--spacing-um must be finite and positive")
    if not np.isfinite(args.center_um):
        raise ValueError("--center-um must be finite")
    indices = np.arange(args.frames, dtype=np.float64) - (args.frames - 1) / 2.0
    return tuple(float(args.center_um + index * spacing) for index in indices)


def resolve_output_dtype(requested: str, source_dtype: np.dtype[Any]) -> str:
    if requested != "source":
        return requested
    if source_dtype == np.dtype(np.uint8):
        return "uint8"
    if source_dtype == np.dtype(np.uint16):
        return "uint16"
    if np.issubdtype(source_dtype, np.floating):
        return "float32"
    raise ValueError(
        f"source volume dtype {source_dtype} is not model-ready; explicitly choose "
        "--output-dtype uint8, uint16 or float32"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_surface_pixels <= 0:
        parser.error("--max-surface-pixels must be positive")
    try:
        storage_options = parse_key_values(args.storage_option, name="storage option")
        axis_indices_raw = parse_key_values(args.axis_index, name="axis index")
        fixed_indices = {key: int(value) for key, value in axis_indices_raw.items()}
        offsets_um = offsets_from_args(args)
        signs = (
            ("positive", "negative")
            if args.normal_sign == "both"
            else (args.normal_sign,)
        )
        if args.require_calibration_pass and args.published_stack is None:
            raise ValueError("--require-calibration-pass requires --published-stack")

        if not args.quiet:
            print("Loading and validating tifxyz coordinates...", flush=True)
        tifxyz = load_tifxyz_asset(
            args.tifxyz,
            storage_options=storage_options,
            maximum_surface_pixels=args.max_surface_pixels,
            resolution=args.surface_resolution,
            output_shape=args.surface_output_shape,
            interpolation=args.surface_interpolation,
        )
        if not args.quiet:
            print("Opening explicit z,y,x volume view...", flush=True)
        opened_volume = open_volume_zyx(
            args.volume,
            array_path=args.volume_array_path,
            axes=args.volume_axes,
            fixed_indices=fixed_indices,
            storage_options=storage_options,
        )
        output_dtype = resolve_output_dtype(
            args.output_dtype, opened_volume.volume_zyx.dtype
        )
        options = RenderOptions(
            voxel_size_zyx_um=args.voxel_size_um,
            offsets_um=offsets_um,
            signs=signs,
            tile_shape=args.tile_size,
            output_dtype=output_dtype,
            fill_value=args.fill_value,
            png_mode=args.png_mode,
            overwrite=args.overwrite,
            hash_outputs=args.hash_outputs,
        )

        def progress(index: int, total: int, details: Mapping[str, Any]) -> None:
            if args.quiet:
                return
            cadence = max(1, total // 20)
            if index == 1 or index == total or index % cadence == 0:
                suffix = (
                    f" roi={tuple(details['roi_shape_zyx'])}"
                    if details.get("fetched")
                    else " no-valid-samples"
                )
                print(f"tile {index}/{total}{suffix}", flush=True)

        manifest = render_surface_to_directory(
            opened_volume.volume_zyx,
            tifxyz.surface,
            args.output,
            options=options,
            tifxyz_manifest=tifxyz.manifest(),
            volume_manifest=opened_volume.manifest(),
            progress=progress,
        )
        if not args.quiet:
            stats = manifest["render_statistics"]
            print(
                f"Rendered {stats['tiles_fetched']}/{stats['tiles_total']} tiles; "
                f"fetched {stats['roi_bytes_fetched'] / 2**30:.3f} GiB in compact ROIs.",
                flush=True,
            )

        if args.published_stack is not None:
            thresholds = CalibrationThresholds(
                valid_mask_iou=args.calibration_iou,
                median_layer_correlation=args.calibration_correlation,
                median_absolute_difference=args.calibration_mad,
                erosion_pixels=args.calibration_erosion,
            )
            calibration = compare_published_stack(
                args.output,
                args.published_stack,
                thresholds=thresholds,
                expected_orientation=args.expected_calibration_orientation,
            )
            if not args.quiet:
                print(
                    f"Calibration {'PASSED' if calibration['passed'] else 'FAILED'}: "
                    f"selected={calibration['selected_orientation']} "
                    f"best={calibration['best_orientation']}",
                    flush=True,
                )
            if args.require_calibration_pass and not calibration["passed"]:
                return 2
        if not args.quiet:
            print(f"Complete manifest: {args.output / 'manifest.json'}", flush=True)
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, IndexError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
