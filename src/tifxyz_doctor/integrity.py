"""Raw TIFXYZ integrity and Python/C++ interoperability checks.

This module deliberately reads the files on disk rather than consuming
``TifxyzData``.  That makes it possible to report defects which a reader may
otherwise normalize away, notably non-canonical sentinels and mask values
whose meaning differs between the current villa Python and C++ readers.

Only Pillow and NumPy are required.  The returned report contains plain JSON
types, is deterministic for unchanged input, and can be serialized with
``allow_nan=False``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FILES = ("x.tif", "y.tif", "z.tif", "meta.json")
SCHEMA_VERSION = "tifxyz-integrity-v1"

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass
class _ImageRead:
    array: np.ndarray | None
    info: dict[str, Any]


class _Findings:
    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        *,
        file: str | None = None,
        count: int | None = None,
    ) -> None:
        item: dict[str, Any] = {
            "severity": severity,
            "code": code,
            "message": message,
        }
        if file is not None:
            item["file"] = file
        if count is not None:
            item["count"] = int(count)
        self._items.append(item)

    def sorted(self) -> list[dict[str, Any]]:
        return sorted(
            self._items,
            key=lambda item: (
                _SEVERITY_ORDER.get(str(item["severity"]), 99),
                str(item["code"]),
                str(item.get("file", "")),
                str(item["message"]),
            ),
        )


def _empty_tiff_info(filename: str) -> dict[str, Any]:
    return {
        "filename": filename,
        "readable": False,
        "shape": None,
        "dtype": None,
        "mode": None,
        "frames": None,
    }


def _read_tiff(
    path: Path,
    *,
    role: str,
    findings: _Findings,
) -> _ImageRead:
    """Read one TIFF without coercing its dtype or dimensionality."""
    info = _empty_tiff_info(path.name)
    try:
        from PIL import Image

        with Image.open(path) as image:
            frames = int(getattr(image, "n_frames", 1))
            mode = str(image.mode)
            image.load()
            array = np.asarray(image)
    except Exception as exc:
        findings.add(
            "error",
            "tiff-unreadable",
            f"{path.name} could not be decoded by Pillow: {type(exc).__name__}: {exc}",
            file=path.name,
        )
        return _ImageRead(None, info)

    info.update(
        {
            "readable": True,
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "mode": mode,
            "frames": frames,
        }
    )

    if frames != 1:
        findings.add(
            "error",
            "tiff-multiple-images",
            f"{path.name} contains {frames} images; a TIFXYZ component must be single-image.",
            file=path.name,
            count=frames,
        )
    if array.ndim != 2:
        findings.add(
            "error",
            "tiff-not-2d",
            f"{path.name} has shape {tuple(array.shape)}; a TIFXYZ component must be 2-D.",
            file=path.name,
        )
        return _ImageRead(None, info)
    is_supported_numeric = (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.bool_)
    )
    if not is_supported_numeric:
        findings.add(
            "error",
            "tiff-nonnumeric",
            f"{path.name} has non-numeric dtype {array.dtype}.",
            file=path.name,
        )
        return _ImageRead(None, info)

    if role == "coordinate" and array.dtype != np.dtype(np.float32):
        findings.add(
            "warning",
            "coordinate-dtype-nonstandard",
            (
                f"{path.name} has dtype {array.dtype}; canonical villa writers emit "
                "float32 coordinates and readers will coerce this input."
            ),
            file=path.name,
        )
    if role == "mask" and array.dtype != np.dtype(np.uint8):
        findings.add(
            "warning",
            "mask-dtype-nonstandard",
            (
                f"{path.name} has dtype {array.dtype}; uint8 with values 0 and 255 "
                "is the portable mask representation."
            ),
            file=path.name,
        )

    return _ImageRead(array, info)


def _is_json_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_float(value: Any) -> float | None:
    if not _is_json_number(value):
        return None
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _count_nonfinite_json_numbers(value: Any) -> int:
    if isinstance(value, float):
        return int(not math.isfinite(value))
    if isinstance(value, dict):
        return sum(_count_nonfinite_json_numbers(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_nonfinite_json_numbers(item) for item in value)
    return 0


def _read_metadata(path: Path, findings: _Findings) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    info: dict[str, Any] = {
        "readable": False,
        "root_type": None,
        "uuid": None,
        "format": None,
        "type": None,
        "scale": None,
        "bbox": None,
        "area": None,
        "area_vx2": None,
        "area_cm2": None,
    }
    try:
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception as exc:
        findings.add(
            "error",
            "metadata-json-invalid",
            f"meta.json could not be parsed: {type(exc).__name__}: {exc}",
            file="meta.json",
        )
        return None, info

    info["readable"] = True
    info["root_type"] = type(metadata).__name__
    if not isinstance(metadata, dict):
        findings.add(
            "error",
            "metadata-not-object",
            "meta.json must contain a JSON object.",
            file="meta.json",
        )
        return None, info

    nonfinite_count = _count_nonfinite_json_numbers(metadata)
    if nonfinite_count:
        findings.add(
            "error",
            "metadata-nonfinite-number",
            (
                f"meta.json contains {nonfinite_count} NaN or infinity value(s), "
                "which are not strict JSON numbers."
            ),
            file="meta.json",
            count=nonfinite_count,
        )

    uuid = metadata.get("uuid")
    if isinstance(uuid, str) and uuid.strip():
        info["uuid"] = uuid
    else:
        findings.add(
            "error",
            "metadata-uuid-invalid",
            "meta.json must contain a non-empty string uuid for the C++ loader.",
            file="meta.json",
        )

    format_value = metadata.get("format")
    if isinstance(format_value, str):
        info["format"] = format_value
    if format_value != "tifxyz":
        findings.add(
            "warning",
            "metadata-format-nonstandard",
            "meta.json format should be the string \"tifxyz\".",
            file="meta.json",
        )

    type_value = metadata.get("type")
    if isinstance(type_value, str):
        info["type"] = type_value
    if type_value != "seg":
        findings.add(
            "warning",
            "metadata-type-nonstandard",
            "meta.json type should be the string \"seg\".",
            file="meta.json",
        )

    raw_scale = metadata.get("scale")
    scale_values = (
        [_finite_float(value) for value in raw_scale[:2]]
        if isinstance(raw_scale, list) and len(raw_scale) >= 2
        else []
    )
    scale_valid = len(scale_values) == 2 and all(
        value is not None and value > 0 for value in scale_values
    )
    if scale_valid:
        # On disk this is [x_scale, y_scale].  Python reverses it internally.
        info["scale"] = [float(scale_values[0]), float(scale_values[1])]
    else:
        findings.add(
            "error",
            "metadata-scale-invalid",
            (
                "meta.json scale must begin with two positive finite numbers "
                "in [x_scale, y_scale] order."
            ),
            file="meta.json",
        )

    raw_bbox = metadata.get("bbox")
    if raw_bbox is not None:
        bbox: np.ndarray | None = None
        try:
            candidate = np.asarray(raw_bbox, dtype=np.float64)
            if candidate.shape == (2, 3):
                bbox = candidate
        except (TypeError, ValueError, OverflowError):
            bbox = None
        if (
            bbox is None
            or not np.isfinite(bbox).all()
            or bool(np.any(bbox[0] > bbox[1]))
        ):
            findings.add(
                "error",
                "metadata-bbox-invalid",
                (
                    "meta.json bbox must be [[min_x,min_y,min_z],"
                    "[max_x,max_y,max_z]] with finite ordered bounds."
                ),
                file="meta.json",
            )
        else:
            info["bbox"] = [
                [float(value) for value in bbox[0]],
                [float(value) for value in bbox[1]],
            ]

    raw_area = metadata.get("area")
    if raw_area is not None:
        parsed_area = _finite_float(raw_area)
        if parsed_area is not None and parsed_area >= 0:
            info["area"] = parsed_area
        else:
            findings.add(
                "warning",
                "metadata-area-invalid",
                "meta.json area, when present, should be a finite non-negative number.",
                file="meta.json",
            )

    for field in ("area_vx2", "area_cm2"):
        raw_value = metadata.get(field)
        if raw_value is None:
            continue
        parsed_value = _finite_float(raw_value)
        if parsed_value is not None and parsed_value >= 0:
            info[field] = parsed_value
        else:
            findings.add(
                "warning",
                f"metadata-{field.replace('_', '-')}-invalid",
                (
                    f"meta.json {field}, when present, should be a finite "
                    "non-negative number."
                ),
                file="meta.json",
            )

    if raw_area is None and (
        info["area_vx2"] is not None or info["area_cm2"] is not None
    ):
        findings.add(
            "warning",
            "metadata-area-schema-divergence",
            (
                "meta.json supplies area_vx2/area_cm2 but no area; the current "
                "Villa Python reader preserves those fields only as extra metadata "
                "and leaves Tifxyz.area unset."
            ),
            file="meta.json",
        )

    return metadata, info


def _mask_relation(
    mask_shape: tuple[int, int],
    coordinate_shape: tuple[int, int],
) -> tuple[str, tuple[int, int] | None]:
    if mask_shape == coordinate_shape:
        return "exact", (1, 1)
    mask_h, mask_w = mask_shape
    height, width = coordinate_shape
    if (
        height > 0
        and width > 0
        and mask_h % height == 0
        and mask_w % width == 0
    ):
        return "integer_multiple", (mask_h // height, mask_w // width)
    return "incompatible", None


def _cpp_mask_at_coordinate_resolution(
    mask: np.ndarray,
    coordinate_shape: tuple[int, int],
    relation: str,
    factors: tuple[int, int] | None,
) -> np.ndarray | None:
    """Model the C++ loader's mask-retain threshold and downsampling."""
    if relation == "incompatible" or factors is None:
        return None

    # Current C++ treats 255 as the retain threshold for all numeric formats.
    # Comparisons with NaN are false, which correctly invalidates them here.
    retained = mask >= 255
    if relation == "exact":
        return np.asarray(retained, dtype=bool)

    factor_y, factor_x = factors
    height, width = coordinate_shape
    # Any invalid high-resolution sample invalidates the corresponding stored
    # vertex, so every sample in a factor block must meet the retain threshold.
    return retained.reshape(height, factor_y, width, factor_x).all(axis=(1, 3))


def _valid_face_count(valid: np.ndarray) -> int:
    if valid.shape[0] < 2 or valid.shape[1] < 2:
        return 0
    faces = (
        valid[:-1, :-1]
        & valid[:-1, 1:]
        & valid[1:, :-1]
        & valid[1:, 1:]
    )
    return int(faces.sum())


def _actual_bbox(
    components: tuple[np.ndarray, np.ndarray, np.ndarray],
    valid: np.ndarray,
) -> list[list[float]] | None:
    if not bool(valid.any()):
        return None
    minima = [float(np.min(component[valid])) for component in components]
    maxima = [float(np.max(component[valid])) for component in components]
    return [minima, maxima]


def _bbox_comparison(
    declared: list[list[float]] | None,
    actual: list[list[float]] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "comparable": False,
        "matches": None,
        "max_absolute_delta": None,
        "tolerance_by_axis": None,
    }
    if declared is None or actual is None:
        return result

    declared_array = np.asarray(declared, dtype=np.float64)
    actual_array = np.asarray(actual, dtype=np.float64)
    magnitude = np.maximum(np.max(np.abs(actual_array), axis=0), 1.0)
    tolerance = 1e-4 + 1e-6 * magnitude
    delta = np.abs(declared_array - actual_array)
    result.update(
        {
            "comparable": True,
            "matches": bool(np.all(delta <= tolerance[None, :])),
            "max_absolute_delta": float(np.max(delta)),
            "tolerance_by_axis": [float(value) for value in tolerance],
        }
    )
    return result


def _empty_report(path: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(path),
        "status": "error",
        "summary": {"errors": 0, "warnings": 0, "info": 0},
        "files": {
            "required": {name: False for name in REQUIRED_FILES},
            "optional_mask": False,
        },
        "metadata": {
            "readable": False,
            "root_type": None,
            "uuid": None,
            "format": None,
            "type": None,
            "scale": None,
            "bbox": None,
            "area": None,
            "area_vx2": None,
            "area_cm2": None,
        },
        "tiffs": {
            "x": _empty_tiff_info("x.tif"),
            "y": _empty_tiff_info("y.tif"),
            "z": _empty_tiff_info("z.tif"),
            "mask": None,
        },
        "coordinates": {
            "shape": None,
            "vertex_count": None,
            "finite_vertex_count": None,
            "nonfinite_vertex_count": None,
            "nonfinite_by_channel": None,
            "canonical_sentinel_vertex_count": None,
            "partial_sentinel_vertex_count": None,
            "nonpositive_z_nonsentinel_vertex_count": None,
        },
        "validity": {
            "mask_present": False,
            "mask_shape_relation": "absent",
            "mask_scale_factors_yx": None,
            "mask_values": None,
            "python_rule": None,
            "python_valid_vertex_count": None,
            "cpp_rule": None,
            "cpp_valid_mask_vertex_count": None,
            "cpp_finite_point_vertex_count": None,
            "python_cpp_valid_mask_disagreement_count": None,
            "python_cpp_finite_point_disagreement_count": None,
            "portable_valid_vertex_count": None,
        },
        "geometry": {
            "portable_valid_face_count": None,
            "actual_bbox": None,
            "bbox_comparison": {
                "comparable": False,
                "matches": None,
                "max_absolute_delta": None,
                "tolerance_by_axis": None,
            },
        },
        "findings": [],
    }


def _finish_report(report: dict[str, Any], findings: _Findings) -> dict[str, Any]:
    items = findings.sorted()
    counts = {
        severity: sum(item["severity"] == severity for item in items)
        for severity in ("error", "warning", "info")
    }
    report["findings"] = items
    report["summary"] = {
        "errors": int(counts["error"]),
        "warnings": int(counts["warning"]),
        "info": int(counts["info"]),
    }
    if counts["error"]:
        report["status"] = "error"
    elif counts["warning"]:
        report["status"] = "warning"
    else:
        report["status"] = "pass"

    # This is an invariant of the API, not merely a property of the dumper.
    json.dumps(report, allow_nan=False, sort_keys=True)
    return report


def audit_tifxyz_integrity(path: str | Path) -> dict[str, Any]:
    """Audit the raw files in one TIFXYZ directory.

    The function is diagnostic: malformed input is represented by findings in
    the returned report instead of being normalized or raising at the first
    defect.  Files which cannot be decoded naturally prevent dependent checks.
    """
    directory = Path(path)
    report = _empty_report(directory)
    findings = _Findings()

    if not directory.is_dir():
        findings.add(
            "error",
            "path-not-directory",
            f"TIFXYZ path is not a directory: {directory}",
        )
        return _finish_report(report, findings)

    required_presence = {
        name: bool((directory / name).is_file()) for name in REQUIRED_FILES
    }
    mask_present = bool((directory / "mask.tif").is_file())
    report["files"] = {
        "required": required_presence,
        "optional_mask": mask_present,
    }
    report["validity"]["mask_present"] = mask_present

    for filename, present in required_presence.items():
        if not present:
            findings.add(
                "error",
                "missing-required-file",
                f"Required TIFXYZ file is missing: {filename}",
                file=filename,
            )

    metadata: dict[str, Any] | None = None
    if required_presence["meta.json"]:
        metadata, metadata_info = _read_metadata(directory / "meta.json", findings)
        report["metadata"] = metadata_info

    coordinate_reads: dict[str, _ImageRead] = {}
    for axis in ("x", "y", "z"):
        filename = f"{axis}.tif"
        if required_presence[filename]:
            coordinate_reads[axis] = _read_tiff(
                directory / filename,
                role="coordinate",
                findings=findings,
            )
            report["tiffs"][axis] = coordinate_reads[axis].info

    mask_read: _ImageRead | None = None
    if mask_present:
        mask_read = _read_tiff(
            directory / "mask.tif",
            role="mask",
            findings=findings,
        )
        report["tiffs"]["mask"] = mask_read.info

    if not all(
        axis in coordinate_reads and coordinate_reads[axis].array is not None
        for axis in ("x", "y", "z")
    ):
        return _finish_report(report, findings)

    components = tuple(
        coordinate_reads[axis].array for axis in ("x", "y", "z")
    )
    assert all(component is not None for component in components)
    x, y, z = components
    assert x is not None and y is not None and z is not None

    shapes = (tuple(x.shape), tuple(y.shape), tuple(z.shape))
    if len(set(shapes)) != 1:
        findings.add(
            "error",
            "coordinate-shape-mismatch",
            f"Coordinate shapes differ: x={shapes[0]}, y={shapes[1]}, z={shapes[2]}.",
        )
        return _finish_report(report, findings)

    coordinate_shape = shapes[0]
    height, width = coordinate_shape
    vertex_count = int(height * width)

    finite_x = np.isfinite(x)
    finite_y = np.isfinite(y)
    finite_z = np.isfinite(z)
    all_finite = finite_x & finite_y & finite_z
    any_nan = np.isnan(x) | np.isnan(y) | np.isnan(z)
    sentinel_x = x == -1
    sentinel_y = y == -1
    sentinel_z = z == -1
    sentinel_all = sentinel_x & sentinel_y & sentinel_z
    sentinel_any = sentinel_x | sentinel_y | sentinel_z
    partial_sentinel = sentinel_any & ~sentinel_all
    z_positive_finite = finite_z & (z > 0)
    nonpositive_z_nonsentinel = (z <= 0) & ~sentinel_all

    nonfinite_by_channel = {
        "x": int((~finite_x).sum()),
        "y": int((~finite_y).sum()),
        "z": int((~finite_z).sum()),
    }
    report["coordinates"] = {
        "shape": [int(height), int(width)],
        "vertex_count": vertex_count,
        "finite_vertex_count": int(all_finite.sum()),
        "nonfinite_vertex_count": int((~all_finite).sum()),
        "nonfinite_by_channel": nonfinite_by_channel,
        "canonical_sentinel_vertex_count": int(sentinel_all.sum()),
        "partial_sentinel_vertex_count": int(partial_sentinel.sum()),
        "nonpositive_z_nonsentinel_vertex_count": int(nonpositive_z_nonsentinel.sum()),
    }

    nonfinite_count = int((~all_finite).sum())
    if nonfinite_count:
        findings.add(
            "error",
            "nonfinite-coordinate",
            f"{nonfinite_count} vertices contain NaN or infinity in at least one coordinate.",
            count=nonfinite_count,
        )
    partial_sentinel_count = int(partial_sentinel.sum())
    if partial_sentinel_count:
        findings.add(
            "error",
            "partial-sentinel",
            (
                f"{partial_sentinel_count} vertices contain -1 in only some channels; "
                "the portable invalid sentinel is (-1, -1, -1)."
            ),
            count=partial_sentinel_count,
        )
    noncanonical_invalid_count = int(nonpositive_z_nonsentinel.sum())
    if noncanonical_invalid_count:
        findings.add(
            "warning",
            "noncanonical-invalid-coordinate",
            (
                f"{noncanonical_invalid_count} vertices have z <= 0 without the "
                "canonical (-1, -1, -1) sentinel; readers will rewrite them."
            ),
            count=noncanonical_invalid_count,
        )

    mask_relation = "absent"
    mask_factors: tuple[int, int] | None = None
    mask_values: dict[str, int] | None = None
    python_mask: np.ndarray | None = None
    cpp_mask: np.ndarray | None = None
    mask = mask_read.array if mask_read is not None else None
    if mask is not None:
        mask_relation, mask_factors = _mask_relation(
            tuple(mask.shape), coordinate_shape
        )
        finite_mask = np.isfinite(mask)
        mask_values = {
            "zero": int((finite_mask & (mask == 0)).sum()),
            "low_nonzero_below_255": int(
                (finite_mask & (mask != 0) & (mask < 255)).sum()
            ),
            "at_least_255": int((finite_mask & (mask >= 255)).sum()),
            "nonfinite": int((~finite_mask).sum()),
        }
        if mask_values["nonfinite"]:
            findings.add(
                "error",
                "mask-nonfinite",
                f"mask.tif contains {mask_values['nonfinite']} NaN or infinity value(s).",
                file="mask.tif",
                count=mask_values["nonfinite"],
            )
        if mask_values["low_nonzero_below_255"]:
            findings.add(
                "error",
                "mask-low-nonzero-interoperability",
                (
                    f"mask.tif contains {mask_values['low_nonzero_below_255']} nonzero "
                    "value(s) below 255; Python treats them as valid while C++ rejects them."
                ),
                file="mask.tif",
                count=mask_values["low_nonzero_below_255"],
            )

        if mask_relation == "exact":
            python_mask = mask != 0
        elif mask_relation == "integer_multiple":
            findings.add(
                "error",
                "mask-shape-python-incompatible",
                (
                    f"mask.tif shape {tuple(mask.shape)} is an integer multiple of "
                    f"coordinate shape {coordinate_shape}; C++ applies it but Python "
                    "ignores it and derives validity from z."
                ),
                file="mask.tif",
            )
        else:
            findings.add(
                "error",
                "mask-shape-incompatible",
                (
                    f"mask.tif shape {tuple(mask.shape)} is not equal to, or an integer "
                    f"multiple of, coordinate shape {coordinate_shape}; both readers "
                    "ignore its validity information."
                ),
                file="mask.tif",
            )
        cpp_mask = _cpp_mask_at_coordinate_resolution(
            mask,
            coordinate_shape,
            mask_relation,
            mask_factors,
        )

    if python_mask is not None:
        python_valid = np.asarray(python_mask, dtype=bool)
        python_rule = "exact-size mask != 0"
    else:
        python_valid = z_positive_finite
        python_rule = "finite(z) and z > 0"

    # Current C++ load order is a hard z <= 0 rejection followed by a mask
    # which can only invalidate.  QuadSurface::validMask additionally rejects
    # NaNs and the full sentinel, but currently does not reject infinity.
    cpp_valid = ~(z <= 0) & ~any_nan & ~sentinel_all
    cpp_rule = "not(z <= 0), then mask >= 255; reject NaN/full sentinel"
    if cpp_mask is not None:
        cpp_valid &= cpp_mask

    # isValidPointSample is stricter than QuadSurface::validMask: every channel
    # must be finite and different from -1.
    cpp_finite_point = cpp_valid & all_finite & ~sentinel_any
    portable_valid = python_valid & cpp_finite_point
    valid_mask_disagreement = python_valid ^ cpp_valid
    finite_point_disagreement = python_valid ^ cpp_finite_point

    if python_mask is not None:
        revalidated = python_mask & ~z_positive_finite
        revalidated_count = int(revalidated.sum())
        if revalidated_count:
            findings.add(
                "error",
                "mask-revalidates-nonpositive-or-nonfinite-z",
                (
                    f"The exact-size mask marks {revalidated_count} vertices valid "
                    "although z is non-positive or non-finite; Python accepts these "
                    "vertices while C++ retains z as a hard validity gate."
                ),
                file="mask.tif",
                count=revalidated_count,
            )

    disagreement_count = int(finite_point_disagreement.sum())
    if disagreement_count:
        findings.add(
            "error",
            "python-cpp-validity-disagreement",
            (
                f"Python validity and C++ finite-point validity disagree at "
                f"{disagreement_count} vertices."
            ),
            count=disagreement_count,
        )

    report["validity"] = {
        "mask_present": mask_present,
        "mask_shape_relation": mask_relation,
        "mask_scale_factors_yx": (
            [int(mask_factors[0]), int(mask_factors[1])]
            if mask_factors is not None
            else None
        ),
        "mask_values": mask_values,
        "python_rule": python_rule,
        "python_valid_vertex_count": int(python_valid.sum()),
        "cpp_rule": cpp_rule,
        "cpp_valid_mask_vertex_count": int(cpp_valid.sum()),
        "cpp_finite_point_vertex_count": int(cpp_finite_point.sum()),
        "python_cpp_valid_mask_disagreement_count": int(
            valid_mask_disagreement.sum()
        ),
        "python_cpp_finite_point_disagreement_count": disagreement_count,
        "portable_valid_vertex_count": int(portable_valid.sum()),
    }

    valid_faces = _valid_face_count(portable_valid)
    actual_bbox = _actual_bbox((x, y, z), portable_valid)
    declared_bbox = report["metadata"]["bbox"]
    bbox_comparison = _bbox_comparison(declared_bbox, actual_bbox)
    report["geometry"] = {
        "portable_valid_face_count": valid_faces,
        "actual_bbox": actual_bbox,
        "bbox_comparison": bbox_comparison,
    }

    if not bool(portable_valid.any()):
        findings.add(
            "error",
            "no-portable-valid-vertex",
            "No vertex is valid under both Python and finite C++ point semantics.",
        )
    if valid_faces == 0:
        findings.add(
            "error",
            "no-valid-face",
            (
                "No 2x2 grid cell has four portable valid vertices; the surface "
                "contains no renderable quad face."
            ),
        )
    if bbox_comparison["comparable"] and not bbox_comparison["matches"]:
        findings.add(
            "warning",
            "metadata-bbox-mismatch",
            (
                "meta.json bbox does not match the bounds of portable valid vertices "
                f"(max absolute delta {bbox_comparison['max_absolute_delta']:.9g})."
            ),
            file="meta.json",
        )

    # Keep the local variable meaningful for static checkers and document that
    # metadata validation is independent from the selected, sanitized fields.
    _ = metadata
    return _finish_report(report, findings)


def dumps_integrity_report(report: dict[str, Any], *, indent: int | None = 2) -> str:
    """Serialize an integrity report as deterministic strict JSON."""
    return json.dumps(
        report,
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
    )
