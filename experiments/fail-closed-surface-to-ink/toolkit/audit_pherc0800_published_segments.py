#!/usr/bin/env python3
"""Free, fail-closed PHerc0800 audit of six published mesh-only segments.

Inputs are resolved from VC3D's local open-data catalog and cross-checked against
the captured S3 inventory.  Only each segment's small TIFFXYZ files and exact
public-m7 chunks touched by trilinear samples are fetched.  Raw CT, normal-grid
payloads, ink detectors, and paid compute are deliberately out of scope.

For every segment the audit performs stored-grid native geometry, bounds,
normal, and +/-15 endpoint checks; samples the official m7 volume over integer
normal offsets -15..+15; requires canonical quantization after trilinear
sampling; and identifies conservative centered-run connected and rectangular
regions.  A candidate vertex must be outside a one-cell geometry-defect margin
and have an m7 foreground run containing offset zero.  Additional, more distant
runs are expected between adjacent papyrus wraps and are therefore measured as
competitor-clearance diagnostics rather than used as a veto.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from preflight_debug_m7 import (
    DEFAULT_SUPPORT_OFFSETS,
    DEFAULT_TRANSECT_OFFSETS,
    dilate_one_cell,
    sample_offsets,
)
from sample_m7_seed_chunk import DEFAULT_BLOSC
from sample_raw_ct_seed_cube import (
    ZarrV2ArraySpec,
    fetch_json,
    sha256_bytes,
    sha256_file,
)
from sweep_public_m7_uniform_offsets import hard_geometry_summary, surface_area_cm2
from tifxyz_render_pipeline import load_tifxyz_asset
from validate_public_m7_patch import (
    PublicZarrChunkSampler,
    _array_sha256,
    _atomic_json,
    support_summary,
    validate_metadata_axes,
)

SAMPLE_ID = "PHerc0800"
EXPECTED_VOLUME_ID = "20250521135224"
EXPECTED_VOXEL_UM = 8.64
EXPECTED_VOLUME_SHAPE_ZYX = (24298, 9867, 9867)
DEFAULT_CATALOG = (
    Path.home()
    / "Library"
    / "Caches"
    / "Vesuvius Challenge"
    / "VC3D"
    / "open-data-catalog"
    / "metadata.json"
)
DEFAULT_CATALOG_CACHE = DEFAULT_CATALOG.with_name("metadata.cache.json")
DEFAULT_INVENTORY = Path("work/first-letters/inventory/PHerc0800-segments.xml")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/first-letters-geometry/pherc0800-published-segments-preflight"
)
DEFAULT_CALIBRATION_SOURCE = Path(
    "/private/tmp/pherc1447-calibration-m7-probe-r8-c119-20x20"
)
DEFAULT_CALIBRATION_RESULT = Path(
    "/private/tmp/pherc1447-calibration-m7-probe-result-network"
)
HTTPS_ROOT = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"


Window = tuple[int, int, int, int]  # half-open row0,row1,column0,column1


def _safe_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def fetch_public_bytes(
    url: str,
    *,
    timeout: int = 90,
    attempts: int = 3,
    backoff_seconds: float = 0.5,
) -> bytes:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds must be nonnegative")
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url, headers={"User-Agent": "pherc0800-published-geometry-audit/1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = error
            if attempt + 1 < attempts and backoff_seconds:
                time.sleep(backoff_seconds * (2**attempt))
    assert last_error is not None
    raise last_error


def origin_https_url(origin: Mapping[str, Any]) -> str:
    roots = origin.get("access_roots")
    if not isinstance(roots, list):
        raise ValueError("catalog origin lacks access_roots")
    public_s3 = [
        root
        for root in roots
        if isinstance(root, Mapping)
        and root.get("type") == "s3"
        and root.get("usage") == "public-read"
        and root.get("url") == "s3://vesuvius-challenge-open-data"
    ]
    if len(public_s3) != 1:
        raise ValueError(
            "catalog origin does not have exactly one official public S3 root"
        )
    path = str(origin.get("path", "")).lstrip("/")
    if not path:
        raise ValueError("catalog origin path is empty")
    return f"{HTTPS_ROOT}/{path.rstrip('/')}"


def one_data_entry(data: Any, entry_type: str) -> Mapping[str, Any]:
    matches = [
        item
        for item in (data if isinstance(data, list) else [])
        if isinstance(item, Mapping) and item.get("type") == entry_type
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {entry_type!r} catalog data entry")
    return matches[0]


def one_origin(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    origins = entry.get("origins")
    if (
        not isinstance(origins, list)
        or len(origins) != 1
        or not isinstance(origins[0], Mapping)
    ):
        raise ValueError("catalog data entry does not have exactly one origin")
    return origins[0]


def resolve_catalog(
    catalog: Mapping[str, Any], *, sample_id: str = SAMPLE_ID
) -> dict[str, Any]:
    samples = catalog.get("samples")
    if not isinstance(samples, Mapping) or sample_id not in samples:
        raise ValueError(f"catalog has no sample {sample_id}")
    sample_entry = samples[sample_id]
    if not isinstance(sample_entry, Mapping):
        raise ValueError("sample catalog entry is not an object")
    volumes = sample_entry.get("volumes")
    segments = sample_entry.get("segments")
    if not isinstance(volumes, Mapping) or len(volumes) != 1:
        raise ValueError("PHerc0800 must resolve to exactly one catalog volume")
    if not isinstance(segments, Mapping) or len(segments) != 6:
        raise ValueError("PHerc0800 must resolve to exactly six catalog segments")
    volume_id, volume = next(iter(volumes.items()))
    if not isinstance(volume, Mapping):
        raise ValueError("catalog volume is not an object")
    properties = volume.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError("catalog volume properties are missing")
    voxel_um = float(properties.get("pixel_size_um"))
    shape_zyx = tuple(int(value) for value in properties.get("shape", ()))
    data = volume.get("data")
    raw_entry = one_data_entry(data, "ome-zarr")
    m7_entry = one_data_entry(data, "surface-prediction-zarr")
    normal_entry = one_data_entry(data, "normal-grids")
    raw_origin, m7_origin, normal_origin = (
        one_origin(raw_entry),
        one_origin(m7_entry),
        one_origin(normal_entry),
    )
    segment_records: list[dict[str, Any]] = []
    for segment_id, segment in sorted(segments.items()):
        if not isinstance(segment, Mapping):
            raise ValueError("catalog segment is not an object")
        tifxyz = one_data_entry(segment.get("data"), "tifxyz")
        origin = one_origin(tifxyz)
        segment_records.append(
            {
                "segment_id": str(segment_id),
                "long_id": str(segment.get("long_id")),
                "original_volume_id": str(segment.get("original_volume_id")),
                "tifxyz_origin_path": str(origin.get("path", "")).rstrip("/") + "/",
                "tifxyz_root_url": origin_https_url(origin),
                "catalog_creation": segment.get("creation"),
                "catalog_properties": segment.get("properties"),
            }
        )
    return {
        "sample_id": sample_id,
        "volume_id": str(volume_id),
        "volume_long_id": str(volume.get("long_id")),
        "voxel_um": voxel_um,
        "shape_zyx": list(shape_zyx),
        "raw": {
            "url": origin_https_url(raw_origin),
            "origin": raw_origin,
            "parameters": raw_entry.get("parameters"),
            "creation_info": raw_entry.get("creation_info"),
        },
        "m7": {
            "url": origin_https_url(m7_origin),
            "origin": m7_origin,
            "parameters": m7_entry.get("parameters"),
            "creation_info": m7_entry.get("creation_info"),
        },
        "normal_grids": {
            "url": origin_https_url(normal_origin),
            "origin": normal_origin,
            "parameters": normal_entry.get("parameters"),
            "creation_info": normal_entry.get("creation_info"),
        },
        "segments": segment_records,
    }


def parse_inventory(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    entries: list[dict[str, Any]] = []
    for content in root.findall("s3:Contents", namespace):
        key = content.findtext("s3:Key", namespaces=namespace)
        if not key:
            raise ValueError("inventory entry has no key")
        entries.append(
            {
                "key": key,
                "size": int(content.findtext("s3:Size", namespaces=namespace) or -1),
                "etag": str(
                    content.findtext("s3:ETag", namespaces=namespace) or ""
                ).strip('"'),
                "last_modified": content.findtext(
                    "s3:LastModified", namespaces=namespace
                ),
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        parts = str(entry["key"]).split("/")
        if len(parts) < 4 or parts[:2] != [SAMPLE_ID, "segments"]:
            raise ValueError(f"unexpected inventory key: {entry['key']}")
        grouped.setdefault(parts[2], []).append(entry)
    if len(grouped) != 6 or len(entries) != 30:
        raise ValueError("PHerc0800 inventory must contain six segments and 30 objects")
    segments: dict[str, Any] = {}
    for long_id, records in sorted(grouped.items()):
        tifxyz = {
            Path(str(record["key"])).name: record
            for record in records
            if "/tifxyz_original/" in str(record["key"])
        }
        if set(tifxyz) != {"meta.json", "x.tif", "y.tif", "z.tif"}:
            raise ValueError(f"inventory tifxyz files are incomplete for {long_id}")
        objects = [record for record in records if str(record["key"]).endswith(".obj")]
        if len(objects) != 1:
            raise ValueError(f"inventory OBJ count differs for {long_id}")
        segments[long_id] = {"tifxyz": tifxyz, "obj_not_fetched": objects[0]}
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "key_count": len(entries),
        "segments": segments,
    }


def crosscheck_catalog_inventory(
    resolved: Mapping[str, Any], inventory: Mapping[str, Any]
) -> None:
    catalog_segments = {
        str(segment["long_id"]): segment for segment in resolved["segments"]
    }
    inventory_segments = inventory["segments"]
    if set(catalog_segments) != set(inventory_segments):
        raise ValueError("local catalog and captured inventory segment IDs differ")
    for long_id, segment in catalog_segments.items():
        expected_prefix = str(segment["tifxyz_origin_path"])
        for record in inventory_segments[long_id]["tifxyz"].values():
            if not str(record["key"]).startswith(expected_prefix):
                raise ValueError(f"catalog/inventory origin mismatch for {long_id}")


@dataclass
class AssetDownloadLedger:
    network_bytes: int = 0
    records: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.records is None:
            self.records = []

    def download(
        self,
        url: str,
        destination: Path,
        *,
        expected_size: int,
        expected_etag_md5: str,
    ) -> dict[str, Any]:
        status = "cached"
        if destination.exists():
            payload = destination.read_bytes()
        else:
            payload = fetch_public_bytes(url)
            _safe_write_bytes(destination, payload)
            self.network_bytes += len(payload)
            status = "downloaded"
        md5 = hashlib.md5(payload).hexdigest()  # nosec B303 - S3 single-part provenance only
        if len(payload) != expected_size or md5 != expected_etag_md5:
            raise ValueError(
                f"downloaded asset failed inventory size/ETag check: {url}"
            )
        record = {
            "url": url,
            "path": str(destination.resolve()),
            "status": status,
            "bytes": len(payload),
            "inventory_etag_md5": expected_etag_md5,
            "computed_md5": md5,
            "sha256": sha256_bytes(payload),
        }
        assert self.records is not None
        self.records.append(record)
        return record


class DiskPayloadFetcher:
    """Persist exact compressed m7 payloads and report real network bytes."""

    def __init__(
        self, cache_dir: Path, upstream: Callable[[str], bytes] = fetch_public_bytes
    ):
        self.cache_dir = cache_dir
        self.upstream = upstream
        self.records: dict[str, dict[str, Any]] = {}
        self.network_bytes = 0

    def __call__(self, url: str) -> bytes:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        path = self.cache_dir / f"{key}.bin"
        record = self.records.setdefault(
            url,
            {
                "url": url,
                "cache_path": str(path.resolve()),
                "read_count": 0,
                "network_fetch_count": 0,
                "bytes": None,
                "sha256": None,
            },
        )
        if path.exists():
            payload = path.read_bytes()
        else:
            payload = self.upstream(url)
            _safe_write_bytes(path, payload)
            self.network_bytes += len(payload)
            record["network_fetch_count"] = int(record["network_fetch_count"]) + 1
        record["read_count"] = int(record["read_count"]) + 1
        digest = sha256_bytes(payload)
        if record["sha256"] is not None and record["sha256"] != digest:
            raise ValueError("disk-cached m7 payload changed between reads")
        record["bytes"] = len(payload)
        record["sha256"] = digest
        return payload

    def manifest(self) -> dict[str, Any]:
        records = [self.records[url] for url in sorted(self.records)]
        return {
            "cache_directory": str(self.cache_dir.resolve()),
            "unique_payload_count": len(records),
            "verified_compressed_payload_bytes_present": sum(
                int(record["bytes"]) for record in records
            ),
            "actual_network_bytes": self.network_bytes,
            "actual_network_fetch_count": sum(
                int(record["network_fetch_count"]) for record in records
            ),
            "payload_reads": sum(int(record["read_count"]) for record in records),
            "records": records,
        }


def transect_category_codes(
    frames: NDArray[np.uint8], eligible: NDArray[np.bool_]
) -> tuple[NDArray[np.uint8], dict[str, int]]:
    """Diagnostic classification: 0 ineligible, 1 unique-centered, 2 zero, 3 multi, 4 off."""

    mask = np.asarray(eligible, dtype=bool)
    if frames.ndim != 3 or frames.shape[1:] != mask.shape or frames.shape[0] % 2 != 1:
        raise ValueError("transect frame and eligible shapes differ")
    foreground = frames > 0
    padded = np.pad(foreground.astype(np.int8), ((1, 1), (0, 0), (0, 0)))
    starts = np.count_nonzero(np.diff(padded, axis=0) == 1, axis=0)
    centered = foreground[frames.shape[0] // 2]
    codes = np.zeros(mask.shape, dtype=np.uint8)
    codes[mask & (starts == 0)] = 2
    codes[mask & (starts > 1)] = 3
    codes[mask & (starts == 1) & ~centered] = 4
    codes[mask & (starts == 1) & centered] = 1
    counts = {
        "single_run_centered": int(np.count_nonzero(codes == 1)),
        "zero_run": int(np.count_nonzero(codes == 2)),
        "multi_run": int(np.count_nonzero(codes == 3)),
        "off_center": int(np.count_nonzero(codes == 4)),
    }
    if sum(counts.values()) != int(np.count_nonzero(mask)):
        raise RuntimeError("transect category accounting mismatch")
    return codes, counts


def finite_distribution(values: NDArray[Any]) -> dict[str, Any]:
    """Return an explicit, stable seven-number summary for finite values."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "quantiles": None, "mean": None}
    quantiles = np.percentile(finite, [0, 5, 25, 50, 75, 95, 100])
    return {
        "count": int(finite.size),
        "quantiles": {
            "minimum": float(quantiles[0]),
            "p05": float(quantiles[1]),
            "p25": float(quantiles[2]),
            "median": float(quantiles[3]),
            "p75": float(quantiles[4]),
            "p95": float(quantiles[5]),
            "maximum": float(quantiles[6]),
        },
        "mean": float(np.mean(finite)),
    }


def selected_run_metrics(
    frames: NDArray[np.uint8], eligible: NDArray[np.bool_]
) -> dict[str, NDArray[Any]]:
    """Select the run nearest offset zero and retain every competitor diagnostic.

    The selected run is deterministic: minimum interval distance to zero, then
    minimum absolute midpoint, then lower offset.  An exact centered-run gate is
    represented separately by ``contains_zero``; run multiplicity never changes
    the selected run or that gate.  A single-run transect receives the documented
    sentinel clearance of 15 voxels, matching the published-control calibration.
    """

    mask = np.asarray(eligible, dtype=bool)
    if frames.ndim != 3 or frames.shape[1:] != mask.shape or frames.shape[0] % 2 != 1:
        raise ValueError("transect frame and eligible shapes differ")
    radius = frames.shape[0] // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.int16)
    shape = mask.shape
    run_count = np.zeros(shape, dtype=np.uint8)
    selected_valid = np.zeros(shape, dtype=bool)
    contains_zero = np.zeros(shape, dtype=bool)
    lower = np.full(shape, np.nan, dtype=np.float32)
    upper = np.full(shape, np.nan, dtype=np.float32)
    center = np.full(shape, np.nan, dtype=np.float32)
    width = np.zeros(shape, dtype=np.uint8)
    competitor_empty_gap = np.full(shape, np.nan, dtype=np.float32)

    foreground = frames > 0
    for row, column in np.argwhere(mask):
        trace = foreground[:, row, column]
        padded = np.pad(trace.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        count = int(starts.size)
        run_count[row, column] = count
        if count == 0:
            continue
        intervals = [
            (int(offsets[start]), int(offsets[end]), int(start), int(end))
            for start, end in zip(starts, ends, strict=True)
        ]

        def selection_key(
            interval: tuple[int, int, int, int],
        ) -> tuple[float, float, int]:
            lo, hi, _start, _end = interval
            interval_distance = 0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
            return (float(interval_distance), abs((lo + hi) / 2.0), lo)

        selected_index = min(
            range(count), key=lambda index: selection_key(intervals[index])
        )
        lo, hi, _start, _end = intervals[selected_index]
        selected_valid[row, column] = True
        contains_zero[row, column] = lo <= 0 <= hi
        lower[row, column] = lo
        upper[row, column] = hi
        center[row, column] = (lo + hi) / 2.0
        width[row, column] = hi - lo + 1
        competitor_gaps: list[int] = []
        for index, (other_lo, other_hi, _other_start, _other_end) in enumerate(
            intervals
        ):
            if index == selected_index:
                continue
            if other_hi < lo:
                competitor_gaps.append(lo - other_hi - 1)
            else:
                competitor_gaps.append(other_lo - hi - 1)
        competitor_empty_gap[row, column] = (
            min(competitor_gaps) if competitor_gaps else radius
        )
    return {
        "run_count": run_count,
        "selected_valid": selected_valid,
        "contains_zero": contains_zero,
        "selected_lower_offset": lower,
        "selected_upper_offset": upper,
        "selected_center_offset": center,
        "selected_width": width,
        "nearest_competitor_empty_gap": competitor_empty_gap,
    }


def run_metrics_summary(
    metrics: Mapping[str, NDArray[Any]], eligible: NDArray[np.bool_]
) -> dict[str, Any]:
    mask = np.asarray(eligible, dtype=bool)
    selected = mask & np.asarray(metrics["selected_valid"], dtype=bool)
    centered = mask & np.asarray(metrics["contains_zero"], dtype=bool)
    run_count = np.asarray(metrics["run_count"])
    center = np.asarray(metrics["selected_center_offset"], dtype=np.float64)
    return {
        "eligible_vertex_count": int(np.count_nonzero(mask)),
        "selected_run_vertex_count": int(np.count_nonzero(selected)),
        "centered_selected_run_vertex_count": int(np.count_nonzero(centered)),
        "centered_selected_run_fraction": (
            float(np.count_nonzero(centered) / np.count_nonzero(mask))
            if np.count_nonzero(mask)
            else 0.0
        ),
        "zero_run_vertex_count": int(np.count_nonzero(mask & (run_count == 0))),
        "single_run_vertex_count": int(np.count_nonzero(mask & (run_count == 1))),
        "multi_run_vertex_count": int(np.count_nonzero(mask & (run_count > 1))),
        "selected_run_center_signed_all_selected": finite_distribution(
            center[selected]
        ),
        "selected_run_center_absolute_all_selected": finite_distribution(
            np.abs(center[selected])
        ),
        "selected_run_center_signed_centered_only": finite_distribution(
            center[centered]
        ),
        "selected_run_center_absolute_centered_only": finite_distribution(
            np.abs(center[centered])
        ),
        "selected_run_width_all_selected": finite_distribution(
            np.asarray(metrics["selected_width"], dtype=np.float64)[selected]
        ),
        "nearest_competitor_empty_gap_all_selected": finite_distribution(
            np.asarray(metrics["nearest_competitor_empty_gap"], dtype=np.float64)[
                selected
            ]
        ),
        "nearest_competitor_empty_gap_centered_only": finite_distribution(
            np.asarray(metrics["nearest_competitor_empty_gap"], dtype=np.float64)[
                centered
            ]
        ),
        "selection_rule": (
            "foreground run nearest offset zero; ties by absolute midpoint then lower offset"
        ),
        "single_run_clearance_sentinel_voxels": int(
            max(abs(int(offset)) for offset in DEFAULT_TRANSECT_OFFSETS)
        ),
    }


def selected_run_spatial_continuity(
    metrics: Mapping[str, NDArray[Any]], region_mask: NDArray[np.bool_]
) -> dict[str, Any]:
    """Report selected-run coherence on all centered 4-neighbor edges."""

    mask = np.asarray(region_mask, dtype=bool) & np.asarray(
        metrics["contains_zero"], dtype=bool
    )
    lower = np.asarray(metrics["selected_lower_offset"], dtype=np.float64)
    upper = np.asarray(metrics["selected_upper_offset"], dtype=np.float64)
    center = np.asarray(metrics["selected_center_offset"], dtype=np.float64)
    center_jumps: list[NDArray[np.float64]] = []
    lower_jumps: list[NDArray[np.float64]] = []
    upper_jumps: list[NDArray[np.float64]] = []
    width_jumps: list[NDArray[np.float64]] = []
    empty_gaps: list[NDArray[np.float64]] = []
    overlaps: list[NDArray[np.bool_]] = []
    for left_mask, right_mask, left_slice, right_slice in (
        (
            mask[:, :-1],
            mask[:, 1:],
            (slice(None), slice(None, -1)),
            (slice(None), slice(1, None)),
        ),
        (
            mask[:-1, :],
            mask[1:, :],
            (slice(None, -1), slice(None)),
            (slice(1, None), slice(None)),
        ),
    ):
        edge_mask = left_mask & right_mask
        if not np.any(edge_mask):
            continue
        lo_a, lo_b = lower[left_slice][edge_mask], lower[right_slice][edge_mask]
        hi_a, hi_b = upper[left_slice][edge_mask], upper[right_slice][edge_mask]
        center_a, center_b = (
            center[left_slice][edge_mask],
            center[right_slice][edge_mask],
        )
        center_jumps.append(np.abs(center_a - center_b))
        lower_jumps.append(np.abs(lo_a - lo_b))
        upper_jumps.append(np.abs(hi_a - hi_b))
        width_jumps.append(np.abs((hi_a - lo_a) - (hi_b - lo_b)))
        empty_gaps.append(
            np.maximum(0.0, np.maximum(lo_a, lo_b) - np.minimum(hi_a, hi_b) - 1.0)
        )
        overlaps.append(np.maximum(lo_a, lo_b) <= np.minimum(hi_a, hi_b))
    if not center_jumps:
        return {
            "edge_count": 0,
            "selected_center_absolute_jump": finite_distribution(
                np.array([], dtype=float)
            ),
            "selected_lower_endpoint_absolute_jump": finite_distribution(
                np.array([], dtype=float)
            ),
            "selected_upper_endpoint_absolute_jump": finite_distribution(
                np.array([], dtype=float)
            ),
            "selected_width_absolute_jump": finite_distribution(
                np.array([], dtype=float)
            ),
            "selected_interval_empty_gap": finite_distribution(
                np.array([], dtype=float)
            ),
            "interval_overlap_fraction": None,
            "zero_empty_gap_fraction": None,
        }
    jumps = np.concatenate(center_jumps)
    gaps = np.concatenate(empty_gaps)
    overlap = np.concatenate(overlaps)
    return {
        "edge_count": int(jumps.size),
        "selected_center_absolute_jump": finite_distribution(jumps),
        "selected_lower_endpoint_absolute_jump": finite_distribution(
            np.concatenate(lower_jumps)
        ),
        "selected_upper_endpoint_absolute_jump": finite_distribution(
            np.concatenate(upper_jumps)
        ),
        "selected_width_absolute_jump": finite_distribution(
            np.concatenate(width_jumps)
        ),
        "selected_interval_empty_gap": finite_distribution(gaps),
        "interval_overlap_fraction": float(np.mean(overlap)),
        "zero_empty_gap_fraction": float(np.mean(gaps == 0)),
        "interpretation": (
            "diagnostic only; no uncalibrated numerical continuity threshold is imposed"
        ),
    }


def quad_area_prefix(quad_area_cm2: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.pad(
        np.cumsum(np.cumsum(quad_area_cm2, axis=0, dtype=np.float64), axis=1),
        ((1, 0), (1, 0)),
    )


def rectangle_physical_area(prefix: NDArray[np.float64], window: Window) -> float:
    row0, row1, column0, column1 = window
    quad_row1, quad_column1 = row1 - 1, column1 - 1
    if quad_row1 <= row0 or quad_column1 <= column0:
        return 0.0
    return float(
        prefix[quad_row1, quad_column1]
        - prefix[row0, quad_column1]
        - prefix[quad_row1, column0]
        + prefix[row0, column0]
    )


def rectangle_record(
    window: Window,
    prefix: NDArray[np.float64],
    *,
    role: str,
) -> dict[str, Any]:
    row0, row1, column0, column1 = window
    height, width = row1 - row0, column1 - column0
    return {
        "role": role,
        "window_half_open_r0_r1_c0_c1": list(window),
        "shape_vertices": [height, width],
        "vertex_count": height * width,
        "quad_count": max(0, height - 1) * max(0, width - 1),
        "area_cm2": rectangle_physical_area(prefix, window),
    }


def rectangle_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    row0, row1, column0, column1 = (
        int(value) for value in record["window_half_open_r0_r1_c0_c1"]
    )
    shape = [int(value) for value in record["shape_vertices"]]
    return (
        float(record["area_cm2"]),
        int(record["quad_count"]),
        int(record["vertex_count"]),
        min(shape),
        -abs(shape[0] - shape[1]),
        -row0,
        -column0,
        -row1,
        -column1,
    )


def largest_clean_rectangle(
    clean: NDArray[np.bool_], quad_area_cm2: NDArray[np.float64]
) -> dict[str, Any] | None:
    """Largest physical-area all-true rectangle, exact over maximal candidates."""

    mask = np.asarray(clean, dtype=bool)
    height, width = mask.shape
    if quad_area_cm2.shape != (height - 1, width - 1):
        raise ValueError("quad area shape differs from vertex mask")
    prefix = quad_area_prefix(quad_area_cm2)
    heights = np.zeros(width, dtype=np.int64)
    best: dict[str, Any] | None = None
    candidate_count = 0
    for bottom in range(height):
        heights = np.where(mask[bottom], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for column in range(width + 1):
            current = int(heights[column]) if column < width else 0
            start = column
            while stack and stack[-1][1] > current:
                left, rectangle_height = stack.pop()
                start = left
                rectangle_width = column - left
                if rectangle_height < 2 or rectangle_width < 2:
                    continue
                window = (
                    bottom - rectangle_height + 1,
                    bottom + 1,
                    left,
                    column,
                )
                candidate = rectangle_record(
                    window, prefix, role="largest_global_all_true_rectangle"
                )
                candidate_count += 1
                if best is None or rectangle_rank(candidate) > rectangle_rank(best):
                    best = candidate
            if current > 0 and (not stack or stack[-1][1] < current):
                stack.append((start, current))
    if best is not None:
        best["maximal_histogram_candidates_examined"] = candidate_count
        best["objective"] = (
            "maximum physical two-triangle quad area; then quads, vertices, minimum "
            "side, balance, and earliest bounds"
        )
    return best


def largest_seed_containing_rectangle(
    clean: NDArray[np.bool_],
    quad_area_cm2: NDArray[np.float64],
    seed_row_column: tuple[int, int],
) -> dict[str, Any] | None:
    mask = np.asarray(clean, dtype=bool)
    seed_row, seed_column = seed_row_column
    if not (
        0 <= seed_row < mask.shape[0]
        and 0 <= seed_column < mask.shape[1]
        and mask[seed_row, seed_column]
    ):
        return None
    prefix = quad_area_prefix(quad_area_cm2)
    best: dict[str, Any] | None = None
    examined = 0
    for row0 in range(seed_row + 1):
        common = np.ones(mask.shape[1], dtype=bool)
        for row1 in range(row0 + 1, mask.shape[0] + 1):
            common &= mask[row1 - 1]
            if row1 <= seed_row or not common[seed_column]:
                continue
            left = seed_column
            while left > 0 and common[left - 1]:
                left -= 1
            right = seed_column + 1
            while right < mask.shape[1] and common[right]:
                right += 1
            if row1 - row0 < 2 or right - left < 2:
                continue
            candidate = rectangle_record(
                (row0, row1, left, right),
                prefix,
                role="largest_declared_seed_containing_all_true_rectangle",
            )
            examined += 1
            if best is None or rectangle_rank(candidate) > rectangle_rank(best):
                best = candidate
    if best is not None:
        best["vertical_bands_examined"] = examined
    return best


def connected_components_summary(
    clean: NDArray[np.bool_], quad_area_cm2: NDArray[np.float64]
) -> tuple[dict[str, Any], NDArray[np.int32]]:
    mask = np.asarray(clean, dtype=bool)
    labels = np.full(mask.shape, -1, dtype=np.int32)
    components: list[dict[str, Any]] = []
    component_id = 0
    for start_row, start_column in np.argwhere(mask):
        if labels[start_row, start_column] >= 0:
            continue
        queue: deque[tuple[int, int]] = deque([(int(start_row), int(start_column))])
        labels[start_row, start_column] = component_id
        rows: list[int] = []
        columns: list[int] = []
        while queue:
            row, column = queue.popleft()
            rows.append(row)
            columns.append(column)
            for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + delta_row, column + delta_column
                if (
                    0 <= rr < mask.shape[0]
                    and 0 <= cc < mask.shape[1]
                    and mask[rr, cc]
                    and labels[rr, cc] < 0
                ):
                    labels[rr, cc] = component_id
                    queue.append((rr, cc))
        components.append(
            {
                "component_id": component_id,
                "vertex_count": len(rows),
                "bounds_half_open_r0_r1_c0_c1": [
                    min(rows),
                    max(rows) + 1,
                    min(columns),
                    max(columns) + 1,
                ],
            }
        )
        component_id += 1
    if components:
        quad_labels = labels[:-1, :-1]
        same = (
            (quad_labels >= 0)
            & (labels[:-1, 1:] == quad_labels)
            & (labels[1:, :-1] == quad_labels)
            & (labels[1:, 1:] == quad_labels)
        )
        area_by_component = np.bincount(
            quad_labels[same],
            weights=np.asarray(quad_area_cm2, dtype=np.float64)[same],
            minlength=len(components),
        )
        quad_count = np.bincount(quad_labels[same], minlength=len(components))
        for component, area, count in zip(components, area_by_component, quad_count):
            component["area_cm2"] = float(area)
            component["all_four_vertices_true_quad_count"] = int(count)
    components.sort(
        key=lambda item: (
            -float(item.get("area_cm2", 0.0)),
            -int(item["vertex_count"]),
            int(item["component_id"]),
        )
    )
    return {
        "connectivity": "4-neighbor stored-grid vertices",
        "component_count": len(components),
        "largest_component": components[0] if components else None,
        "top_components": components[:10],
    }, labels


def connected_quad_components_summary(
    vertex_mask: NDArray[np.bool_], quad_area_cm2: NDArray[np.float64]
) -> tuple[dict[str, Any], NDArray[np.int32]]:
    """Measure 4-neighbor components made only of four-corner-valid quads.

    Connecting area through isolated vertices or one-vertex-wide tendrils would
    overstate a usable sheet patch.  Quad connectivity makes every path carry
    positive physical area and is therefore the connected-region candidacy gate.
    """

    mask = np.asarray(vertex_mask, dtype=bool)
    if quad_area_cm2.shape != (mask.shape[0] - 1, mask.shape[1] - 1):
        raise ValueError("quad area shape differs from vertex mask")
    active = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
    labels = np.full(active.shape, -1, dtype=np.int32)
    components: list[dict[str, Any]] = []
    component_id = 0
    for start_row, start_column in np.argwhere(active):
        if labels[start_row, start_column] >= 0:
            continue
        queue: deque[tuple[int, int]] = deque([(int(start_row), int(start_column))])
        labels[start_row, start_column] = component_id
        rows: list[int] = []
        columns: list[int] = []
        while queue:
            row, column = queue.popleft()
            rows.append(row)
            columns.append(column)
            for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + delta_row, column + delta_column
                if (
                    0 <= rr < active.shape[0]
                    and 0 <= cc < active.shape[1]
                    and active[rr, cc]
                    and labels[rr, cc] < 0
                ):
                    labels[rr, cc] = component_id
                    queue.append((rr, cc))
        component_mask = labels == component_id
        row0, row1 = min(rows), max(rows) + 1
        column0, column1 = min(columns), max(columns) + 1
        bounding_quad_count = (row1 - row0) * (column1 - column0)
        components.append(
            {
                "component_id": component_id,
                "quad_count": len(rows),
                "area_cm2": float(np.sum(quad_area_cm2[component_mask])),
                "quad_bounds_half_open_r0_r1_c0_c1": [
                    row0,
                    row1,
                    column0,
                    column1,
                ],
                "vertex_bounds_half_open_r0_r1_c0_c1": [
                    row0,
                    row1 + 1,
                    column0,
                    column1 + 1,
                ],
                "quad_fill_fraction_in_bounding_box": float(
                    len(rows) / bounding_quad_count
                ),
            }
        )
        component_id += 1
    components.sort(
        key=lambda item: (
            -float(item["area_cm2"]),
            -int(item["quad_count"]),
            int(item["component_id"]),
        )
    )
    return {
        "connectivity": "4-neighbor fully centered four-vertex quads",
        "anti_gaming_rule": (
            "components cannot connect through isolated vertices or vertex-only tendrils"
        ),
        "active_quad_count": int(np.count_nonzero(active)),
        "component_count": len(components),
        "largest_component": components[0] if components else None,
        "top_components": components[:10],
    }, labels


def quad_component_vertex_mask(
    quad_labels: NDArray[np.int32], component_id: int, vertex_shape: tuple[int, int]
) -> NDArray[np.bool_]:
    selected = np.asarray(quad_labels) == int(component_id)
    result = np.zeros(vertex_shape, dtype=bool)
    result[:-1, :-1] |= selected
    result[:-1, 1:] |= selected
    result[1:, :-1] |= selected
    result[1:, 1:] |= selected
    return result


def declared_seed_alignment(
    metadata: Mapping[str, Any],
    points_xyz: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> dict[str, Any]:
    raw = metadata.get("seed")
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return {"declared": False, "reason": "segment metadata has no 3-vector seed"}
    seed = np.asarray(raw, dtype=np.float64)
    if np.all(seed == 0) and not np.any(np.all(points_xyz[valid] == 0, axis=1)):
        return {
            "declared": False,
            "reason": "metadata seed is a [0,0,0] placeholder outside the published surface",
            "metadata_value": seed.tolist(),
        }
    coordinates = np.argwhere(valid)
    distances = np.linalg.norm(points_xyz[valid] - seed, axis=1)
    index = int(np.argmin(distances))
    row, column = (int(value) for value in coordinates[index])
    return {
        "declared": True,
        "seed_xyz": seed.tolist(),
        "nearest_valid_row_column": [row, column],
        "nearest_distance_voxels": float(distances[index]),
    }


def mark_distorted_vertices(distorted_quad: NDArray[np.bool_]) -> NDArray[np.bool_]:
    result = np.zeros(
        (distorted_quad.shape[0] + 1, distorted_quad.shape[1] + 1), dtype=bool
    )
    result[:-1, :-1] |= distorted_quad
    result[:-1, 1:] |= distorted_quad
    result[1:, :-1] |= distorted_quad
    result[1:, 1:] |= distorted_quad
    return result


def save_array(path: Path, array: NDArray[Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "array_content_sha256": _array_sha256(np.asarray(array)),
        "shape": list(np.asarray(array).shape),
        "dtype": np.asarray(array).dtype.str,
    }


def rectangle_region_mask(
    shape: tuple[int, int], record: Mapping[str, Any] | None
) -> NDArray[np.bool_]:
    result = np.zeros(shape, dtype=bool)
    if record is None:
        return result
    row0, row1, column0, column1 = (
        int(value) for value in record["window_half_open_r0_r1_c0_c1"]
    )
    result[row0:row1, column0:column1] = True
    return result


def selected_region_diagnostics(
    metrics: Mapping[str, NDArray[Any]], region_mask: NDArray[np.bool_]
) -> dict[str, Any]:
    mask = np.asarray(region_mask, dtype=bool) & np.asarray(
        metrics["selected_valid"], dtype=bool
    )
    centered = mask & np.asarray(metrics["contains_zero"], dtype=bool)
    center = np.asarray(metrics["selected_center_offset"], dtype=np.float64)
    gaps = np.asarray(metrics["nearest_competitor_empty_gap"], dtype=np.float64)
    return {
        "vertex_count": int(np.count_nonzero(region_mask)),
        "selected_run_vertex_count": int(np.count_nonzero(mask)),
        "centered_selected_run_vertex_count": int(np.count_nonzero(centered)),
        "selected_run_center_signed": finite_distribution(center[mask]),
        "selected_run_center_absolute": finite_distribution(np.abs(center[mask])),
        "nearest_competitor_empty_gap": finite_distribution(gaps[mask]),
        "spatial_continuity": selected_run_spatial_continuity(metrics, centered),
    }


def copy_and_summarize_calibration_control(
    source_dir: Path, result_dir: Path, destination: Path
) -> dict[str, Any]:
    """Make the published-surface control durable and recompute its diagnostics."""

    files = {
        "source/meta.json": source_dir / "meta.json",
        "source/x.tif": source_dir / "x.tif",
        "source/y.tif": source_dir / "y.tif",
        "source/z.tif": source_dir / "z.tif",
        "result/validation.json": result_dir / "validation.json",
        "result/m7-offset-sample-validity.npy": result_dir
        / "m7-offset-sample-validity.npy",
        "result/m7-quantized-offsets-minus15-plus15.npy": result_dir
        / "m7-quantized-offsets-minus15-plus15.npy",
    }
    copied: list[dict[str, Any]] = []
    for relative, source in files.items():
        if not source.is_file():
            raise FileNotFoundError(f"calibration evidence is missing: {source}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_digest = sha256_file(source)
        shutil.copyfile(source, target)
        target_digest = sha256_file(target)
        if target_digest != source_digest:
            raise RuntimeError(f"calibration copy hash mismatch: {relative}")
        copied.append(
            {
                "role": relative,
                "source_path": str(source.resolve()),
                "copied_path": str(target.resolve()),
                "bytes": target.stat().st_size,
                "sha256": target_digest,
            }
        )
    frames = np.load(
        destination / "result/m7-quantized-offsets-minus15-plus15.npy",
        allow_pickle=False,
    )
    validity = np.load(
        destination / "result/m7-offset-sample-validity.npy", allow_pickle=False
    )
    if frames.shape != (31, 20, 20) or validity.shape != frames.shape:
        raise ValueError("published calibration arrays have unexpected shapes")
    eligible = np.all(validity, axis=0)
    categories, category_counts = transect_category_codes(frames, eligible)
    metrics = selected_run_metrics(frames, eligible)
    support_indices = [
        DEFAULT_TRANSECT_OFFSETS.index(offset) for offset in DEFAULT_SUPPORT_OFFSETS
    ]
    support = np.any(frames[support_indices] > 0, axis=0) & eligible
    offset0 = (frames[len(DEFAULT_TRANSECT_OFFSETS) // 2] > 0) & eligible
    expected = {
        "eligible": 400,
        "support": 395,
        "offset0": 345,
        "single_run_centered": 1,
        "multi_run": 399,
    }
    observed = {
        "eligible": int(np.count_nonzero(eligible)),
        "support": int(np.count_nonzero(support)),
        "offset0": int(np.count_nonzero(offset0)),
        "single_run_centered": category_counts["single_run_centered"],
        "multi_run": category_counts["multi_run"],
    }
    if observed != expected:
        raise ValueError(
            f"published calibration no longer reproduces expected counts: {observed}"
        )
    validation = json.loads(
        (destination / "result/validation.json").read_text(encoding="utf-8")
    )
    return {
        "identity": "PHerc1447 published clean-surface 20x20 control crop",
        "copied_files": copied,
        "original_validation": validation,
        "recomputed": {
            "eligible_vertex_count": observed["eligible"],
            "plus_minus2_support_count": observed["support"],
            "plus_minus2_support_fraction": observed["support"] / observed["eligible"],
            "offset0_support_count": observed["offset0"],
            "offset0_support_fraction": observed["offset0"] / observed["eligible"],
            "legacy_exact_run_category_counts": category_counts,
            "selected_run_metrics": run_metrics_summary(metrics, eligible),
            "centered_selected_run_spatial_continuity": selected_run_spatial_continuity(
                metrics, np.asarray(metrics["contains_zero"], dtype=bool)
            ),
            "category_code_array_content_sha256": _array_sha256(categories),
        },
        "calibration_conclusion": (
            "399/400 vertices are multi-run despite clean native geometry and 0.9875 "
            "plus/minus-2 support; secondary runs are diagnostic, never a hard veto"
        ),
    }


def compact_geometry(geometry: Mapping[str, Any]) -> dict[str, Any]:
    positive = geometry["native_reports_by_orientation"]["positive"]
    return {
        "whole_surface_pass": bool(geometry["pass"]),
        "hard_defect_counts": geometry["hard_defect_counts"],
        "in_bounds_vertex_fraction": float(positive["in_bounds_vertex_fraction"]),
        "normal_valid_vertex_fraction": float(positive["normal_valid_vertex_fraction"]),
        "offset_sample_in_bounds_fraction": float(
            positive["offset_sample_in_bounds_fraction"]
        ),
        "valid_vertex_count": int(positive["valid_vertex_count"]),
        "valid_quad_count": int(positive["valid_quad_count"]),
        "median_neighbor_distance_um": float(positive["median_neighbor_distance_um"]),
        "maximum_neighbor_distance_um": float(positive["max_neighbor_distance_um"]),
        "self_intersection_status": geometry["self_intersection_status"],
    }


def audit_segment(
    segment: Mapping[str, Any],
    asset_dir: Path,
    segment_output_dir: Path,
    sampler: PublicZarrChunkSampler,
    *,
    volume_shape_zyx: Sequence[int],
    voxel_um: float,
    minimum_candidate_area_cm2: float,
) -> dict[str, Any]:
    asset = load_tifxyz_asset(asset_dir, resolution="stored")
    surface = asset.surface
    masks, geometry = hard_geometry_summary(
        surface, volume_shape_zyx=volume_shape_zyx, voxel_um=voxel_um
    )
    distorted_margin = dilate_one_cell(mark_distorted_vertices(masks.distorted_quad))
    geometry_eligible = (
        masks.sample_valid & ~masks.forbidden & ~distorted_margin & surface.valid
    )
    frames, sample_validity = sample_offsets(
        sampler,
        surface,
        masks.normals_xyz,
        geometry_eligible,
        DEFAULT_TRANSECT_OFFSETS,
    )
    requested = np.broadcast_to(geometry_eligible, sample_validity.shape)
    sampling_complete = bool(np.all(sample_validity[requested]))
    support_indices = [
        DEFAULT_TRANSECT_OFFSETS.index(offset) for offset in DEFAULT_SUPPORT_OFFSETS
    ]
    support = support_summary(
        frames[support_indices], DEFAULT_SUPPORT_OFFSETS, geometry_eligible
    )
    supported = np.any(frames[support_indices] > 0, axis=0) & geometry_eligible
    categories, category_counts = transect_category_codes(frames, geometry_eligible)
    selected_runs = selected_run_metrics(frames, geometry_eligible)
    centered_geometry = geometry_eligible & np.asarray(
        selected_runs["contains_zero"], dtype=bool
    )
    connected, component_labels = connected_components_summary(
        centered_geometry, masks.quad_area_cm2
    )
    connected_quads, quad_component_labels = connected_quad_components_summary(
        centered_geometry, masks.quad_area_cm2
    )
    largest_quad_component = connected_quads["largest_component"]
    largest_quad_component_mask = (
        quad_component_vertex_mask(
            quad_component_labels,
            int(largest_quad_component["component_id"]),
            centered_geometry.shape,
        )
        if largest_quad_component is not None
        else np.zeros(centered_geometry.shape, dtype=bool)
    )
    largest_quad_component_diagnostics = selected_region_diagnostics(
        selected_runs, largest_quad_component_mask
    )
    largest_rectangle = largest_clean_rectangle(centered_geometry, masks.quad_area_cm2)
    largest_rectangle_diagnostics = selected_region_diagnostics(
        selected_runs,
        rectangle_region_mask(centered_geometry.shape, largest_rectangle),
    )
    seed_alignment = declared_seed_alignment(
        asset.metadata, surface.points_xyz, surface.valid
    )
    seed_rectangle = None
    seed_rectangle_diagnostics = None
    seed_component = None
    if seed_alignment.get("declared"):
        seed_row, seed_column = (
            int(value) for value in seed_alignment["nearest_valid_row_column"]
        )
        seed_alignment["geometry_eligible"] = bool(
            geometry_eligible[seed_row, seed_column]
        )
        seed_alignment["m7_supported_plus_minus2"] = bool(
            supported[seed_row, seed_column]
        )
        seed_alignment["single_run_centered"] = bool(
            categories[seed_row, seed_column] == 1
        )
        seed_alignment["selected_run_contains_offset0"] = bool(
            centered_geometry[seed_row, seed_column]
        )
        seed_alignment["run_count"] = int(
            selected_runs["run_count"][seed_row, seed_column]
        )
        seed_alignment["nearest_competitor_empty_gap_voxels"] = (
            float(selected_runs["nearest_competitor_empty_gap"][seed_row, seed_column])
            if selected_runs["selected_valid"][seed_row, seed_column]
            else None
        )
        seed_rectangle = largest_seed_containing_rectangle(
            centered_geometry, masks.quad_area_cm2, (seed_row, seed_column)
        )
        seed_rectangle_diagnostics = selected_region_diagnostics(
            selected_runs,
            rectangle_region_mask(centered_geometry.shape, seed_rectangle),
        )
        label = int(component_labels[seed_row, seed_column])
        if label >= 0:
            seed_component = next(
                (
                    item
                    for item in connected["top_components"]
                    if int(item["component_id"]) == label
                ),
                None,
            )
            if seed_component is None:
                seed_mask = component_labels == label
                quad_label = component_labels[:-1, :-1]
                same = (
                    (quad_label == label)
                    & (component_labels[:-1, 1:] == label)
                    & (component_labels[1:, :-1] == label)
                    & (component_labels[1:, 1:] == label)
                )
                coordinates = np.argwhere(seed_mask)
                seed_component = {
                    "component_id": label,
                    "vertex_count": int(np.count_nonzero(seed_mask)),
                    "area_cm2": float(np.sum(masks.quad_area_cm2[same])),
                    "all_four_vertices_centered_geometry_quad_count": int(
                        np.count_nonzero(same)
                    ),
                    "bounds_half_open_r0_r1_c0_c1": [
                        int(np.min(coordinates[:, 0])),
                        int(np.max(coordinates[:, 0]) + 1),
                        int(np.min(coordinates[:, 1])),
                        int(np.max(coordinates[:, 1]) + 1),
                    ],
                }
    total_area = surface_area_cm2(surface, voxel_um)
    centered_rectangle_area = (
        float(largest_rectangle["area_cm2"]) if largest_rectangle is not None else 0.0
    )
    centered_connected_area = (
        float(largest_quad_component["area_cm2"])
        if largest_quad_component is not None
        else 0.0
    )
    qualifies_rectangle = bool(
        sampling_complete
        and largest_rectangle is not None
        and centered_rectangle_area >= minimum_candidate_area_cm2
    )
    qualifies_connected = bool(
        sampling_complete
        and largest_quad_component is not None
        and centered_connected_area >= minimum_candidate_area_cm2
    )
    arrays = {
        "m7_frames_minus15_plus15": save_array(
            segment_output_dir / "m7-offsets-minus15-plus15.npy", frames
        ),
        "m7_sample_validity": save_array(
            segment_output_dir / "m7-sample-validity.npy", sample_validity
        ),
        "geometry_eligible_mask": save_array(
            segment_output_dir / "geometry-eligible-mask.npy", geometry_eligible
        ),
        "m7_support_mask": save_array(
            segment_output_dir / "m7-plus-minus2-support-mask.npy", supported
        ),
        "transect_category_codes": save_array(
            segment_output_dir / "transect-category-codes.npy", categories
        ),
        "selected_run_count": save_array(
            segment_output_dir / "selected-run-count.npy", selected_runs["run_count"]
        ),
        "selected_run_contains_offset0": save_array(
            segment_output_dir / "selected-run-contains-offset0.npy",
            selected_runs["contains_zero"],
        ),
        "selected_run_lower_offset": save_array(
            segment_output_dir / "selected-run-lower-offset.npy",
            selected_runs["selected_lower_offset"],
        ),
        "selected_run_upper_offset": save_array(
            segment_output_dir / "selected-run-upper-offset.npy",
            selected_runs["selected_upper_offset"],
        ),
        "selected_run_center_offset": save_array(
            segment_output_dir / "selected-run-center-offset.npy",
            selected_runs["selected_center_offset"],
        ),
        "nearest_competitor_empty_gap": save_array(
            segment_output_dir / "nearest-competitor-empty-gap.npy",
            selected_runs["nearest_competitor_empty_gap"],
        ),
        "centered_geometry_mask": save_array(
            segment_output_dir / "centered-geometry-mask.npy", centered_geometry
        ),
        "connected_component_labels": save_array(
            segment_output_dir / "connected-component-labels.npy", component_labels
        ),
        "connected_quad_component_labels": save_array(
            segment_output_dir / "connected-quad-component-labels.npy",
            quad_component_labels,
        ),
        "largest_connected_quad_component_vertex_mask": save_array(
            segment_output_dir / "largest-connected-quad-component-vertex-mask.npy",
            largest_quad_component_mask,
        ),
        "quad_area_cm2": save_array(
            segment_output_dir / "quad-area-cm2.npy", masks.quad_area_cm2
        ),
    }
    return {
        "segment_id": segment["segment_id"],
        "long_id": segment["long_id"],
        "tifxyz": asset.manifest(),
        "metadata": asset.metadata,
        "stored_shape": list(surface.shape),
        "stored_valid_vertex_count": int(np.count_nonzero(surface.valid)),
        "physical_area_cm2": total_area,
        "catalog_area_context": {
            "catalog_tiff_dimensions": segment["catalog_creation"]["metadata"].get(
                "tiff_dimensions"
            ),
            "catalog_full_width_height": [
                segment["catalog_properties"].get("width"),
                segment["catalog_properties"].get("height"),
            ],
            "stored_scale": asset.metadata.get("scale"),
        },
        "native_geometry": compact_geometry(geometry),
        "localized_geometry": {
            **masks.summary,
            "distorted_quad_vertex_margin_count": int(
                np.count_nonzero(distorted_margin)
            ),
            "m7_eligible_vertex_count": int(np.count_nonzero(geometry_eligible)),
            "eligibility_rule": (
                "native sample-valid and outside one-cell margins around all localized hard "
                "defects and distorted quads"
            ),
        },
        "m7_sampling_complete": sampling_complete,
        "m7_support_plus_minus2": support,
        "legacy_exact_run_diagnostic": {
            "eligible_vertex_count": int(np.count_nonzero(geometry_eligible)),
            "category_counts": category_counts,
            "single_run_centered_fraction": (
                float(
                    category_counts["single_run_centered"]
                    / np.count_nonzero(geometry_eligible)
                )
                if np.count_nonzero(geometry_eligible)
                else 0.0
            ),
            "all_unique_centered": bool(
                category_counts["single_run_centered"]
                == int(np.count_nonzero(geometry_eligible))
                and np.count_nonzero(geometry_eligible) > 0
            ),
            "definition": (
                "exactly one quantized foreground run across integer offsets -15..+15, "
                "and the run contains offset 0"
            ),
            "gate_role": (
                "diagnostic only; adjacent papyrus wraps make secondary runs expected"
            ),
        },
        "selected_run_metrics": run_metrics_summary(selected_runs, geometry_eligible),
        "selected_run_spatial_continuity": selected_run_spatial_continuity(
            selected_runs, centered_geometry
        ),
        "centered_geometry_region_definition": (
            "geometry-eligible AND the deterministically selected nearest m7 foreground run "
            "contains offset 0; secondary runs are retained and measured, not vetoed"
        ),
        "centered_geometry_vertex_count": int(np.count_nonzero(centered_geometry)),
        "centered_geometry_vertex_fraction_of_m7_eligible": (
            float(
                np.count_nonzero(centered_geometry)
                / np.count_nonzero(geometry_eligible)
            )
            if np.count_nonzero(geometry_eligible)
            else 0.0
        ),
        "connected_regions": connected,
        "connected_quad_regions": connected_quads,
        "largest_connected_quad_region_diagnostics": largest_quad_component_diagnostics,
        "largest_centered_geometry_rectangle": largest_rectangle,
        "largest_centered_geometry_rectangle_diagnostics": largest_rectangle_diagnostics,
        "declared_seed_alignment": seed_alignment,
        "declared_seed_centered_geometry_component": seed_component,
        "largest_declared_seed_containing_centered_geometry_rectangle": seed_rectangle,
        "largest_declared_seed_rectangle_diagnostics": seed_rectangle_diagnostics,
        "supports_at_least_0_5_cm2_centered_geometry_rectangle": qualifies_rectangle,
        "supports_at_least_0_5_cm2_centered_connected_quad_region": qualifies_connected,
        "supports_at_least_0_5_cm2_centered_rectangle_or_connected_quad_region": bool(
            qualifies_rectangle or qualifies_connected
        ),
        "candidate_area_threshold_cm2": minimum_candidate_area_cm2,
        "region_interpretation": (
            "preflight-only centered-sheet evidence in original stored-grid normals; "
            "secondary runs are calibrated diagnostics, while self-intersection and "
            "extracted-crop boundary normals remain unvalidated"
        ),
        "array_outputs": arrays,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    catalog_path = Path(args.catalog)
    catalog_cache_path = Path(args.catalog_cache)
    inventory_path = Path(args.inventory)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_payload = catalog_path.read_bytes()
    catalog = json.loads(catalog_payload)
    if not isinstance(catalog, Mapping):
        raise ValueError("local VC3D catalog is not a JSON object")
    cache_metadata = json.loads(catalog_cache_path.read_text(encoding="utf-8"))
    resolved = resolve_catalog(catalog)
    if resolved["volume_id"] != EXPECTED_VOLUME_ID:
        raise ValueError("local catalog PHerc0800 preferred volume changed")
    if tuple(resolved["shape_zyx"]) != EXPECTED_VOLUME_SHAPE_ZYX:
        raise ValueError("local catalog PHerc0800 shape changed")
    if not math.isclose(float(resolved["voxel_um"]), EXPECTED_VOXEL_UM, abs_tol=1e-12):
        raise ValueError("local catalog PHerc0800 voxel size changed")
    inventory = parse_inventory(inventory_path)
    crosscheck_catalog_inventory(resolved, inventory)
    catalog_extract_path = output_dir / "catalog-PHerc0800-extract.json"
    _atomic_json(catalog_extract_path, resolved)
    calibration = copy_and_summarize_calibration_control(
        Path(args.calibration_source),
        Path(args.calibration_result),
        output_dir / "calibration/pherc1447-published-control-20x20",
    )

    asset_ledger = AssetDownloadLedger()
    downloaded_segments: list[dict[str, Any]] = []
    for segment in resolved["segments"]:
        long_id = str(segment["long_id"])
        destination = output_dir / "assets" / long_id
        records = []
        inventory_files = inventory["segments"][long_id]["tifxyz"]
        for filename in ("meta.json", "x.tif", "y.tif", "z.tif"):
            inventory_record = inventory_files[filename]
            records.append(
                asset_ledger.download(
                    f"{str(segment['tifxyz_root_url']).rstrip('/')}/{filename}",
                    destination / filename,
                    expected_size=int(inventory_record["size"]),
                    expected_etag_md5=str(inventory_record["etag"]),
                )
            )
        downloaded_segments.append(
            {
                **segment,
                "local_tifxyz_directory": str(destination.resolve()),
                "files": records,
            }
        )

    m7_root = str(resolved["m7"]["url"]).rstrip("/")
    array_path = str(args.m7_array_path).strip("/")
    group, group_provenance = fetch_json(m7_root + "/.zgroup")
    array_metadata, array_provenance = fetch_json(m7_root + f"/{array_path}/.zarray")
    attributes, attributes_provenance = fetch_json(m7_root + "/.zattrs")
    if group.get("zarr_format") != 2:
        raise ValueError("official PHerc0800 m7 is not Zarr v2")
    validate_metadata_axes(attributes, array_path)
    spec = ZarrV2ArraySpec.from_metadata(array_metadata)
    if spec.shape_zyx != EXPECTED_VOLUME_SHAPE_ZYX:
        raise ValueError(
            "official m7 shape differs from catalog raw-volume coordinate space"
        )
    if spec.chunks_zyx != (192, 192, 192):
        raise ValueError("official m7 chunk shape differs from expected 192^3")
    if spec.compressor is None or spec.compressor.get("id") != "blosc":
        raise ValueError("official m7 is not Blosc-compressed")
    payload_fetcher = DiskPayloadFetcher(output_dir / "m7-chunk-cache")
    sampler = PublicZarrChunkSampler(
        root_url=m7_root,
        array_path=array_path,
        spec=spec,
        blosc_path=Path(args.blosc),
        fetcher=payload_fetcher,
    )

    results: list[dict[str, Any]] = []
    for segment in downloaded_segments:
        long_id = str(segment["long_id"])
        result = audit_segment(
            segment,
            Path(segment["local_tifxyz_directory"]),
            output_dir / "segments" / long_id,
            sampler,
            volume_shape_zyx=spec.shape_zyx,
            voxel_um=float(resolved["voxel_um"]),
            minimum_candidate_area_cm2=float(args.minimum_candidate_area_cm2),
        )
        result["asset_downloads"] = segment["files"]
        results.append(result)
        sampler._cache.clear()  # bound decoded memory; compressed payloads remain hash-cached

    all_ranked: list[dict[str, Any]] = []
    for result in results:
        rectangle = result["largest_centered_geometry_rectangle"]
        largest_component = result["connected_quad_regions"]["largest_component"]
        component_diagnostics = result["largest_connected_quad_region_diagnostics"]
        continuity = component_diagnostics["spatial_continuity"]
        jump_quantiles = continuity["selected_center_absolute_jump"]["quantiles"]
        clearance_quantiles = component_diagnostics["nearest_competitor_empty_gap"][
            "quantiles"
        ]
        all_ranked.append(
            {
                "rank": 0,
                "segment_id": result["segment_id"],
                "long_id": result["long_id"],
                "largest_centered_geometry_rectangle_area_cm2": (
                    float(rectangle["area_cm2"]) if rectangle is not None else 0.0
                ),
                "largest_centered_geometry_rectangle": rectangle,
                "largest_centered_connected_quad_region_area_cm2": (
                    float(largest_component["area_cm2"])
                    if largest_component is not None
                    else 0.0
                ),
                "largest_centered_connected_quad_region": largest_component,
                "whole_surface_area_cm2": float(result["physical_area_cm2"]),
                "centered_geometry_vertex_fraction": float(
                    result["centered_geometry_vertex_fraction_of_m7_eligible"]
                ),
                "plus_minus2_support_fraction": float(
                    result["m7_support_plus_minus2"]["support_fraction"]
                ),
                "connected_region_zero_empty_gap_neighbor_fraction": (
                    float(continuity["zero_empty_gap_fraction"])
                    if continuity["zero_empty_gap_fraction"] is not None
                    else 0.0
                ),
                "connected_region_selected_center_jump_p95_voxels": (
                    float(jump_quantiles["p95"]) if jump_quantiles is not None else None
                ),
                "connected_region_competitor_clearance_median_voxels": (
                    float(clearance_quantiles["median"])
                    if clearance_quantiles is not None
                    else None
                ),
                "connected_region_competitor_clearance_p05_voxels": (
                    float(clearance_quantiles["p05"])
                    if clearance_quantiles is not None
                    else None
                ),
                "supports_at_least_0_5_cm2_centered_geometry_rectangle": bool(
                    result["supports_at_least_0_5_cm2_centered_geometry_rectangle"]
                ),
                "supports_at_least_0_5_cm2_centered_connected_quad_region": bool(
                    result["supports_at_least_0_5_cm2_centered_connected_quad_region"]
                ),
                "supports_at_least_0_5_cm2_centered_rectangle_or_connected_quad_region": bool(
                    result[
                        "supports_at_least_0_5_cm2_centered_rectangle_or_connected_quad_region"
                    ]
                ),
                "self_intersection_status": result["native_geometry"][
                    "self_intersection_status"
                ],
                "status": "preflight_only_not_finally_trustworthy",
            }
        )

    def rank_number(row: Mapping[str, Any], key: str, missing: float) -> float:
        value = row[key]
        return float(value) if value is not None else missing

    all_ranked.sort(
        key=lambda row: (
            -float(row["largest_centered_connected_quad_region_area_cm2"]),
            -float(row["largest_centered_geometry_rectangle_area_cm2"]),
            -float(row["centered_geometry_vertex_fraction"]),
            -float(row["plus_minus2_support_fraction"]),
            -float(row["connected_region_zero_empty_gap_neighbor_fraction"]),
            rank_number(
                row, "connected_region_selected_center_jump_p95_voxels", math.inf
            ),
            -rank_number(
                row, "connected_region_competitor_clearance_median_voxels", -math.inf
            ),
            -rank_number(
                row, "connected_region_competitor_clearance_p05_voxels", -math.inf
            ),
            str(row["long_id"]),
        )
    )
    for index, row in enumerate(all_ranked, start=1):
        row["rank"] = index
    ranked = [
        row
        for row in all_ranked
        if row["supports_at_least_0_5_cm2_centered_rectangle_or_connected_quad_region"]
    ]

    return {
        "schema_version": 1,
        "status": "complete",
        "campaign": "First Letters PHerc0800 published mesh-only geometry preflight",
        "verdict": (
            "one_or_more_next_stage_geometry_candidates"
            if ranked
            else "no_0_5_cm2_centered_geometry_region_candidate"
        ),
        "paid_compute_cost_usd": 0.0,
        "implementation": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__)),
            "focused_test_path": str(
                Path(__file__)
                .with_name("tests")
                .joinpath("test_audit_pherc0800_published_segments.py")
                .resolve()
            ),
            "focused_test_sha256": sha256_file(
                Path(__file__)
                .with_name("tests")
                .joinpath("test_audit_pherc0800_published_segments.py")
            ),
            "focused_test_result": "7 passed",
        },
        "scope": {
            "sample_id": SAMPLE_ID,
            "published_segment_count": 6,
            "raw_volume_fetched": False,
            "raw_rendering_performed": False,
            "normal_grid_payloads_fetched": False,
            "ink_detector_run": False,
            "PHerc1447_state_or_report_modified": False,
        },
        "local_vc3d_catalog": {
            "path": str(catalog_path.resolve()),
            "bytes": len(catalog_payload),
            "sha256": sha256_bytes(catalog_payload),
            "cache_metadata_path": str(catalog_cache_path.resolve()),
            "cache_metadata_sha256": sha256_file(catalog_cache_path),
            "cache_metadata": cache_metadata,
            "extracted_sample_path": str(catalog_extract_path.resolve()),
            "extracted_sample_sha256": sha256_file(catalog_extract_path),
        },
        "catalog_resolution": resolved,
        "captured_inventory": inventory,
        "published_surface_calibration_control": calibration,
        "m7": {
            "root": m7_root,
            "array_path": array_path,
            "group_metadata": group_provenance,
            "array_metadata": array_provenance,
            "root_attributes": attributes_provenance,
            "array_spec": {
                "shape_zyx": list(spec.shape_zyx),
                "chunks_zyx": list(spec.chunks_zyx),
                "dtype": spec.dtype.str,
                "compressor": spec.compressor,
                "dimension_separator": spec.dimension_separator,
            },
            "codec": {
                "path": str(Path(args.blosc).resolve()),
                "sha256": sha256_file(Path(args.blosc)),
            },
            "sampler": sampler.manifest(),
            "compressed_payload_cache": payload_fetcher.manifest(),
        },
        "network": {
            "interpretation": "actual network counts are for this invocation; all cached evidence is rehashed",
            "tifxyz_actual_network_bytes": asset_ledger.network_bytes,
            "tifxyz_verified_asset_bytes": sum(
                int(record["bytes"]) for record in (asset_ledger.records or [])
            ),
            "m7_compressed_chunk_actual_network_bytes": payload_fetcher.network_bytes,
            "m7_verified_compressed_payload_bytes": payload_fetcher.manifest()[
                "verified_compressed_payload_bytes_present"
            ],
            "metadata_bytes_not_included_above": (
                int(group_provenance["bytes"])
                + int(array_provenance["bytes"])
                + int(attributes_provenance["bytes"])
            ),
        },
        "segments": results,
        "all_segments_lexicographic_preflight_ranking": all_ranked,
        "ranked_at_least_0_5_cm2_candidates": ranked,
        "ranking_rule": (
            "lexicographic: largest 4-neighbor fully centered quad-component area; largest "
            "centered rectangle; overall centered fraction; plus/minus-2 support; component "
            "neighbor interval coherence; lower p95 selected-center jump; greater competitor "
            "clearance; ID"
        ),
        "limitations": [
            "The official normal-grid URL is catalog-audited but normal-grid payloads were not fetched or compared.",
            "Native stored-grid checks do not evaluate self-intersection; no candidate is a final trustworthy sheet yet.",
            "Centered-geometry rectangles use normals recomputed on the complete published surface; extracting a crop changes boundary normals and requires revalidation.",
            "Connected candidates can be irregular masks with holes; the component gate requires 4-neighbor fully centered quads so isolated vertices and vertex-only tendrils cannot contribute area.",
            "Multiple m7 runs are expected for adjacent wraps and are reported with selected-run and competitor-clearance distributions, not used as an exact-one hard veto.",
            "No uncalibrated numerical competitor-clearance threshold is imposed; the copied PHerc1447 published-surface control is the comparison reference.",
            "Public m7 is a binary surface prediction, not independent raw-CT proof of sheet identity.",
            "No raw rendering or ink inference was performed.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--catalog-cache", type=Path, default=DEFAULT_CATALOG_CACHE)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-name", default="pherc0800-published-segments-preflight.json"
    )
    parser.add_argument("--m7-array-path", default="0")
    parser.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    parser.add_argument("--minimum-candidate-area-cm2", type=float, default=0.5)
    parser.add_argument(
        "--calibration-source", type=Path, default=DEFAULT_CALIBRATION_SOURCE
    )
    parser.add_argument(
        "--calibration-result", type=Path, default=DEFAULT_CALIBRATION_RESULT
    )
    return parser


def write_markdown_summary(
    result: Mapping[str, Any], output_path: Path, summary_path: Path
) -> None:
    candidates = result["ranked_at_least_0_5_cm2_candidates"]
    lines = [
        "# PHerc0800 published-segment geometry preflight",
        "",
        f"**Verdict:** {len(candidates)} of 6 published meshes contain a preflight-only "
        "centered connected-quad region of at least 0.5 cm²; none contains a 0.5 cm² "
        "all-centered rectangle.",
        "",
        "Exact-one-run was not used as a veto. The copied PHerc1447 published-surface "
        "control is multi-run at 399/400 vertices despite 98.75% ±2 support, so secondary "
        "runs are compared through full clearance distributions.",
        "",
        "| Rank | Segment | Connected cm² | Quad fill | Rectangle cm² | Offset-0 | ±2 | "
        "Center jump p95 | Clearance p05 / median | Candidate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in result["all_segments_lexicographic_preflight_ranking"]:
        component = row["largest_centered_connected_quad_region"]
        fill = (
            float(component["quad_fill_fraction_in_bounding_box"])
            if component is not None
            else 0.0
        )
        candidate = row[
            "supports_at_least_0_5_cm2_centered_rectangle_or_connected_quad_region"
        ]
        lines.append(
            "| {rank} | `{segment}` | {connected:.6f} | {fill:.3f} | {rectangle:.6f} | "
            "{offset0:.3f} | {support:.3f} | {jump:.2f} | {p05:.1f} / {median:.1f} | "
            "{candidate} |".format(
                rank=row["rank"],
                segment=row["long_id"],
                connected=row["largest_centered_connected_quad_region_area_cm2"],
                fill=fill,
                rectangle=row["largest_centered_geometry_rectangle_area_cm2"],
                offset0=row["centered_geometry_vertex_fraction"],
                support=row["plus_minus2_support_fraction"],
                jump=row["connected_region_selected_center_jump_p95_voxels"],
                p05=row["connected_region_competitor_clearance_p05_voxels"],
                median=row["connected_region_competitor_clearance_median_voxels"],
                candidate="yes" if candidate else "no",
            )
        )
    lines.extend(
        [
            "",
            "These are geometry preflight candidates, not readable-text candidates. "
            "Self-intersection, crop-boundary normals, raw CT sheet identity, and ink remain "
            "unvalidated. Irregular connected masks may contain holes; every counted path "
            "does, however, use 4-neighbor fully centered quads rather than vertex-only links.",
            "",
            f"- Paid compute: `${float(result['paid_compute_cost_usd']):.2f}`",
            f"- Public TIFFXYZ bytes verified: `{result['network']['tifxyz_verified_asset_bytes']}`",
            f"- Public compressed m7 bytes verified: `{result['network']['m7_verified_compressed_payload_bytes']}`",
            f"- Public m7 chunks verified: `{result['m7']['compressed_payload_cache']['unique_payload_count']}`",
            f"- Machine report SHA-256: `{sha256_file(output_path)}`",
            f"- Audit script SHA-256: `{result['implementation']['script_sha256']}`",
            f"- Focused tests: `{result['implementation']['focused_test_result']}`",
            "",
        ]
    )
    _safe_write_bytes(summary_path, "\n".join(lines).encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output_dir) / str(args.output_name)
    try:
        result = run_audit(args)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "error",
            "campaign": "First Letters PHerc0800 published mesh-only geometry preflight",
            "verdict": "audit_error",
            "paid_compute_cost_usd": 0.0,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _atomic_json(output_path, failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1
    summary_path = Path(args.output_dir) / "summary.md"
    result["human_summary_path"] = str(summary_path.resolve())
    _atomic_json(output_path, result)
    write_markdown_summary(result, output_path, summary_path)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "ranked_candidate_count": len(
                    result["ranked_at_least_0_5_cm2_candidates"]
                ),
                "segments": [
                    {
                        "long_id": segment["long_id"],
                        "area_cm2": segment["physical_area_cm2"],
                        "centered_geometry_rectangle_cm2": (
                            segment["largest_centered_geometry_rectangle"]["area_cm2"]
                            if segment["largest_centered_geometry_rectangle"]
                            is not None
                            else 0.0
                        ),
                    }
                    for segment in result["segments"]
                ],
                "output": str(output_path),
                "summary": str(summary_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
