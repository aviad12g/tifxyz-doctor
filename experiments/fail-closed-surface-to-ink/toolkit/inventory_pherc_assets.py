#!/usr/bin/env python3
"""Create a deterministic inventory of public PHerc assets in S3.

The inventory is built from unauthenticated, fully paginated S3 ListObjectsV2
requests.  A delimiter walk proves the complete asset-prefix set without
entering enormous Zarr chunk trees.  Compact mesh/TIFF-XYZ and TIFF-stack
assets are recursively enumerated, while Zarr and normal-grid containers are
recorded by prefix with an explicit ``prefix_only`` coverage marker.  It covers
the three scopes needed by the First Letters workflow:

* ``<scroll>/volumes/`` (raw volume Zarrs),
* ``<scroll>/representations/predictions/`` (including normal grids), and
* ``<scroll>/segments/`` (mesh/TIFF-XYZ variants and surface volumes).

Every enumerated object record preserves its S3 key, Size, ETag, LastModified,
storage class, and advertised checksum metadata.  Stable SHA-256 digests over
both object records and discovered asset identities make snapshots comparable.

For reproducible/offline runs, pass one or more XML files or directories with
``--fixtures``.  Fixture XML is an ordinary S3 ListObjectsV2 response.  A
multi-page fixture should include ``ContinuationToken`` on pages after the
first; filename order is accepted as a fallback for captured responses that
omit it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Iterable, Sequence
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


DEFAULT_ENDPOINT = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
SCHEMA_VERSION = 2
GEOMETRY_EXTENSIONS = {
    ".glb",
    ".gltf",
    ".mesh",
    ".npz",
    ".obj",
    ".off",
    ".ply",
    ".stl",
    ".vtk",
    ".vtp",
}
TIFF_EXTENSIONS = {".tif", ".tiff"}
TIFFXYZ_RE = re.compile(r"tif[-_]?xyz", re.IGNORECASE)


class InventoryError(RuntimeError):
    """Raised when a listing is incomplete, contradictory, or malformed."""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str | None:
    for child in element:
        if _local_name(child.tag) == name:
            return child.text
    return None


def _required_child_text(element: ET.Element, name: str) -> str:
    value = _child_text(element, name)
    if value is None:
        raise InventoryError(f"S3 Contents entry is missing {name}")
    return value


@dataclasses.dataclass(frozen=True, order=True)
class ObjectRecord:
    """Metadata supplied by S3 for one immutable object version snapshot."""

    key: str
    size: int
    etag: str
    last_modified: str
    storage_class: str | None = None
    checksum_algorithms: tuple[str, ...] = ()
    checksum_type: str | None = None

    def to_dict(self) -> dict[str, object]:
        # Keep optional fields in the schema.  A null says that S3 did not
        # advertise the field, which is different from losing it in parsing.
        return {
            "key": self.key,
            "size": self.size,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "storage_class": self.storage_class,
            "checksum_algorithms": list(self.checksum_algorithms),
            "checksum_type": self.checksum_type,
        }


@dataclasses.dataclass(frozen=True)
class ListPage:
    prefix: str
    delimiter: str | None
    continuation_token: str | None
    next_continuation_token: str | None
    is_truncated: bool
    objects: tuple[ObjectRecord, ...]
    common_prefixes: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Listing:
    prefix: str
    pages: int
    objects: tuple[ObjectRecord, ...]
    common_prefixes: tuple[str, ...] = ()
    delimiter: str | None = None


def parse_list_objects_v2_xml(data: bytes | str) -> ListPage:
    """Parse one namespaced or namespace-free ListObjectsV2 response."""

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise InventoryError(f"invalid ListObjectsV2 XML: {exc}") from exc
    if _local_name(root.tag) != "ListBucketResult":
        raise InventoryError(
            f"expected ListBucketResult, got {_local_name(root.tag)!r}"
        )

    prefix_text = _child_text(root, "Prefix")
    if prefix_text is None:
        raise InventoryError("ListObjectsV2 response is missing Prefix")
    prefix = prefix_text
    delimiter = _child_text(root, "Delimiter")
    continuation_token = _child_text(root, "ContinuationToken")
    next_token = _child_text(root, "NextContinuationToken")
    truncated_value = _child_text(root, "IsTruncated")
    if truncated_value is None:
        raise InventoryError("ListObjectsV2 response is missing IsTruncated")
    truncated_text = truncated_value.strip().lower()
    if truncated_text not in {"true", "false"}:
        raise InventoryError(f"invalid IsTruncated value: {truncated_text!r}")

    records: list[ObjectRecord] = []
    common_prefixes: list[str] = []
    for child in root:
        tag = _local_name(child.tag)
        if tag == "Contents":
            size_text = _required_child_text(child, "Size")
            try:
                size = int(size_text)
            except ValueError as exc:
                raise InventoryError(f"invalid object Size {size_text!r}") from exc
            if size < 0:
                raise InventoryError(f"negative object Size {size}")
            checksum_algorithms = tuple(
                item.text or ""
                for item in child
                if _local_name(item.tag) == "ChecksumAlgorithm"
            )
            records.append(
                ObjectRecord(
                    key=_required_child_text(child, "Key"),
                    size=size,
                    etag=_required_child_text(child, "ETag"),
                    last_modified=_required_child_text(child, "LastModified"),
                    storage_class=_child_text(child, "StorageClass"),
                    checksum_algorithms=checksum_algorithms,
                    checksum_type=_child_text(child, "ChecksumType"),
                )
            )
        elif tag == "CommonPrefixes":
            common_prefix = _child_text(child, "Prefix")
            if common_prefix is not None:
                common_prefixes.append(common_prefix)

    return ListPage(
        prefix=prefix,
        delimiter=delimiter,
        continuation_token=continuation_token,
        next_continuation_token=next_token,
        is_truncated=truncated_text == "true",
        objects=tuple(records),
        common_prefixes=tuple(common_prefixes),
    )


class HttpListObjectsV2Source:
    """Unauthenticated S3 ListObjectsV2 transport."""

    mode = "public-s3"

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        timeout: float = 60.0,
        retries: int = 4,
    ) -> None:
        self.endpoint = endpoint.rstrip("/") + "/"
        self.timeout = timeout
        self.retries = retries

    def fetch(
        self,
        prefix: str,
        continuation_token: str | None,
        delimiter: str | None = None,
    ) -> ListPage:
        params = {"list-type": "2", "prefix": prefix}
        if continuation_token is not None:
            params["continuation-token"] = continuation_token
        if delimiter is not None:
            params["delimiter"] = delimiter
        url = self.endpoint + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "pherc-immutable-inventory/1"},
        )

        for attempt in range(self.retries + 1):
            try:
                # No Authorization header, signed query parameter, AWS SDK, or
                # credential lookup is involved here.
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return parse_list_objects_v2_xml(response.read())
            except urllib.error.HTTPError as exc:
                retryable = exc.code in {408, 429, 500, 502, 503, 504}
                if not retryable or attempt == self.retries:
                    raise InventoryError(f"S3 listing failed for {prefix}: {exc}") from exc
            except urllib.error.URLError as exc:
                if attempt == self.retries:
                    raise InventoryError(f"S3 listing failed for {prefix}: {exc}") from exc
            time.sleep(min(2**attempt, 8))
        raise AssertionError("unreachable")


class FixtureListObjectsV2Source:
    """Serve ListObjectsV2 pages from captured XML without network access."""

    mode = "xml-fixtures"

    def __init__(self, paths: Sequence[Path | str]) -> None:
        files: list[Path] = []
        for raw_path in paths:
            path = Path(raw_path)
            if path.is_dir():
                files.extend(sorted(path.rglob("*.xml")))
            elif path.is_file():
                files.append(path)
            else:
                raise InventoryError(f"fixture path does not exist: {path}")
        if not files:
            raise InventoryError("no XML fixture files found")

        grouped: dict[tuple[str, str | None], list[tuple[Path, ListPage]]] = {}
        for path in sorted(set(files)):
            page = parse_list_objects_v2_xml(path.read_bytes())
            grouped.setdefault((page.prefix, page.delimiter), []).append((path, page))

        self._pages: dict[tuple[str, str | None, str | None], ListPage] = {}
        self.fixture_count = len(files)
        for (prefix, delimiter), entries in grouped.items():
            self._index_group(prefix, delimiter, entries)

    def _index_group(
        self,
        prefix: str,
        delimiter: str | None,
        entries: Sequence[tuple[Path, ListPage]],
    ) -> None:
        explicit: dict[str | None, tuple[Path, ListPage]] = {}
        unassigned: list[tuple[Path, ListPage]] = []
        for path, page in entries:
            token = page.continuation_token
            if token is None:
                unassigned.append((path, page))
            else:
                if token in explicit:
                    raise InventoryError(
                        f"duplicate fixture page for {prefix} continuation token {token!r}"
                    )
                explicit[token] = (path, page)

        # The first response has no ContinuationToken.  Captured responses from
        # some tools also omit it on later pages, so deterministically chain
        # those pages by filename using the preceding NextContinuationToken.
        first_candidates = list(unassigned)
        if not first_candidates and entries:
            raise InventoryError(f"fixtures for {prefix} have no first page")
        first_path, first_page = first_candidates.pop(0)
        ordered: list[tuple[str | None, Path, ListPage]] = [(None, first_path, first_page)]
        unassigned = first_candidates
        token = first_page.next_continuation_token
        seen_tokens: set[str | None] = {None}
        while token is not None:
            if token in seen_tokens:
                raise InventoryError(f"fixture continuation-token cycle for {prefix}")
            seen_tokens.add(token)
            if token in explicit:
                path, page = explicit.pop(token)
            elif unassigned:
                path, page = unassigned.pop(0)
            else:
                # Leave the chain incomplete.  fetch() will produce a precise
                # error if a run actually needs the missing response.
                break
            ordered.append((token, path, page))
            token = page.next_continuation_token

        if explicit or unassigned:
            leftovers = [str(path) for path, _ in explicit.values()] + [
                str(path) for path, _ in unassigned
            ]
            raise InventoryError(
                f"unreachable or ambiguous fixture pages for {prefix}: {leftovers}"
            )
        for token, _path, page in ordered:
            self._pages[(prefix, delimiter, token)] = page

    def fetch(
        self,
        prefix: str,
        continuation_token: str | None,
        delimiter: str | None = None,
    ) -> ListPage:
        try:
            return self._pages[(prefix, delimiter, continuation_token)]
        except KeyError as exc:
            raise InventoryError(
                "missing XML fixture for "
                f"prefix={prefix!r}, delimiter={delimiter!r}, "
                f"continuation_token={continuation_token!r}"
            ) from exc


def list_all_objects(
    source: object, prefix: str, delimiter: str | None = None
) -> Listing:
    """Fetch every page for *prefix*, rejecting incomplete/unstable listings.

    With a delimiter, completeness applies to all direct objects and all
    CommonPrefixes at that hierarchy level.  Descendants beneath each common
    prefix are not enumerated unless the caller explicitly walks them.
    """

    token: str | None = None
    seen_tokens: set[str] = set()
    records: dict[str, ObjectRecord] = {}
    common_prefixes: set[str] = set()
    pages = 0
    while True:
        page = source.fetch(prefix, token, delimiter)  # type: ignore[attr-defined]
        pages += 1
        if page.prefix != prefix:
            raise InventoryError(
                f"S3 returned prefix {page.prefix!r} while listing {prefix!r}"
            )
        if (
            page.continuation_token is not None
            and page.continuation_token != token
        ):
            raise InventoryError(
                f"S3 echoed continuation token {page.continuation_token!r} "
                f"while {token!r} was requested for {prefix}"
            )
        if page.delimiter != delimiter:
            raise InventoryError(
                f"S3 returned delimiter {page.delimiter!r} while {delimiter!r} "
                f"was requested for {prefix}"
            )
        for record in page.objects:
            if not record.key.startswith(prefix):
                raise InventoryError(
                    f"object {record.key!r} falls outside requested prefix {prefix!r}"
                )
            previous = records.get(record.key)
            if previous is not None:
                detail = "conflicting metadata" if previous != record else "duplicate key"
                raise InventoryError(f"{detail} across pages for {record.key}")
            records[record.key] = record
        for common_prefix in page.common_prefixes:
            if not common_prefix.startswith(prefix):
                raise InventoryError(
                    f"common prefix {common_prefix!r} falls outside {prefix!r}"
                )
            if delimiter and not common_prefix.endswith(delimiter):
                raise InventoryError(
                    f"common prefix {common_prefix!r} lacks delimiter {delimiter!r}"
                )
            if common_prefix in common_prefixes:
                raise InventoryError(
                    f"duplicate CommonPrefix across pages for {common_prefix}"
                )
            common_prefixes.add(common_prefix)

        if not page.is_truncated:
            if page.next_continuation_token:
                raise InventoryError(
                    f"non-truncated response for {prefix} unexpectedly has a next token"
                )
            break
        next_token = page.next_continuation_token
        if not next_token:
            raise InventoryError(
                f"truncated response for {prefix} has no NextContinuationToken"
            )
        if next_token in seen_tokens:
            raise InventoryError(f"repeated continuation token while listing {prefix}")
        seen_tokens.add(next_token)
        token = next_token

    return Listing(
        prefix=prefix,
        pages=pages,
        objects=tuple(sorted(records.values())),
        common_prefixes=tuple(sorted(common_prefixes)),
        delimiter=delimiter,
    )


def _records_dict(records: Iterable[ObjectRecord]) -> list[dict[str, object]]:
    return [record.to_dict() for record in sorted(records)]


def _object_digest(records: Iterable[ObjectRecord]) -> str:
    canonical = json.dumps(
        _records_dict(records),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _asset_format(name: str) -> str:
    lower = name.lower().rstrip("/")
    if "normal-grids" in lower or "normal_grids" in lower:
        return "normal_grids"
    if lower.endswith(".zarr"):
        return "zarr"
    if lower.endswith(".tifs"):
        return "tiff_stack"
    if any(lower.endswith(extension) for extension in TIFF_EXTENSIONS):
        return "tiff"
    return "other"


def _summarize_asset(
    prefix: str,
    records: Iterable[ObjectRecord],
    *,
    asset_format: str | None = None,
    inventory_mode: str = "recursive_objects",
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    ordered = tuple(sorted(records))
    result: dict[str, object] = {
        "name": prefix.rstrip("/").rsplit("/", 1)[-1],
        "prefix": prefix,
        "format": asset_format or _asset_format(prefix),
        "object_count": len(ordered),
        "total_bytes": sum(record.size for record in ordered),
        "latest_last_modified": max(
            (record.last_modified for record in ordered), default=None
        ),
        "objects_sha256": _object_digest(ordered),
        "object_inventory": {
            "mode": inventory_mode,
            "descendant_objects_enumerated": inventory_mode == "recursive_objects",
            "asset_prefix_discovery_complete": True,
        },
        "objects": _records_dict(ordered),
    }
    if extra:
        result.update(extra)
    return result


def _container_root(
    record: ObjectRecord,
    base_prefix: str,
    suffixes: Sequence[str],
) -> str | None:
    relative = record.key.removeprefix(base_prefix)
    parts = relative.split("/")
    for index, part in enumerate(parts):
        if any(part.lower().endswith(suffix) for suffix in suffixes):
            return base_prefix + "/".join(parts[: index + 1]) + "/"
    return None


def _group_container_assets(
    records: Iterable[ObjectRecord],
    base_prefix: str,
    suffixes: Sequence[str],
    *,
    known_prefixes: Iterable[str] = (),
    prefix_only: Iterable[str] = (),
) -> tuple[list[dict[str, object]], list[ObjectRecord]]:
    groups: dict[str, list[ObjectRecord]] = {
        prefix: [] for prefix in sorted(set(known_prefixes))
    }
    prefix_only_set = set(prefix_only)
    other: list[ObjectRecord] = []
    for record in records:
        root = _container_root(record, base_prefix, suffixes)
        if root is None:
            other.append(record)
        else:
            groups.setdefault(root, []).append(record)
    assets = [
        _summarize_asset(
            root,
            groups[root],
            inventory_mode=(
                "prefix_only" if root in prefix_only_set else "recursive_objects"
            ),
        )
        for root in sorted(groups)
    ]
    return assets, sorted(other)


def _infer_transform(name: str) -> str | None:
    lower = Path(name).stem.lower()
    for transform in ("flattened", "normalized", "original"):
        if lower == transform or lower.endswith("_" + transform):
            return transform
    if TIFFXYZ_RE.search(lower):
        suffix = TIFFXYZ_RE.split(lower, maxsplit=1)[-1].strip("-_")
        return suffix or None
    return None


def _build_mesh_assets(
    records: Sequence[ObjectRecord],
    mesh_prefix: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[ObjectRecord]]:
    tifxyz_groups: dict[str, list[ObjectRecord]] = {}
    geometry_records: list[ObjectRecord] = []
    other: list[ObjectRecord] = []

    for record in sorted(records):
        relative = record.key.removeprefix(mesh_prefix)
        parts = relative.split("/")
        tifxyz_index = next(
            (index for index, part in enumerate(parts) if TIFFXYZ_RE.search(part)),
            None,
        )
        if tifxyz_index is not None:
            root = mesh_prefix + "/".join(parts[: tifxyz_index + 1]) + "/"
            tifxyz_groups.setdefault(root, []).append(record)
        elif Path(record.key).suffix.lower() in GEOMETRY_EXTENSIONS:
            geometry_records.append(record)
        else:
            other.append(record)

    geometry_assets: list[dict[str, object]] = []
    geometry_transforms: set[str] = set()
    for record in geometry_records:
        transform = _infer_transform(record.key)
        if transform:
            geometry_transforms.add(transform)
        geometry_assets.append(
            _summarize_asset(
                record.key,
                [record],
                asset_format=Path(record.key).suffix.lower().lstrip("/.") or "mesh",
                extra={
                    "variant_id": record.key.removeprefix(mesh_prefix),
                    "transform": transform,
                },
            )
        )

    tifxyz_assets: list[dict[str, object]] = []
    for root in sorted(tifxyz_groups):
        group = tifxyz_groups[root]
        axis_keys: dict[str, list[str]] = {"x": [], "y": [], "z": []}
        for record in group:
            stem = Path(record.key).stem.lower()
            if stem in axis_keys and Path(record.key).suffix.lower() in TIFF_EXTENSIONS:
                axis_keys[stem].append(record.key)
        axis_keys = {axis: keys for axis, keys in axis_keys.items() if keys}
        axes = sorted(axis_keys)
        transform = _infer_transform(root.rstrip("/").rsplit("/", 1)[-1])
        tifxyz_assets.append(
            _summarize_asset(
                root,
                group,
                asset_format="tifxyz",
                extra={
                    "variant_id": root.removeprefix(mesh_prefix).rstrip("/"),
                    "transform": transform,
                    "coordinate_axes_present": axes,
                    "coordinate_keys": axis_keys,
                    "complete": axes == ["x", "y", "z"],
                    "has_meta_json": any(
                        Path(record.key).name.lower() == "meta.json" for record in group
                    ),
                    "has_geometry_counterpart": bool(
                        transform and transform in geometry_transforms
                    ),
                },
            )
        )
    return tifxyz_assets, geometry_assets, sorted(other)


def _build_raw_tifxyz_assets(
    records: Sequence[ObjectRecord],
    raw_prefix: str,
) -> tuple[list[dict[str, object]], set[str]]:
    """Classify flat TIFF-XYZ maps published beneath ``segments/raw/``."""

    by_parent: dict[str, list[ObjectRecord]] = {}
    all_by_parent: dict[str, list[ObjectRecord]] = {}
    core_names = {"meta.json", "x.tif", "x.tiff", "y.tif", "y.tiff", "z.tif", "z.tiff"}
    for record in sorted(records):
        if not record.key.startswith(raw_prefix):
            continue
        parent = record.key.rsplit("/", 1)[0] + "/"
        all_by_parent.setdefault(parent, []).append(record)
        if Path(record.key).name.lower() in core_names:
            by_parent.setdefault(parent, []).append(record)

    assets: list[dict[str, object]] = []
    assigned_keys: set[str] = set()
    for parent in sorted(by_parent):
        group = by_parent[parent]
        axes = sorted(
            {
                Path(record.key).stem.lower()
                for record in group
                if Path(record.key).stem.lower() in {"x", "y", "z"}
                and Path(record.key).suffix.lower() in TIFF_EXTENSIONS
            }
        )
        if not axes:
            continue
        assigned_keys.update(record.key for record in group)
        variant_id = parent.removeprefix(raw_prefix).rstrip("/")
        related_previews = [
            record.key
            for record in all_by_parent.get(parent, [])
            if Path(record.key).suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        assets.append(
            _summarize_asset(
                parent,
                group,
                asset_format="tifxyz",
                extra={
                    "variant_id": variant_id,
                    "basename": variant_id.rsplit("/", 1)[-1],
                    "coordinate_axes_present": axes,
                    "complete": axes == ["x", "y", "z"],
                    "has_meta_json": any(
                        Path(record.key).name.lower() == "meta.json"
                        for record in group
                    ),
                    "nested_auxiliary": "/" in variant_id,
                    "related_preview_keys": sorted(related_previews),
                },
            )
        )
    return assets, assigned_keys


def _build_surface_assets(
    records: Sequence[ObjectRecord],
    surface_prefix: str,
    *,
    known_prefixes: Iterable[str] = (),
    prefix_only: Iterable[str] = (),
) -> list[dict[str, object]]:
    groups: dict[str, list[ObjectRecord]] = {
        prefix: [] for prefix in sorted(set(known_prefixes))
    }
    prefix_only_set = set(prefix_only)
    for record in sorted(records):
        relative = record.key.removeprefix(surface_prefix)
        first_component = relative.split("/", 1)[0]
        if not first_component:
            first_component = "(prefix-object)"
        suffix = "/" if "/" in relative else ""
        root = surface_prefix + first_component + suffix
        groups.setdefault(root, []).append(record)
    return [
        _summarize_asset(
            root,
            groups[root],
            inventory_mode=(
                "prefix_only" if root in prefix_only_set else "recursive_objects"
            ),
        )
        for root in sorted(groups)
    ]


def _build_segments(
    records: Sequence[ObjectRecord],
    segments_prefix: str,
    *,
    known_surface_prefixes: Iterable[str] = (),
    prefix_only_surface_prefixes: Iterable[str] = (),
    known_segment_ids: Iterable[str] = (),
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[ObjectRecord]]:
    known_segment_ids = tuple(sorted(set(known_segment_ids)))
    known_by_length = sorted(known_segment_ids, key=len, reverse=True)

    def split_segment(relative: str) -> tuple[str, str] | None:
        for known_id in known_by_length:
            marker = known_id + "/"
            if relative.startswith(marker):
                return known_id, relative[len(marker) :]
        parts = relative.split("/", 1)
        if not known_by_length and len(parts) == 2 and parts[0]:
            return parts[0], parts[1]
        return None

    per_segment: dict[str, dict[str, list[ObjectRecord]]] = {}
    top_level_other: list[ObjectRecord] = []
    for record in sorted(records):
        relative = record.key.removeprefix(segments_prefix)
        split = split_segment(relative)
        if split is None:
            top_level_other.append(record)
            continue
        segment_id, within_segment = split
        category = within_segment.split("/", 1)[0]
        buckets = per_segment.setdefault(
            segment_id, {"mesh": [], "surface-volumes": [], "other": []}
        )
        if category in {"mesh", "surface-volumes"}:
            buckets[category].append(record)
        else:
            buckets["other"].append(record)

    for segment_id in known_segment_ids:
        per_segment.setdefault(
            segment_id, {"mesh": [], "surface-volumes": [], "other": []}
        )

    surface_prefixes_by_segment: dict[str, list[str]] = {}
    for prefix in known_surface_prefixes:
        relative = prefix.removeprefix(segments_prefix)
        split = split_segment(relative)
        if split is None:
            raise InventoryError(
                f"surface asset {prefix!r} has no known segment prefix"
            )
        segment_id, _within_segment = split
        surface_prefixes_by_segment.setdefault(segment_id, []).append(prefix)
    prefix_only_surface_set = set(prefix_only_surface_prefixes)

    segments: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for segment_id in sorted(per_segment):
        buckets = per_segment[segment_id]
        segment_prefix = f"{segments_prefix}{segment_id}/"
        mesh_prefix = segment_prefix + "mesh/"
        surface_prefix = segment_prefix + "surface-volumes/"
        tifxyz, geometry, other_mesh = _build_mesh_assets(
            buckets["mesh"], mesh_prefix
        )
        surface_volumes = _build_surface_assets(
            buckets["surface-volumes"],
            surface_prefix,
            known_prefixes=surface_prefixes_by_segment.get(segment_id, []),
            prefix_only=(
                prefix
                for prefix in surface_prefixes_by_segment.get(segment_id, [])
                if prefix in prefix_only_surface_set
            ),
        )
        complete_tifxyz = [asset for asset in tifxyz if asset["complete"]]
        surface_formats = sorted({str(asset["format"]) for asset in surface_volumes})
        capabilities = {
            "has_geometry_mesh": bool(geometry),
            "has_tifxyz": bool(tifxyz),
            "has_complete_tifxyz": bool(complete_tifxyz),
            "has_surface_volume": bool(surface_volumes),
            "has_surface_zarr": "zarr" in surface_formats,
            "has_surface_tiff_stack": "tiff_stack" in surface_formats,
            "direct_surface_screening_ready": bool(surface_volumes),
            "surface_extraction_required": bool(complete_tifxyz)
            and not surface_volumes,
            "tifxyz_variants": len(tifxyz),
            "complete_tifxyz_variants": len(complete_tifxyz),
            "geometry_variants": len(geometry),
            "surface_volume_variants": len(surface_volumes),
        }
        segment = {
            "segment_id": segment_id,
            "prefix": segment_prefix,
            "capabilities": capabilities,
            "tifxyz_variants": tifxyz,
            "geometry_variants": geometry,
            "surface_volumes": surface_volumes,
            "other_mesh_objects": _records_dict(other_mesh),
            "other_objects": _records_dict(buckets["other"]),
        }
        segments.append(segment)

        if not surface_volumes:
            for asset in complete_tifxyz:
                candidates.append(
                    {
                        "segment_id": segment_id,
                        "variant_id": asset["variant_id"],
                        "prefix": asset["prefix"],
                        "transform": asset["transform"],
                        "object_count": asset["object_count"],
                        "total_bytes": asset["total_bytes"],
                        "objects_sha256": asset["objects_sha256"],
                        "has_geometry_counterpart": asset[
                            "has_geometry_counterpart"
                        ],
                        "reason": "complete_tifxyz_no_published_surface_volume",
                    }
                )
    return segments, candidates, sorted(top_level_other)


def build_inventory(
    scroll: str,
    listings: Sequence[Listing],
    *,
    source_mode: str,
    endpoint: str = DEFAULT_ENDPOINT,
    discovered_assets: dict[str, object] | None = None,
    listing_audit: dict[str, object] | None = None,
) -> dict[str, object]:
    """Classify exhaustive listings into a deterministic inventory report."""

    if not scroll or "/" in scroll:
        raise InventoryError("--scroll must be one S3 path component")
    expected_prefixes = {
        f"{scroll}/volumes/",
        f"{scroll}/representations/predictions/",
        f"{scroll}/segments/",
    }
    by_prefix = {listing.prefix: listing for listing in listings}
    missing = expected_prefixes - by_prefix.keys()
    extra = by_prefix.keys() - expected_prefixes
    if missing or extra or len(by_prefix) != len(listings):
        raise InventoryError(
            f"listing scopes do not match expected prefixes; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    volume_prefix = f"{scroll}/volumes/"
    prediction_prefix = f"{scroll}/representations/predictions/"
    segments_prefix = f"{scroll}/segments/"
    discovered_assets = discovered_assets or {}
    raw_prefixes = tuple(discovered_assets.get("raw_volumes", ()))
    prediction_prefixes = tuple(discovered_assets.get("predictions", ()))
    surface_prefixes = tuple(discovered_assets.get("surface_volumes", ()))
    prefix_only = set(discovered_assets.get("prefix_only", ()))
    known_segment_ids = tuple(discovered_assets.get("segment_ids", ()))

    raw_volumes, other_volume_objects = _group_container_assets(
        by_prefix[volume_prefix].objects,
        volume_prefix,
        (".zarr",),
        known_prefixes=raw_prefixes,
        prefix_only=(prefix for prefix in raw_prefixes if prefix in prefix_only),
    )
    predictions, other_prediction_objects = _group_container_assets(
        by_prefix[prediction_prefix].objects,
        prediction_prefix,
        (
            ".normal-grids",
            "normal-grids",
            "normal_grids",
            ".zarr",
            ".tifs",
            ".tif",
            ".tiff",
        ),
        known_prefixes=prediction_prefixes,
        prefix_only=(
            prefix for prefix in prediction_prefixes if prefix in prefix_only
        ),
    )
    raw_tifxyz, raw_tifxyz_keys = _build_raw_tifxyz_assets(
        by_prefix[segments_prefix].objects,
        segments_prefix + "raw/",
    )
    classified_segment_records = tuple(
        record
        for record in by_prefix[segments_prefix].objects
        if record.key not in raw_tifxyz_keys
    )
    segments, candidates, other_segment_objects = _build_segments(
        classified_segment_records,
        segments_prefix,
        known_surface_prefixes=surface_prefixes,
        prefix_only_surface_prefixes=(
            prefix for prefix in surface_prefixes if prefix in prefix_only
        ),
        known_segment_ids=known_segment_ids,
    )

    all_records = tuple(
        sorted(record for listing in listings for record in listing.objects)
    )
    keys = [record.key for record in all_records]
    if len(keys) != len(set(keys)):
        raise InventoryError("the three inventory scopes contain duplicate object keys")

    surface_assets = [
        asset for segment in segments for asset in segment["surface_volumes"]
    ]
    tifxyz_assets = [
        asset for segment in segments for asset in segment["tifxyz_variants"]
    ]
    geometry_assets = [
        asset for segment in segments for asset in segment["geometry_variants"]
    ]
    segment_by_id = {str(segment["segment_id"]): segment for segment in segments}
    unseen_raw_tifxyz: list[dict[str, object]] = []
    nested_unmatched_raw_tifxyz: list[dict[str, object]] = []

    def raw_candidate_reference(asset: dict[str, object]) -> dict[str, object]:
        return {
            "variant_id": asset["variant_id"],
            "basename": asset["basename"],
            "prefix": asset["prefix"],
            "object_count": asset["object_count"],
            "total_bytes": asset["total_bytes"],
            "objects_sha256": asset["objects_sha256"],
            "nested_auxiliary": asset["nested_auxiliary"],
            "related_preview_keys": asset["related_preview_keys"],
            "visibility": asset["visibility"],
        }

    for asset in raw_tifxyz:
        basename = str(asset["basename"])
        matches = sorted(
            segment_id
            for segment_id in segment_by_id
            if segment_id == basename or segment_id.endswith("-" + basename)
        )
        surface_matches = [
            segment_id
            for segment_id in matches
            if segment_by_id[segment_id]["capabilities"]["has_surface_volume"]
        ]
        asset["published_segment_matches"] = matches
        asset["published_surface_segment_matches"] = surface_matches
        if surface_matches:
            asset["visibility"] = "published_surface_available"
        elif matches:
            asset["visibility"] = "published_mesh_only"
        elif asset["nested_auxiliary"]:
            asset["visibility"] = "nested_unmatched_auxiliary"
            if asset["complete"]:
                nested_unmatched_raw_tifxyz.append(raw_candidate_reference(asset))
        else:
            asset["visibility"] = "raw_only_unseen"
            if asset["complete"]:
                unseen_raw_tifxyz.append(raw_candidate_reference(asset))
    other_objects_total = (
        len(other_volume_objects)
        + len(other_prediction_objects)
        + len(other_segment_objects)
        + sum(
            len(segment["other_mesh_objects"]) + len(segment["other_objects"])
            for segment in segments
        )
    )
    all_assets = (
        raw_volumes
        + predictions
        + surface_assets
        + tifxyz_assets
        + geometry_assets
        + raw_tifxyz
    )
    summary = {
        "objects_total": len(all_records),
        "indexed_objects_total": len(all_records),
        "bytes_total": sum(record.size for record in all_records),
        "latest_last_modified": max(
            (record.last_modified for record in all_records), default=None
        ),
        "raw_volume_zarrs": len(raw_volumes),
        "prediction_assets": len(predictions),
        "normal_grid_assets": sum(
            asset["format"] == "normal_grids" for asset in predictions
        ),
        "segments_total": len(segments),
        "segments_with_geometry_mesh": sum(
            bool(segment["capabilities"]["has_geometry_mesh"])
            for segment in segments
        ),
        "segments_with_tifxyz": sum(
            bool(segment["capabilities"]["has_tifxyz"]) for segment in segments
        ),
        "segments_with_complete_tifxyz": sum(
            bool(segment["capabilities"]["has_complete_tifxyz"])
            for segment in segments
        ),
        "tifxyz_variants": len(tifxyz_assets),
        "complete_tifxyz_variants": sum(
            bool(asset["complete"]) for asset in tifxyz_assets
        ),
        "geometry_variants": len(geometry_assets),
        "surface_volume_assets": len(surface_assets),
        "surface_tiff_stacks": sum(
            asset["format"] == "tiff_stack" for asset in surface_assets
        ),
        "surface_zarrs": sum(asset["format"] == "zarr" for asset in surface_assets),
        "segments_direct_surface_screening_ready": sum(
            bool(segment["capabilities"]["direct_surface_screening_ready"])
            for segment in segments
        ),
        "segments_requiring_surface_extraction": sum(
            bool(segment["capabilities"]["surface_extraction_required"])
            for segment in segments
        ),
        "unscreened_tifxyz_only_meshes": len(candidates),
        "raw_tifxyz_variants": len(raw_tifxyz),
        "complete_raw_tifxyz_variants": sum(
            bool(asset["complete"]) for asset in raw_tifxyz
        ),
        "raw_tifxyz_with_published_segment": sum(
            bool(asset["published_segment_matches"]) for asset in raw_tifxyz
        ),
        "unseen_raw_tifxyz_candidates": len(unseen_raw_tifxyz),
        "nested_unmatched_raw_tifxyz_variants": len(
            nested_unmatched_raw_tifxyz
        ),
        "other_objects_retained": other_objects_total,
        "asset_prefixes_total": len(all_assets),
        "prefix_only_assets": sum(
            asset["object_inventory"]["mode"] == "prefix_only"
            for asset in all_assets
        ),
        "recursively_enumerated_assets": sum(
            asset["object_inventory"]["mode"] == "recursive_objects"
            for asset in all_assets
        ),
    }
    asset_identity = [
        {
            "prefix": asset["prefix"],
            "format": asset["format"],
            "inventory_mode": asset["object_inventory"]["mode"],
            "objects_sha256": asset["objects_sha256"],
        }
        for asset in sorted(
            all_assets,
            key=lambda item: str(item["prefix"]),
        )
    ]
    inventory_sha256 = hashlib.sha256(
        json.dumps(
            asset_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    default_pages = {
        prefix: by_prefix[prefix].pages for prefix in sorted(expected_prefixes)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scroll": scroll,
        "source": {
            "mode": source_mode,
            "endpoint": endpoint if source_mode == "public-s3" else None,
            "api": "S3 ListObjectsV2",
            "authentication": "none",
            "scope_prefixes": sorted(expected_prefixes),
            "pages_by_prefix": default_pages,
            "listing_audit": listing_audit,
            "scalable_discovery": bool(discovered_assets),
            "large_container_descendants_enumerated": not bool(prefix_only),
        },
        "snapshot": {
            "inventory_sha256": inventory_sha256,
            "objects_sha256": _object_digest(all_records),
            "object_count": len(all_records),
            "total_bytes": sum(record.size for record in all_records),
            "latest_last_modified": summary["latest_last_modified"],
        },
        "candidate_definitions": {
            "unscreened_tifxyz_only_meshes": (
                "Complete x/y/z TIFF coordinate-map variants whose segment has no "
                "published surface-volume asset. These require extraction from a raw "
                "volume before the direct surface-stack screening workflow can run."
            ),
            "unseen_raw_tifxyz_candidates": (
                "Complete top-level TIFF-XYZ maps under segments/raw with no "
                "matching published segment. Nested auxiliary maps are reported "
                "separately and are not counted as primary candidates."
            ),
        },
        "summary": summary,
        "raw_volumes": raw_volumes,
        "other_volume_objects": _records_dict(other_volume_objects),
        "predictions": predictions,
        "other_prediction_objects": _records_dict(other_prediction_objects),
        "segments": segments,
        "raw_tifxyz_variants": raw_tifxyz,
        "other_segment_objects": _records_dict(other_segment_objects),
        "unscreened_tifxyz_only_meshes": candidates,
        "unseen_raw_tifxyz_candidates": unseen_raw_tifxyz,
        "nested_unmatched_raw_tifxyz_variants": nested_unmatched_raw_tifxyz,
    }


def _inventory_scroll_recursive(
    scroll: str,
    source: object,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    workers: int = 3,
) -> dict[str, object]:
    prefixes = (
        f"{scroll}/volumes/",
        f"{scroll}/representations/predictions/",
        f"{scroll}/segments/",
    )
    if workers < 1:
        raise InventoryError("workers must be at least 1")
    # Continuation pages within one scope are necessarily sequential.  The
    # three disjoint scopes are independent and safe to enumerate in parallel.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, len(prefixes))
    ) as pool:
        listings = list(pool.map(lambda prefix: list_all_objects(source, prefix), prefixes))
    return build_inventory(
        scroll,
        listings,
        source_mode=str(source.mode),  # type: ignore[attr-defined]
        endpoint=endpoint,
    )


class ListingCollector:
    """Thread-safe collector for complete, auditable prefix listings."""

    def __init__(self, source: object) -> None:
        self.source = source
        self._lock = threading.Lock()
        self._listings: list[Listing] = []

    def list(self, prefix: str, delimiter: str | None = None) -> Listing:
        listing = list_all_objects(self.source, prefix, delimiter)
        with self._lock:
            self._listings.append(listing)
        return listing

    def audit(self) -> dict[str, object]:
        with self._lock:
            listings = tuple(self._listings)
        requests = [
            {
                "prefix": listing.prefix,
                "delimiter": listing.delimiter,
                "pages": listing.pages,
                "direct_objects": len(listing.objects),
                "common_prefixes": len(listing.common_prefixes),
            }
            for listing in sorted(
                listings,
                key=lambda item: (item.prefix, item.delimiter or ""),
            )
        ]
        return {
            "strategy": "paginated_delimiter_walk",
            "request_count": len(requests),
            "page_count": sum(int(item["pages"]) for item in requests),
            "direct_objects_seen": sum(
                int(item["direct_objects"]) for item in requests
            ),
            "common_prefixes_seen": sum(
                int(item["common_prefixes"]) for item in requests
            ),
            "all_responses_reached_IsTruncated_false": True,
            "requests": requests,
        }

    def pages_under(self, prefix: str) -> int:
        with self._lock:
            return sum(
                listing.pages
                for listing in self._listings
                if listing.prefix.startswith(prefix)
            )


def _prefix_has_suffix(prefix: str, suffixes: Sequence[str]) -> bool:
    name = prefix.rstrip("/").rsplit("/", 1)[-1].lower()
    return any(name.endswith(suffix.lower()) for suffix in suffixes)


def _discover_container_tree(
    collector: ListingCollector,
    root_prefix: str,
    suffixes: Sequence[str],
    *,
    max_depth: int = 8,
    max_prefixes: int = 10_000,
) -> tuple[list[str], list[ObjectRecord]]:
    """Walk hierarchy prefixes and stop before entering recognized containers."""

    queue: list[tuple[str, int]] = [(root_prefix, 0)]
    visited: set[str] = set()
    assets: set[str] = set()
    direct_objects: dict[str, ObjectRecord] = {}
    while queue:
        prefix, depth = queue.pop(0)
        if prefix in visited:
            raise InventoryError(f"prefix discovery cycle at {prefix}")
        visited.add(prefix)
        if len(visited) > max_prefixes:
            raise InventoryError(
                f"prefix discovery exceeded {max_prefixes} nodes below {root_prefix}; "
                "refusing to risk walking a chunk hierarchy"
            )
        listing = collector.list(prefix, "/")
        for record in listing.objects:
            previous = direct_objects.get(record.key)
            if previous is not None and previous != record:
                raise InventoryError(f"conflicting metadata for {record.key}")
            direct_objects[record.key] = record
        for child_prefix in listing.common_prefixes:
            if _prefix_has_suffix(child_prefix, suffixes):
                assets.add(child_prefix)
                continue
            if depth >= max_depth:
                raise InventoryError(
                    f"asset discovery exceeded depth {max_depth} at {child_prefix}; "
                    "refusing to descend into a possible chunk hierarchy"
                )
            queue.append((child_prefix, depth + 1))
    return sorted(assets), sorted(direct_objects.values())


def _merge_records(groups: Iterable[Iterable[ObjectRecord]]) -> tuple[ObjectRecord, ...]:
    records: dict[str, ObjectRecord] = {}
    for group in groups:
        for record in group:
            previous = records.get(record.key)
            if previous is not None:
                detail = "conflicting metadata" if previous != record else "duplicate key"
                raise InventoryError(f"{detail} while merging {record.key}")
            records[record.key] = record
    return tuple(sorted(records.values()))


def _surface_assets_for_segment(
    collector: ListingCollector,
    segment_prefix: str,
) -> tuple[list[str], tuple[ObjectRecord, ...], set[str]]:
    surface_prefix = segment_prefix + "surface-volumes/"
    assets, direct_objects = _discover_container_tree(
        collector,
        surface_prefix,
        (".zarr", ".tifs", ".tif", ".tiff"),
    )
    object_groups: list[Iterable[ObjectRecord]] = [direct_objects]
    prefix_only: set[str] = set()
    for asset_prefix in assets:
        asset_format = _asset_format(asset_prefix)
        if asset_format == "tiff_stack":
            # Public flattened stacks contain tens of layers, not millions of
            # chunks, so retaining every layer's immutable metadata is safe.
            object_groups.append(collector.list(asset_prefix).objects)
        else:
            prefix_only.add(asset_prefix)
    return assets, _merge_records(object_groups), prefix_only


def _discover_segment_prefixes(
    collector: ListingCollector,
    segments_prefix: str,
    *,
    max_depth: int = 4,
    max_prefixes: int = 2_000,
) -> tuple[list[str], tuple[ObjectRecord, ...]]:
    """Find segment roots, including nested collections such as ``raw/``."""

    root = collector.list(segments_prefix, "/")
    queue: list[tuple[str, int]] = [
        (prefix, 0) for prefix in root.common_prefixes
    ]
    visited: set[str] = set()
    segment_prefixes: set[str] = set()
    direct_object_groups: list[Iterable[ObjectRecord]] = [root.objects]
    while queue:
        prefix, depth = queue.pop(0)
        if prefix in visited:
            raise InventoryError(f"segment-prefix discovery cycle at {prefix}")
        visited.add(prefix)
        if len(visited) > max_prefixes:
            raise InventoryError(
                f"segment discovery exceeded {max_prefixes} hierarchy nodes"
            )
        listing = collector.list(prefix, "/")
        direct_object_groups.append(listing.objects)
        child_names = {
            child.removeprefix(prefix).rstrip("/").split("/", 1)[0]
            for child in listing.common_prefixes
        }
        if child_names.intersection({"mesh", "surface-volumes"}):
            segment_prefixes.add(prefix)
            continue
        if depth >= max_depth and listing.common_prefixes:
            raise InventoryError(
                f"segment discovery exceeded depth {max_depth} at {prefix}"
            )
        for child in listing.common_prefixes:
            if _prefix_has_suffix(
                child, (".zarr", ".normal-grids", "normal-grids")
            ):
                raise InventoryError(
                    f"refusing to descend into container {child} during segment discovery"
                )
            queue.append((child, depth + 1))
    return sorted(segment_prefixes), _merge_records(direct_object_groups)


def inventory_scroll(
    scroll: str,
    source: object,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    workers: int = 8,
    scalable: bool = True,
) -> dict[str, object]:
    """Inventory one scroll without descending into large chunk containers."""

    if not scalable:
        return _inventory_scroll_recursive(
            scroll, source, endpoint=endpoint, workers=workers
        )
    if workers < 1:
        raise InventoryError("workers must be at least 1")
    if not scroll or "/" in scroll:
        raise InventoryError("--scroll must be one S3 path component")

    volume_prefix = f"{scroll}/volumes/"
    prediction_prefix = f"{scroll}/representations/predictions/"
    segments_prefix = f"{scroll}/segments/"
    collector = ListingCollector(source)

    raw_assets, volume_objects = _discover_container_tree(
        collector, volume_prefix, (".zarr",)
    )
    prediction_assets, prediction_objects = _discover_container_tree(
        collector,
        prediction_prefix,
        (
            ".normal-grids",
            "normal-grids",
            "normal_grids",
            ".zarr",
            ".tifs",
            ".tif",
            ".tiff",
        ),
    )
    prediction_object_groups: list[Iterable[ObjectRecord]] = [prediction_objects]
    for asset_prefix in prediction_assets:
        if _asset_format(asset_prefix) == "tiff_stack":
            prediction_object_groups.append(collector.list(asset_prefix).objects)
    prediction_objects = list(_merge_records(prediction_object_groups))
    segment_prefixes, segment_hierarchy_objects = _discover_segment_prefixes(
        collector, segments_prefix
    )
    segment_ids = [
        prefix.removeprefix(segments_prefix).rstrip("/")
        for prefix in segment_prefixes
    ]

    def crawl_segment(
        segment_prefix: str,
    ) -> tuple[str, tuple[ObjectRecord, ...], list[str], set[str]]:
        mesh_objects = collector.list(segment_prefix + "mesh/").objects
        surfaces, surface_objects, surface_prefix_only = _surface_assets_for_segment(
            collector, segment_prefix
        )
        return (
            segment_prefix,
            _merge_records((mesh_objects, surface_objects)),
            surfaces,
            surface_prefix_only,
        )

    segment_results: list[
        tuple[str, tuple[ObjectRecord, ...], list[str], set[str]]
    ] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, max(1, len(segment_prefixes)))
    ) as pool:
        segment_results = list(pool.map(crawl_segment, segment_prefixes))

    surface_assets: list[str] = []
    prefix_only: set[str] = set(raw_assets)
    prefix_only.update(
        prefix
        for prefix in prediction_assets
        if _asset_format(prefix) in {"zarr", "normal_grids"}
    )
    segment_object_groups: list[Iterable[ObjectRecord]] = [
        segment_hierarchy_objects
    ]
    for _segment_prefix, objects, surfaces, surface_prefix_only in segment_results:
        segment_object_groups.append(objects)
        surface_assets.extend(surfaces)
        prefix_only.update(surface_prefix_only)

    volume_listing = Listing(
        prefix=volume_prefix,
        pages=collector.pages_under(volume_prefix),
        objects=tuple(volume_objects),
    )
    prediction_listing = Listing(
        prefix=prediction_prefix,
        pages=collector.pages_under(prediction_prefix),
        objects=tuple(prediction_objects),
    )
    segments_listing = Listing(
        prefix=segments_prefix,
        pages=collector.pages_under(segments_prefix),
        objects=_merge_records(segment_object_groups),
    )
    discovered_assets = {
        "raw_volumes": raw_assets,
        "predictions": prediction_assets,
        "surface_volumes": sorted(surface_assets),
        "prefix_only": sorted(prefix_only),
        "segment_ids": sorted(segment_ids),
    }
    report = build_inventory(
        scroll,
        [volume_listing, prediction_listing, segments_listing],
        source_mode=str(source.mode),  # type: ignore[attr-defined]
        endpoint=endpoint,
        discovered_assets=discovered_assets,
        listing_audit=collector.audit(),
    )
    report["source"]["coverage_policy"] = {
        "asset_prefixes": "complete paginated delimiter discovery",
        "mesh_and_tifxyz_objects": "complete recursive object metadata",
        "tiff_stack_objects": "complete recursive object metadata",
        "zarr_and_normal_grid_descendants": (
            "prefix identity only; chunk objects intentionally not enumerated"
        ),
    }
    return report


def write_report(report: dict[str, object], output: str, *, force: bool = False) -> None:
    payload = json.dumps(
        report, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    if output == "-":
        sys.stdout.write(payload)
        return
    target = Path(output)
    if target.exists() and not force:
        raise InventoryError(
            f"refusing to overwrite immutable snapshot {target}; pass --force to replace it"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory immutable public PHerc S3 assets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scroll", default="PHerc1447", help="S3 scroll prefix")
    parser.add_argument(
        "--output",
        default="-",
        metavar="JSON",
        help="output JSON path, or - for stdout",
    )
    parser.add_argument(
        "--fixtures",
        action="append",
        default=[],
        metavar="PATH",
        help="offline ListObjectsV2 XML file/directory; repeatable",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="parallel compact per-segment listing operations",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output snapshot",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.fixtures:
            source: object = FixtureListObjectsV2Source(args.fixtures)
        else:
            source = HttpListObjectsV2Source(
                args.endpoint, timeout=args.timeout, retries=args.retries
            )
        report = inventory_scroll(
            args.scroll, source, endpoint=args.endpoint, workers=args.workers
        )
        write_report(report, args.output, force=args.force)
    except (InventoryError, OSError) as exc:
        print(f"inventory error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
