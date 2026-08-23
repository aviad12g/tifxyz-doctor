#!/usr/bin/env python3
"""Metadata-only, fail-closed audit for a prospective control supplement.

The audit reads only a JSON requirements record and a JSONL manifest.  It
never opens an NPZ, CT volume, label, prediction, probability, panel, or
scientific endpoint.  Passing this audit is necessary but not sufficient to
supersede the active PHerc1218 quarantine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class AuditError(RuntimeError):
    """The proposed control supplement is not safe to accept."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_requirements(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    observed = body.pop("payload_sha256", None)
    require(observed == canonical_sha256(body), "requirements payload SHA-256 mismatch")
    require(payload.get("schema_version") == "1.0", "unexpected requirements schema")
    require(
        payload.get("status")
        == "PROSPECTIVE_CONTROL_REQUIREMENTS_DO_NOT_SUPERSEDE_QUARANTINE",
        "requirements do not preserve the quarantine",
    )
    require(payload.get("quarantine_superseded") is False, "requirements pre-authorize reopening")
    require(payload.get("scientific_endpoints_scored") is False, "requirements record scored endpoints")
    require(payload.get("confirmation_outputs_inspected") is False, "requirements record opened confirmation")
    return payload


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AuditError(f"manifest line {line_number} is invalid JSON") from error
        require(isinstance(row, dict), f"manifest line {line_number} is not an object")
        row["_line_number"] = line_number
        rows.append(row)
    require(rows, "control manifest is empty")
    return rows


def safe_npz_path(value: Any, line_number: int) -> str:
    require(isinstance(value, str) and value.endswith(".npz"), f"line {line_number}: invalid file")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts, f"line {line_number}: unsafe file path")
    return value


def audit_controls(
    requirements: dict[str, Any],
    manifest_path: Path,
    *,
    dataset_ref: str,
    dataset_version: int,
    dataset_id: int | None,
    expected_manifest_sha256: str,
    expected_manifest_bytes: int,
) -> dict[str, Any]:
    require(SHA256_RE.fullmatch(expected_manifest_sha256) is not None, "invalid expected manifest SHA-256")
    require(dataset_version > 0, "dataset version must be positive")
    require(dataset_ref.count("/") == 1, "dataset ref must be owner/slug")
    actual_sha256 = sha256_file(manifest_path)
    require(actual_sha256 == expected_manifest_sha256, "manifest SHA-256 mismatch")
    require(manifest_path.stat().st_size == expected_manifest_bytes, "manifest byte count mismatch")

    rules = requirements["control_rules"]
    max_empty = float(rules["maximum_ct_empty_frac"])
    minimum_eligible = int(rules["minimum_eligible_controls"])
    minimum_slabs = int(rules["minimum_distinct_source_slabs"])
    maximum_slab_share = float(rules["maximum_single_slab_share"])
    minimum_z_span_fraction = float(rules["minimum_z_span_fraction_of_contact_range"])
    reference_min = float(rules["contact_reference_z_min_level1"])
    reference_max = float(rules["contact_reference_z_max_level1"])
    require(0 <= max_empty <= 1, "invalid emptiness threshold")
    require(minimum_eligible >= 48, "minimum eligible controls silently relaxed")
    require(minimum_slabs >= 2, "spatial slab requirement silently relaxed")
    require(0 < maximum_slab_share < 1, "invalid single-slab share ceiling")
    require(0 < minimum_z_span_fraction <= 1, "invalid z-span requirement")
    require(reference_max > reference_min, "invalid contact reference z range")

    rows = load_manifest(manifest_path)
    files: set[str] = set()
    hashes: set[str] = set()
    eligible: list[dict[str, Any]] = []
    for row in rows:
        line_number = int(row.pop("_line_number"))
        require(row.get("arm") == "control", f"line {line_number}: non-control row in supplement")
        filename = safe_npz_path(row.get("file"), line_number)
        digest = row.get("sha256")
        require(isinstance(digest, str) and SHA256_RE.fullmatch(digest), f"line {line_number}: invalid SHA-256")
        require(filename not in files, f"line {line_number}: duplicate file path")
        require(digest not in hashes, f"line {line_number}: duplicate file SHA-256")
        files.add(filename)
        hashes.add(digest)
        try:
            empty_fraction = float(row["ct_empty_frac"])
            center_z = float(row["center_z_level1"])
        except (KeyError, TypeError, ValueError) as error:
            raise AuditError(f"line {line_number}: invalid control metadata") from error
        require(0 <= empty_fraction <= 1, f"line {line_number}: ct_empty_frac out of range")
        slab = row.get("source_slab")
        require(isinstance(slab, (str, int)) and str(slab), f"line {line_number}: missing source_slab")
        if empty_fraction <= max_empty:
            eligible.append(
                {
                    "center_z_level1": center_z,
                    "ct_empty_frac": empty_fraction,
                    "file": filename,
                    "sha256": digest,
                    "source_slab": str(slab),
                }
            )

    require(len(eligible) >= minimum_eligible, "too few eligible controls")
    slab_counts = Counter(row["source_slab"] for row in eligible)
    require(len(slab_counts) >= minimum_slabs, "too few distinct eligible control slabs")
    largest_slab_share = max(slab_counts.values()) / len(eligible)
    require(largest_slab_share <= maximum_slab_share, "eligible controls are concentrated in one slab")
    observed_z_span = max(row["center_z_level1"] for row in eligible) - min(
        row["center_z_level1"] for row in eligible
    )
    reference_z_span = reference_max - reference_min
    observed_z_span_fraction = observed_z_span / reference_z_span
    require(
        observed_z_span_fraction >= minimum_z_span_fraction,
        "eligible control z span is too narrow relative to contacts",
    )

    eligible = sorted(eligible, key=lambda row: row["file"])
    eligible_payload_sha256 = canonical_sha256(eligible)
    result = {
        "schema_version": "1.0",
        "status": "SUPPLEMENTAL_CONTROLS_METADATA_AUDIT_PASSED_QUARANTINE_STILL_ACTIVE",
        "dataset": {
            "provider": "kaggle",
            "dataset_id": dataset_id,
            "ref": dataset_ref,
            "version": dataset_version,
            "manifest_bytes": expected_manifest_bytes,
            "manifest_sha256": actual_sha256,
        },
        "inventory": {
            "manifest_controls": len(rows),
            "eligible_controls": len(eligible),
            "distinct_eligible_source_slabs": len(slab_counts),
            "largest_eligible_slab_share": largest_slab_share,
            "eligible_z_span_fraction_of_contact_range": observed_z_span_fraction,
        },
        "eligible_control_identities": eligible,
        "eligible_control_identities_payload_sha256": eligible_payload_sha256,
        "requirements_payload_sha256": requirements["payload_sha256"],
        "quarantine_superseded": False,
        "npz_ct_labels_predictions_probabilities_panels_or_endpoints_opened": False,
    }
    result["payload_sha256"] = canonical_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-ref", required=True)
    parser.add_argument("--dataset-version", type=int, required=True)
    parser.add_argument("--dataset-id", type=int)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--manifest-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        requirements = load_requirements(args.requirements)
        result = audit_controls(
            requirements,
            args.manifest,
            dataset_ref=args.dataset_ref,
            dataset_version=args.dataset_version,
            dataset_id=args.dataset_id,
            expected_manifest_sha256=args.manifest_sha256,
            expected_manifest_bytes=args.manifest_bytes,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            require(not args.output.exists(), "output path must start absent")
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except (AuditError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
