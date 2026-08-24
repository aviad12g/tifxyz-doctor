#!/usr/bin/env python3
"""Fail-closed metadata-only validator for the GapBalance Stage-0 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


BANDS = ("0-2", "2-4", "4-6", "6-10", "10+")
PANEL_BANDS = BANDS[:4]
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_SOURCE_CHECKPOINT = (
    "f1990a02ac91889c1f989522ae0e45421a91cb666320448aaf579d42b081636f"
)
EXPECTED_CONTROL_HASHES = {
    "11": "7a2e6168f32b3a3389bdc2b43b39a6654568a6da467e71948f523e7cbef6248f",
    "23": "d40c4b856c9a65cc2de127e98d37d08633e03e6b7b5e375bec8991b1bf9487b3",
    "47": "fd86b40b3b25f45f86f2ba43668997d4351fec7d3ebd35fa41856e4c0d505e44",
}


class ContractError(RuntimeError):
    """The frozen Stage-0 contract or manifest is not safe to execute."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def require_no_active_quarantine(contract: dict[str, Any], path: Path) -> None:
    """Reject a withdrawn holdout before any manifest or payload access."""
    if not path.exists():
        return
    quarantine = json.loads(path.read_text(encoding="utf-8"))
    body = dict(quarantine)
    observed = body.pop("payload_sha256", None)
    require(observed == canonical_sha256(body), "holdout quarantine payload mismatch")
    require(
        quarantine.get("status")
        == "BLOCKED_REAL_HOLDOUT_NOT_CERTIFIED",
        "unknown holdout quarantine status",
    )
    require(
        quarantine.get("quarantine_status")
        == "ACTIVE_AUTHOR_WITHDRAWAL_ALL_PHERC1218_VERSIONS_QUARANTINED",
        "holdout quarantine is not active",
    )
    real = contract.get("confirmation", {}).get("real", {})
    affected = quarantine.get("affected_identity", {})
    require(
        real.get("ref") == affected.get("ref"),
        "holdout quarantine does not match the contract dataset",
    )
    require(
        quarantine.get("confirmation_gate", {}).get("real_confirmation_may_open")
        is False,
        "active holdout quarantine does not fail closed",
    )
    require(
        quarantine.get("compute_gate", {}).get("runpod_restart_permitted") is False,
        "blocked holdout does not stop paid development compute",
    )
    raise ContractError("PHerc1218 holdout is quarantined by the dataset author")


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    require(contract.get("schema_version") == "1.0", "unexpected schema_version")
    encoded = json.dumps(contract, sort_keys=True)
    require("PENDING_" not in encoded, "contract still contains PENDING placeholders")
    require(
        contract.get("status") == "PUBLIC_FROZEN_READY_FOR_EXPLICIT_TRAINING_APPROVAL",
        "contract status is not public-frozen",
    )

    experiment = contract.get("experiment", {})
    require(
        experiment.get("source_checkpoint_sha256") == EXPECTED_SOURCE_CHECKPOINT,
        "source checkpoint changed",
    )
    require(
        experiment.get("arms") == {"control": 1.0, "gap2": 2.0, "gap4": 4.0},
        "arm weights changed",
    )
    require(experiment.get("matched_seeds") == [11, 23, 47], "matched seeds changed")
    require(experiment.get("new_training_jobs") == 6, "job count is not exactly six")
    require(
        experiment.get("reused_control_checkpoint_sha256") == EXPECTED_CONTROL_HASHES,
        "reused control checkpoint identity changed",
    )

    development = contract.get("development", {})
    require(
        development.get("synthetic_seeds") == [400, 401, 402, 403],
        "development synthetic seeds changed",
    )
    confirmation = contract.get("confirmation", {})
    require(
        confirmation.get("synthetic_seeds") == [500, 501, 502, 503, 504],
        "confirmation synthetic seeds changed",
    )

    execution = contract.get("execution", {})
    require(execution.get("training_authorized") is False, "protocol cannot authorize training")
    require(execution.get("paid_compute_authorized") is False, "protocol cannot authorize spend")

    real = confirmation.get("real", {})
    require(real.get("provider") == "kaggle", "unexpected holdout provider")
    require(
        real.get("ref") == "jhjeong0815/pherc1218-tight-contact-val",
        "unexpected holdout dataset",
    )
    require(real.get("dataset_id") == 11704096, "unexpected holdout dataset id")
    require(isinstance(real.get("version"), int) and real["version"] >= 2, "corrected version required")
    require(SHA256_RE.fullmatch(str(real.get("manifest_sha256", ""))) is not None, "invalid manifest hash")
    inventory = real.get("expected_inventory")
    require(isinstance(inventory, dict), "expected inventory is not frozen")
    require(
        inventory.get("contact_bands")
        and set(inventory["contact_bands"]) == set(BANDS),
        "expected contact-band inventory is incomplete",
    )
    require(
        all(isinstance(inventory["contact_bands"][band], int) for band in BANDS),
        "expected contact-band counts must be integers",
    )
    require(isinstance(inventory.get("control"), int), "expected control count is missing")
    for field in (
        "both_instances_present",
        "eligible_contact",
        "eligible_contact_under_4_voxels",
        "eligible_control",
    ):
        require(isinstance(inventory.get(field), int), f"expected {field} count is missing")

    provider_inventory = real.get("provider_inventory")
    require(isinstance(provider_inventory, dict), "provider inventory is not frozen")
    expected_npz = sum(inventory["contact_bands"].values()) + inventory["control"]
    require(provider_inventory.get("npz_files") == expected_npz, "provider NPZ count changed")
    require(provider_inventory.get("all_files") == expected_npz + 10, "provider file count changed")
    require(
        isinstance(provider_inventory.get("total_bytes"), int)
        and provider_inventory["total_bytes"] > 0,
        "provider byte count is not frozen",
    )
    require(
        isinstance(real.get("manifest_bytes"), int) and real["manifest_bytes"] > 0,
        "manifest byte count is not frozen",
    )

    gates = contract.get("gates", {})
    require(gates.get("fusion_delta_pp_max_pooled") == -5.0, "fusion gate changed")
    require(gates.get("detection_delta_pp_min_pooled") == -2.0, "detection gate changed")
    require(gates.get("false_split_delta_pp_max_pooled") == 2.0, "split gate changed")
    require(gates.get("real_blend_noninferiority_margin") == -0.005, "blend gate changed")
    require(gates.get("real_toposcore_noninferiority_margin") == -0.005, "TopoScore gate changed")
    return real


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(f"manifest line {line_number} is invalid JSON") from error
        require(isinstance(row, dict), f"manifest line {line_number} is not an object")
        rows.append(row)
    require(rows, "manifest is empty")
    return rows


def validate_path(value: Any, line_number: int) -> str:
    require(isinstance(value, str) and value.endswith(".npz"), f"line {line_number}: invalid file")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts, f"line {line_number}: unsafe file path")
    return value


def validate_manifest(
    contract: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    real = validate_contract(contract)
    expected_sha = real["manifest_sha256"]
    actual_sha = sha256_file(manifest_path)
    require(actual_sha == expected_sha, "manifest SHA-256 mismatch")
    require(manifest_path.stat().st_size == real["manifest_bytes"], "manifest byte count mismatch")

    rows = load_manifest(manifest_path)
    files: set[str] = set()
    hashes: set[str] = set()
    contact_counts: Counter[str] = Counter()
    both_instances_present = 0
    eligible_contacts: list[dict[str, Any]] = []
    eligible_controls: list[dict[str, Any]] = []

    for index, row in enumerate(rows, 1):
        filename = validate_path(row.get("file"), index)
        digest = row.get("sha256")
        require(isinstance(digest, str) and SHA256_RE.fullmatch(digest), f"line {index}: invalid SHA-256")
        require(filename not in files, f"line {index}: duplicate file path")
        require(digest not in hashes, f"line {index}: duplicate file SHA-256")
        files.add(filename)
        hashes.add(digest)
        require(row.get("arm") in {"crops", "control"}, f"line {index}: invalid arm")
        try:
            empty_fraction = float(row["ct_empty_frac"])
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"line {index}: invalid ct_empty_frac") from error
        require(0.0 <= empty_fraction <= 1.0, f"line {index}: ct_empty_frac out of range")

        if row["arm"] == "crops":
            band = row.get("band")
            require(band in BANDS, f"line {index}: invalid contact band")
            require(isinstance(row.get("both_instances_present"), bool), f"line {index}: missing instance flag")
            both_instances_present += int(row["both_instances_present"])
            contact_counts[band] += 1
            if row["both_instances_present"] and empty_fraction <= 0.10:
                eligible_contacts.append(row)
        elif empty_fraction <= 0.10:
            eligible_controls.append(row)

    expected = real["expected_inventory"]
    require(dict(contact_counts) == expected["contact_bands"], "contact-band inventory mismatch")
    control_count = sum(row["arm"] == "control" for row in rows)
    require(control_count == expected["control"], "control inventory mismatch")
    require(
        both_instances_present == expected["both_instances_present"],
        "both-instance inventory mismatch",
    )

    eligibility = real["eligibility"]
    under_four = sum(row["band"] in {"0-2", "2-4"} for row in eligible_contacts)
    require(len(eligible_contacts) >= eligibility["minimum_contact"], "too few eligible contacts")
    require(under_four >= eligibility["minimum_contact_under_4_voxels"], "too few eligible contacts below four voxels")
    require(len(eligible_controls) >= eligibility["minimum_control"], "too few eligible controls")
    require(len(eligible_contacts) == expected["eligible_contact"], "eligible-contact inventory mismatch")
    require(
        under_four == expected["eligible_contact_under_4_voxels"],
        "eligible under-four inventory mismatch",
    )
    require(len(eligible_controls) == expected["eligible_control"], "eligible-control inventory mismatch")

    selected_files = sorted(row["file"] for row in eligible_contacts + eligible_controls)
    selected_sha = hashlib.sha256(("\n".join(selected_files) + "\n").encode("utf-8")).hexdigest()
    eligible_file_identities = {
        "contact": sorted(
            (
                {
                    "band": row["band"],
                    "ct_empty_frac": float(row["ct_empty_frac"]),
                    "file": row["file"],
                    "sha256": row["sha256"],
                }
                for row in eligible_contacts
            ),
            key=lambda row: row["file"],
        ),
        "control": sorted(
            (
                {
                    "ct_empty_frac": float(row["ct_empty_frac"]),
                    "file": row["file"],
                    "sha256": row["sha256"],
                }
                for row in eligible_controls
            ),
            key=lambda row: row["file"],
        ),
    }
    panels = {}
    for band in PANEL_BANDS:
        candidates = [row for row in eligible_contacts if row["band"] == band]
        require(candidates, f"no eligible fixed-panel candidate in band {band}")
        panels[band] = min(
            candidates,
            key=lambda row: (
                hashlib.sha256((actual_sha + "\0" + row["file"]).encode("utf-8")).hexdigest(),
                row["file"],
            ),
        )["file"]

    result = {
        "schema_version": "1.0",
        "status": "STAGE0_MANIFEST_VALIDATED_RESULT_BLIND",
        "dataset": {
            "dataset_id": real["dataset_id"],
            "ref": real["ref"],
            "version": real["version"],
            "manifest_sha256": actual_sha,
        },
        "inventory": {
            "rows": len(rows),
            "contact_bands": {band: contact_counts[band] for band in BANDS},
            "control": control_count,
            "eligible_contact": len(eligible_contacts),
            "eligible_contact_under_4_voxels": under_four,
            "eligible_control": len(eligible_controls),
        },
        "selected_files_sha256": selected_sha,
        "eligible_file_identities": eligible_file_identities,
        "eligible_file_identities_payload_sha256": canonical_sha256(eligible_file_identities),
        "fixed_panels": panels,
        "fixed_panels_payload_sha256": canonical_sha256(panels),
        "npz_opened_or_parsed": False,
    }
    result["payload_sha256"] = canonical_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()

    try:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        require_no_active_quarantine(
            contract, args.contract.with_name("PHERC1218_HOLDOUT_QUARANTINE.json")
        )
        if args.contract_only:
            validate_contract(contract)
            result = {"status": "CONTRACT_VALIDATED", "npz_opened_or_parsed": False}
        else:
            require(args.manifest is not None, "--manifest is required")
            result = validate_manifest(contract, args.manifest)
    except (ContractError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True))
        return 2

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        require(not args.output.exists(), "output path must be absent")
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
