#!/usr/bin/env python3
"""Freeze the result-blind private-Kaggle CPU retry after tooling publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    content = dict(payload)
    content.pop("payload_sha256", None)
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("payload_sha256") != canonical_sha256(payload):
        raise RuntimeError(f"invalid hashed JSON: {path}")
    return payload


def identity(path: Path) -> dict:
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def tree_identity(root: Path) -> dict:
    records = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"frozen public tree contains symlink: {path}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        records.append(f"{sha256_file(path)}  {size}  {path.relative_to(root).as_posix()}\n")
        total += size
    return {
        "files": len(records),
        "bytes": total,
        "ledger_sha256": hashlib.sha256("".join(records).encode()).hexdigest(),
    }


def write_hashed(path: Path, payload: dict) -> dict:
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return content


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--public-commit", required=True)
    parser.add_argument("--transport-receipt", type=Path, required=True)
    parser.add_argument("--prior-receipt", type=Path, required=True)
    parser.add_argument("--puller", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--deployer", type=Path, required=True)
    parser.add_argument("--runtime-preparer", type=Path, required=True)
    parser.add_argument("--runtime-bootstrap-plan", type=Path, required=True)
    parser.add_argument("--runtime-predecessor-plan", type=Path, required=True)
    parser.add_argument("--metric-archive", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--scoring-assets", type=Path, required=True)
    parser.add_argument("--metric-runtime", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("private-Kaggle retry plan target must start absent")
    predecessor = load_hashed(args.predecessor)
    if predecessor.get("payload_sha256") != "86d8c883008f8aee61cf4e967666505938b5b26c6c6563ffe674d2d41d70f918":
        raise RuntimeError("wrong panel-provenance predecessor plan")
    transport_receipt = load_hashed(args.transport_receipt)
    prior_receipt = load_hashed(args.prior_receipt)
    if transport_receipt.get("status") != "private real-transport dataset version 1 ready":
        raise RuntimeError("private Kaggle transport dataset is not READY")
    remaining = 1.8450102378888895
    prior_guarded = 13.5 - remaining
    plan = deepcopy(predecessor)
    plan["public_runpod_commit"] = args.public_commit
    plan["budget"] = {
        "absolute_cap_usd": 13.5,
        "prior_guarded_spend_upper_bound_usd": prior_guarded,
        "compute_cutoff_usd": remaining,
        "guard_seconds": 300,
        "maximum_provider_wall_seconds_at_price_ceiling": int(remaining / 1.28 * 3600),
        "maximum_runtime_seconds_before_guard_at_price_ceiling": int(remaining / 1.28 * 3600) - 300,
        "reserve_usd": 0.0,
        "on_cutoff": "stop the CPU Pod, preserve sealed volume artifacts, and never score a partial result",
    }
    plan["private_transport_puller"] = identity(args.puller)
    plan["private_transport_wrapper"] = identity(args.wrapper)
    plan["private_transport_deployer"] = identity(args.deployer)
    plan["runtime_preparer"] = identity(args.runtime_preparer)
    plan["runtime_bootstrap_plan"] = identity(args.runtime_bootstrap_plan)
    plan["runtime_predecessor_plan"] = identity(args.runtime_predecessor_plan)
    plan["public_metric_archive"] = identity(args.metric_archive)
    plan["scoring_assets_tree"] = tree_identity(args.scoring_assets)
    plan["metric_runtime_tree"] = tree_identity(args.metric_runtime)
    plan["private_kaggle_transport"] = {
        "dataset_id": "aviadcohen1/vesuvius-fusion-real-heldout-transport-v1",
        "dataset_version": 1,
        "dataset_handle": "aviadcohen1/vesuvius-fusion-real-heldout-transport-v1/versions/1",
        "dataset_private": True,
        "expected_total_files": 284,
        "expected_total_bytes": 5_791_288_517,
        "staging_manifest": transport_receipt["staging_manifest"],
        "upload_receipt_payload_sha256": transport_receipt["payload_sha256"],
        "remote_inventory_exact_path_and_size_match": True,
        "kagglehub_wheelhouse_tree": tree_identity(args.wheelhouse),
        "credential_lifecycle": "upload once over the accepted SSH endpoint with mode 0600; delete immediately after private version-1 download; never log or publish credential content",
        "npz_payloads_opened_or_parsed": False,
        "transport_only": True,
    }
    plan["private_kaggle_transport_retry"] = {
        "schema_version": "1.0",
        "status": "result-blind exact private-Kaggle transport retry frozen before CPU resume",
        "prior_receipt": {
            "file": args.prior_receipt.name,
            "payload_sha256": prior_receipt["payload_sha256"],
            "provider_state": "EXITED",
            "scientific_outputs_inspected": False,
        },
        "correction": {
            "same_private_sealed_input_identities": True,
            "private_dataset_version_exactly_one": True,
            "downloaded_files_rehashed_before_executor": True,
            "fresh_absent_transport_input_status_and_work_roots": True,
            "empty_or_partial_predecessor_workspace_excluded": True,
            "ephemeral_provider_credential_removed_after_download": True,
        },
        "scientific_gate": {
            "cache_model_metric_threshold_seed_panel_endpoint_gate_aggregation_order_or_claim_changed": False,
            "result_probability_endpoint_panel_or_npz_opened": False,
            "test_time_tuning_permitted": False,
        },
    }
    unchanged = (
        "sealed_real_inputs", "embedded_real_launcher", "remote_executor",
        "public_metric_verifier", "frozen_thresholds", "scoring_assets", "runtime",
        "scientific_gate", "public_execution_plan", "public_cache_delivery",
    )
    if any(plan[key] != predecessor[key] for key in unchanged):
        raise RuntimeError("scientific predecessor identity changed during transport freeze")
    frozen = write_hashed(args.out, plan)
    print(frozen["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
