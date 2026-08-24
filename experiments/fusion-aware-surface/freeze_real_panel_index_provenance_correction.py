#!/usr/bin/env python3
"""Freeze the result-blind real-panel score-index provenance correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREDECESSOR_PLAN = {
    "commit": "07a82d5f211609ac96ec266c41a5c5a3c5b1f2bd",
    "file": "heldout_execution_plan.json",
    "bytes": 49885,
    "sha256": "b3fc826aa00d4d1f9a18472fba54a93d3b89d89f5b3e0270df28c110ac746481",
    "payload_sha256": "cbaff81b679f0d90e7d7cffa37ce5acce447b89ae88b322472e508cdd48d6479",
}
PREDECESSOR_DELIVERY = {
    "commit": "5ba626c37b7ca7899f65c43fa24359c9b6851306",
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 21015,
    "sha256": "cb008acbca91405e60df8e1813eb4c11b6d5fce6cfae632ec8cac13b49cc6170",
    "payload_sha256": "ada3295cffd573a867ce2e189988b30d4bf0e80bb9ad5b73ba66bf5e32c23a20",
}
PREDECESSOR_RENDERER = {
    "file": "render_real_panels.py",
    "bytes": 18206,
    "sha256": "e05b746c17f0a2b2b33a60b41b72408c799204bbca533667b8169ca415343d15",
}
PREDECESSOR_STAGER = {
    "file": "stage_heldout_for_scoring.py",
    "bytes": 19922,
    "sha256": "1fa4aec460bdd7425b70c35f5d432fc41ac5b66a36908e1604dc3f080223950d",
}
CACHE_JOB_PLAN = {
    "commit": "46d030245c14ecb2be63796a308581c462fc2e29",
    "file": "heldout_execution_plan.json",
    "bytes": 20443,
    "sha256": "29d08a7430c8870582a96579fcc77737ac699aef64efc159ba94ede87009da5d",
    "payload_sha256": "3826bf641941b5d70fa52f9a23fc1711ae538be8fc974dbba1b5f0db8f18afdd",
}
FAILED_ATTEMPT_RECEIPT = {
    "file": "provider_receipt.json",
    "payload_sha256": "755d5bcfb6306125cca82acc8a06969a14abc4bab6c5c0df97a7c269d271f133",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


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


def write_hashed(path: Path, payload: dict) -> None:
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_hashed(path)


def require_commit(value: str, label: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be 40 lowercase hexadecimal characters")
    return value


def correction_record(public_tooling_commit: str, corrected_renderer: dict) -> dict:
    if set(corrected_renderer) != {"file", "bytes", "sha256"}:
        raise ValueError("corrected renderer identity schema mismatch")
    return {
        "schema_version": "1.0",
        "status": "result-blind real-panel score-index provenance correction",
        "public_tooling_commit": require_commit(public_tooling_commit, "public tooling commit"),
        "predecessor_execution_plan": PREDECESSOR_PLAN,
        "predecessor_cache_delivery": PREDECESSOR_DELIVERY,
        "predecessor_panel_renderer": PREDECESSOR_RENDERER,
        "corrected_panel_renderer": corrected_renderer,
        "failed_attempt": {
            "provider": "RunPod CPU",
            "pod_id": "z0vqb3dt43vkga",
            "receipt": FAILED_ATTEMPT_RECEIPT,
            "stage": "fixed real-panel public-context validation after pinned metric build and before panel or scorer output",
            "failure_message": "real score-input index provenance mismatch",
            "scientific_result_created_or_opened": False,
            "cache_npz_probability_endpoint_or_panel_opened": False,
        },
        "cause": (
            "the stager correctly recorded the hash-bound original cache-job execution-plan "
            "identity, while the fixed panel validator's exact expected schema omitted that "
            "operational provenance field"
        ),
        "correction": {
            "required_score_input_field": "cache_job_execution_plan",
            "expected_identity_derived_from": CACHE_JOB_PLAN,
            "materialized_file_alias": "cache_job_execution_plan.json",
            "field_must_match_exactly": True,
            "unknown_or_tampered_fields_rejected": True,
            "stager_scientific_projection_allowlist_extended_for_this_record": True,
        },
        "scientific_gate": {
            "cache_or_manifest_changed": False,
            "model_metric_threshold_seed_panel_endpoint_gate_aggregation_or_claim_changed": False,
            "test_time_tuning_permitted": False,
            "scientific_outputs_inspected": False,
        },
    }


def freeze_plan(args: argparse.Namespace) -> None:
    if identity(args.plan) != {key: PREDECESSOR_PLAN[key] for key in ("file", "bytes", "sha256")}:
        raise RuntimeError("execution plan is not the exact panel-provenance predecessor")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PREDECESSOR_PLAN["payload_sha256"]:
        raise RuntimeError("predecessor execution-plan payload mismatch")
    if plan.get("real_panel_renderer") != PREDECESSOR_RENDERER:
        raise RuntimeError("predecessor panel-renderer identity mismatch")
    if plan.get("one_shot_scoring_stager") != PREDECESSOR_STAGER:
        raise RuntimeError("predecessor scoring-stager identity mismatch")
    if (
        plan.get("result_blind_scoring_asset_correction", {}).get(
            "predecessor_public_execution_plan"
        )
        != CACHE_JOB_PLAN
    ):
        raise RuntimeError("cache-job execution-plan identity mismatch")
    corrected_renderer = identity(args.renderer)
    plan["one_shot_scoring_stager"] = identity(args.stager)
    plan["real_panel_renderer"] = corrected_renderer
    plan["result_blind_real_panel_index_provenance_correction"] = correction_record(
        args.public_tooling_commit, corrected_renderer
    )
    write_hashed(args.plan, plan)
    print("REAL_PANEL_INDEX_PROVENANCE_PLAN_CORRECTED")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {
        key: PREDECESSOR_DELIVERY[key] for key in ("file", "bytes", "sha256")
    }:
        raise RuntimeError("delivery is not the exact panel-provenance predecessor")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != PREDECESSOR_DELIVERY["payload_sha256"]:
        raise RuntimeError("predecessor cache-delivery payload mismatch")
    plan = load_hashed(args.plan)
    record = plan.get("result_blind_real_panel_index_provenance_correction")
    if record != correction_record(args.public_tooling_commit, plan["real_panel_renderer"]):
        raise RuntimeError("panel-provenance correction is absent or changed")
    public_plan_commit = require_commit(args.public_plan_commit, "public plan commit")
    delivery["public_execution_plan"] = identity(args.plan) | {
        "commit": public_plan_commit,
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_real_panel_index_provenance_correction"] = {
        "public_execution_plan_commit": public_plan_commit,
        "public_tooling_commit": record["public_tooling_commit"],
        "exact_cache_job_provenance_required": True,
        "scientific_contract_changed": False,
        "scientific_result_created_or_opened": False,
        "retry_permitted_after_full_preflight": True,
    }
    write_hashed(args.delivery, delivery)
    args.alias.write_bytes(args.delivery.read_bytes())
    if sha256_file(args.alias) != sha256_file(args.delivery):
        raise RuntimeError("delivery alias differs byte-for-byte")
    print("REAL_PANEL_INDEX_PROVENANCE_DELIVERY_CORRECTED")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--stager", type=Path, required=True)
    plan.add_argument("--renderer", type=Path, required=True)
    plan.add_argument("--public-tooling-commit", required=True)
    plan.set_defaults(function=freeze_plan)
    delivery = subparsers.add_parser("delivery")
    delivery.add_argument("--delivery", type=Path, required=True)
    delivery.add_argument("--alias", type=Path, required=True)
    delivery.add_argument("--plan", type=Path, required=True)
    delivery.add_argument("--public-tooling-commit", required=True)
    delivery.add_argument("--public-plan-commit", required=True)
    delivery.set_defaults(function=freeze_delivery)
    args = parser.parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
