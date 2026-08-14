#!/usr/bin/env python3
"""Freeze the CPU-only retry after the result-blind panel-provenance correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREDECESSOR_RUNPOD_PLAN = {
    "file": "runpod_real_scoring_metric_verifier_plan.json",
    "bytes": 15400,
    "sha256": "f4635f53a343f423e44a9e2ed8716d93a9f8741476e2b0b2442fda5830d4efce",
    "payload_sha256": "8c2322ed872901f08541b5aef11b43881db7af73d5063cd5cbe055fd20408863",
    "public_commit": "c96c4424560bd2cee3e4f3d986a2075a54cf6a2d",
}
PUBLIC_PLAN = {
    "file": "heldout_execution_plan.json",
    "bytes": 52907,
    "sha256": "bed684b2fb1d4d505bffc8bc4350036d7745dd90acb4031f18018f753f4cc560",
    "payload_sha256": "0154df3bcf8568b5c52ad36e9d9eb68594062d57f78c5426bf7c9015129be096",
    "commit": "414aa55ebba2c273e45b685a989663ab387be4cb",
}
PUBLIC_DELIVERY = {
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 21422,
    "sha256": "6033e8a5e4309d1b4b0f8223f63f0c3215e1866269e00517f9bb83ab315717b1",
    "payload_sha256": "f07325081390f8c88cfd9531bbbeb28fc053b6e27b954af21d9fb1d2e3117210",
    "commit": "0712bb152957dc67af52c8704850819dbefdaa59",
}
EMBEDDED_LAUNCHER = {
    "file": "one_shot_scoring_launcher.py",
    "bytes": 31522,
    "sha256": "74ffaa3d1a65ca2e82b560ae62d63c72d039dba00caa9d9d98e95f3645de26ab",
}
FAILED_RECEIPT = {
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
    if path.exists():
        raise RuntimeError("corrected RunPod plan target must start absent")
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_hashed(path)


def correction_record() -> dict:
    return {
        "schema_version": "1.0",
        "status": "result-blind CPU retry after exact real-panel score-index provenance correction",
        "predecessor_runpod_plan": PREDECESSOR_RUNPOD_PLAN,
        "failed_attempt": {
            "provider": "RunPod CPU",
            "pod_id": "z0vqb3dt43vkga",
            "receipt": FAILED_RECEIPT,
            "stage": "fixed real-panel public-context validation after pinned metric build and before panel or scorer output",
            "failure_message": "real score-input index provenance mismatch",
            "scientific_outputs_inspected": False,
        },
        "public_execution_plan": PUBLIC_PLAN,
        "public_cache_delivery": PUBLIC_DELIVERY,
        "embedded_real_launcher": EMBEDDED_LAUNCHER,
        "correction": {
            "score_input_cache_job_execution_plan_required": True,
            "score_input_cache_job_execution_plan_exactly_bound": True,
            "unknown_or_tampered_score_input_fields_rejected": True,
            "new_private_input_upload_required": False,
            "existing_verified_sealed_input_volume_reused": True,
        },
        "scientific_gate": {
            "cache_model_metric_threshold_seed_panel_selection_endpoint_gate_aggregation_or_claim_changed": False,
            "test_time_tuning_permitted": False,
            "result_probability_endpoint_panel_or_npz_opened": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--public-plan", type=Path, required=True)
    parser.add_argument("--public-delivery", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if identity(args.predecessor) != {
        key: PREDECESSOR_RUNPOD_PLAN[key] for key in ("file", "bytes", "sha256")
    }:
        raise RuntimeError("predecessor RunPod plan identity mismatch")
    predecessor = load_hashed(args.predecessor)
    if predecessor["payload_sha256"] != PREDECESSOR_RUNPOD_PLAN["payload_sha256"]:
        raise RuntimeError("predecessor RunPod plan payload mismatch")
    public_plan = load_hashed(args.public_plan)
    public_delivery = load_hashed(args.public_delivery)
    if identity(args.public_plan) != {
        key: PUBLIC_PLAN[key] for key in ("file", "bytes", "sha256")
    } or public_plan["payload_sha256"] != PUBLIC_PLAN["payload_sha256"]:
        raise RuntimeError("corrected public execution-plan identity mismatch")
    if identity(args.public_delivery) != {
        key: PUBLIC_DELIVERY[key] for key in ("file", "bytes", "sha256")
    } or public_delivery["payload_sha256"] != PUBLIC_DELIVERY["payload_sha256"]:
        raise RuntimeError("corrected public delivery identity mismatch")
    if public_delivery.get("public_execution_plan") != PUBLIC_PLAN:
        raise RuntimeError("corrected public delivery does not bind the corrected plan")
    if identity(args.launcher) != EMBEDDED_LAUNCHER:
        raise RuntimeError("corrected embedded launcher identity mismatch")
    corrected = dict(predecessor)
    corrected["public_execution_plan"] = PUBLIC_PLAN
    corrected["public_cache_delivery"] = PUBLIC_DELIVERY
    corrected["embedded_real_launcher"] = EMBEDDED_LAUNCHER
    corrected["result_blind_real_panel_index_provenance_retry"] = correction_record()
    write_hashed(args.out, corrected)
    print("RUNPOD_REAL_PANEL_PROVENANCE_RETRY_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
