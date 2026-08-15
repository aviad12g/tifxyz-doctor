#!/usr/bin/env python3
"""Freeze the result-blind RunPod-real/Kaggle-synthetic validation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    content = dict(payload)
    observed = content.pop("payload_sha256", None)
    if observed != canonical_sha256(content):
        raise RuntimeError(f"embedded payload SHA-256 mismatch: {path}")
    return payload


def identity(path: Path, payload: dict | None = None) -> dict:
    record = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if payload is not None:
        record["payload_sha256"] = payload["payload_sha256"]
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--scientific-plan", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--runpod-plan", type=Path, required=True)
    parser.add_argument("--collection-plan", type=Path, required=True)
    parser.add_argument("--result-sources", type=Path, required=True)
    parser.add_argument("--package-index", type=Path, required=True)
    parser.add_argument("--pair-receipt", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--panel-source", type=Path, required=True)
    parser.add_argument("--visual-recorder", type=Path, required=True)
    parser.add_argument("--operational-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("validation plan must start absent")

    hashed = {
        "scientific_plan": load_hashed(args.scientific_plan),
        "delivery": load_hashed(args.delivery),
        "runpod_plan": load_hashed(args.runpod_plan),
        "collection_plan": load_hashed(args.collection_plan),
        "result_sources": load_hashed(args.result_sources),
        "package_index": load_hashed(args.package_index),
        "pair_receipt": load_hashed(args.pair_receipt),
        "thresholds": load_hashed(args.thresholds),
        "panel_source": load_hashed(args.panel_source),
    }
    scientific = hashed["scientific_plan"]
    delivery = hashed["delivery"]
    runpod = hashed["runpod_plan"]
    index = hashed["package_index"]
    synthetic_packages = [
        record for record in index.get("packages", []) if record.get("mode") == "synthetic"
    ]
    if len(synthetic_packages) != 1:
        raise RuntimeError("expected exactly one synthetic package")
    synthetic_package = synthetic_packages[0]
    real_imports = scientific[
        "result_blind_scorer_public_import_dependency_correction"
    ]["public_import_dependencies"]
    real_metric_worker = scientific[
        "result_blind_scorer_public_metric_worker_correction"
    ]["public_metric_worker"]
    payload = {
        "schema_version": "1.0",
        "status": "RunPod-real and Kaggle-synthetic final validation frozen before scientific result access",
        "validator": identity(args.validator),
        "renderer": identity(args.renderer),
        "freezer": identity(Path(__file__).resolve()),
        "sources": {
            "scientific_plan": identity(args.scientific_plan, scientific),
            "delivery": identity(args.delivery, delivery),
            "runpod_plan": identity(args.runpod_plan, hashed["runpod_plan"]),
            "collection_plan": identity(args.collection_plan, hashed["collection_plan"]),
            "result_sources": identity(args.result_sources, hashed["result_sources"]),
            "package_index": identity(args.package_index, index),
            "pair_receipt": identity(args.pair_receipt, hashed["pair_receipt"]),
            "thresholds": identity(args.thresholds, hashed["thresholds"]),
            "split": identity(args.split),
            "panel_source": identity(args.panel_source, hashed["panel_source"]),
            "visual_recorder": identity(args.visual_recorder),
            "operational_audit": identity(args.operational_audit),
        },
        "expected_run_provenance": {
            "real": {
                "public_execution_plan": delivery["public_execution_plan"],
                "public_cache_delivery": {
                    "commit": runpod["public_runpod_commit"],
                    "file_sha256": sha256_file(args.delivery),
                    "payload_sha256": delivery["payload_sha256"],
                },
                "launcher": scientific["one_shot_scoring_launcher"],
                "scorer": scientific["one_shot_scorers"]["real"]["script"],
                "public_imports": real_imports,
                "public_metric_worker": real_metric_worker,
            },
            "synthetic": {
                "public_execution_plan": index["public_execution_plan"],
                "public_cache_delivery": index["public_cache_delivery"],
                "launcher": index["launcher"],
                "scorer": scientific["one_shot_scorers"]["synthetic"]["script"],
                "package_launcher_sha256": synthetic_package["files"][
                    "one_shot_scoring_launcher.py"
                ]["sha256"],
            },
            "threshold_binding": scientific["threshold_binding"],
            "panel_renderer": scientific["real_panel_renderer"],
        },
        "scientific_gate": {
            "paired_collection_completed_before_validation_freeze": True,
            "real_and_synthetic_results_opened_or_parsed_by_freezer": False,
            "panels_opened_or_inspected_by_freezer": False,
            "all_seven_preregistered_gates_must_be_reported": True,
            "all_four_panels_and_twelve_comparisons_must_be_reported": True,
            "adverse_null_and_failed_outcomes_must_be_published": True,
            "result_dependent_tuning_or_selection_permitted": False,
            "official_form_submission_permitted_without_aviad_review": False,
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("final-validation payload SHA-256:", payload["payload_sha256"])
    print("RUNPOD_KAGGLE_FINAL_VALIDATION_FROZEN_RESULT_BLIND")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
