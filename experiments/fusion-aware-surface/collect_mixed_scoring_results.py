#!/usr/bin/env python3
"""Collect sealed real v5 and synthetic v4 artifacts only after both complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import collect_heldout_delivery_inputs as heldout
import collect_scoring_results as base
import orchestrate_heldout_queue as queue
import orchestrate_scoring_pair as pair


MODES = ("real", "synthetic")


def identity(path: Path) -> dict:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": queue.sha256_file(path),
    }


def package_by_mode(packages: list[dict], mode: str) -> dict:
    records = [record for record in packages if record.get("mode") == mode]
    if len(records) != 1:
        raise RuntimeError(f"expected exactly one {mode} package")
    return records[0]


def load_mixed_receipt(path: Path, real_index_path: Path, real_index: dict) -> dict:
    receipt = queue.load_canonical(path)
    if receipt.get("status") != (
        "accelerated real v5 accepted; completed synthetic v4 retained sealed"
    ):
        raise RuntimeError("wrong mixed scorer receipt status")
    if receipt.get("package_index") != {
        "file": real_index_path.name,
        "bytes": real_index_path.stat().st_size,
        "sha256": queue.sha256_file(real_index_path),
        "payload_sha256": real_index["payload_sha256"],
    }:
        raise RuntimeError("mixed receipt points to another real package index")
    if receipt.get("scientific_gate") != {
        "real_v4_result_opened_downloaded_or_used": False,
        "synthetic_v4_result_opened_downloaded_or_used": False,
        "synthetic_scientific_scorer_repeated": False,
        "threshold_model_seed_panel_endpoint_gate_or_claim_changed": False,
        "real_retry_runs_panels_before_parallel_scoring": True,
    }:
        raise RuntimeError("mixed receipt scientific gate mismatch")
    return receipt


def verify_server_source(
    *, kaggle: str, root: Path, index: dict, package: dict, expected_version: int
) -> dict:
    kernel_id = package["kaggle_kernel_id"]
    status = queue.kernel_status(kaggle, kernel_id)
    if status != "COMPLETE":
        raise RuntimeError(f"{package['mode']}: scorer is not COMPLETE ({status})")
    version, source = heldout.current_kernel_version_and_source(kernel_id)
    if version != expected_version:
        raise RuntimeError(f"{package['mode']}: latest scorer version changed")
    local_source = (
        root / package["directory"] / index["launcher"]["file"]
    ).read_bytes()
    if source != local_source:
        raise RuntimeError(f"{package['mode']}: server source differs from accepted package")
    return {
        "kernel_id": kernel_id,
        "kernel_version": version,
        "source_sha256": queue.sha256_bytes(source),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-packages", type=Path, required=True)
    parser.add_argument("--real-index", type=Path, required=True)
    parser.add_argument("--synthetic-packages", type=Path, required=True)
    parser.add_argument("--synthetic-index", type=Path, required=True)
    parser.add_argument("--mixed-receipt", type=Path, required=True)
    parser.add_argument("--v4-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kaggle", default="kaggle")
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("mixed scoring collector output must start absent")
    real_index, real_packages = pair.validate_packages(
        args.real_packages, args.real_index
    )
    synthetic_index, synthetic_packages = pair.validate_packages(
        args.synthetic_packages, args.synthetic_index
    )
    real_package = package_by_mode(real_packages, "real")
    synthetic_package = package_by_mode(synthetic_packages, "synthetic")
    mixed = load_mixed_receipt(args.mixed_receipt, args.real_index, real_index)
    real_record = mixed.get("real_retry")
    retained = mixed.get("retained_synthetic")
    if (
        not isinstance(real_record, dict)
        or real_record.get("kernel_id") != real_package["kaggle_kernel_id"]
        or real_record.get("kernel_version") != 5
        or real_record.get("package_ledger_sha256")
        != real_package["files"]["KERNEL_SHA256SUMS"]["sha256"]
    ):
        raise RuntimeError("mixed receipt real-v5 binding mismatch")
    v4 = pair.load_receipt(args.v4_receipt, args.synthetic_index, synthetic_index)
    accepted_v4 = pair.receipt_map(v4, synthetic_packages)
    synthetic_record = accepted_v4["synthetic"]
    if retained != {
        "mode": "synthetic",
        "kernel_id": synthetic_package["kaggle_kernel_id"],
        "kernel_version": 4,
        "status": "COMPLETE",
        "result_opened_downloaded_or_used": False,
    } or synthetic_record["kernel_version"] != 4:
        raise RuntimeError("mixed receipt retained-synthetic binding mismatch")

    sources = {
        "real": verify_server_source(
            kaggle=args.kaggle,
            root=args.real_packages,
            index=real_index,
            package=real_package,
            expected_version=5,
        ),
        "synthetic": verify_server_source(
            kaggle=args.kaggle,
            root=args.synthetic_packages,
            index=synthetic_index,
            package=synthetic_package,
            expected_version=4,
        ),
    }
    args.out.mkdir(parents=True)
    downloads = args.out / "downloads"
    downloads.mkdir()
    artifacts = {}
    for mode in MODES:
        destination = downloads / mode
        base.download_selected(args.kaggle, sources[mode]["kernel_id"], destination)
        result, run, panel_manifest, panels = base.validate_download(destination, mode)
        artifacts[mode] = {
            **sources[mode],
            "result": str(result),
            "run_manifest": str(run),
            "panel_manifest": str(panel_manifest) if panel_manifest else None,
            "panel_images": [str(path) for path in panels],
        }
        print(f"SEALED_MIXED_SCORING_ARTIFACTS_COLLECTED mode={mode}")
    collector = Path(__file__).resolve()
    payload = {
        "schema_version": "1.0",
        "status": "sealed real v5 and synthetic v4 artifacts jointly collected",
        "real_package_index": identity(args.real_index),
        "synthetic_package_index": identity(args.synthetic_index),
        "mixed_receipt": identity(args.mixed_receipt),
        "v4_receipt": identity(args.v4_receipt),
        "artifacts": artifacts,
        "collector": identity(collector),
        "scientific_gate": {
            "real_v5_and_synthetic_v4_complete_before_first_download": True,
            "server_sources_match_accepted_packages": True,
            "synthetic_scorer_repeated": False,
            "kernel_logs_opened_or_read": False,
            "results_opened_or_read_by_collector": False,
            "panel_images_opened_or_read_by_collector": False,
            "independent_validation_required_next": True,
        },
    }
    payload["payload_sha256"] = queue.canonical_sha256(payload)
    output = args.out / "mixed_scoring_result_sources.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("mixed scoring source payload SHA-256:", payload["payload_sha256"])
    print("BOTH_SEALED_MIXED_RESULTS_READY_FOR_INDEPENDENT_VALIDATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
