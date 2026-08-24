#!/usr/bin/env python3
"""Generate the two private one-shot scorer packages from public freezes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ASSET_SOURCE = "aviadcohen1/vesuvius-fusion-aware-training-assets/1"
METRIC_SOURCE = "sohier/vesuvius-metric-resources/1"
METRIC_RUNTIME_SOURCE = "aviadcohen1/vesuvius-metric-runtime-cp312/1"
MODES = ("real", "synthetic")
EMBEDDED_CONFIG_PREFIX = b"# SCORING_JOB_CONFIG_HEX="


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
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    observed = payload.get("payload_sha256")
    content = dict(payload)
    content.pop("payload_sha256", None)
    if observed != canonical_sha256(content):
        raise RuntimeError(f"embedded payload SHA-256 mismatch: {path}")
    return payload


def require_commit(value: str, label: str) -> str:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be 40 lowercase hexadecimal characters")
    return value


def local_identity(path: Path) -> dict:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_public_inputs(
    *,
    plan_path: Path,
    delivery_path: Path,
    public_plan_commit: str,
    launcher: Path,
    stager: Path,
    metric_preparer: Path,
    panel_renderer: Path,
    generator: Path,
) -> tuple[dict, dict]:
    plan = load_hashed(plan_path)
    delivery = load_hashed(delivery_path)
    if plan.get("status") != "held-out execution plan frozen before held-out inference":
        raise RuntimeError("wrong held-out execution-plan status")
    if delivery.get("status") != (
        "all 14 publicly planned held-out caches sealed before one-shot scoring"
    ):
        raise RuntimeError("wrong held-out delivery status")
    if delivery.get("public_execution_plan") != {
        "commit": public_plan_commit,
        "file": plan_path.name,
        "bytes": plan_path.stat().st_size,
        "sha256": sha256_file(plan_path),
        "payload_sha256": plan["payload_sha256"],
    }:
        raise RuntimeError("held-out delivery points to another public plan")
    expected_jobs = plan.get("real_test_jobs", []) + plan.get("synthetic_ray_jobs", [])
    if (
        len(plan.get("real_test_jobs", [])) != 7
        or len(plan.get("synthetic_ray_jobs", [])) != 7
        or len(expected_jobs) != 14
    ):
        raise RuntimeError("public plan does not contain the frozen 7+7 job matrix")
    expected_order = [job["job_id"] for job in expected_jobs]
    if delivery.get("job_order") != expected_order:
        raise RuntimeError("delivery job order differs from public plan")
    delivered = delivery.get("jobs")
    if not isinstance(delivered, list) or len(delivered) != 14:
        raise RuntimeError("delivery does not contain 14 job identities")
    for expected, observed in zip(expected_jobs, delivered, strict=True):
        if (
            observed.get("job_id") != expected["job_id"]
            or observed.get("mode") != expected["mode"]
            or observed.get("run") != expected["run"]
            or observed.get("kernel_id")
            != f"aviadcohen1/vesuvius-fusion-{expected['job_id']}"
            or not isinstance(observed.get("kernel_version"), int)
            or observed["kernel_version"] <= 0
        ):
            raise RuntimeError(f"delivery job identity mismatch: {expected['job_id']}")
    if delivery.get("threshold_binding") != plan.get("threshold_binding"):
        raise RuntimeError("delivery threshold binding differs from public plan")
    if delivery.get("counts") != {
        "jobs": 14,
        "real_jobs": 7,
        "synthetic_jobs": 7,
        "real_probability_caches": 266,
        "synthetic_ray_caches": 700,
    }:
        raise RuntimeError("delivery cache counts mismatch")
    checks = {
        "one_shot_scoring_launcher": local_identity(launcher),
        "one_shot_scoring_stager": local_identity(stager),
        "metric_runtime_preparer": local_identity(metric_preparer),
        "real_panel_renderer": local_identity(panel_renderer),
        "scoring_package_generator": local_identity(generator),
        "scoring_pair_controller": plan.get("scoring_pair_controller"),
        "scoring_result_collector": plan.get("scoring_result_collector"),
    }
    for key, expected in checks.items():
        if key in {"scoring_pair_controller", "scoring_result_collector"}:
            if not isinstance(expected, dict) or set(expected) != {
                "file",
                "bytes",
                "sha256",
            }:
                raise RuntimeError(f"public plan identity is invalid for {key}")
            continue
        if plan.get(key) != expected:
            raise RuntimeError(f"public plan identity mismatch for {key}")
    if delivery.get("scientific_gate") != {
        "all_cache_jobs_completed": True,
        "thresholds_publicly_frozen_before_cache_inference": True,
        "scientific_endpoints_scored": False,
        "scientific_endpoints_printed": False,
        "cache_npz_payloads_opened_or_inspected_by_delivery_freezer": False,
        "one_shot_scoring_permitted_after_this_public_freeze": True,
    }:
        raise RuntimeError("public delivery scientific gate mismatch")
    return plan, delivery


def metadata_for(mode: str, plan: dict, delivery: dict) -> dict:
    expected_mode = "real_test_cache" if mode == "real" else "synthetic_ray_cache"
    sources = [
        f"{record['kernel_id']}/{record['kernel_version']}"
        for record in delivery["jobs"]
        if record["mode"] == expected_mode
    ]
    if len(sources) != 7:
        raise RuntimeError(f"delivery does not contain seven {mode} cache kernels")
    threshold = plan["threshold_binding"]
    kernel_sources = [
        f"{threshold['kernel_id']}/{threshold['kernel_version']}",
        *sources,
    ]
    datasets = [ASSET_SOURCE]
    if mode == "real":
        datasets.extend([METRIC_SOURCE, METRIC_RUNTIME_SOURCE])
    return {
        "id": f"aviadcohen1/vesuvius-fusion-one-shot-{mode}-scoring",
        "title": f"Vesuvius Fusion One Shot {mode.title()} Scoring",
        "code_file": "one_shot_scoring_launcher.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": datasets,
        "kernel_sources": kernel_sources,
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "Gpu",
    }


def write_package(
    *,
    mode: str,
    root: Path,
    plan: dict,
    delivery: dict,
    plan_path: Path,
    delivery_path: Path,
    public_plan_commit: str,
    public_delivery_commit: str,
    launcher: Path,
) -> dict:
    destination = root / mode
    destination.mkdir()
    config = {
        "schema_version": "1.0",
        "mode": mode,
        "public_plan_commit": public_plan_commit,
        "public_plan_file_sha256": sha256_file(plan_path),
        "public_delivery_commit": public_delivery_commit,
        "public_delivery_file_sha256": sha256_file(delivery_path),
    }
    config["payload_sha256"] = canonical_sha256(config)
    config_raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    launcher_path = destination / launcher.name
    launcher_path.write_bytes(
        EMBEDDED_CONFIG_PREFIX
        + config_raw.hex().encode("ascii")
        + b"\n"
        + launcher.read_bytes()
    )
    metadata = metadata_for(mode, plan, delivery)
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(destination.iterdir())
    }
    ledger = destination / "KERNEL_SHA256SUMS"
    ledger.write_text(
        "".join(f"{record['sha256']}  {name}\n" for name, record in files.items()),
        encoding="utf-8",
    )
    files[ledger.name] = {"bytes": ledger.stat().st_size, "sha256": sha256_file(ledger)}
    return {
        "mode": mode,
        "kaggle_kernel_id": metadata["id"],
        "directory": destination.name,
        "required_cache_kernel_count": 7,
        "threshold_kernel": metadata["kernel_sources"][0],
        "files": files,
        "embedded_job_config": {
            "encoding": "hex in first source comment",
            "bytes": len(config_raw),
            "sha256": hashlib.sha256(config_raw).hexdigest(),
            "payload_sha256": config["payload_sha256"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--public-plan-commit", required=True)
    parser.add_argument("--public-delivery-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--launcher",
        type=Path,
        default=Path(__file__).with_name("one_shot_scoring_launcher.py"),
    )
    parser.add_argument(
        "--stager",
        type=Path,
        default=Path(__file__).with_name("stage_heldout_for_scoring.py"),
    )
    parser.add_argument(
        "--metric-preparer",
        type=Path,
        default=Path(__file__).with_name("prepare_metric_runtime.py"),
    )
    parser.add_argument(
        "--panel-renderer",
        type=Path,
        default=Path(__file__).with_name("render_real_panels.py"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"output root must start absent: {args.out}")
    public_plan_commit = require_commit(args.public_plan_commit, "public plan commit")
    public_delivery_commit = require_commit(
        args.public_delivery_commit, "public delivery commit"
    )
    generator = Path(__file__).resolve()
    plan, delivery = validate_public_inputs(
        plan_path=args.plan,
        delivery_path=args.delivery,
        public_plan_commit=public_plan_commit,
        launcher=args.launcher,
        stager=args.stager,
        metric_preparer=args.metric_preparer,
        panel_renderer=args.panel_renderer,
        generator=generator,
    )
    args.out.mkdir(parents=True)
    packages = [
        write_package(
            mode=mode,
            root=args.out,
            plan=plan,
            delivery=delivery,
            plan_path=args.plan,
            delivery_path=args.delivery,
            public_plan_commit=public_plan_commit,
            public_delivery_commit=public_delivery_commit,
            launcher=args.launcher,
        )
        for mode in MODES
    ]
    payload = {
        "schema_version": "1.0",
        "status": "two one-shot scorer packages generated after public cache-delivery freeze",
        "public_execution_plan": {
            "commit": public_plan_commit,
            "file_sha256": sha256_file(args.plan),
            "payload_sha256": plan["payload_sha256"],
        },
        "public_cache_delivery": {
            "commit": public_delivery_commit,
            "file_sha256": sha256_file(args.delivery),
            "payload_sha256": delivery["payload_sha256"],
        },
        "generator": local_identity(generator),
        "launcher": plan["one_shot_scoring_launcher"],
        "pair_controller": plan["scoring_pair_controller"],
        "result_collector": plan["scoring_result_collector"],
        "package_count": 2,
        "packages": packages,
        "scientific_gate": {
            "all_14_cache_jobs_publicly_frozen": True,
            "scientific_results_opened_or_inspected": False,
            "both_scorer_packages_generated_together": True,
            "both_scorers_must_be_launched_before_either_result_is_opened": True,
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    index = args.out / "generated_scoring_packages_index.json"
    index.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("one-shot scoring packages:", len(packages))
    print("scoring-package index payload SHA-256:", payload["payload_sha256"])
    print("BOTH_ONE_SHOT_SCORING_PACKAGES_GENERATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
