#!/usr/bin/env python3
"""Mechanically generate the 14 private held-out Kaggle job packages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ASSET_SOURCE = "aviadcohen1/vesuvius-fusion-aware-training-assets/1"
EMBEDDED_CONFIG_PREFIX = b"# HELDOUT_JOB_CONFIG_HEX="


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


def validate_payload(payload: dict) -> None:
    observed = payload.get("payload_sha256")
    content = dict(payload)
    content.pop("payload_sha256", None)
    if observed != canonical_sha256(content):
        raise RuntimeError("execution-plan payload SHA-256 mismatch")


def slug_for(job: dict) -> str:
    job_id = job["job_id"]
    slug = "vesuvius-fusion-" + job_id
    if len(slug) > 70:
        raise RuntimeError(f"generated Kaggle slug is too long: {slug}")
    return slug


def title_for(job: dict) -> str:
    if job["mode"] == "real_test_cache":
        return f"Vesuvius Fusion Real Test {job['run']}"
    return f"Vesuvius Fusion Synthetic {job['run']} All Shards"


def metadata_for(job: dict, threshold_binding: dict) -> dict:
    kernel_sources = [
        f"{threshold_binding['kernel_id']}/{threshold_binding['kernel_version']}"
    ]
    if job.get("kernel_id") is not None:
        kernel_sources.append(f"{job['kernel_id']}/{job['kernel_version']}")
    return {
        "id": f"aviadcohen1/{slug_for(job)}",
        "title": title_for(job),
        "code_file": "heldout_cache_launcher.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [ASSET_SOURCE],
        "kernel_sources": kernel_sources,
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "Gpu",
    }


def write_package(
    *,
    root: Path,
    launcher: Path,
    plan: dict,
    plan_file_sha256: str,
    public_plan_commit: str,
    job: dict,
) -> dict:
    destination = root / job["job_id"]
    destination.mkdir()
    config = {
        "schema_version": "1.0",
        "job_id": job["job_id"],
        "public_plan_commit": public_plan_commit,
        "public_plan_file_sha256": plan_file_sha256,
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
    metadata = metadata_for(job, plan["threshold_binding"])
    metadata_path = destination / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    files = {}
    for path in sorted(destination.iterdir()):
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    ledger = destination / "KERNEL_SHA256SUMS"
    ledger.write_text(
        "".join(f"{record['sha256']}  {name}\n" for name, record in files.items()),
        encoding="utf-8",
    )
    files[ledger.name] = {"bytes": ledger.stat().st_size, "sha256": sha256_file(ledger)}
    return {
        "job_id": job["job_id"],
        "mode": job["mode"],
        "kaggle_kernel_id": metadata["id"],
        "directory": destination.name,
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
    parser.add_argument("--public-plan-commit", required=True)
    parser.add_argument(
        "--launcher",
        type=Path,
        default=Path(__file__).with_name("heldout_cache_launcher.py"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"output root must start absent: {args.out}")
    if len(args.public_plan_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.public_plan_commit
    ):
        raise ValueError(
            "public plan commit must be 40 lowercase hexadecimal characters"
        )
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_payload(plan)
    if plan.get("status") != "held-out execution plan frozen before held-out inference":
        raise RuntimeError("wrong execution-plan status")
    if plan.get("heldout_cache_launcher") != {
        "file": args.launcher.name,
        "bytes": args.launcher.stat().st_size,
        "sha256": sha256_file(args.launcher),
    }:
        raise RuntimeError("package launcher differs from the public execution plan")
    generator_path = Path(__file__).resolve()
    if plan.get("heldout_package_generator") != {
        "file": generator_path.name,
        "bytes": generator_path.stat().st_size,
        "sha256": sha256_file(generator_path),
    }:
        raise RuntimeError("package generator differs from the public execution plan")
    jobs = plan.get("real_test_jobs", []) + plan.get("synthetic_ray_jobs", [])
    if (
        len(plan.get("real_test_jobs", [])) != 7
        or len(plan.get("synthetic_ray_jobs", [])) != 7
    ):
        raise RuntimeError("execution plan does not contain the frozen 7+7 job matrix")
    if len({job["job_id"] for job in jobs}) != 14:
        raise RuntimeError("execution plan job identities are not unique")
    args.out.mkdir(parents=True)
    records = [
        write_package(
            root=args.out,
            launcher=args.launcher,
            plan=plan,
            plan_file_sha256=sha256_file(args.plan),
            public_plan_commit=args.public_plan_commit,
            job=job,
        )
        for job in jobs
    ]
    payload = {
        "schema_version": "1.0",
        "status": "all private held-out cache packages generated from public plan",
        "public_plan_commit": args.public_plan_commit,
        "public_plan_file_sha256": sha256_file(args.plan),
        "public_plan_payload_sha256": plan["payload_sha256"],
        "launcher": plan["heldout_cache_launcher"],
        "package_generator": {
            "file": generator_path.name,
            "bytes": generator_path.stat().st_size,
            "sha256": sha256_file(generator_path),
        },
        "package_count": len(records),
        "packages": records,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    index = args.out / "generated_packages_index.json"
    index.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("held-out packages:", len(records))
    print("package-index payload SHA-256:", payload["payload_sha256"])
    print("ALL_HELDOUT_JOB_PACKAGES_GENERATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
