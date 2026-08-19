#!/usr/bin/env python3
"""Materialize exactly the six frozen Gap2/Gap4 Kaggle job packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_JOBS = {
    ("gap2", 11),
    ("gap2", 23),
    ("gap2", 47),
    ("gap4", 11),
    ("gap4", 23),
    ("gap4", 47),
}
EXPECTED_CONTROLS = {
    "11": "7a2e6168f32b3a3389bdc2b43b39a6654568a6da467e71948f523e7cbef6248f",
    "23": "d40c4b856c9a65cc2de127e98d37d08633e03e6b7b5e375bec8991b1bf9487b3",
    "47": "fd86b40b3b25f45f86f2ba43668997d4351fec7d3ebd35fa41856e4c0d505e44",
}
KERNEL_ID_RE = re.compile(r"aviadcohen1/vesuvius-gapbalance-gap[24]-seed-(11|23|47)\Z")
PREFLIGHT_KERNEL_ID = "aviadcohen1/vesuvius-gapbalance-verify-only"


class PlanError(RuntimeError):
    """The frozen free-compute plan or local source tree changed."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def load_and_validate_plan(plan_path: Path, repo_root: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    require(plan.get("schema_version") == "1.0", "unexpected plan schema")
    require(
        plan.get("status") == "PUBLIC_FROZEN_READY_FOR_FREE_KAGGLE_VERIFY_ONLY",
        "plan is not frozen for verify-only",
    )
    execution = plan.get("execution", {})
    require(execution.get("free_kaggle_execution_authorized") is True, "free execution is not authorized")
    require(execution.get("paid_compute_authorized") is False, "paid compute must remain unauthorized")
    require(execution.get("form_submission_authorized") is False, "form submission must remain unauthorized")
    require(execution.get("provider_cost_usd") == 0, "provider cost must be zero")
    require(plan.get("control_reuse") == EXPECTED_CONTROLS, "control checkpoint identities changed")
    require(
        plan.get("asset_bundle", {}).get("expanded_archive_identity")
        == {
            "bytes": 2_005_739_905,
            "files": 400,
            "sha256": "981448a526d2e04cb59b0b1f40331fa901ece5c0514161733814c5b7ea823021",
        },
        "expanded Kaggle archive identity changed",
    )

    preflight = plan.get("preflight")
    require(
        preflight == {
            "arm": "gap2",
            "enable_gpu": False,
            "kernel_id": PREFLIGHT_KERNEL_ID,
            "seed": 11,
            "title": "Vesuvius GapBalance Verify Only",
            "verify_only": True,
        },
        "CPU verify-only preflight changed",
    )

    jobs = plan.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 6, "plan must contain exactly six jobs")
    observed = {(job.get("arm"), job.get("seed")) for job in jobs}
    require(observed == EXPECTED_JOBS, "job matrix changed")
    kernel_ids = [job.get("kernel_id") for job in jobs]
    require(len(set(kernel_ids)) == 6, "kernel IDs are not unique")
    require(all(isinstance(value, str) and KERNEL_ID_RE.fullmatch(value) for value in kernel_ids), "invalid kernel ID")

    protocol = plan.get("protocol", {})
    for path_key, hash_key in (
        ("contract_path", "contract_file_sha256"),
        ("holdout_selection_path", "holdout_selection_file_sha256"),
        ("protocol_path", "protocol_file_sha256"),
    ):
        path = repo_root / protocol[path_key]
        require(path.is_file(), f"missing protocol artifact: {path}")
        require(sha256_file(path) == protocol[hash_key], f"protocol artifact changed: {path}")

    projection = plan.get("public_projection", {})
    launcher = repo_root / projection.get("launcher_file", "")
    require(launcher.is_file(), "public launcher is missing")
    require(sha256_file(launcher) == projection.get("launcher_sha256"), "public launcher changed")
    return plan


def render_launcher(template: str, arm: str, seed: int, verify_only: bool) -> str:
    arm_marker = 'FROZEN_ARM = "__FROZEN_ARM__"'
    seed_marker = 'FROZEN_SEED = "__FROZEN_SEED__"'
    verify_marker = 'FROZEN_VERIFY_ONLY = "__FROZEN_VERIFY_ONLY__"'
    require(template.count(arm_marker) == 1, "arm marker is not unique")
    require(template.count(seed_marker) == 1, "seed marker is not unique")
    require(template.count(verify_marker) == 1, "verify-only marker is not unique")
    rendered = template.replace(arm_marker, f'FROZEN_ARM = "{arm}"')
    rendered = rendered.replace(seed_marker, f'FROZEN_SEED = "{seed}"')
    rendered = rendered.replace(verify_marker, f"FROZEN_VERIFY_ONLY = {verify_only!r}")
    require(
        arm_marker not in rendered and seed_marker not in rendered and verify_marker not in rendered,
        "rendered launcher still has assignment placeholders",
    )
    return rendered


def prepare(plan_path: Path, repo_root: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), f"output must start absent: {output}")
    plan = load_and_validate_plan(plan_path, repo_root)
    launcher_path = repo_root / plan["public_projection"]["launcher_file"]
    template = launcher_path.read_text(encoding="utf-8")
    output.mkdir(parents=True)
    job_records = []

    preflight = plan["preflight"]
    preflight_root = output / "preflight"
    preflight_root.mkdir()
    preflight_launcher = preflight_root / "kernel_launcher.py"
    preflight_launcher.write_text(
        render_launcher(template, preflight["arm"], preflight["seed"], True),
        encoding="utf-8",
    )
    preflight_metadata = {
        "id": preflight["kernel_id"],
        "title": preflight["title"],
        "code_file": "kernel_launcher.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [plan["asset_bundle"]["dataset_id"]],
        "competition_sources": [],
        "kernel_sources": [],
    }
    preflight_metadata_path = preflight_root / "kernel-metadata.json"
    preflight_metadata_path.write_text(json.dumps(preflight_metadata, indent=2) + "\n", encoding="utf-8")
    for job in plan["jobs"]:
        arm = job["arm"]
        seed = job["seed"]
        job_root = output / f"{arm}-seed{seed}"
        job_root.mkdir()
        launcher = job_root / "kernel_launcher.py"
        launcher.write_text(render_launcher(template, arm, seed, False), encoding="utf-8")
        metadata = {
            "id": job["kernel_id"],
            "title": job["title"],
            "code_file": "kernel_launcher.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [plan["asset_bundle"]["dataset_id"]],
            "competition_sources": [],
            "kernel_sources": [],
        }
        metadata_path = job_root / "kernel-metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        job_records.append({
            "arm": arm,
            "seed": seed,
            "kernel_id": job["kernel_id"],
            "launcher_sha256": sha256_file(launcher),
            "metadata_sha256": sha256_file(metadata_path),
        })
    receipt = {
        "schema_version": "1.0",
        "status": "CPU_PREFLIGHT_AND_SIX_FREE_KAGGLE_JOB_PACKAGES_PREPARED_NOT_PUSHED",
        "plan_file_sha256": sha256_file(plan_path),
        "plan_payload_sha256": canonical_sha256(plan),
        "preflight": {
            "kernel_id": preflight["kernel_id"],
            "launcher_sha256": sha256_file(preflight_launcher),
            "metadata_sha256": sha256_file(preflight_metadata_path),
            "verify_only": True,
            "enable_gpu": False,
        },
        "jobs": job_records,
        "paid_compute_authorized": False,
        "confirmation_outputs_inspected": False,
    }
    receipt["payload_sha256"] = canonical_sha256(receipt)
    receipt_path = output / "PREPARATION_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = prepare(args.plan.resolve(), args.repo_root.resolve(), args.output.resolve())
    except (PlanError, OSError, json.JSONDecodeError, KeyError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
