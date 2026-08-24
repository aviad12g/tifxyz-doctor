#!/usr/bin/env python3
"""Verify the blocked, result-blind GapBalance resume package.

This script is deliberately incapable of launching providers, scoring caches,
or opening confirmation inputs.  A successful run means only that the public
resume materials are internally consistent and ready for a later, explicit
quarantine-supersession decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class ReadinessError(RuntimeError):
    """The blocked resume package is not internally consistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hashed_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    observed = body.pop("payload_sha256", None)
    require(observed == canonical_sha256(body), f"embedded payload mismatch: {path.name}")
    return payload


def verify(root: Path) -> dict[str, Any]:
    resume_path = root / "GAPBALANCE_KAGGLE_RESUME_DRAFT.json"
    requirements_path = root / "GAPBALANCE_SUPPLEMENTAL_CONTROL_REQUIREMENTS.json"
    quarantine_path = root / "PHERC1218_HOLDOUT_QUARANTINE.json"
    plan_path = root / "GAPBALANCE_KAGGLE_DEVELOPMENT_PLAN.json"
    resume = load_hashed_json(resume_path)
    requirements = load_hashed_json(requirements_path)
    quarantine = load_hashed_json(quarantine_path)

    require(
        resume.get("status") == "DRAFT_BLOCKED_PENDING_HOLDOUT_SUPERSESSION",
        "resume draft is not blocked",
    )
    require(resume.get("activation", {}).get("launch_permitted") is False, "resume draft permits launch")
    require(resume.get("activation", {}).get("scoring_permitted") is False, "resume draft permits scoring")
    require(resume.get("activation", {}).get("confirmation_access_permitted") is False, "resume draft permits confirmation")
    require(resume.get("provider") == "free Kaggle GPU", "resume provider is not uniform free Kaggle")
    require(resume.get("provider_cost_usd") == 0, "resume draft authorizes spending")
    require(resume["source_plan"]["file_sha256"] == sha256_file(plan_path), "source plan file changed")
    plan = load_hashed_json(plan_path)
    require(resume["source_plan"]["payload_sha256"] == plan["payload_sha256"], "source plan payload changed")
    require(
        resume["active_quarantine_payload_sha256"] == quarantine["payload_sha256"],
        "resume draft is not bound to the active quarantine",
    )
    require(quarantine.get("status") == "BLOCKED_REAL_HOLDOUT_NOT_CERTIFIED", "quarantine status changed")
    require(requirements.get("quarantine_superseded") is False, "control requirements supersede quarantine")

    completed = resume.get("completed_cache_jobs", [])
    pending = resume.get("pending_cache_jobs", [])
    completed_ids = {job["job_id"] for job in completed}
    pending_ids = {job["job_id"] for job in pending}
    require(len(completed) == 5 and len(completed_ids) == 5, "completed cache ledger must contain five jobs")
    require(len(pending) == 8 and len(pending_ids) == 8, "pending launch ledger must contain eight jobs")
    require(not completed_ids & pending_ids, "completed and pending jobs overlap")
    expected_ids = {job["job_id"] for job in plan["jobs"]}
    require(completed_ids | pending_ids == expected_ids, "resume ledger does not cover the exact 13 jobs")
    require(
        all(job.get("provider") == "kaggle" for job in completed + pending),
        "resume ledger mixes providers",
    )
    require(
        [job["job_id"] for job in pending] == resume["frozen_launch_order"],
        "pending jobs differ from frozen launch order",
    )
    require(resume.get("cross_provider_partial_mix_permitted") is False, "cross-provider partial mix enabled")
    require(resume.get("scientific_endpoints_scored") is False, "resume draft records scoring")
    require(resume.get("confirmation_outputs_inspected") is False, "resume draft records confirmation access")

    for record in resume["chain"]:
        require(record.get("executed") is False, "resume-chain step was marked executed")
    require(
        [record["step"] for record in resume["chain"]]
        == [
            "metadata_only_holdout_reassessment",
            "public_quarantine_supersession",
            "complete_uniform_kaggle_development_cache",
            "mechanical_cache_verification",
            "one_shot_development_scoring",
            "public_candidate_and_threshold_freeze",
            "sealed_confirmation_cache_and_scoring",
            "publish_all_outcomes",
        ],
        "resume chain changed",
    )

    result = {
        "schema_version": "1.0",
        "status": "BLOCKED_PACKAGE_READY_FOR_LATER_HOLDOUT_REASSESSMENT",
        "completed_cache_jobs": len(completed),
        "pending_cache_jobs": len(pending),
        "uniform_provider": "kaggle",
        "provider_actions_executed": False,
        "scientific_endpoints_scored": False,
        "confirmation_outputs_inspected": False,
        "quarantine_active": True,
    }
    result["payload_sha256"] = canonical_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.root)
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            require(not args.output.exists(), "output path must start absent")
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except (ReadinessError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
