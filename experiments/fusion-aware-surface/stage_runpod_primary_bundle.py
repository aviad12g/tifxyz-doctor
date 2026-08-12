#!/usr/bin/env python3
"""Stage the exact sealed RunPod primary inputs without duplicating large files."""

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


def canonical(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def stage_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replacement-plan", type=Path, default=here / "runpod_synthetic_replacement_plan.json")
    parser.add_argument("--heldout-plan", type=Path, default=here / "heldout_execution_plan.json")
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--job-id", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"staging output must start absent: {args.output}")
    args.output.mkdir(parents=True)
    replacement = load_json(args.replacement_plan)
    heldout = load_json(args.heldout_plan)
    index = load_json(args.packages / "generated_packages_index.json")
    expected_payload = replacement["payload_sha256"]
    body = dict(replacement)
    body.pop("payload_sha256")
    if canonical(body) != expected_payload:
        raise RuntimeError("replacement plan payload mismatch")
    all_jobs = [item["job_id"] for item in replacement["jobs"]]
    selected = args.job_id or all_jobs
    if any(job not in all_jobs for job in selected) or len(set(selected)) != len(selected):
        raise RuntimeError("invalid staged job selection")
    replacement_jobs = {item["job_id"]: item for item in replacement["jobs"]}
    heldout_jobs = {item["job_id"]: item for item in heldout["synthetic_ray_jobs"]}
    packages = {item["job_id"]: item for item in index["packages"]}
    records = []

    def add(source: Path, relative: str, expected_sha256: str | None = None) -> None:
        if not source.is_file():
            raise RuntimeError(f"staging source absent: {source}")
        digest = expected_sha256 or sha256_file(source)
        destination = args.output / relative
        stage_link(source, destination)
        records.append({"bytes": source.stat().st_size, "path": relative, "sha256": digest})

    ledger = args.assets / "SOURCE_SHA256SUMS"
    add(ledger, "input/assets/SOURCE_SHA256SUMS", replacement["assets"]["ledger_sha256"])
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        add(args.assets / relative, f"input/assets/{relative}", digest)
    add(
        args.assets / "model" / "Model_epoch499.pth",
        "input/assets/model/Model_epoch499.pth",
        replacement["assets"]["source_checkpoint_sha256"],
    )
    threshold = heldout["threshold_binding"]
    add(
        args.heldout_plan.parent / "frozen_thresholds.json",
        "input/threshold-freeze/frozen_thresholds.json",
        threshold["frozen_thresholds"]["sha256"],
    )
    add(
        args.heldout_plan.parent / "threshold_run_manifest.json",
        "input/threshold-freeze/threshold_run_manifest.json",
        threshold["threshold_run_manifest"]["sha256"],
    )

    checkpoint_by_digest = {}
    for path in args.checkpoints.rglob("*.pth"):
        digest = sha256_file(path)
        if digest in checkpoint_by_digest:
            raise RuntimeError(f"duplicate checkpoint digest: {digest}")
        checkpoint_by_digest[digest] = path
    for job_id in selected:
        job = replacement_jobs[job_id]
        package = packages[job_id]
        launcher_identity = package["files"]["heldout_cache_launcher.py"]
        if launcher_identity["sha256"] != job["launcher_sha256"]:
            raise RuntimeError(f"launcher binding mismatch: {job_id}")
        add(
            args.packages / package["directory"] / "heldout_cache_launcher.py",
            f"launchers/{job_id}/heldout_cache_launcher.py",
            job["launcher_sha256"],
        )
        if job["model_state_sha256"] is None:
            continue
        source = checkpoint_by_digest.get(job["model_state_sha256"])
        if source is None:
            raise RuntimeError(f"checkpoint absent: {job_id}")
        add(source, f"input/{job['run']}/{source.name}", job["model_state_sha256"])
        original = heldout_jobs[job_id]
        mount = source.parents[3]
        manifests = list(mount.rglob("training_run_manifest.json"))
        if len(manifests) != 1:
            raise RuntimeError(f"training manifest ambiguity: {job_id}")
        add(
            manifests[0],
            f"input/{job['run']}/training_run_manifest.json",
            original["training_run_manifest_sha256"],
        )

    for filename in (
        "runpod_synthetic_replacement_plan.json",
        "runpod_launch_frozen_job.py",
        "runpod_execute_primary.py",
    ):
        if filename == "runpod_synthetic_replacement_plan.json":
            add(args.replacement_plan.parent / filename, f"controller/{filename}")
        else:
            identities = {
                item["file"]: item for item in replacement["tooling"].values()
            }
            add(
                args.replacement_plan.parent / filename,
                f"controller/{filename}",
                identities[filename]["sha256"],
            )
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate staged destination")
    records.sort(key=lambda item: item["path"])
    manifest = {
        "files": records,
        "job_ids": selected,
        "plan_payload_sha256": replacement["payload_sha256"],
        "schema_version": "1.0",
        "scientific_outputs_present": False,
        "status": "RunPod primary input bundle staged; held-out inference not started",
    }
    manifest["payload_sha256"] = canonical(manifest)
    (args.output / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "file_count": len(records),
        "job_ids": selected,
        "payload_sha256": manifest["payload_sha256"],
        "staged_bytes": sum(item["bytes"] for item in records),
        "status": "RUNPOD_PRIMARY_BUNDLE_STAGED",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
