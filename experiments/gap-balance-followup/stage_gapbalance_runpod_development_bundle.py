#!/usr/bin/env python3
"""Stage exact result-blind inputs for the frozen RunPod development run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREFIX = b"# GAPBALANCE_DEVELOPMENT_JOB_CONFIG_HEX="
CONTROLLER_FILES = (
    "GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json",
    "run_gapbalance_runpod_development_job.py",
    "execute_gapbalance_runpod_development.py",
)


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


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("payload_sha256", None)
    if canonical(body) != expected:
        raise RuntimeError(f"payload SHA-256 mismatch: {path}")
    return payload


def embedded_config(path: Path) -> dict:
    first = path.read_bytes().split(b"\n", 1)[0]
    if not first.startswith(PREFIX):
        raise RuntimeError(f"generated launcher config is absent: {path}")
    payload = json.loads(bytes.fromhex(first[len(PREFIX):].decode("ascii")))
    body = dict(payload)
    expected = body.pop("payload_sha256", None)
    if canonical(body) != expected:
        raise RuntimeError(f"generated launcher config hash mismatch: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=here / "GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json")
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--assets-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("staging output must start absent")
    args.output.mkdir(parents=True)
    plan = load_hashed(args.plan)
    jobs = plan["jobs"]
    records = []

    def add(
        source: Path,
        relative: str,
        expected_sha256: str | None = None,
        expected_bytes: int | None = None,
        *,
        verify_content: bool = True,
    ) -> None:
        if not source.is_file():
            raise RuntimeError(f"staging source absent: {source}")
        if not verify_content and expected_sha256 is None:
            raise RuntimeError("trusted staging identity requires an expected SHA-256")
        digest = sha256_file(source) if verify_content else expected_sha256
        size = source.stat().st_size
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError(f"staging SHA-256 mismatch: {source}")
        if expected_bytes is not None and size != expected_bytes:
            raise RuntimeError(f"staging byte-size mismatch: {source}")
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source.resolve())
        records.append({"path": relative, "bytes": size, "sha256": digest})

    prior_manifest = load_hashed(args.assets_manifest)
    asset_identities = {
        record["path"].removeprefix("input/assets/"): record
        for record in prior_manifest.get("files", [])
        if record["path"].startswith("input/assets/")
    }
    asset_files = [path for path in sorted(args.assets.rglob("*")) if path.is_file()]
    if not asset_files or not (args.assets / "SOURCE_SHA256SUMS").is_file():
        raise RuntimeError("frozen asset root is incomplete")
    if {path.relative_to(args.assets).as_posix() for path in asset_files} != set(asset_identities):
        raise RuntimeError("asset file set differs from the verified prior bundle")
    for source in asset_files:
        relative = source.relative_to(args.assets).as_posix()
        identity = asset_identities[relative]
        add(
            source,
            f"input/assets/{relative}",
            identity["sha256"],
            int(identity["bytes"]),
            verify_content=False,
        )

    frozen_runs = {}
    for job in jobs:
        launcher = args.packages / job["job_id"] / "gapbalance_development_kaggle_launcher.py"
        config = embedded_config(launcher)
        if (
            config.get("job_id") != job["job_id"]
            or config.get("payload_sha256") != job["config_payload_sha256"]
            or sha256_file(launcher) != job["launcher_sha256"]
        ):
            raise RuntimeError(f"private launcher binding mismatch: {job['job_id']}")
        add(
            launcher,
            f"launchers/{job['job_id']}/gapbalance_development_kaggle_launcher.py",
            job["launcher_sha256"],
        )
        for run, identity in config["runs"].items():
            existing = frozen_runs.setdefault(run, identity)
            if existing != identity:
                raise RuntimeError(f"inconsistent frozen run identity: {run}")

    checkpoint_candidates = []
    manifest_candidates = []
    for root in args.checkpoint_root:
        checkpoint_candidates.extend(path for path in root.rglob("*.pth") if path.is_file())
        manifest_candidates.extend(
            path for path in root.rglob("training_run_manifest.json") if path.is_file()
        )
    checkpoints_by_hash = {}
    manifests_by_hash = {}
    for path in checkpoint_candidates:
        checkpoints_by_hash.setdefault(sha256_file(path), []).append(path)
    for path in manifest_candidates:
        manifests_by_hash.setdefault(sha256_file(path), []).append(path)
    for run, identity in sorted(frozen_runs.items()):
        states = checkpoints_by_hash.get(identity["checkpoint_sha256"], [])
        manifests = manifests_by_hash.get(identity["training_manifest_sha256"], [])
        if len(states) != 1 or len(manifests) != 1:
            raise RuntimeError(f"exact local checkpoint/manifest pair not unique: {run}")
        add(
            states[0],
            f"input/{run}/{states[0].name}",
            identity["checkpoint_sha256"],
            int(identity["checkpoint_bytes"]),
            verify_content=False,
        )
        add(
            manifests[0],
            f"input/{run}/training_run_manifest.json",
            identity["training_manifest_sha256"],
            int(identity["training_manifest_bytes"]),
        )

    for filename in CONTROLLER_FILES:
        source = args.plan if filename == args.plan.name else here / filename
        add(source, f"controller/{filename}")
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate bundle destination")
    records.sort(key=lambda item: item["path"])
    manifest = {
        "schema_version": "1.0",
        "status": "RunPod GapBalance development bundle staged; endpoints not scored",
        "plan_payload_sha256": plan["payload_sha256"],
        "job_ids": [job["job_id"] for job in jobs],
        "files": records,
        "scientific_outputs_present": False,
        "pherc1218_present": False,
        "confirmation_seeds_500_504_present": False,
    }
    manifest["payload_sha256"] = canonical(manifest)
    (args.output / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "file_count": len(records),
        "jobs": len(jobs),
        "logical_bytes": sum(record["bytes"] for record in records),
        "payload_sha256": manifest["payload_sha256"],
        "status": "RUNPOD_DEVELOPMENT_BUNDLE_STAGED_NOT_SCORED",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
