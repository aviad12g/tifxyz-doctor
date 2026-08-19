"""Prepare the exact private Kaggle packages for frozen development caching."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LAUNCHER_PATH = HERE / "gapbalance_development_kaggle_launcher.py"
FREEZE_PATH = HERE / "GAPBALANCE_TRAINING_FREEZE.json"
ASSET_SOURCE = "aviadcohen1/vesuvius-fusion-aware-training-assets/1"
RUNS = tuple(
    f"{arm}_seed{seed}"
    for arm in ("control", "gap2", "gap4")
    for seed in (11, 23, 47)
)
PUBLIC_SOURCE_FILES = (
    "experiments/gap-balance-followup/gapbalance_development.py",
    "experiments/gap-balance-followup/cache_gapbalance_real_development.py",
    "experiments/gap-balance-followup/cache_gapbalance_synthetic_development.py",
    "experiments/fusion-aware-surface/inference.py",
    "experiments/fusion-aware-surface/normalization.py",
    "experiments/fusion-aware-surface/fusion_ray_readout.py",
)
CONTROL_MANIFESTS = {
    "control_seed11": (2481, "18b52d4fa2895307fa282a3b26e9cdeeeafd4204991c8ed4636fe366065a6f82"),
    "control_seed23": (2481, "9ad56528323f821e60c98faff48265c327a4ede351148e0596a14737aa493f2d"),
    "control_seed47": (2481, "2eede7befa96f01aed362040705da55643fffe287b11c9759596e5197f99ecb2"),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def git_bytes(commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cannot read {relative} at {commit}: {result.stderr.decode().strip()}"
        )
    return result.stdout


def launcher_identity(commit: str) -> dict[str, Any]:
    relative = LAUNCHER_PATH.relative_to(REPO).as_posix()
    raw = git_bytes(commit, relative)
    if raw != LAUNCHER_PATH.read_bytes():
        raise RuntimeError("working launcher differs from the requested public commit")
    return {
        "file": LAUNCHER_PATH.name,
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def public_source_hashes(commit: str) -> dict[str, str]:
    hashes = {}
    for relative in PUBLIC_SOURCE_FILES:
        raw = git_bytes(commit, relative)
        if raw != (REPO / relative).read_bytes():
            raise RuntimeError(f"working source differs from {commit}: {relative}")
        hashes[relative] = sha256_bytes(raw)
    return hashes


def training_records() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    body = dict(freeze)
    observed = body.pop("payload_sha256", None)
    if observed != canonical_sha256(body):
        raise RuntimeError("training-freeze embedded SHA-256 mismatch")
    if set(freeze.get("runs", {})) != set(RUNS):
        raise RuntimeError("training freeze does not contain the exact nine runs")
    records = {}
    for run in RUNS:
        source = freeze["runs"][run]
        if run in CONTROL_MANIFESTS:
            manifest_bytes, manifest_sha256 = CONTROL_MANIFESTS[run]
            checkpoint_bytes = 409_569_594
        else:
            manifest = source["files"]["training_run_manifest"]
            manifest_bytes = int(manifest["bytes"])
            manifest_sha256 = manifest["sha256"]
            checkpoint_bytes = int(source["files"]["checkpoint"]["bytes"])
        records[run] = {
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_sha256": source["checkpoint_sha256"],
            "kernel_id": source["kernel"]["id"],
            "kernel_version": int(source["kernel"]["version"]),
            "training_manifest_bytes": manifest_bytes,
            "training_manifest_sha256": manifest_sha256,
        }
    return freeze, records


def job_specs(records: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        {
            "job_id": "gapbalance-development-real",
            "kernel_slug": "vesuvius-gapbalance-development-real-all-runs",
            "mode": "real",
            "runs": {run: records[run] for run in RUNS},
            "title": "Vesuvius GapBalance Development Real All Runs",
        }
    ]
    for seed in (11, 23, 47):
        seed_runs = tuple(f"{arm}_seed{seed}" for arm in ("control", "gap2", "gap4"))
        for shard in range(4):
            specs.append(
                {
                    "job_id": f"gapbalance-development-synthetic-seed{seed}-shard{shard:02d}",
                    "kernel_slug": f"gapbalance-dev-syn-s{seed}-q{shard}",
                    "mode": "synthetic",
                    "runs": {run: records[run] for run in seed_runs},
                    "seed": seed,
                    "shard_index": shard,
                    "title": f"GapBalance Dev Syn S{seed} Q{shard}",
                }
            )
    return specs


def write_package(
    root: Path,
    spec: dict[str, Any],
    commit: str,
    sources: dict[str, str],
    launcher: dict[str, Any],
) -> dict[str, Any]:
    destination = root / spec["job_id"]
    destination.mkdir(parents=True, exist_ok=False)
    config = {
        "schema_version": "1.0",
        "job_id": spec["job_id"],
        "mode": spec["mode"],
        "seed": spec.get("seed"),
        "shard_index": spec.get("shard_index"),
        "public_source_commit": commit,
        "public_source_files": sources,
        "launcher": launcher,
        "runs": spec["runs"],
        "scientific_endpoints_scored": False,
        "confirmation_outputs_inspected": False,
    }
    config["payload_sha256"] = canonical_sha256(config)
    config_raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    launcher_raw = LAUNCHER_PATH.read_bytes()
    generated = (
        b"# GAPBALANCE_DEVELOPMENT_JOB_CONFIG_HEX="
        + config_raw.hex().encode("ascii")
        + b"\n"
        + launcher_raw
    )
    code = destination / LAUNCHER_PATH.name
    code.write_bytes(generated)
    kernel_sources = [
        f"{record['kernel_id']}/{record['kernel_version']}"
        for record in spec["runs"].values()
    ]
    metadata = {
        "id": f"aviadcohen1/{spec['kernel_slug']}",
        "title": spec["title"],
        "code_file": code.name,
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
    metadata_path = destination / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    sums = destination / "KERNEL_SHA256SUMS"
    sums.write_text(
        f"{sha256_file(code)}  {code.name}\n"
        f"{sha256_file(metadata_path)}  {metadata_path.name}\n",
        encoding="utf-8",
    )
    return {
        "job_id": spec["job_id"],
        "kernel_id": metadata["id"],
        "mode": spec["mode"],
        "seed": spec.get("seed"),
        "shard_index": spec.get("shard_index"),
        "runs": list(spec["runs"]),
        "config_payload_sha256": config["payload_sha256"],
        "launcher_sha256": sha256_file(code),
        "metadata_sha256": sha256_file(metadata_path),
        "package_sums_sha256": sha256_file(sums),
        "kernel_sources": kernel_sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    commit = args.public_source_commit
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("public source commit must be an exact lowercase Git SHA")
    if args.out.exists():
        raise RuntimeError(f"job-package output must start absent: {args.out}")
    args.out.mkdir(parents=True)

    freeze, records = training_records()
    launcher = launcher_identity(commit)
    sources = public_source_hashes(commit)
    jobs = [
        write_package(args.out, spec, commit, sources, launcher)
        for spec in job_specs(records)
    ]
    payload = {
        "schema_version": "1.0",
        "status": "GapBalance development cache packages prepared before endpoint access",
        "public_source_commit": commit,
        "public_source_files": sources,
        "launcher": launcher,
        "training_freeze_payload_sha256": freeze["payload_sha256"],
        "jobs": jobs,
        "scientific_endpoints_scored": False,
        "confirmation_outputs_inspected": False,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    manifest = args.out / "development_job_packages_manifest.json"
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("development-package payload SHA-256:", payload["payload_sha256"])
    print("development package count:", len(jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
