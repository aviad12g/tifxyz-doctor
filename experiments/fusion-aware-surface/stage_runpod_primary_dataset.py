#!/usr/bin/env python3
"""Stage verified RunPod primary caches as one private Kaggle dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def safe_relative(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"invalid {label}")
    path = Path(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError(f"unsafe {label}: {value}")
    return path


def safe_link(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"source must be one regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"dataset destination collision: {destination}")
    os.link(source, destination)


def stage_job(source: dict, destination_root: Path) -> tuple[int, int, list[dict]]:
    job_id_path = safe_relative(source["job_id"], "job ID")
    if len(job_id_path.parts) != 1:
        raise RuntimeError("job ID must be one path component")
    job_id = job_id_path.name
    index_identity = source["job_index_identity"]
    index_path = Path(source["job_index"])
    if (
        not index_path.is_file()
        or index_path.stat().st_size != index_identity["bytes"]
        or sha256_file(index_path) != index_identity["sha256"]
    ):
        raise RuntimeError(f"{job_id}: verified index identity changed")
    index = load_hashed(index_path)
    if index["payload_sha256"] != index_identity["payload_sha256"]:
        raise RuntimeError(f"{job_id}: verified index payload changed")
    root = index_path.parent
    relative_root = Path("primary") / job_id_path
    staged = []
    files = [
        {
            "relative": Path("heldout_job_index.json"),
            "bytes": index_identity["bytes"],
            "sha256": index_identity["sha256"],
        }
    ]
    for record in index["cache_manifests"]:
        files.append(
            {
            "relative": safe_relative(record["file"], "cache-manifest file"),
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
        )
    run = safe_relative(index["job"]["run"], "run directory")
    if len(run.parts) != 1:
        raise RuntimeError("run directory must be one path component")
    for record in index["sealed_cache_files"]:
        files.append(
            {
                "relative": run / safe_relative(record["file"], "sealed-cache file"),
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
        )
    for record in files:
        source_path = root / record["relative"]
        if (
            not source_path.is_file()
            or source_path.stat().st_size != record["bytes"]
            or sha256_file(source_path) != record["sha256"]
        ):
            raise RuntimeError(f"{job_id}: source identity changed: {record['relative']}")
        destination = destination_root / relative_root / record["relative"]
        safe_link(source_path, destination)
        if destination.stat().st_size != record["bytes"] or sha256_file(destination) != record["sha256"]:
            raise RuntimeError(f"{job_id}: staged identity mismatch: {record['relative']}")
        staged.append(
            {
                "path": (relative_root / record["relative"]).as_posix(),
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
        )
    return len(index["cache_manifests"]), len(index["sealed_cache_files"]), staged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--dataset-id",
        default="aviadcohen1/vesuvius-fusion-runpod-primary-caches",
    )
    parser.add_argument(
        "--title",
        default="Vesuvius Fusion RunPod Primary Caches",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("dataset staging directory must start absent")
    if args.dataset_id.count("/") != 1 or not all(args.dataset_id.split("/")):
        raise RuntimeError("invalid Kaggle dataset ID")
    records = load_hashed(args.records)
    if records.get("status") != (
        "seven authoritative RunPod primary deliveries verified without opening NPZ"
    ):
        raise RuntimeError("wrong RunPod delivery-record status")
    if records.get("counts") != {
        "jobs": 7,
        "cache_manifests": 70,
        "sealed_cache_files": 700,
    }:
        raise RuntimeError("wrong RunPod delivery counts")
    if records.get("scientific_gate") != {
        "all_primary_jobs_complete": True,
        "all_copied_cache_hashes_verified": True,
        "npz_cache_payloads_opened_or_inspected": False,
        "scientific_endpoints_opened_or_read": False,
        "runpod_is_authoritative_primary": True,
    }:
        raise RuntimeError("RunPod delivery blind gate mismatch")
    args.out.mkdir(parents=True)
    manifest_count = cache_count = 0
    staged = []
    for source in records["jobs"]:
        manifests, caches, identities = stage_job(source, args.out)
        manifest_count += manifests
        cache_count += caches
        staged.extend(identities)
        print(f"RUNPOD_DATASET_JOB_STAGED job={source['job_id']}", flush=True)
    if (manifest_count, cache_count, len(staged)) != (70, 700, 777):
        raise RuntimeError("staged RunPod dataset aggregate mismatch")
    ledger = args.out / "DATASET_SHA256SUMS"
    ledger.write_text(
        "".join(f"{record['sha256']}  {record['path']}\n" for record in staged),
        encoding="utf-8",
    )
    dataset_metadata = {
        "id": args.dataset_id,
        "title": args.title,
        "isPrivate": True,
        "licenses": [{"name": "other"}],
    }
    (args.out / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    stager = Path(__file__).resolve()
    payload = {
        "schema_version": "1.0",
        "status": "authoritative RunPod caches staged for private hash-bound transport",
        "dataset_id": args.dataset_id,
        "source_records": {
            "file": args.records.name,
            "bytes": args.records.stat().st_size,
            "sha256": sha256_file(args.records),
            "payload_sha256": records["payload_sha256"],
        },
        "counts": {
            "jobs": 7,
            "cache_manifests": manifest_count,
            "sealed_cache_files": cache_count,
            "staged_scientific_files": len(staged),
        },
        "dataset_ledger": {
            "file": ledger.name,
            "bytes": ledger.stat().st_size,
            "sha256": sha256_file(ledger),
        },
        "stager": {
            "file": stager.name,
            "bytes": stager.stat().st_size,
            "sha256": sha256_file(stager),
        },
        "scientific_gate": {
            "all_staged_files_rehashed": True,
            "npz_payloads_opened_or_inspected": False,
            "scientific_endpoints_opened_or_read": False,
            "dataset_private": True,
            "transport_changes_execution_origin": False,
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    (args.out / "runpod_primary_dataset_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RUNPOD_PRIMARY_PRIVATE_DATASET_STAGED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
