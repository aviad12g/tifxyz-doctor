#!/usr/bin/env python3
"""Stage the exact sealed real held-out caches for private Kaggle transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath


DATASET_ID = "aviadcohen1/vesuvius-fusion-real-heldout-transport-v1"
EXPECTED_COUNTS = {"jobs": 7, "manifests": 7, "sealed_npz_caches": 266}
EXPECTED_SOURCE_FILES = 281
EXPECTED_TOTAL_FILES = 284


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


def safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise RuntimeError(f"unsafe dataset path: {value!r}")
    return Path(*pure.parts)


def expected_records(source: Path, manifest: dict) -> dict[str, str]:
    if manifest.get("status") != (
        "seven sealed Kaggle real-cache deliveries collected for authorized RunPod transport"
    ):
        raise RuntimeError("wrong sealed real-input status")
    counts = manifest.get("counts") or {}
    if {key: counts.get(key) for key in EXPECTED_COUNTS} != EXPECTED_COUNTS:
        raise RuntimeError("wrong sealed real-input counts")
    if manifest.get("scientific_gate") != {
        "all_cache_hashes_verified": True,
        "kernel_logs_downloaded_or_read": False,
        "npz_payloads_opened_or_parsed": False,
        "only_expected_npz_files_downloaded": True,
        "scientific_endpoints_opened_or_read": False,
    }:
        raise RuntimeError("sealed real-input scientific gate mismatch")
    records = {
        "runpod_real_input_manifest.json": sha256_file(
            source / "runpod_real_input_manifest.json"
        )
    }
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != EXPECTED_COUNTS["jobs"]:
        raise RuntimeError("wrong sealed real-input job count")
    npz_count = 0
    for job in jobs:
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or safe_relative(job_id).as_posix() != job_id:
            raise RuntimeError("unsafe or missing real job id")
        root = Path("jobs") / job_id
        for key in ("job_index", "cache_manifest"):
            identity = job.get(key)
            if not isinstance(identity, dict):
                raise RuntimeError(f"missing {key} identity")
            relative = (root / safe_relative(identity["file"])).as_posix()
            path = source / relative
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != identity["bytes"]
                or sha256_file(path) != identity["sha256"]
            ):
                raise RuntimeError(f"sealed identity mismatch: {relative}")
            records[relative] = identity["sha256"]
        caches = job.get("sealed_cache_files")
        if not isinstance(caches, list):
            raise RuntimeError("missing sealed cache identities")
        for identity in caches:
            relative = (root / "sealed-caches" / safe_relative(identity["file"])).as_posix()
            path = source / relative
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != identity["bytes"]
                or sha256_file(path) != identity["sha256"]
            ):
                raise RuntimeError(f"sealed cache identity mismatch: {relative}")
            records[relative] = identity["sha256"]
            npz_count += 1
    if npz_count != EXPECTED_COUNTS["sealed_npz_caches"] or len(records) != EXPECTED_SOURCE_FILES:
        raise RuntimeError("wrong sealed real transport record count")
    observed = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
    }
    if observed != set(records):
        raise RuntimeError("sealed real source contains an unexpected file set")
    return records


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def write_hashed(path: Path, payload: dict) -> dict:
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return content


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("transport staging output must start absent")
    source_manifest_path = args.source / "runpod_real_input_manifest.json"
    source_manifest = load_hashed(source_manifest_path)
    records = expected_records(args.source, source_manifest)
    args.out.mkdir(parents=True)
    for relative in sorted(records):
        link_or_copy(args.source / safe_relative(relative), args.out / safe_relative(relative))
    ledger = args.out / "REAL_TRANSPORT_SHA256SUMS"
    ledger.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in sorted(records.items())),
        encoding="utf-8",
    )
    metadata = {
        "id": DATASET_ID,
        "isPrivate": True,
        "licenses": [{"name": "other"}],
        "title": "Vesuvius Fusion Real Held-Out Transport V1",
    }
    (args.out / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = write_hashed(
        args.out / "real_transport_dataset_manifest.json",
        {
            "schema_version": "1.0",
            "status": "exact sealed real held-out caches staged for private Kaggle transport",
            "dataset_id": DATASET_ID,
            "source_manifest": {
                "file": source_manifest_path.name,
                "bytes": source_manifest_path.stat().st_size,
                "sha256": sha256_file(source_manifest_path),
                "payload_sha256": source_manifest["payload_sha256"],
            },
            "counts": {
                **EXPECTED_COUNTS,
                "staged_source_files": len(records),
                "total_dataset_files": EXPECTED_TOTAL_FILES,
            },
            "dataset_ledger": {
                "file": ledger.name,
                "bytes": ledger.stat().st_size,
                "sha256": sha256_file(ledger),
            },
            "scientific_gate": {
                "all_staged_files_rehashed": True,
                "npz_payloads_opened_or_inspected": False,
                "scientific_endpoints_opened_or_read": False,
                "dataset_private": True,
                "transport_only": True,
            },
        },
    )
    observed = [path for path in args.out.rglob("*") if path.is_file()]
    if len(observed) != EXPECTED_TOTAL_FILES:
        raise RuntimeError("wrong staged transport total file count")
    print(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "files": len(observed),
                "manifest_payload_sha256": manifest["payload_sha256"],
                "source_files": len(records),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
