#!/usr/bin/env python3
"""Collect the seven sealed real-cache trees for a RunPod scoring retry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
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


def identity(path: Path) -> dict:
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def download_caches(kaggle: str, kernel_id: str, names: list[str], destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"download destination must start absent: {destination}")
    pattern = r"(^|/)(" + "|".join(re.escape(name) for name in names) + r")$"
    completed = subprocess.run(
        [
            kaggle,
            "kernels",
            "output",
            kernel_id,
            "-p",
            str(destination),
            "--file-pattern",
            pattern,
            "--page-size",
            "1000",
            "--page-token",
            "",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    if completed.returncode != 0:
        message = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
        raise RuntimeError(f"sealed-cache download failed for {kernel_id}: {message}")


def locate_single(root: Path, name: str) -> Path:
    matches = [path.resolve() for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name!r} under {root}; found {matches}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kaggle", default="kaggle")
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("RunPod real-input collection must start absent")

    sources = load_hashed(args.source_records)
    delivery = load_hashed(args.delivery)
    if sources.get("status") != "seven real held-out delivery indexes and manifests collected without NPZ":
        raise RuntimeError("wrong real delivery-source record status")
    real_jobs = [record for record in delivery.get("jobs", []) if record.get("mode") == "real_test_cache"]
    if len(real_jobs) != 7 or delivery.get("counts", {}).get("real_probability_caches") != 266:
        raise RuntimeError("public delivery does not bind seven real jobs and 266 caches")
    source_jobs = sources.get("jobs")
    if not isinstance(source_jobs, list) or [item.get("job_id") for item in source_jobs] != [
        item.get("job_id") for item in real_jobs
    ]:
        raise RuntimeError("real source-record order differs from public delivery")

    args.out.mkdir(parents=True)
    jobs_root = args.out / "jobs"
    jobs_root.mkdir()
    records = []
    aggregate_files = 0
    aggregate_bytes = 0
    for source, delivered in zip(source_jobs, real_jobs, strict=True):
        origin = source.get("execution_origin")
        if origin != delivered.get("execution_origin"):
            raise RuntimeError(f"{source.get('job_id')}: provider origin mismatch")
        index_source = Path(source["job_index"])
        if identity(index_source) != {
            key: delivered["job_index"][key] for key in ("file", "bytes", "sha256")
        }:
            raise RuntimeError(f"{source['job_id']}: local index identity mismatch")
        index = load_hashed(index_source)
        caches = index.get("sealed_cache_files")
        manifests = index.get("cache_manifests")
        if not isinstance(caches, list) or len(caches) != 38 or not isinstance(manifests, list) or len(manifests) != 1:
            raise RuntimeError(f"{source['job_id']}: index count mismatch")
        job_root = jobs_root / source["job_id"]
        cache_download = job_root / "download"
        job_root.mkdir()
        download_caches(args.kaggle, origin["kernel_id"], [item["file"] for item in caches], cache_download)
        cache_root = job_root / "sealed-caches"
        cache_root.mkdir()
        observed = []
        for expected in caches:
            downloaded = locate_single(cache_download, expected["file"])
            if downloaded.stat().st_size != expected["bytes"] or sha256_file(downloaded) != expected["sha256"]:
                raise RuntimeError(f"{source['job_id']}: sealed cache identity mismatch: {expected['file']}")
            destination = cache_root / expected["file"]
            shutil.move(str(downloaded), destination)
            observed.append(identity(destination))
        unexpected = [path for path in cache_download.rglob("*") if path.is_file()]
        if unexpected:
            raise RuntimeError(f"{source['job_id']}: unexpected downloaded files: {unexpected}")
        shutil.rmtree(cache_download)
        index_destination = job_root / index_source.name
        shutil.copy2(index_source, index_destination)
        manifest_source = locate_single(index_source.parent, manifests[0]["file"])
        if identity(manifest_source) != {
            key: manifests[0][key] for key in ("file", "bytes", "sha256")
        }:
            raise RuntimeError(f"{source['job_id']}: local manifest identity mismatch")
        shutil.copy2(manifest_source, job_root / manifest_source.name)
        aggregate_files += len(observed)
        aggregate_bytes += sum(item["bytes"] for item in observed)
        records.append(
            {
                "job_id": source["job_id"],
                "execution_origin": origin,
                "job_index": identity(index_destination) | {"payload_sha256": index["payload_sha256"]},
                "cache_manifest": identity(job_root / manifest_source.name) | {
                    "payload_sha256": manifests[0]["payload_sha256"]
                },
                "sealed_cache_files": observed,
            }
        )
        print(f"RUNPOD_REAL_SEALED_INPUT_COLLECTED job={source['job_id']}", flush=True)
    if aggregate_files != 266 or aggregate_bytes != 5_791_045_122:
        raise RuntimeError("RunPod real-input aggregate mismatch")
    payload = {
        "schema_version": "1.0",
        "status": "seven sealed Kaggle real-cache deliveries collected for authorized RunPod transport",
        "public_delivery": identity(args.delivery) | {"payload_sha256": delivery["payload_sha256"]},
        "source_records": identity(args.source_records) | {"payload_sha256": sources["payload_sha256"]},
        "counts": {"jobs": 7, "manifests": 7, "sealed_npz_caches": 266, "sealed_npz_bytes": aggregate_bytes},
        "jobs": records,
        "collector": identity(Path(__file__).resolve()),
        "scientific_gate": {
            "npz_payloads_opened_or_parsed": False,
            "scientific_endpoints_opened_or_read": False,
            "kernel_logs_downloaded_or_read": False,
            "only_expected_npz_files_downloaded": True,
            "all_cache_hashes_verified": True,
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    (args.out / "runpod_real_input_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RUNPOD_REAL_SEALED_INPUTS_READY_WITHOUT_NPZ_ACCESS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
