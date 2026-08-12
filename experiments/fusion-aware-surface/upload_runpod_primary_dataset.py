#!/usr/bin/env python3
"""Upload a verified RunPod cache tree as one private nested Kaggle dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path, PurePosixPath


KAGGLEHUB_VERSION = "1.0.2"
KAGGLEHUB_SOURCE_HASHES = {
    "datasets.py": "53d1cf09f135bb39dc0aa5d894c938c32dbce133d2c5578e92a90483f324ab03",
    "gcs_upload.py": "9873897f1825980bdf0c808162e67031d1d84aa633c9f52677553168b0aaf95f",
    "datasets_helpers.py": "00b786848f516b64692557e10a7f4bbbdfbd2ade0d9fa6ea4cf64e4463c0de3b",
}
EXPECTED_DATASET_ID = "aviadcohen1/vesuvius-fusion-runpod-primary-caches"
EXPECTED_SCIENTIFIC_FILE_COUNT = 777
EXPECTED_TOTAL_FILE_COUNT = 780


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


def parse_ledger(path: Path) -> dict[str, str]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative in records
        ):
            raise RuntimeError("invalid or duplicate private-dataset ledger record")
        safe_relative(relative)
        records[relative] = digest
    return records


def validate_staging(root: Path) -> tuple[dict, dict[str, str]]:
    manifest_path = root / "runpod_primary_dataset_manifest.json"
    ledger_path = root / "DATASET_SHA256SUMS"
    metadata_path = root / "dataset-metadata.json"
    manifest = load_hashed(manifest_path)
    if manifest.get("status") != (
        "authoritative RunPod caches staged for private hash-bound transport"
    ):
        raise RuntimeError("wrong private-dataset staging status")
    if manifest.get("dataset_id") != EXPECTED_DATASET_ID:
        raise RuntimeError("wrong private-dataset ID")
    if manifest.get("counts") != {
        "jobs": 7,
        "cache_manifests": 70,
        "sealed_cache_files": 700,
        "staged_scientific_files": EXPECTED_SCIENTIFIC_FILE_COUNT,
    }:
        raise RuntimeError("wrong private-dataset staging counts")
    if manifest.get("scientific_gate") != {
        "all_staged_files_rehashed": True,
        "npz_payloads_opened_or_inspected": False,
        "scientific_endpoints_opened_or_read": False,
        "dataset_private": True,
        "transport_changes_execution_origin": False,
    }:
        raise RuntimeError("private-dataset staging blind gate mismatch")
    if manifest.get("dataset_ledger") != {
        "file": ledger_path.name,
        "bytes": ledger_path.stat().st_size,
        "sha256": sha256_file(ledger_path),
    }:
        raise RuntimeError("private-dataset ledger identity mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata != {
        "id": EXPECTED_DATASET_ID,
        "isPrivate": True,
        "licenses": [{"name": "other"}],
        "title": "Vesuvius Fusion RunPod Primary Caches",
    }:
        raise RuntimeError("private-dataset metadata mismatch")
    ledger = parse_ledger(ledger_path)
    if len(ledger) != EXPECTED_SCIENTIFIC_FILE_COUNT:
        raise RuntimeError("private-dataset ledger count mismatch")
    for relative, digest in ledger.items():
        path = root / safe_relative(relative)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise RuntimeError(f"private-dataset staged identity mismatch: {relative}")
    expected = set(ledger) | {
        manifest_path.name,
        ledger_path.name,
        metadata_path.name,
    }
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"private-dataset staging contains a symlink: {path}")
        if path.is_file():
            observed.add(path.relative_to(root).as_posix())
    if observed != expected or len(observed) != EXPECTED_TOTAL_FILE_COUNT:
        raise RuntimeError("private-dataset staging file set mismatch")
    return manifest, ledger


def validate_kagglehub_runtime():
    if importlib.metadata.version("kagglehub") != KAGGLEHUB_VERSION:
        raise RuntimeError("wrong KaggleHub upload runtime version")
    import kagglehub
    import kagglehub.datasets as datasets
    import kagglehub.datasets_helpers as datasets_helpers
    import kagglehub.gcs_upload as gcs_upload

    package = Path(kagglehub.__file__).resolve().parent
    modules = {
        "datasets.py": Path(datasets.__file__).resolve(),
        "gcs_upload.py": Path(gcs_upload.__file__).resolve(),
        "datasets_helpers.py": Path(datasets_helpers.__file__).resolve(),
    }
    if any(path.parent != package for path in modules.values()):
        raise RuntimeError("KaggleHub upload modules do not share one package root")
    if {name: sha256_file(path) for name, path in modules.items()} != KAGGLEHUB_SOURCE_HASHES:
        raise RuntimeError("KaggleHub upload runtime source identity mismatch")
    if gcs_upload.MAX_FILES_TO_UPLOAD != 50:
        raise RuntimeError("unexpected KaggleHub nested-upload threshold")
    return kagglehub, gcs_upload


def run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)


def require_absent_dataset(kaggle: str) -> None:
    result = run_cli(
        [
            kaggle,
            "datasets",
            "list",
            "-m",
            "-s",
            EXPECTED_DATASET_ID.split("/", 1)[1],
            "--csv",
        ]
    )
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0 or combined.strip() != "No datasets found":
        raise RuntimeError("private transport dataset is not provably absent")


def write_receipt(path: Path, payload: dict) -> None:
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def upload(args: argparse.Namespace) -> None:
    if args.receipt.exists():
        raise RuntimeError("private-dataset upload receipt already exists; use status")
    manifest, ledger = validate_staging(args.staging)
    require_absent_dataset(args.kaggle)
    kagglehub, gcs_upload = validate_kagglehub_runtime()
    uploader = Path(__file__).resolve()
    receipt = {
        "schema_version": "1.0",
        "status": "RunPod private dataset upload intent recorded before provider mutation",
        "dataset_id": EXPECTED_DATASET_ID,
        "expected_version": 1,
        "staging_manifest": {
            "file": "runpod_primary_dataset_manifest.json",
            "bytes": (args.staging / "runpod_primary_dataset_manifest.json").stat().st_size,
            "sha256": sha256_file(args.staging / "runpod_primary_dataset_manifest.json"),
            "payload_sha256": manifest["payload_sha256"],
        },
        "file_counts": {
            "scientific": len(ledger),
            "total": EXPECTED_TOTAL_FILE_COUNT,
        },
        "runtime": {
            "package": "kagglehub",
            "version": KAGGLEHUB_VERSION,
            "source_sha256": KAGGLEHUB_SOURCE_HASHES,
            "nested_file_threshold_override": 1000,
        },
        "uploader": {
            "file": uploader.name,
            "bytes": uploader.stat().st_size,
            "sha256": sha256_file(uploader),
        },
        "provider_status": None,
        "scientific_gate": {
            "npz_payloads_opened_or_inspected": False,
            "scientific_endpoints_opened_or_read": False,
            "nested_paths_preserved_without_archive_transport": True,
            "dataset_private": True,
            "runpod_primary_authority_changed": False,
        },
    }
    write_receipt(args.receipt, receipt)
    gcs_upload.MAX_FILES_TO_UPLOAD = 1000
    kagglehub.dataset_upload(
        EXPECTED_DATASET_ID,
        str(args.staging),
        version_notes="Frozen RunPod primary cache transport; scientific outputs remained sealed",
    )
    receipt["status"] = "RunPod private dataset upload accepted; provider processing pending"
    receipt["provider_status"] = "processing"
    write_receipt(args.receipt, receipt)
    print("RUNPOD_PRIMARY_PRIVATE_DATASET_UPLOAD_ACCEPTED", flush=True)


def status(args: argparse.Namespace) -> None:
    receipt = load_hashed(args.receipt)
    if (
        receipt.get("dataset_id") != EXPECTED_DATASET_ID
        or receipt.get("expected_version") != 1
        or receipt.get("staging_manifest", {}).get("payload_sha256")
        != validate_staging(args.staging)[0]["payload_sha256"]
    ):
        raise RuntimeError("private-dataset receipt binding mismatch")
    result = run_cli(
        [
            args.kaggle,
            "datasets",
            "status",
            EXPECTED_DATASET_ID,
            "--format",
            "json",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError("private-dataset provider status query failed")
    provider = json.loads(result.stdout.strip())
    if provider.get("current_version_number") != 1:
        raise RuntimeError("private-dataset provider version is not exactly one")
    receipt["provider_status"] = provider["status"]
    if provider["status"] == "ready":
        receipt["status"] = "RunPod private dataset version 1 ready for sealed scoring transport"
    write_receipt(args.receipt, receipt)
    print(json.dumps(provider, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("upload", "status"))
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--kaggle", default="kaggle")
    args = parser.parse_args()
    if args.command == "upload":
        upload(args)
    else:
        status(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
