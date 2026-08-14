#!/usr/bin/env python3
"""Upload and monitor the exact private real-cache Kaggle transport dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path, PurePosixPath


DATASET_ID = "aviadcohen1/vesuvius-fusion-real-heldout-transport-v1"
EXPECTED_SOURCE_FILES = 281
EXPECTED_TOTAL_FILES = 284
KAGGLEHUB_VERSION = "1.0.2"
KAGGLEHUB_SOURCE_HASHES = {
    "datasets.py": "53d1cf09f135bb39dc0aa5d894c938c32dbce133d2c5578e92a90483f324ab03",
    "gcs_upload.py": "9873897f1825980bdf0c808162e67031d1d84aa633c9f52677553168b0aaf95f",
    "datasets_helpers.py": "00b786848f516b64692557e10a7f4bbbdfbd2ade0d9fa6ea4cf64e4463c0de3b",
}


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
            raise RuntimeError("invalid or duplicate real-transport ledger record")
        safe_relative(relative)
        records[relative] = digest
    return records


def validate_staging(root: Path) -> tuple[dict, dict[str, str]]:
    manifest_path = root / "real_transport_dataset_manifest.json"
    ledger_path = root / "REAL_TRANSPORT_SHA256SUMS"
    metadata_path = root / "dataset-metadata.json"
    manifest = load_hashed(manifest_path)
    if manifest.get("status") != (
        "exact sealed real held-out caches staged for private Kaggle transport"
    ) or manifest.get("dataset_id") != DATASET_ID:
        raise RuntimeError("wrong private real-transport staging identity")
    if manifest.get("counts") != {
        "jobs": 7,
        "manifests": 7,
        "sealed_npz_caches": 266,
        "staged_source_files": EXPECTED_SOURCE_FILES,
        "total_dataset_files": EXPECTED_TOTAL_FILES,
    }:
        raise RuntimeError("wrong private real-transport staging counts")
    if manifest.get("scientific_gate") != {
        "all_staged_files_rehashed": True,
        "npz_payloads_opened_or_inspected": False,
        "scientific_endpoints_opened_or_read": False,
        "dataset_private": True,
        "transport_only": True,
    }:
        raise RuntimeError("private real-transport blind gate mismatch")
    if manifest.get("dataset_ledger") != {
        "file": ledger_path.name,
        "bytes": ledger_path.stat().st_size,
        "sha256": sha256_file(ledger_path),
    }:
        raise RuntimeError("private real-transport ledger identity mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata != {
        "id": DATASET_ID,
        "isPrivate": True,
        "licenses": [{"name": "other"}],
        "title": "Vesuvius Fusion Real Held-Out Transport V1",
    }:
        raise RuntimeError("private real-transport metadata mismatch")
    ledger = parse_ledger(ledger_path)
    if len(ledger) != EXPECTED_SOURCE_FILES:
        raise RuntimeError("private real-transport ledger count mismatch")
    for relative, digest in ledger.items():
        path = root / safe_relative(relative)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise RuntimeError(f"private real-transport identity mismatch: {relative}")
    expected = set(ledger) | {manifest_path.name, ledger_path.name, metadata_path.name}
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != expected or len(observed) != EXPECTED_TOTAL_FILES:
        raise RuntimeError("private real-transport file set mismatch")
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
    result = run_cli([kaggle, "datasets", "list", "-m", "-s", DATASET_ID.split("/", 1)[1], "--csv"])
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0 or combined.strip() != "No datasets found":
        raise RuntimeError("private real-transport dataset is not provably absent")


def write_receipt(path: Path, payload: dict) -> None:
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def upload(args: argparse.Namespace) -> None:
    if args.receipt.exists():
        raise RuntimeError("private real-transport receipt already exists; use status")
    manifest, ledger = validate_staging(args.staging)
    require_absent_dataset(args.kaggle)
    kagglehub, gcs_upload = validate_kagglehub_runtime()
    uploader = Path(__file__).resolve()
    receipt = {
        "schema_version": "1.0",
        "status": "private real-transport upload intent recorded before provider mutation",
        "dataset_id": DATASET_ID,
        "expected_version": 1,
        "staging_manifest": {
            "file": "real_transport_dataset_manifest.json",
            "bytes": (args.staging / "real_transport_dataset_manifest.json").stat().st_size,
            "sha256": sha256_file(args.staging / "real_transport_dataset_manifest.json"),
            "payload_sha256": manifest["payload_sha256"],
        },
        "file_counts": {"source": len(ledger), "total": EXPECTED_TOTAL_FILES},
        "runtime": {
            "package": "kagglehub",
            "version": KAGGLEHUB_VERSION,
            "source_sha256": KAGGLEHUB_SOURCE_HASHES,
            "nested_file_threshold_override": 500,
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
            "transport_only": True,
        },
    }
    write_receipt(args.receipt, receipt)
    gcs_upload.MAX_FILES_TO_UPLOAD = 500
    kagglehub.dataset_upload(
        DATASET_ID,
        str(args.staging),
        version_notes="Exact sealed real held-out caches; private transport only",
    )
    receipt["status"] = "private real-transport upload accepted; provider processing pending"
    receipt["provider_status"] = "processing"
    write_receipt(args.receipt, receipt)
    print("PRIVATE_REAL_TRANSPORT_UPLOAD_ACCEPTED", flush=True)


def status(args: argparse.Namespace) -> None:
    receipt = load_hashed(args.receipt)
    manifest, _ = validate_staging(args.staging)
    if (
        receipt.get("dataset_id") != DATASET_ID
        or receipt.get("expected_version") != 1
        or receipt.get("staging_manifest", {}).get("payload_sha256") != manifest["payload_sha256"]
    ):
        raise RuntimeError("private real-transport receipt binding mismatch")
    result = run_cli([args.kaggle, "datasets", "status", DATASET_ID, "--format", "json"])
    if result.returncode != 0:
        raise RuntimeError("private real-transport status query failed")
    provider = json.loads(result.stdout.strip())
    if provider.get("current_version_number") != 1:
        raise RuntimeError("private real-transport provider version is not exactly one")
    receipt["provider_status"] = provider["status"]
    if provider["status"] == "ready":
        receipt["status"] = "private real-transport dataset version 1 ready"
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
