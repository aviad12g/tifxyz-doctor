#!/usr/bin/env python3
"""Pull and verify the exact private real-cache transport on CPU-only RunPod."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


DATASET_ID = "aviadcohen1/vesuvius-fusion-real-heldout-transport-v1"
DATASET_VERSION = 1
DATASET_HANDLE = f"{DATASET_ID}/versions/{DATASET_VERSION}"
EXPECTED_SOURCE_FILES = 281
EXPECTED_TOTAL_FILES = 284
EXPECTED_TOTAL_BYTES = 5_791_288_517
EXPECTED_PACKAGES = {
    "certifi": "2026.7.22",
    "charset-normalizer": "3.5.0",
    "idna": "3.18",
    "kagglehub": "1.0.2",
    "kagglesdk": "0.1.37",
    "packaging": "26.3",
    "protobuf": "7.35.1",
    "PyYAML": "6.0.3",
    "requests": "2.34.2",
    "tqdm": "4.70.0",
    "urllib3": "2.7.0",
}
KAGGLEHUB_SOURCE_HASHES = {
    "datasets.py": "53d1cf09f135bb39dc0aa5d894c938c32dbce133d2c5578e92a90483f324ab03",
    "datasets_helpers.py": "00b786848f516b64692557e10a7f4bbbdfbd2ade0d9fa6ea4cf64e4463c0de3b",
    "gcs_upload.py": "9873897f1825980bdf0c808162e67031d1d84aa633c9f52677553168b0aaf95f",
}
KAGGLEHUB_COMPLETION_MARKER = Path(
    ".complete/datasets/aviadcohen1/vesuvius-fusion-real-heldout-transport-v1/1/bundle.complete"
)


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


def file_identity(record: dict) -> dict:
    return {field: record[field] for field in ("file", "bytes", "sha256")}


def safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise RuntimeError(f"unsafe transport path: {value!r}")
    return Path(*pure.parts)


def tree_identity(root: Path) -> dict:
    records = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("runtime tree contains a symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        records.append(f"{sha256_file(path)}  {size}  {relative}\n")
        total += size
    return {
        "files": len(records),
        "bytes": total,
        "ledger_sha256": hashlib.sha256("".join(records).encode()).hexdigest(),
    }


def install_runtime(python: Path, wheelhouse: Path, root: Path) -> None:
    if root.exists():
        raise RuntimeError("KaggleHub runtime root must start absent")
    wheels = sorted(wheelhouse.glob("*.whl"))
    if len(wheels) != len(EXPECTED_PACKAGES):
        raise RuntimeError("KaggleHub wheelhouse file count mismatch")
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--disable-pip-version-check",
            "--target",
            str(root),
            *map(str, wheels),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def validate_runtime(root: Path) -> None:
    distributions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions(path=[str(root)])
    }
    if {name: distributions.get(name) for name in EXPECTED_PACKAGES} != EXPECTED_PACKAGES:
        raise RuntimeError("KaggleHub runtime package versions mismatch")
    package = root / "kagglehub"
    if {
        name: sha256_file(package / name)
        for name in KAGGLEHUB_SOURCE_HASHES
    } != KAGGLEHUB_SOURCE_HASHES:
        raise RuntimeError("KaggleHub runtime source identity mismatch")


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
            raise RuntimeError("invalid or duplicate private-transport ledger record")
        safe_relative(relative)
        records[relative] = digest
    return records


def expected_source_records(root: Path, manifest: dict) -> dict[str, tuple[int, str]]:
    records = {
        "runpod_real_input_manifest.json": (
            manifest["source_manifest"]["bytes"],
            manifest["source_manifest"]["sha256"],
        )
    }
    source = load_hashed(root / "runpod_real_input_manifest.json")
    if source["payload_sha256"] != manifest["source_manifest"]["payload_sha256"]:
        raise RuntimeError("private transport source-manifest payload mismatch")
    jobs = source.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 7:
        raise RuntimeError("private transport source-manifest job count mismatch")
    npz = 0
    for job in jobs:
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or safe_relative(job_id).as_posix() != job_id:
            raise RuntimeError("unsafe private transport job id")
        job_root = Path("jobs") / job_id
        for key in ("job_index", "cache_manifest"):
            record = job[key]
            relative = (job_root / safe_relative(record["file"])).as_posix()
            records[relative] = (record["bytes"], record["sha256"])
        for record in job["sealed_cache_files"]:
            relative = (
                job_root / "sealed-caches" / safe_relative(record["file"])
            ).as_posix()
            records[relative] = (record["bytes"], record["sha256"])
            npz += 1
    if len(records) != EXPECTED_SOURCE_FILES or npz != 266:
        raise RuntimeError("private transport source record count mismatch")
    return records


def validate_download(root: Path, transport: dict) -> dict[str, tuple[int, str]]:
    manifest_path = root / "real_transport_dataset_manifest.json"
    ledger_path = root / "REAL_TRANSPORT_SHA256SUMS"
    metadata_path = root / "dataset-metadata.json"
    if identity(manifest_path) != file_identity(transport["staging_manifest"]):
        raise RuntimeError("private transport staging-manifest identity mismatch")
    manifest = load_hashed(manifest_path)
    if manifest["payload_sha256"] != transport["staging_manifest"]["payload_sha256"]:
        raise RuntimeError("private transport staging-manifest payload mismatch")
    if manifest.get("dataset_id") != DATASET_ID or manifest.get("scientific_gate") != {
        "all_staged_files_rehashed": True,
        "dataset_private": True,
        "npz_payloads_opened_or_inspected": False,
        "scientific_endpoints_opened_or_read": False,
        "transport_only": True,
    }:
        raise RuntimeError("private transport scientific gate mismatch")
    if identity(ledger_path) != {
        "file": manifest["dataset_ledger"]["file"],
        "bytes": manifest["dataset_ledger"]["bytes"],
        "sha256": manifest["dataset_ledger"]["sha256"],
    }:
        raise RuntimeError("private transport ledger identity mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata != {
        "id": DATASET_ID,
        "isPrivate": True,
        "licenses": [{"name": "other"}],
        "title": "Vesuvius Fusion Real Held-Out Transport V1",
    }:
        raise RuntimeError("private transport metadata mismatch")
    ledger = parse_ledger(ledger_path)
    expected = expected_source_records(root, manifest)
    if ledger != {relative: digest for relative, (_, digest) in expected.items()}:
        raise RuntimeError("private transport ledger/source-manifest disagreement")
    expected_names = set(expected) | {manifest_path.name, ledger_path.name, metadata_path.name}
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != expected_names or len(observed) != EXPECTED_TOTAL_FILES:
        raise RuntimeError("private transport downloaded file set mismatch")
    if sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) != EXPECTED_TOTAL_BYTES:
        raise RuntimeError("private transport downloaded byte count mismatch")
    for relative, (size, digest) in expected.items():
        path = root / safe_relative(relative)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != size
            or sha256_file(path) != digest
        ):
            raise RuntimeError(f"private transport downloaded identity mismatch: {relative}")
    return expected


def remove_kagglehub_completion_marker(root: Path) -> None:
    """Remove only KaggleHub's pinned, non-dataset completion marker."""
    completion_root = root / ".complete"
    marker = root / KAGGLEHUB_COMPLETION_MARKER
    observed = {
        path.relative_to(root)
        for path in completion_root.rglob("*")
        if path.is_file() or path.is_symlink()
    } if completion_root.is_dir() and not completion_root.is_symlink() else set()
    if (
        observed != {KAGGLEHUB_COMPLETION_MARKER}
        or not marker.is_file()
        or marker.is_symlink()
        or marker.stat().st_size != 0
    ):
        raise RuntimeError("unexpected KaggleHub completion-marker layout")
    shutil.rmtree(completion_root)


def materialize_input(download: Path, target: Path, records: dict[str, tuple[int, str]]) -> None:
    if target.exists():
        raise RuntimeError("sealed real input root must start absent")
    target.mkdir(parents=True)
    for relative in sorted(records):
        source = download / safe_relative(relative)
        destination = target / safe_relative(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    observed = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    }
    if observed != set(records):
        raise RuntimeError("materialized sealed input file set mismatch")


def write_status(path: Path, state: str, **extra: object) -> None:
    payload = {
        "schema_version": "1.0",
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_outputs_inspected": False,
        **extra,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def remove_credentials(path: Path) -> None:
    if path.exists():
        path.unlink()
    try:
        path.parent.rmdir()
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--status-root", type=Path, required=True)
    args = parser.parse_args()
    if args.status_root.exists():
        raise RuntimeError("private transport status root must start absent")
    args.status_root.mkdir(parents=True)
    status_path = args.status_root / "transport-status.json"
    try:
        plan = load_hashed(args.plan)
        transport = plan["private_kaggle_transport"]
        if (
            transport.get("dataset_id") != DATASET_ID
            or transport.get("dataset_version") != DATASET_VERSION
            or transport.get("dataset_handle") != DATASET_HANDLE
            or transport.get("expected_total_files") != EXPECTED_TOTAL_FILES
            or transport.get("expected_total_bytes") != EXPECTED_TOTAL_BYTES
        ):
            raise RuntimeError("private Kaggle transport contract mismatch")
        if identity(Path(__file__).resolve()) != plan["private_transport_puller"]:
            raise RuntimeError("private transport puller differs from frozen plan")
        observed_python = subprocess.run(
            [str(args.python), "-c", "import sys; print(sys.version_info[:3])"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if observed_python != "(3, 12, 13)":
            raise RuntimeError("private transport Python version mismatch")
        if tree_identity(args.wheelhouse) != transport["kagglehub_wheelhouse_tree"]:
            raise RuntimeError("KaggleHub wheelhouse identity mismatch")
        install_runtime(args.python, args.wheelhouse, args.runtime_root)
        validate_runtime(args.runtime_root)
        if (
            not args.credentials.is_file()
            or args.credentials.is_symlink()
            or stat.S_IMODE(args.credentials.stat().st_mode) & 0o077
        ):
            raise RuntimeError("ephemeral Kaggle credentials are absent or too permissive")
        write_status(
            status_path,
            "DOWNLOADING_PRIVATE_TRANSPORT",
            plan_payload_sha256=plan["payload_sha256"],
            dataset_handle=DATASET_HANDLE,
        )
        credential_record = json.loads(args.credentials.read_text(encoding="utf-8"))
        if (
            not isinstance(credential_record, dict)
            or set(credential_record) != {"username", "key"}
            or not isinstance(credential_record["username"], str)
            or not credential_record["username"]
            or not isinstance(credential_record["key"], str)
            or not credential_record["key"]
        ):
            raise RuntimeError("ephemeral Kaggle credential JSON has the wrong schema")
        os.environ.pop("KAGGLE_API_TOKEN", None)
        os.environ.pop("KAGGLE_USERNAME", None)
        os.environ.pop("KAGGLE_KEY", None)
        os.environ["KAGGLEHUB_VERBOSITY"] = "error"
        sys.path.insert(0, str(args.runtime_root))
        import kagglehub  # noqa: PLC0415
        from kagglehub.auth import set_kaggle_credentials  # noqa: PLC0415

        set_kaggle_credentials(credential_record["username"], credential_record["key"])
        credential_record.clear()
        remove_credentials(args.credentials)
        if kagglehub.whoami(verbose=False).get("username") != DATASET_ID.partition("/")[0]:
            raise RuntimeError("ephemeral Kaggle credential owner mismatch")

        resolved = Path(
            kagglehub.dataset_download(
                DATASET_HANDLE,
                force_download=True,
                output_dir=str(args.download_root),
            )
        )
        if resolved.resolve() != args.download_root.resolve():
            raise RuntimeError("private transport download resolved to an unexpected root")
        remove_kagglehub_completion_marker(args.download_root)
        records = validate_download(args.download_root, transport)
        materialize_input(args.download_root, args.input_root, records)
        write_status(
            status_path,
            "PRIVATE_TRANSPORT_VERIFIED",
            plan_payload_sha256=plan["payload_sha256"],
            dataset_handle=DATASET_HANDLE,
            verified_files=EXPECTED_TOTAL_FILES,
            verified_bytes=EXPECTED_TOTAL_BYTES,
            materialized_source_files=EXPECTED_SOURCE_FILES,
            credentials_removed=True,
        )
        (args.status_root / "PRIVATE_TRANSPORT_VERIFIED").write_text(
            plan["payload_sha256"] + "\n", encoding="utf-8"
        )
        return 0
    except Exception as error:
        remove_credentials(args.credentials)
        write_status(
            status_path,
            "ERROR",
            error_type=type(error).__name__,
            error_message=str(error),
            credentials_removed=not args.credentials.exists(),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
