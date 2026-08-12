"""Result-blind, fixed-order controller for the 14 held-out cache kernels.

The controller validates every generated package against the package index,
queries only Kaggle operational status, and fills at most two active GPU slots.
It never downloads kernel outputs or reads scientific artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ACTIVE_STATUSES = {"PENDING", "QUEUED", "RUNNING"}
FAILED_STATUSES = {"ERROR", "CANCELED", "CANCELLED"}
EXPECTED_PACKAGE_COUNT = 14
MAX_ACTIVE = 2
EMBEDDED_CONFIG_PREFIX = b"# HELDOUT_JOB_CONFIG_HEX="
STATUS_PATTERN = re.compile(r"KernelWorkerStatus[.]([A-Z_]+)")
VERSION_PATTERN = re.compile(r"Kernel version\s+(\d+)\s+successfully pushed")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def load_canonical(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    observed = payload.get("payload_sha256")
    content = dict(payload)
    content.pop("payload_sha256", None)
    if observed != canonical_sha256(content):
        raise RuntimeError(f"canonical payload SHA-256 mismatch: {path}")
    return payload


def validate_packages(root: Path, index_path: Path) -> tuple[dict, list[dict]]:
    index = load_canonical(index_path)
    if (
        index.get("status")
        != "all private held-out cache packages generated from public plan"
    ):
        raise RuntimeError("wrong generated-package index status")
    packages = index.get("packages")
    if not isinstance(packages, list) or len(packages) != EXPECTED_PACKAGE_COUNT:
        raise RuntimeError("generated-package index must contain exactly 14 packages")
    if index.get("package_count") != EXPECTED_PACKAGE_COUNT:
        raise RuntimeError("generated-package count field mismatch")
    identities = [record.get("kaggle_kernel_id") for record in packages]
    if len(set(identities)) != EXPECTED_PACKAGE_COUNT or not all(
        isinstance(identity, str) and identity.startswith("aviadcohen1/")
        for identity in identities
    ):
        raise RuntimeError("generated Kaggle kernel identities are invalid")
    job_ids = [record.get("job_id") for record in packages]
    if len(set(job_ids)) != EXPECTED_PACKAGE_COUNT:
        raise RuntimeError("generated job identities are not unique")
    actual_directories = {path.name for path in root.iterdir() if path.is_dir()}
    expected_directories = {record["directory"] for record in packages}
    if actual_directories != expected_directories:
        raise RuntimeError("package-root directory set differs from package index")
    launcher = index.get("launcher")
    if (
        not isinstance(launcher, dict)
        or set(launcher) != {"file", "bytes", "sha256"}
        or launcher.get("file") != "heldout_cache_launcher.py"
    ):
        raise RuntimeError("generated-package launcher identity is invalid")
    for record in packages:
        if set(record) != {
            "job_id",
            "mode",
            "kaggle_kernel_id",
            "directory",
            "files",
            "embedded_job_config",
        }:
            raise RuntimeError(
                f"{record.get('job_id')}: package record schema mismatch"
            )
        directory = root / record["directory"]
        files = record.get("files")
        if not isinstance(files, dict) or "KERNEL_SHA256SUMS" not in files:
            raise RuntimeError(f"{record['job_id']}: incomplete package file ledger")
        observed_names = {path.name for path in directory.iterdir() if path.is_file()}
        if observed_names != set(files):
            raise RuntimeError(f"{record['job_id']}: package file set mismatch")
        for name, identity in files.items():
            path = directory / name
            if path.stat().st_size != int(identity.get("bytes", -1)):
                raise RuntimeError(f"{record['job_id']}: byte-size mismatch for {name}")
            if sha256_file(path) != identity.get("sha256"):
                raise RuntimeError(f"{record['job_id']}: SHA-256 mismatch for {name}")
        expected_ledger = "".join(
            f"{files[name]['sha256']}  {name}\n"
            for name in sorted(files)
            if name != "KERNEL_SHA256SUMS"
        )
        if (directory / "KERNEL_SHA256SUMS").read_text(
            encoding="utf-8"
        ) != expected_ledger:
            raise RuntimeError(f"{record['job_id']}: checksum ledger content mismatch")
        wrapper = (directory / launcher["file"]).read_bytes()
        first_line, separator, base_source = wrapper.partition(b"\n")
        if not first_line.startswith(EMBEDDED_CONFIG_PREFIX) or not separator:
            raise RuntimeError(f"{record['job_id']}: embedded config line is absent")
        if (
            len(base_source) != launcher["bytes"]
            or sha256_bytes(base_source) != launcher["sha256"]
        ):
            raise RuntimeError(f"{record['job_id']}: base launcher identity mismatch")
        try:
            config_raw = bytes.fromhex(
                first_line[len(EMBEDDED_CONFIG_PREFIX) :].decode("ascii")
            )
            config = json.loads(config_raw)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"{record['job_id']}: embedded config cannot be decoded"
            ) from error
        config_content = dict(config)
        observed_payload = config_content.pop("payload_sha256", None)
        if observed_payload != canonical_sha256(config_content):
            raise RuntimeError(f"{record['job_id']}: embedded config payload mismatch")
        if config != {
            "schema_version": "1.0",
            "job_id": record["job_id"],
            "public_plan_commit": index["public_plan_commit"],
            "public_plan_file_sha256": index["public_plan_file_sha256"],
            "payload_sha256": observed_payload,
        }:
            raise RuntimeError(f"{record['job_id']}: embedded config identity mismatch")
        expected_config = record["embedded_job_config"]
        if expected_config != {
            "encoding": "hex in first source comment",
            "bytes": len(config_raw),
            "sha256": sha256_bytes(config_raw),
            "payload_sha256": observed_payload,
        }:
            raise RuntimeError(f"{record['job_id']}: embedded config ledger mismatch")
    return index, packages


def run_cli(command: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def owned_kernel_exists(kaggle: str, kernel_id: str) -> bool:
    result = run_cli(
        [
            kaggle,
            "kernels",
            "list",
            "--mine",
            "--page-size",
            "200",
            "--search",
            kernel_id.rsplit("/", 1)[-1],
            "--csv",
        ]
    )
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle owned-kernel lookup failed for {kernel_id}: {combined.strip()}"
        )
    rows = csv.DictReader(io.StringIO(result.stdout))
    return any(row.get("ref") == kernel_id for row in rows)


def kernel_status(
    kaggle: str, kernel_id: str, *, allow_unaccepted_absence: bool
) -> str | None:
    result = run_cli([kaggle, "kernels", "status", kernel_id])
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        lowered = combined.lower()
        if "404" in lowered or "not found" in lowered:
            return None
        if (
            allow_unaccepted_absence
            and "permission 'kernels.get' was denied" in lowered
            and not owned_kernel_exists(kaggle, kernel_id)
        ):
            return None
        raise RuntimeError(
            f"Kaggle status query failed for {kernel_id}: {combined.strip()}"
        )
    match = STATUS_PATTERN.search(combined)
    if match is None:
        raise RuntimeError(f"Kaggle status output was not recognized for {kernel_id}")
    return match.group(1)


def empty_receipt(index_path: Path, index: dict) -> dict:
    payload = {
        "schema_version": "1.0",
        "status": "held-out cache queue acceptance receipt",
        "generated_packages_index": {
            "file": index_path.name,
            "bytes": index_path.stat().st_size,
            "sha256": sha256_file(index_path),
            "payload_sha256": index["payload_sha256"],
        },
        "accepted": [],
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def load_receipt(path: Path, index_path: Path, index: dict) -> dict:
    if not path.exists():
        return empty_receipt(index_path, index)
    payload = load_canonical(path)
    expected_index = {
        "file": index_path.name,
        "bytes": index_path.stat().st_size,
        "sha256": sha256_file(index_path),
        "payload_sha256": index["payload_sha256"],
    }
    if payload.get("status") != "held-out cache queue acceptance receipt":
        raise RuntimeError("queue receipt status mismatch")
    if payload.get("generated_packages_index") != expected_index:
        raise RuntimeError("queue receipt points to another package index")
    accepted = payload.get("accepted")
    if not isinstance(accepted, list):
        raise TypeError("queue receipt accepted records are invalid")
    return payload


def write_receipt(path: Path, receipt: dict) -> None:
    content = dict(receipt)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def receipt_map(receipt: dict, packages: list[dict]) -> dict[str, dict]:
    records = receipt["accepted"]
    mapped = {record.get("job_id"): record for record in records}
    if len(mapped) != len(records):
        raise RuntimeError("queue receipt contains duplicate job identities")
    expected_prefix = [record["job_id"] for record in packages[: len(records)]]
    if [record.get("job_id") for record in records] != expected_prefix:
        raise RuntimeError("queue receipt does not preserve the fixed package order")
    packages_by_job = {record["job_id"]: record for record in packages}
    for record in records:
        if set(record) != {
            "job_id",
            "kernel_id",
            "kernel_version",
            "package_ledger_sha256",
            "accepted_at_utc",
        }:
            raise RuntimeError("queue receipt record schema mismatch")
        package = packages_by_job[record["job_id"]]
        if record["kernel_id"] != package["kaggle_kernel_id"]:
            raise RuntimeError(f"{record['job_id']}: receipt kernel identity mismatch")
        if (
            not isinstance(record["kernel_version"], int)
            or record["kernel_version"] <= 0
        ):
            raise RuntimeError(f"{record['job_id']}: receipt kernel version is invalid")
        if (
            record["package_ledger_sha256"]
            != package["files"]["KERNEL_SHA256SUMS"]["sha256"]
        ):
            raise RuntimeError(f"{record['job_id']}: receipt package ledger mismatch")
        if not isinstance(record["accepted_at_utc"], str):
            raise TypeError(f"{record['job_id']}: receipt timestamp is invalid")
    return mapped


def validate_existing_prefix(
    packages: list[dict], statuses: list[str | None], accepted: dict[str, dict]
) -> None:
    missing_seen = False
    for record, status in zip(packages, statuses, strict=True):
        job_id = record["job_id"]
        receipt_record = accepted.get(job_id)
        if status is None:
            missing_seen = True
            if receipt_record is not None:
                raise RuntimeError(
                    f"{job_id}: receipt exists but Kaggle kernel is absent"
                )
            continue
        if missing_seen:
            raise RuntimeError(f"{job_id}: existing kernel violates fixed push order")
        if receipt_record is None:
            raise RuntimeError(f"{job_id}: existing kernel lacks an acceptance receipt")
        if receipt_record.get("kernel_id") != record["kaggle_kernel_id"]:
            raise RuntimeError(f"{job_id}: receipt kernel identity mismatch")
        if not isinstance(receipt_record.get("kernel_version"), int):
            raise TypeError(f"{job_id}: receipt kernel version is invalid")


def push_package(kaggle: str, root: Path, record: dict) -> tuple[int, str]:
    directory = root / record["directory"]
    result = run_cli([kaggle, "kernels", "push", "-p", str(directory)], timeout=300)
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    match = VERSION_PATTERN.search(combined)
    if result.returncode != 0 or match is None:
        raise RuntimeError(
            f"{record['job_id']}: kernel push was not accepted: {combined.strip()}"
        )
    return int(match.group(1)), combined.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--kaggle", default="kaggle")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    index, packages = validate_packages(args.packages, args.index)
    receipt = load_receipt(args.receipt, args.index, index)
    accepted = receipt_map(receipt, packages)
    statuses = [
        kernel_status(
            args.kaggle,
            record["kaggle_kernel_id"],
            allow_unaccepted_absence=record["job_id"] not in accepted,
        )
        for record in packages
    ]
    validate_existing_prefix(packages, statuses, accepted)

    for record, status in zip(packages, statuses, strict=True):
        if status in FAILED_STATUSES:
            raise RuntimeError(
                f"{record['job_id']}: operational failure ({status}); inspect only logs"
            )
        if (
            status is not None
            and status != "COMPLETE"
            and status not in ACTIVE_STATUSES
        ):
            raise RuntimeError(f"{record['job_id']}: unsupported status {status}")

    active = sum(status in ACTIVE_STATUSES for status in statuses)
    if active > MAX_ACTIVE:
        raise RuntimeError(f"active GPU job count exceeds frozen cap: {active}")
    print(f"HELDOUT_QUEUE_STATUS complete={statuses.count('COMPLETE')} active={active}")
    if args.dry_run:
        print("HELDOUT_QUEUE_DRY_RUN_OK")
        return 0

    for index_position, (record, status) in enumerate(
        zip(packages, statuses, strict=True)
    ):
        if active >= MAX_ACTIVE:
            break
        if status is not None:
            continue
        version, _ = push_package(args.kaggle, args.packages, record)
        package_ledger = record["files"]["KERNEL_SHA256SUMS"]
        receipt["accepted"].append(
            {
                "job_id": record["job_id"],
                "kernel_id": record["kaggle_kernel_id"],
                "kernel_version": version,
                "package_ledger_sha256": package_ledger["sha256"],
                "accepted_at_utc": datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
            }
        )
        write_receipt(args.receipt, receipt)
        statuses[index_position] = "PENDING"
        active += 1
        print(
            f"HELDOUT_QUEUE_ACCEPTED job={record['job_id']} "
            f"version={version} url=https://www.kaggle.com/code/{record['kaggle_kernel_id']}"
        )

    if all(status == "COMPLETE" for status in statuses):
        print("ALL_HELDOUT_CACHE_KERNELS_COMPLETE")
    else:
        print("HELDOUT_QUEUE_NORMAL_PROGRESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
