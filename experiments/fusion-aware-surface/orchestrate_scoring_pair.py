"""Result-blind controller that launches both one-shot scorers as a pair."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import orchestrate_heldout_queue as queue

MODES = ("real", "synthetic")
EMBEDDED_CONFIG_PREFIX = b"# SCORING_JOB_CONFIG_HEX="


def validate_packages(root: Path, index_path: Path) -> tuple[dict, list[dict]]:
    index = queue.load_canonical(index_path)
    if index.get("status") != (
        "two one-shot scorer packages generated after public cache-delivery freeze"
    ):
        raise RuntimeError("wrong scoring-package index status")
    packages = index.get("packages")
    if (
        not isinstance(packages, list)
        or len(packages) != 2
        or [record.get("mode") for record in packages] != list(MODES)
        or index.get("package_count") != 2
    ):
        raise RuntimeError("scoring-package index does not contain the fixed pair")
    for key in ("launcher", "pair_controller", "result_collector"):
        identity = index.get(key)
        if not isinstance(identity, dict) or set(identity) != {
            "file",
            "bytes",
            "sha256",
        }:
            raise RuntimeError(f"scoring-package {key} identity is invalid")
    local_controller = Path(__file__).resolve()
    if index["pair_controller"] != {
        "file": local_controller.name,
        "bytes": local_controller.stat().st_size,
        "sha256": queue.sha256_file(local_controller),
    }:
        raise RuntimeError("running scoring-pair controller differs from public plan")
    actual_directories = {path.name for path in root.iterdir() if path.is_dir()}
    expected_directories = {record["directory"] for record in packages}
    if actual_directories != expected_directories:
        raise RuntimeError("scoring package-root directory set mismatch")

    for record in packages:
        if set(record) != {
            "mode",
            "kaggle_kernel_id",
            "directory",
            "required_cache_kernel_count",
            "threshold_kernel",
            "files",
            "embedded_job_config",
        }:
            raise RuntimeError(f"{record.get('mode')}: scoring package schema mismatch")
        if record["kaggle_kernel_id"] != (
            f"aviadcohen1/vesuvius-fusion-one-shot-{record['mode']}-scoring"
        ):
            raise RuntimeError(f"{record['mode']}: scoring kernel identity mismatch")
        if record["required_cache_kernel_count"] != 7:
            raise RuntimeError(f"{record['mode']}: cache-kernel count mismatch")
        directory = root / record["directory"]
        files = record["files"]
        observed_names = {path.name for path in directory.iterdir() if path.is_file()}
        if observed_names != set(files):
            raise RuntimeError(f"{record['mode']}: scoring package file set mismatch")
        for name, identity in files.items():
            path = directory / name
            if path.stat().st_size != identity.get("bytes") or queue.sha256_file(
                path
            ) != identity.get("sha256"):
                raise RuntimeError(
                    f"{record['mode']}: file identity mismatch for {name}"
                )
        expected_ledger = "".join(
            f"{files[name]['sha256']}  {name}\n"
            for name in sorted(files)
            if name != "KERNEL_SHA256SUMS"
        )
        if (directory / "KERNEL_SHA256SUMS").read_text(
            encoding="utf-8"
        ) != expected_ledger:
            raise RuntimeError(f"{record['mode']}: checksum ledger mismatch")
        wrapper = (directory / index["launcher"]["file"]).read_bytes()
        first_line, separator, base_source = wrapper.partition(b"\n")
        if not first_line.startswith(EMBEDDED_CONFIG_PREFIX) or not separator:
            raise RuntimeError(f"{record['mode']}: embedded config line is absent")
        if (
            len(base_source) != index["launcher"]["bytes"]
            or queue.sha256_bytes(base_source) != index["launcher"]["sha256"]
        ):
            raise RuntimeError(f"{record['mode']}: base launcher identity mismatch")
        try:
            config_raw = bytes.fromhex(
                first_line[len(EMBEDDED_CONFIG_PREFIX) :].decode("ascii")
            )
            config = json.loads(config_raw)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"{record['mode']}: embedded config cannot be decoded"
            ) from error
        config_content = dict(config)
        observed_payload = config_content.pop("payload_sha256", None)
        if observed_payload != queue.canonical_sha256(config_content):
            raise RuntimeError(f"{record['mode']}: embedded config payload mismatch")
        expected_config = {
            "schema_version": "1.0",
            "mode": record["mode"],
            "public_plan_commit": index["public_execution_plan"]["commit"],
            "public_plan_file_sha256": index["public_execution_plan"]["file_sha256"],
            "public_delivery_commit": index["public_cache_delivery"]["commit"],
            "public_delivery_file_sha256": index["public_cache_delivery"][
                "file_sha256"
            ],
            "payload_sha256": observed_payload,
        }
        if config != expected_config:
            raise RuntimeError(f"{record['mode']}: embedded config identity mismatch")
        if record["embedded_job_config"] != {
            "encoding": "hex in first source comment",
            "bytes": len(config_raw),
            "sha256": queue.sha256_bytes(config_raw),
            "payload_sha256": observed_payload,
        }:
            raise RuntimeError(f"{record['mode']}: embedded config ledger mismatch")
    return index, packages


def empty_receipt(index_path: Path, index: dict) -> dict:
    payload = {
        "schema_version": "1.0",
        "status": "paired one-shot scorer acceptance receipt",
        "scoring_package_index": {
            "file": index_path.name,
            "bytes": index_path.stat().st_size,
            "sha256": queue.sha256_file(index_path),
            "payload_sha256": index["payload_sha256"],
        },
        "accepted": [],
    }
    payload["payload_sha256"] = queue.canonical_sha256(payload)
    return payload


def load_receipt(path: Path, index_path: Path, index: dict) -> dict:
    receipt = (
        queue.load_canonical(path)
        if path.exists()
        else empty_receipt(index_path, index)
    )
    if receipt.get("status") != "paired one-shot scorer acceptance receipt":
        raise RuntimeError("scoring receipt status mismatch")
    if receipt.get("scoring_package_index") != {
        "file": index_path.name,
        "bytes": index_path.stat().st_size,
        "sha256": queue.sha256_file(index_path),
        "payload_sha256": index["payload_sha256"],
    }:
        raise RuntimeError("scoring receipt points to another package index")
    accepted = receipt.get("accepted")
    if not isinstance(accepted, list):
        raise TypeError("scoring receipt accepted records are invalid")
    return receipt


def receipt_map(receipt: dict, packages: list[dict]) -> dict[str, dict]:
    records = receipt["accepted"]
    if [record.get("mode") for record in records] != list(MODES[: len(records)]):
        raise RuntimeError("scoring receipt does not preserve paired launch order")
    mapped = {record.get("mode"): record for record in records}
    if len(mapped) != len(records):
        raise RuntimeError("scoring receipt contains duplicate modes")
    packages_by_mode = {record["mode"]: record for record in packages}
    for record in records:
        if set(record) != {
            "mode",
            "kernel_id",
            "kernel_version",
            "package_ledger_sha256",
            "accepted_at_utc",
        }:
            raise RuntimeError("scoring receipt record schema mismatch")
        package = packages_by_mode[record["mode"]]
        if record["kernel_id"] != package["kaggle_kernel_id"]:
            raise RuntimeError(f"{record['mode']}: receipt kernel identity mismatch")
        if (
            not isinstance(record["kernel_version"], int)
            or record["kernel_version"] <= 0
        ):
            raise TypeError(f"{record['mode']}: receipt kernel version is invalid")
        if (
            record["package_ledger_sha256"]
            != package["files"]["KERNEL_SHA256SUMS"]["sha256"]
        ):
            raise RuntimeError(f"{record['mode']}: receipt package ledger mismatch")
    return mapped


def write_receipt(path: Path, receipt: dict) -> None:
    content = dict(receipt)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = queue.canonical_sha256(content)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def push_package(kaggle: str, root: Path, record: dict) -> int:
    directory = root / record["directory"]
    result = queue.run_cli(
        [kaggle, "kernels", "push", "-p", str(directory)], timeout=300
    )
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    match = queue.VERSION_PATTERN.search(combined)
    if result.returncode != 0 or match is None:
        raise RuntimeError(
            f"{record['mode']}: scorer push was not accepted: {combined.strip()}"
        )
    return int(match.group(1))


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
        queue.kernel_status(args.kaggle, record["kaggle_kernel_id"])
        for record in packages
    ]
    missing_seen = False
    for record, status in zip(packages, statuses, strict=True):
        receipt_record = accepted.get(record["mode"])
        if status is None:
            missing_seen = True
            if receipt_record is not None:
                raise RuntimeError(
                    f"{record['mode']}: receipt exists but kernel is absent"
                )
            continue
        if missing_seen:
            raise RuntimeError("scoring kernels violate paired launch order")
        if receipt_record is None:
            raise RuntimeError(f"{record['mode']}: existing kernel lacks receipt")
        if status in queue.FAILED_STATUSES:
            raise RuntimeError(
                f"{record['mode']}: scorer operational failure ({status}); inspect only logs"
            )
        if status != "COMPLETE" and status not in queue.ACTIVE_STATUSES:
            raise RuntimeError(f"{record['mode']}: unsupported scorer status {status}")
    print(
        "SCORING_PAIR_STATUS "
        f"complete={statuses.count('COMPLETE')} "
        f"active={sum(status in queue.ACTIVE_STATUSES for status in statuses)}"
    )
    if args.dry_run:
        print("SCORING_PAIR_DRY_RUN_OK")
        return 0

    for index_position, (record, status) in enumerate(
        zip(packages, statuses, strict=True)
    ):
        if status is not None:
            continue
        version = push_package(args.kaggle, args.packages, record)
        receipt["accepted"].append(
            {
                "mode": record["mode"],
                "kernel_id": record["kaggle_kernel_id"],
                "kernel_version": version,
                "package_ledger_sha256": record["files"]["KERNEL_SHA256SUMS"]["sha256"],
                "accepted_at_utc": datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
            }
        )
        write_receipt(args.receipt, receipt)
        statuses[index_position] = "PENDING"
        print(
            f"SCORING_PAIR_ACCEPTED mode={record['mode']} version={version} "
            f"url=https://www.kaggle.com/code/{record['kaggle_kernel_id']}"
        )

    if len(receipt["accepted"]) != 2:
        raise RuntimeError("both scorer kernels were not accepted")
    print("BOTH_ONE_SHOT_SCORERS_ACCEPTED_BEFORE_RESULT_ACCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
