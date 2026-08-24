from __future__ import annotations

import json
from pathlib import Path

import pytest

import collect_runpod_real_kaggle_synthetic_results as mixed
import orchestrate_heldout_queue as queue


PLAN_PAYLOAD = "a" * 64


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_canonical(path: Path, payload: dict) -> dict:
    content = dict(payload)
    content["payload_sha256"] = queue.canonical_sha256(payload)
    write_json(path, content)
    return content


def make_real_source(root: Path) -> dict[str, str]:
    for relative in mixed.REAL_EXPECTED:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith("status.json"):
            if "transport-status" in relative:
                payload = {
                    "state": "PRIVATE_TRANSPORT_VERIFIED",
                    "plan_payload_sha256": PLAN_PAYLOAD,
                    "scientific_outputs_inspected": False,
                    "credentials_removed": True,
                }
            else:
                payload = {
                    "state": "COMPLETE",
                    "returncode": 0,
                    "plan_payload_sha256": PLAN_PAYLOAD,
                    "scientific_outputs_inspected": False,
                }
            write_json(path, payload)
        else:
            path.write_bytes((relative + "\n").encode())
    ledger = {
        relative: queue.sha256_file(root / relative)
        for relative in sorted(mixed.REAL_EXPECTED)
    }
    (root / "SHA256SUMS").write_text(
        "".join(f"{digest}  ./{relative}\n" for relative, digest in ledger.items()),
        encoding="utf-8",
    )
    return ledger


def test_validate_real_source_hashes_without_parsing_scientific_files(
    tmp_path: Path,
) -> None:
    ledger = make_real_source(tmp_path)
    validated = mixed.validate_real_source(
        tmp_path, expected_plan_payload=PLAN_PAYLOAD, expected_ledger=ledger
    )
    assert len(validated["artifacts"]) == 7
    assert len(validated["operational_statuses"]) == 8


def test_validate_real_source_rejects_changed_sealed_result(tmp_path: Path) -> None:
    ledger = make_real_source(tmp_path)
    (tmp_path / mixed.REAL_ARTIFACTS[0]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        mixed.validate_real_source(
            tmp_path, expected_plan_payload=PLAN_PAYLOAD, expected_ledger=ledger
        )


def make_synthetic_source(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "packages"
    directory = root / "synthetic"
    directory.mkdir(parents=True)
    launcher = directory / "one_shot_scoring_launcher.py"
    launcher.write_text("print('sealed')\n", encoding="utf-8")
    metadata = directory / "kernel-metadata.json"
    metadata.write_text("{}\n", encoding="utf-8")
    ledger_file = directory / "KERNEL_SHA256SUMS"
    ledger_file.write_text("ledger\n", encoding="utf-8")
    files = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": queue.sha256_file(path),
        }
        for path in (ledger_file, metadata, launcher)
    }
    index_path = root / "generated_scoring_packages_index.json"
    index = write_canonical(
        index_path,
        {
            "schema_version": "1.0",
            "status": "two one-shot scorer packages generated after public cache-delivery freeze",
            "launcher": {"file": launcher.name},
            "packages": [
                {
                    "mode": "synthetic",
                    "directory": "synthetic",
                    "kaggle_kernel_id": mixed.SYNTHETIC_KERNEL,
                    "files": files,
                }
            ],
        },
    )
    receipt_path = tmp_path / "receipt.json"
    write_canonical(
        receipt_path,
        {
            "schema_version": "1.0",
            "status": "paired one-shot scorer acceptance receipt",
            "scoring_package_index": {
                "file": index_path.name,
                "bytes": index_path.stat().st_size,
                "sha256": queue.sha256_file(index_path),
                "payload_sha256": index["payload_sha256"],
            },
            "accepted": [
                {
                    "mode": "synthetic",
                    "kernel_id": mixed.SYNTHETIC_KERNEL,
                    "kernel_version": 4,
                    "package_ledger_sha256": files["KERNEL_SHA256SUMS"][
                        "sha256"
                    ],
                }
            ],
        },
    )
    return root, index_path, receipt_path


def test_validate_synthetic_source_binds_version_four_package(
    tmp_path: Path,
) -> None:
    root, index, receipt = make_synthetic_source(tmp_path)
    validated = mixed.validate_synthetic_source(root, index, receipt)
    assert validated["launcher_identity"]["file"] == "one_shot_scoring_launcher.py"
    assert validated["receipt"]["accepted"][0]["kernel_version"] == 4


def test_read_sha256s_rejects_duplicate_record(tmp_path: Path) -> None:
    ledger = tmp_path / "SHA256SUMS"
    ledger.write_text(f"{'a' * 64}  ./same\n{'b' * 64}  ./same\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate"):
        mixed.read_sha256s(ledger)
