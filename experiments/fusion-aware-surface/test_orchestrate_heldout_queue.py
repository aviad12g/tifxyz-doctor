from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import orchestrate_heldout_queue as queue
import pytest


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[dict]]:
    packages_root = tmp_path / "packages"
    packages_root.mkdir()
    base_launcher = b'"""fixture launcher"""\n'
    launcher_identity = {
        "file": "heldout_cache_launcher.py",
        "bytes": len(base_launcher),
        "sha256": queue.sha256_bytes(base_launcher),
    }
    package_records = []
    for index in range(14):
        job_id = f"job{index:02d}"
        directory = packages_root / job_id
        directory.mkdir()
        files = {}
        config = {
            "schema_version": "1.0",
            "job_id": job_id,
            "public_plan_commit": "a" * 40,
            "public_plan_file_sha256": "b" * 64,
        }
        config["payload_sha256"] = queue.canonical_sha256(config)
        config_raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        launcher_path = directory / "heldout_cache_launcher.py"
        launcher_path.write_bytes(
            queue.EMBEDDED_CONFIG_PREFIX + config_raw.hex().encode("ascii") + b"\n" + base_launcher
        )
        files[launcher_path.name] = {
            "bytes": launcher_path.stat().st_size,
            "sha256": queue.sha256_file(launcher_path),
        }
        for name in ("kernel-metadata.json",):
            path = directory / name
            path.write_text(f"{job_id}:{name}\n", encoding="utf-8")
            files[name] = {
                "bytes": path.stat().st_size,
                "sha256": queue.sha256_file(path),
            }
        ledger = directory / "KERNEL_SHA256SUMS"
        ledger.write_text(
            "".join(f"{files[name]['sha256']}  {name}\n" for name in sorted(files)),
            encoding="utf-8",
        )
        files[ledger.name] = {
            "bytes": ledger.stat().st_size,
            "sha256": queue.sha256_file(ledger),
        }
        package_records.append(
            {
                "job_id": job_id,
                "mode": "real_test_cache" if index < 7 else "synthetic_ray_cache",
                "kaggle_kernel_id": f"aviadcohen1/vesuvius-fusion-{job_id}",
                "directory": job_id,
                "files": files,
                "embedded_job_config": {
                    "encoding": "hex in first source comment",
                    "bytes": len(config_raw),
                    "sha256": queue.sha256_bytes(config_raw),
                    "payload_sha256": config["payload_sha256"],
                },
            }
        )
    payload = {
        "schema_version": "1.0",
        "status": "all private held-out cache packages generated from public plan",
        "public_plan_commit": "a" * 40,
        "public_plan_file_sha256": "b" * 64,
        "public_plan_payload_sha256": "c" * 64,
        "launcher": launcher_identity,
        "package_count": 14,
        "packages": package_records,
    }
    payload["payload_sha256"] = queue.canonical_sha256(payload)
    index_path = packages_root / "generated_packages_index.json"
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return packages_root, index_path, tmp_path / "receipt.json", package_records


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    packages: Path,
    index: Path,
    receipt: Path,
    *extra_args: str,
) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "orchestrate_heldout_queue.py",
            "--packages",
            str(packages),
            "--index",
            str(index),
            "--receipt",
            str(receipt),
            *extra_args,
        ],
    )
    return queue.main()


def test_queue_fills_two_slots_and_resumes_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packages, index, receipt, records = _write_fixture(tmp_path)
    statuses = {record["kaggle_kernel_id"]: None for record in records}
    pushed = []

    monkeypatch.setattr(
        queue,
        "kernel_status",
        lambda _kaggle, kernel_id, **_kwargs: statuses[kernel_id],
    )

    def fake_push(_kaggle: str, _root: Path, record: dict) -> tuple[int, str]:
        pushed.append(record["job_id"])
        statuses[record["kaggle_kernel_id"]] = "PENDING"
        return 1, "Kernel version 1 successfully pushed"

    monkeypatch.setattr(queue, "push_package", fake_push)
    assert _run_main(monkeypatch, packages, index, receipt) == 0
    assert pushed == ["job00", "job01"]
    first_receipt = queue.load_canonical(receipt)
    assert [record["job_id"] for record in first_receipt["accepted"]] == [
        "job00",
        "job01",
    ]

    statuses[records[0]["kaggle_kernel_id"]] = "COMPLETE"
    statuses[records[1]["kaggle_kernel_id"]] = "COMPLETE"
    assert _run_main(monkeypatch, packages, index, receipt) == 0
    assert pushed == ["job00", "job01", "job02", "job03"]


def test_queue_rejects_existing_kernel_after_missing_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packages, index, receipt, records = _write_fixture(tmp_path)
    statuses = {record["kaggle_kernel_id"]: None for record in records}
    statuses[records[1]["kaggle_kernel_id"]] = "COMPLETE"
    monkeypatch.setattr(
        queue,
        "kernel_status",
        lambda _kaggle, kernel_id, **_kwargs: statuses[kernel_id],
    )
    with pytest.raises(RuntimeError, match="lacks an acceptance receipt|fixed push order"):
        _run_main(monkeypatch, packages, index, receipt)


def test_queue_rejects_package_byte_tamper(tmp_path: Path) -> None:
    packages, index, _, _ = _write_fixture(tmp_path)
    (packages / "job07" / "kernel-metadata.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="byte-size mismatch|SHA-256 mismatch"):
        queue.validate_packages(packages, index)


def test_queue_rejects_more_than_two_active_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packages, index, receipt, records = _write_fixture(tmp_path)
    index_payload = queue.load_canonical(index)
    receipt_payload = queue.empty_receipt(index, index_payload)
    for record in records[:3]:
        receipt_payload["accepted"].append(
            {
                "job_id": record["job_id"],
                "kernel_id": record["kaggle_kernel_id"],
                "kernel_version": 1,
                "package_ledger_sha256": record["files"]["KERNEL_SHA256SUMS"][
                    "sha256"
                ],
                "accepted_at_utc": "2026-08-12T00:00:00Z",
            }
        )
    queue.write_receipt(receipt, receipt_payload)
    statuses = {record["kaggle_kernel_id"]: None for record in records}
    for record in records[:3]:
        statuses[record["kaggle_kernel_id"]] = "RUNNING"
    monkeypatch.setattr(
        queue,
        "kernel_status",
        lambda _kaggle, kernel_id, **_kwargs: statuses[kernel_id],
    )
    with pytest.raises(RuntimeError, match="active GPU job count exceeds frozen cap"):
        _run_main(monkeypatch, packages, index, receipt)


def test_queue_failure_prevents_later_pushes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packages, index, receipt, records = _write_fixture(tmp_path)
    index_payload = queue.load_canonical(index)
    receipt_payload = queue.empty_receipt(index, index_payload)
    first = records[0]
    receipt_payload["accepted"].append(
        {
            "job_id": first["job_id"],
            "kernel_id": first["kaggle_kernel_id"],
            "kernel_version": 1,
            "package_ledger_sha256": first["files"]["KERNEL_SHA256SUMS"]["sha256"],
            "accepted_at_utc": "2026-08-12T00:00:00Z",
        }
    )
    queue.write_receipt(receipt, receipt_payload)
    statuses = {record["kaggle_kernel_id"]: None for record in records}
    statuses[first["kaggle_kernel_id"]] = "ERROR"
    pushed = []
    monkeypatch.setattr(
        queue,
        "kernel_status",
        lambda _kaggle, kernel_id, **_kwargs: statuses[kernel_id],
    )
    monkeypatch.setattr(
        queue,
        "push_package",
        lambda _kaggle, _root, record: pushed.append(record["job_id"]),
    )
    with pytest.raises(RuntimeError, match="operational failure"):
        _run_main(monkeypatch, packages, index, receipt)
    assert pushed == []


def test_permission_denied_is_absent_only_for_unaccepted_owned_slug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            subprocess.CompletedProcess(
                [], 1, "", "Permission 'kernels.get' was denied"
            ),
            subprocess.CompletedProcess(
                [], 0, "ref,title,author,lastRunTime,totalVotes\n", ""
            ),
        ]
    )
    monkeypatch.setattr(queue, "run_cli", lambda *args, **kwargs: next(responses))

    assert (
        queue.kernel_status(
            "kaggle",
            "aviadcohen1/vesuvius-fusion-real-test-baseline",
            allow_unaccepted_absence=True,
        )
        is None
    )


def test_two_argument_status_call_retains_frozen_tool_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            subprocess.CompletedProcess(
                [], 1, "", "Permission 'kernels.get' was denied"
            ),
            subprocess.CompletedProcess(
                [], 0, "ref,title,author,lastRunTime,totalVotes\n", ""
            ),
        ]
    )
    monkeypatch.setattr(queue, "run_cli", lambda *args, **kwargs: next(responses))
    assert queue.kernel_status("kaggle", "aviadcohen1/not-created") is None


def test_permission_denied_never_masks_an_accepted_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        queue,
        "run_cli",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 1, "", "Permission 'kernels.get' was denied"
        ),
    )

    with pytest.raises(RuntimeError, match="status query failed"):
        queue.kernel_status(
            "kaggle",
            "aviadcohen1/vesuvius-fusion-real-test-baseline",
            allow_unaccepted_absence=False,
        )


def test_permission_denied_never_masks_an_existing_owned_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            subprocess.CompletedProcess(
                [], 1, "", "Permission 'kernels.get' was denied"
            ),
            subprocess.CompletedProcess(
                [],
                0,
                (
                    "ref,title,author,lastRunTime,totalVotes\n"
                    "aviadcohen1/vesuvius-fusion-real-test-baseline,Title,Aviad,,0\n"
                ),
                "",
            ),
        ]
    )
    monkeypatch.setattr(queue, "run_cli", lambda *args, **kwargs: next(responses))

    with pytest.raises(RuntimeError, match="status query failed"):
        queue.kernel_status(
            "kaggle",
            "aviadcohen1/vesuvius-fusion-real-test-baseline",
            allow_unaccepted_absence=True,
        )


def test_explicit_retry_accepts_only_contiguous_failed_unaccepted_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packages, index, receipt, records = _write_fixture(tmp_path)
    statuses = {record["kaggle_kernel_id"]: None for record in records}
    statuses[records[0]["kaggle_kernel_id"]] = "ERROR"
    statuses[records[1]["kaggle_kernel_id"]] = "ERROR"
    pushed = []
    monkeypatch.setattr(
        queue,
        "kernel_status",
        lambda _kaggle, kernel_id, **_kwargs: statuses[kernel_id],
    )

    def fake_push(_kaggle: str, _root: Path, record: dict) -> tuple[int, str]:
        pushed.append(record["job_id"])
        statuses[record["kaggle_kernel_id"]] = "PENDING"
        return 2, "Kernel version 2 successfully pushed"

    monkeypatch.setattr(queue, "push_package", fake_push)
    assert (
        _run_main(
            monkeypatch,
            packages,
            index,
            receipt,
            "--retry-failed-unaccepted",
        )
        == 0
    )
    assert pushed == ["job00", "job01"]
    accepted = queue.load_canonical(receipt)["accepted"]
    assert [record["kernel_version"] for record in accepted] == [2, 2]


def test_retry_rejects_failed_kernel_after_missing_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packages, index, receipt, records = _write_fixture(tmp_path)
    statuses = {record["kaggle_kernel_id"]: None for record in records}
    statuses[records[1]["kaggle_kernel_id"]] = "ERROR"
    monkeypatch.setattr(
        queue,
        "kernel_status",
        lambda _kaggle, kernel_id, **_kwargs: statuses[kernel_id],
    )
    with pytest.raises(RuntimeError, match="failed retry violates fixed push order"):
        _run_main(
            monkeypatch,
            packages,
            index,
            receipt,
            "--retry-failed-unaccepted",
        )
