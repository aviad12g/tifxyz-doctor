from __future__ import annotations

import json
import sys
from pathlib import Path

import collect_heldout_delivery_inputs as collector
import generate_heldout_job_packages as generator
import orchestrate_heldout_queue as queue

RUNS = (
    "baseline",
    "control_seed11",
    "control_seed23",
    "control_seed47",
    "gap8_seed11",
    "gap8_seed23",
    "gap8_seed47",
)


def test_manifest_download_suppresses_automatic_kernel_log(
    tmp_path: Path, monkeypatch
) -> None:
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(collector.subprocess, "run", fake_run)
    destination = tmp_path / "selected"
    collector.download_selected("kaggle", "owner/kernel", destination)
    command = observed["command"]
    assert command[command.index("--page-size") + 1] == "100"
    assert command[command.index("--page-token") + 1] == ""
    assert "--force" not in command


def _identity(path: Path) -> dict:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": queue.sha256_file(path),
    }


def _jobs(mode: str) -> list[dict]:
    records = []
    for run in RUNS:
        if mode == "real_test_cache":
            records.append(
                {
                    "job_id": f"real-test-{run.replace('_', '-')}",
                    "mode": mode,
                    "run": run,
                    "kernel_id": None,
                    "kernel_version": None,
                    "expected_cache_manifest": f"cache_manifest_{run}_test.json",
                }
            )
        else:
            records.append(
                {
                    "job_id": f"synthetic-{run.replace('_', '-')}-all-shards",
                    "mode": mode,
                    "run": run,
                    "kernel_id": None,
                    "kernel_version": None,
                    "expected_cache_manifests": [
                        f"synthetic_manifest_{run}_shard{shard:02d}.json"
                        for shard in range(10)
                    ],
                }
            )
    return records


def test_collector_binds_versions_sources_and_downloads_no_npz(
    tmp_path: Path, monkeypatch
) -> None:
    root = Path(__file__).resolve().parent
    plan = {
        "schema_version": "1.0",
        "status": "held-out execution plan frozen before held-out inference",
        "heldout_cache_launcher": _identity(root / "heldout_cache_launcher.py"),
        "heldout_package_generator": _identity(
            root / "generate_heldout_job_packages.py"
        ),
        "heldout_delivery_collector": _identity(
            root / "collect_heldout_delivery_inputs.py"
        ),
        "threshold_binding": {
            "kernel_id": "aviadcohen1/vesuvius-fusion-aware-freeze-thresholds",
            "kernel_version": 6,
        },
        "real_test_jobs": _jobs("real_test_cache"),
        "synthetic_ray_jobs": _jobs("synthetic_ray_cache"),
    }
    plan["payload_sha256"] = queue.canonical_sha256(plan)
    plan_path = tmp_path / "heldout_execution_plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    package_root = tmp_path / "packages"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_heldout_job_packages.py",
            "--plan",
            str(plan_path),
            "--public-plan-commit",
            "a" * 40,
            "--out",
            str(package_root),
        ],
    )
    assert generator.main() == 0
    package_index_path = package_root / "generated_packages_index.json"
    package_index = queue.load_canonical(package_index_path)
    receipt = queue.empty_receipt(package_index_path, package_index)
    for record in package_index["packages"]:
        receipt["accepted"].append(
            {
                "job_id": record["job_id"],
                "kernel_id": record["kaggle_kernel_id"],
                "kernel_version": 1,
                "package_ledger_sha256": record["files"]["KERNEL_SHA256SUMS"]["sha256"],
                "accepted_at_utc": "2026-08-12T00:00:00Z",
            }
        )
    receipt_path = tmp_path / "queue_receipt.json"
    queue.write_receipt(receipt_path, receipt)
    package_by_kernel = {
        record["kaggle_kernel_id"]: record for record in package_index["packages"]
    }
    job_by_kernel = {
        f"aviadcohen1/vesuvius-fusion-{job['job_id']}": job
        for job in plan["real_test_jobs"] + plan["synthetic_ray_jobs"]
    }
    monkeypatch.setattr(queue, "kernel_status", lambda _kaggle, _kernel: "COMPLETE")

    def fake_current(kernel_id: str) -> tuple[int, bytes]:
        package = package_by_kernel[kernel_id]
        source = (
            package_root / package["directory"] / "heldout_cache_launcher.py"
        ).read_bytes()
        return 1, source

    def fake_download(_kaggle: str, kernel_id: str, destination: Path) -> None:
        destination.mkdir()
        job = job_by_kernel[kernel_id]
        (destination / "heldout_job_index.json").write_text("{}\n", encoding="utf-8")
        names = (
            [job["expected_cache_manifest"]]
            if job["mode"] == "real_test_cache"
            else job["expected_cache_manifests"]
        )
        for name in names:
            (destination / name).write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(collector, "current_kernel_version_and_source", fake_current)
    monkeypatch.setattr(collector, "download_selected", fake_download)
    output = tmp_path / "collector-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_heldout_delivery_inputs.py",
            "--plan",
            str(plan_path),
            "--public-plan-commit",
            "a" * 40,
            "--packages",
            str(package_root),
            "--package-index",
            str(package_index_path),
            "--queue-receipt",
            str(receipt_path),
            "--out",
            str(output),
        ],
    )
    assert collector.main() == 0
    sources = queue.load_canonical(output / "delivery_source_records.json")
    assert len(sources["jobs"]) == 14
    assert sources["scientific_gate"]["npz_probability_caches_downloaded"] is False
    assert not list(output.rglob("*.npz"))
