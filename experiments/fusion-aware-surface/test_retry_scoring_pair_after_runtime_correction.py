from __future__ import annotations

import json
import sys
from pathlib import Path

import orchestrate_heldout_queue as queue
import retry_scoring_pair_after_runtime_correction as retry


def _write_hashed(path: Path, payload: dict) -> dict:
    content = dict(payload)
    content["payload_sha256"] = queue.canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return content


def test_retry_accepts_exact_v4_pair_after_recorded_runtime_failure(
    tmp_path: Path, monkeypatch
) -> None:
    plan_path = tmp_path / "heldout_execution_plan.json"
    plan = _write_hashed(
        plan_path,
        {
            "result_blind_scoring_asset_correction": {
                "failed_attempts": retry.expected_attempts(1)
            },
            "result_blind_scoring_job_plan_compatibility_correction": {
                "failed_attempts": retry.expected_attempts(2)
            },
            "result_blind_scoring_runtime_transport_correction": {
                "failed_attempts": retry.expected_attempts(3),
                "correction": {
                    "torch": "2.5.1+cpu",
                    "torchvision": "0.20.1+cpu",
                    "other_runtime_versions_changed": False,
                    "primary_index": "https://download.pytorch.org/whl/cpu",
                    "extra_index": "https://pypi.org/simple",
                    "binary_wheels_only": True,
                    "scorer_kernel_accelerator": "none",
                    "cache_manifest_or_npz_changed": False,
                },
                "python312_linux_resolution_preflight": {
                    "status": "all exact top-level requirements and transitive wheels resolved",
                    "files": 30,
                    "bytes": 260266954,
                    "sorted_wheel_ledger_sha256": (
                        "7e8967a004c7e6e97a1e223f33f8edb78731c1d1dfa7c21ea52df9c63747bfec"
                    ),
                },
                "scientific_gate": {
                    "scientific_result_created_or_opened": False,
                    "held_out_input_staging_started_in_failed_attempts": False,
                    "scorer_invocation_started_in_failed_attempts": False,
                },
            },
            "result_blind_scoring_runtime_projection_correction": {
                "local_preflight": {"provider_scorer_version_created": False},
                "scientific_gate": {"scientific_result_created_or_opened": False},
            },
        },
    )
    index_path = tmp_path / "generated_scoring_packages_index.json"
    index = _write_hashed(index_path, {"placeholder": True})
    index.update(
        {
            "public_execution_plan": {
                "commit": retry.PUBLIC_PLAN_COMMIT,
                "file_sha256": queue.sha256_file(plan_path),
                "payload_sha256": plan["payload_sha256"],
            },
            "public_cache_delivery": {"commit": retry.PUBLIC_DELIVERY_COMMIT},
        }
    )
    packages = [
        {
            "mode": mode,
            "kaggle_kernel_id": retry.KERNEL_IDS[mode],
            "files": {"KERNEL_SHA256SUMS": {"sha256": mode * 8}},
        }
        for mode in retry.pair.MODES
    ]
    monkeypatch.setattr(retry.pair, "validate_packages", lambda *_args: (index, packages))
    monkeypatch.setattr(retry.queue, "kernel_status", lambda *_args: "ERROR")
    monkeypatch.setattr(
        retry.heldout,
        "current_kernel_version_and_source",
        lambda _kernel: (3, b"sealed"),
    )
    monkeypatch.setattr(retry.pair, "push_package", lambda *_args: 4)
    receipt = tmp_path / "retry_receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retry_scoring_pair_after_runtime_correction.py",
            "--packages",
            str(tmp_path),
            "--index",
            str(index_path),
            "--plan",
            str(plan_path),
            "--receipt",
            str(receipt),
        ],
    )
    assert retry.main() == 0
    saved = queue.load_canonical(receipt)
    assert [record["mode"] for record in saved["accepted"]] == ["real", "synthetic"]
    assert all(record["kernel_version"] == 4 for record in saved["accepted"])
    assert saved["scientific_gate"][
        "v1_v2_v3_scientific_results_created_or_opened"
    ] is False
