from __future__ import annotations

import json
import sys
from pathlib import Path

import orchestrate_heldout_queue as queue
import retry_scoring_pair_after_fail_closed as retry


def _write_hashed(path: Path, payload: dict) -> dict:
    content = dict(payload)
    content["payload_sha256"] = queue.canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return content


def test_retry_accepts_exact_v2_pair_after_recorded_pre_staging_errors(
    tmp_path: Path, monkeypatch
) -> None:
    correction = {
        "failed_attempts": [
            {"mode": mode, **retry.FAILED_ATTEMPTS[mode]} for mode in retry.pair.MODES
        ],
        "failure_stage": "asset-ledger validation before held-out input staging",
        "scientific_gate": {
            "held_out_input_staging_started_in_failed_attempts": False,
            "scorer_invocation_started_in_failed_attempts": False,
            "cache_npz_opened_or_inspected": False,
            "scientific_result_created_or_opened": False,
            "threshold_seed_panel_gate_or_claim_changed": False,
            "retry_is_first_scientific_scorer_invocation": True,
        },
    }
    plan_path = tmp_path / "heldout_execution_plan.json"
    plan = _write_hashed(
        plan_path, {"result_blind_scoring_asset_correction": correction}
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
            "kaggle_kernel_id": retry.FAILED_ATTEMPTS[mode]["kernel_id"],
            "files": {"KERNEL_SHA256SUMS": {"sha256": mode * 8}},
        }
        for mode in retry.pair.MODES
    ]
    monkeypatch.setattr(retry.pair, "validate_packages", lambda *_args: (index, packages))
    monkeypatch.setattr(retry.queue, "kernel_status", lambda *_args: "ERROR")
    monkeypatch.setattr(
        retry.heldout,
        "current_kernel_version_and_source",
        lambda _kernel: (1, b"sealed"),
    )
    monkeypatch.setattr(retry.pair, "push_package", lambda *_args: 2)
    receipt = tmp_path / "retry_receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retry_scoring_pair_after_fail_closed.py",
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
    assert all(record["kernel_version"] == 2 for record in saved["accepted"])
    assert saved["scientific_gate"]["v1_scientific_results_created_or_opened"] is False
