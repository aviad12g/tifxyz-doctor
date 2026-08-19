from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "gapbalance_development", ROOT / "gapbalance_development.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "gapbalance_development_kaggle_launcher",
    ROOT / "gapbalance_development_kaggle_launcher.py",
)
assert LAUNCHER_SPEC is not None and LAUNCHER_SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(LAUNCHER_SPEC)
LAUNCHER_SPEC.loader.exec_module(LAUNCHER)


def count_row(*, neighbours=100, detected=100, fused=20, controls=100, splits=5):
    return {
        "neighbour_sites": neighbours,
        "detected_neighbour_sites": detected,
        "fused_detected_sites": fused,
        "control_sites": controls,
        "false_split_sites": splits,
    }


def development_inputs(gap2_fused=14, gap4_fused=10):
    synthetic = {}
    real = {}
    for run in MODULE.RUNS:
        arm, _ = MODULE.run_identity(run)
        fused = {"control": 20, "gap2": gap2_fused, "gap4": gap4_fused}[arm]
        synthetic[run] = {
            "primary": count_row(fused=fused),
            "single_sheet_control": count_row(fused=0),
        }
        real[run] = {"blend": 0.80, "toposcore": 0.75}
    return synthetic, real


def test_frozen_identifiers_and_synthetic_grid():
    assert MODULE.RUNS == (
        "control_seed11",
        "control_seed23",
        "control_seed47",
        "gap2_seed11",
        "gap2_seed23",
        "gap2_seed47",
        "gap4_seed11",
        "gap4_seed23",
        "gap4_seed47",
    )
    cells = MODULE.synthetic_cells()
    assert len(cells) == 80
    assert sum(cell["kind"] == "primary" for cell in cells) == 64
    assert sum(cell["kind"] == "single_sheet_control" for cell in cells) == 16
    assert {cell["seed"] for cell in cells} == {400, 401, 402, 403}


def test_threshold_ties_nearest_half_then_lower():
    means = {threshold: 0.1 for threshold in MODULE.THRESHOLDS}
    means[0.4] = means[0.6] = 0.9
    assert MODULE.choose_threshold(means) == 0.4
    means[0.5] = 0.9
    assert MODULE.choose_threshold(means) == 0.5
    with pytest.raises(ValueError):
        MODULE.choose_threshold({0.5: 1.0})


def test_raw_counts_are_pooled_before_rates():
    pooled = MODULE.pool_counts(
        [
            count_row(neighbours=100, detected=50, fused=10, controls=10, splits=1),
            count_row(neighbours=10, detected=10, fused=10, controls=90, splits=9),
        ]
    )
    assert pooled["site_center_detection_rate"] == 60 / 110
    assert pooled["conditional_fusion_rate"] == 20 / 60
    assert pooled["false_split_rate"] == 10 / 100


def test_larger_eligible_reduction_selects_gap4():
    synthetic, real = development_inputs(gap2_fused=14, gap4_fused=10)
    decision = MODULE.select_candidate(synthetic, real)
    assert decision["eligible_arms"] == ["gap2", "gap4"]
    assert decision["selected_arm"] == "gap4"
    assert decision["confirmation_may_open"] is True


def test_under_one_point_reduction_difference_selects_gap2():
    synthetic, real = development_inputs(gap2_fused=14, gap4_fused=14)
    synthetic["gap4_seed11"]["primary"]["fused_detected_sites"] = 13
    decision = MODULE.select_candidate(synthetic, real)
    assert decision["selected_arm"] == "gap2"
    assert "less than 1pp" in decision["reason"]


@pytest.mark.parametrize(
    ("mutation", "constraint"),
    [
        ("fusion", "conditional_fusion_reduction_at_least_5pp"),
        ("detection", "detection_decline_no_worse_than_2pp"),
        ("split", "false_split_increase_no_worse_than_2pp"),
        ("blend", "mean_real_development_blend_delta_at_least_minus_0_005"),
        ("toposcore", "mean_real_development_toposcore_delta_at_least_minus_0_005"),
    ],
)
def test_each_frozen_constraint_fails_closed(mutation, constraint):
    synthetic, real = development_inputs(gap2_fused=14, gap4_fused=10)
    for seed in MODULE.SEEDS:
        run = f"gap4_seed{seed}"
        if mutation == "fusion":
            synthetic[run]["primary"]["fused_detected_sites"] = 16
        elif mutation == "detection":
            synthetic[run]["primary"].update(
                neighbour_sites=100,
                detected_neighbour_sites=97,
                fused_detected_sites=10,
            )
        elif mutation == "split":
            synthetic[run]["single_sheet_control"]["false_split_sites"] = 8
        elif mutation == "blend":
            real[run]["blend"] = 0.794
        elif mutation == "toposcore":
            real[run]["toposcore"] = 0.744
    summary = MODULE.summarize_arm("gap4", synthetic, real)
    assert summary["constraints"][constraint] is False
    assert summary["eligible"] is False


def test_neither_eligible_keeps_confirmation_sealed():
    synthetic, real = development_inputs(gap2_fused=20, gap4_fused=20)
    decision = MODULE.select_candidate(synthetic, real)
    assert decision["selected_arm"] is None
    assert decision["confirmation_may_open"] is False


def test_training_freeze_contains_exact_nine_runs_and_valid_payload_hash():
    path = ROOT / "GAPBALANCE_TRAINING_FREEZE.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = payload.pop("payload_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert observed == hashlib.sha256(canonical).hexdigest()
    assert tuple(payload["runs"]) == MODULE.RUNS
    assert payload["confirmation_outputs_inspected"] is False
    for run, record in payload["runs"].items():
        arm, seed = MODULE.run_identity(run)
        assert (record["arm"], record["seed"]) == (arm, seed)
        assert len(record["checkpoint_sha256"]) == 64


def test_launcher_base_identity_and_path_guard():
    identity = LAUNCHER.launcher_source_identity()
    source = ROOT / "gapbalance_development_kaggle_launcher.py"
    assert identity == {
        "file": source.name,
        "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert LAUNCHER.safe_relative(ROOT, "nested/file.json") == (
        ROOT / "nested/file.json"
    ).resolve()
    with pytest.raises(RuntimeError):
        LAUNCHER.safe_relative(ROOT, "../escape")
    with pytest.raises(RuntimeError):
        LAUNCHER.load_job_config()


def test_public_development_plan_is_complete_and_result_blind():
    path = ROOT / "GAPBALANCE_KAGGLE_DEVELOPMENT_PLAN.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = payload.pop("payload_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert observed == hashlib.sha256(canonical).hexdigest()
    assert payload["scientific_endpoints_scored"] is False
    assert payload["confirmation_outputs_inspected"] is False
    assert payload["execution"] == {
        "concurrent_gpu_jobs_maximum": 2,
        "form_submission_authorized": False,
        "paid_compute_authorized": False,
        "provider": "free Kaggle GPU",
        "provider_cost_usd": 0,
        "real_cache_jobs": 1,
        "scientific_endpoint_scoring_jobs": 1,
        "synthetic_cache_jobs": 12,
    }
    jobs = payload["jobs"]
    assert len(jobs) == 13
    assert sum(job["mode"] == "real" for job in jobs) == 1
    synthetic = [job for job in jobs if job["mode"] == "synthetic"]
    assert {(job["seed"], job["shard_index"]) for job in synthetic} == {
        (seed, shard) for seed in MODULE.SEEDS for shard in range(4)
    }
    assert len({job["job_id"] for job in jobs}) == 13
    assert len({job["kernel_id"] for job in jobs}) == 13
    provider_ids = payload["provider_resolved_kernel_ids"]
    assert set(provider_ids) == {job["job_id"] for job in jobs}
    assert len(set(provider_ids.values())) == 13
    assert provider_ids["gapbalance-development-real"] == (
        "aviadcohen1/vesuvius-gapbalance-development-real-all-runs"
    )
    assert all(len(value.split("/", 1)[1]) <= 50 for value in provider_ids.values())
    for job in jobs:
        for key in (
            "config_payload_sha256",
            "launcher_sha256",
            "metadata_sha256",
            "package_sums_sha256",
        ):
            assert len(job[key]) == 64
    training = ROOT / "GAPBALANCE_TRAINING_FREEZE.json"
    assert payload["training_freeze"]["file_sha256"] == hashlib.sha256(
        training.read_bytes()
    ).hexdigest()
    assert payload["scoring_freeze"]["pure_rule_file_sha256"] == hashlib.sha256(
        (ROOT / "gapbalance_development.py").read_bytes()
    ).hexdigest()
    assert payload["scoring_freeze"]["scorer_file_sha256"] == hashlib.sha256(
        (ROOT / "score_gapbalance_development.py").read_bytes()
    ).hexdigest()


def test_synthetic_operational_fix_changes_only_provider_metadata():
    plan_path = ROOT / "GAPBALANCE_KAGGLE_DEVELOPMENT_PLAN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    fix_path = ROOT / "GAPBALANCE_DEVELOPMENT_OPERATIONAL_FIX.json"
    fix = json.loads(fix_path.read_text(encoding="utf-8"))
    observed = fix.pop("payload_sha256")
    canonical = json.dumps(fix, sort_keys=True, separators=(",", ":")).encode()
    assert observed == hashlib.sha256(canonical).hexdigest()
    assert plan["synthetic_operational_fix"] == {
        "file": "experiments/gap-balance-followup/GAPBALANCE_DEVELOPMENT_OPERATIONAL_FIX.json",
        "file_sha256": hashlib.sha256(fix_path.read_bytes()).hexdigest(),
        "payload_sha256": observed,
    }
    assert fix["failure"]["scientific_endpoints_computed"] is False
    assert fix["confirmation_outputs_inspected"] is False
    original = {job["job_id"]: job for job in plan["jobs"]}
    corrected = fix["synthetic_jobs"]
    assert len(corrected) == 12
    assert len({job["job_id"] for job in corrected}) == 12
    assert len({job["kernel_id"] for job in corrected}) == 12
    for job in corrected:
        before = original[job["job_id"]]
        assert job["config_payload_sha256"] == before["config_payload_sha256"]
        assert job["launcher_sha256"] == before["launcher_sha256"]
        assert job["kernel_id"] == plan["provider_resolved_kernel_ids"][job["job_id"]]
        assert len(job["kernel_id"].split("/", 1)[1]) <= 50


def test_import_closure_fix_is_complete_and_result_blind():
    plan = json.loads(
        (ROOT / "GAPBALANCE_KAGGLE_DEVELOPMENT_PLAN.json").read_text(
            encoding="utf-8"
        )
    )
    fix_path = ROOT / "GAPBALANCE_DEVELOPMENT_IMPORT_FIX.json"
    fix = json.loads(fix_path.read_text(encoding="utf-8"))
    observed = fix.pop("payload_sha256")
    canonical = json.dumps(fix, sort_keys=True, separators=(",", ":")).encode()
    assert observed == hashlib.sha256(canonical).hexdigest()
    assert plan["import_closure_fix"] == {
        "file": "experiments/gap-balance-followup/GAPBALANCE_DEVELOPMENT_IMPORT_FIX.json",
        "file_sha256": hashlib.sha256(fix_path.read_bytes()).hexdigest(),
        "payload_sha256": observed,
    }
    assert fix["scientific_code_changed"] is False
    assert fix["scientific_endpoints_computed"] is False
    assert fix["confirmation_outputs_inspected"] is False
    assert {failure["job_id"] for failure in fix["failures"]} == {
        "gapbalance-development-real",
        "gapbalance-development-synthetic-seed11-shard00",
    }
    assert {
        "experiments/fusion-aware-surface/train_fusion_aware.py",
        "experiments/fusion-aware-surface/fusion_loss.py",
        "experiments/fusion-aware-surface/gap_supervision.py",
    } <= set(fix["frozen_public_source_files"])
    jobs = fix["jobs"]
    assert len(jobs) == 13
    assert {job["job_id"] for job in jobs} == {
        job["job_id"] for job in plan["jobs"]
    }
    assert len({job["config_payload_sha256"] for job in jobs}) == 13
    assert len({job["launcher_sha256"] for job in jobs}) == 13
