import copy
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FREEZE = load_module("freeze_gapbalance_runpod_development", "freeze_gapbalance_runpod_development.py")
RETRY = load_module("freeze_gapbalance_runpod_allocation_retry", "freeze_gapbalance_runpod_allocation_retry.py")
MULTI = load_module("freeze_gapbalance_runpod_multi_pod_retry", "freeze_gapbalance_runpod_multi_pod_retry.py")
EGRESS = load_module("freeze_gapbalance_runpod_egress_resume", "freeze_gapbalance_runpod_egress_resume.py")
WRAPPER = load_module("run_gapbalance_runpod_development_job", "run_gapbalance_runpod_development_job.py")
ORCH = load_module("orchestrate_gapbalance_runpod_development", "orchestrate_gapbalance_runpod_development.py")


def test_public_plan_has_exact_cap_waves_and_blind_gates():
    plan = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json")
    assert plan["budget"] == {
        "absolute_campaign_cap_usd": 25.0,
        "authorization_date": "2026-08-20",
        "confirmation_reserve_usd": 13.0,
        "development_billing_cutoff_usd": 12.0,
        "forms_authorized": False,
        "guard_seconds": 300,
        "on_cutoff": "stop the exact pod, preserve partial operational artifacts, do not score or select partial development output",
        "user_authorization": "Aviad: ok go",
    }
    assert [len(wave) for wave in plan["execution"]["waves"]] == [7, 5]
    assert [job for wave in plan["execution"]["waves"] for job in wave] == [
        job["job_id"] for job in plan["jobs"]
    ]
    assert len(plan["jobs"]) == 12
    assert plan["authority"]["partial_primary_scoring_or_selection_permitted"] is False
    assert all(value is False for value in plan["sealed_gates"].values())


def test_plan_hash_rejects_budget_or_authority_mutation(tmp_path):
    path = HERE / "GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json"
    plan = json.loads(path.read_text())
    altered = copy.deepcopy(plan)
    altered["budget"]["development_billing_cutoff_usd"] = 25.0
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="payload mismatch"):
        WRAPPER.load_plan(changed)
    altered = copy.deepcopy(plan)
    altered["authority"]["partial_primary_scoring_or_selection_permitted"] = True
    changed.write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="payload mismatch"):
        WRAPPER.load_plan(changed)


def test_import_fix_supplies_exact_12_synthetic_jobs():
    fix = FREEZE.load_hashed(HERE / "GAPBALANCE_DEVELOPMENT_IMPORT_FIX.json")
    jobs = [job for job in fix["jobs"] if job["job_id"].startswith("gapbalance-development-synthetic-")]
    assert len(jobs) == 12
    assert [job["job_id"] for job in jobs] == [
        f"gapbalance-development-synthetic-seed{seed}-shard{shard:02d}"
        for seed in (11, 23, 47)
        for shard in range(4)
    ]


def test_allocation_retry_preserves_plan_budget_authority_and_sealed_gates():
    plan = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json")
    retry = RETRY.load_hashed(HERE / "GAPBALANCE_RUNPOD_ALLOCATION_RETRY.json")
    assert retry["runpod_development_plan_payload_sha256"] == plan["payload_sha256"]
    assert retry["failed_zero_cost_attempt"]["spend_usd"] == 0.0
    assert retry["failed_zero_cost_attempt"]["allocation_started"] is False
    assert retry["budget"] == plan["budget"]
    assert retry["authority"] == plan["authority"]
    assert retry["sealed_gates"] == plan["sealed_gates"]
    assert retry["authorized_retry"]["gpu_count"] == 7
    assert retry["authorized_retry"]["maximum_accepted_aggregate_hourly_rate_usd"] == 2.38


def test_multi_pod_retry_preserves_all_jobs_and_seven_gpu_ceiling():
    plan = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json")
    retry = MULTI.load_hashed(HERE / "GAPBALANCE_RUNPOD_MULTI_POD_RETRY.json")
    job_ids = [job["job_id"] for job in plan["jobs"]]
    assert retry["runpod_development_plan_payload_sha256"] == plan["payload_sha256"]
    assert retry["failed_zero_cost_attempt"]["spend_usd"] == 0.0
    assert retry["budget"] == plan["budget"]
    assert retry["authority"] == plan["authority"]
    assert retry["sealed_gates"] == plan["sealed_gates"]
    assert retry["provider_contract"]["aggregate_hourly_ceiling_usd"] == 2.38
    for layout in retry["allowed_layouts_in_order"]:
        assert sum(partition["gpu_count"] for partition in layout["partitions"]) == 7
        assert [job for partition in layout["partitions"] for job in partition["jobs"]] == job_ids
        for partition in layout["partitions"]:
            assert [job for wave in partition["waves"] for job in wave] == partition["jobs"]
            assert len(partition["waves"]) <= 2
            assert max(map(len, partition["waves"])) <= partition["gpu_count"]


def test_egress_resume_excludes_sealed_holdout_and_accounts_prior_spend():
    plan = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json")
    egress = EGRESS.load_hashed(HERE / "GAPBALANCE_RUNPOD_EGRESS_RESUME.json")
    assert egress["runpod_development_plan_payload_sha256"] == plan["payload_sha256"]
    assert egress["egress"]["replication_count"] == 3
    assert egress["egress"]["pherc1218_included"] is False
    assert egress["egress"]["confirmation_seeds_500_504_included"] is False
    assert egress["egress"]["scientific_outputs_included"] is False
    assert egress["resume"]["prior_conservative_spend_usd"] == pytest.approx(0.070296)
    assert egress["resume"]["remaining_development_cutoff_usd"] == pytest.approx(11.929704)
    assert egress["budget"] == plan["budget"]
    assert egress["sealed_gates"] == plan["sealed_gates"]


def test_result_blind_resume_failure_preserves_stopped_state_and_budget():
    plan = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json")
    failure = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_RESUME_FAILURE.json")
    assert failure["runpod_development_plan_payload_sha256"] == plan["payload_sha256"]
    assert failure["status"] == "RESULT_BLIND_EXACT_RUNPOD_RESUME_FAILED_NO_ACTIVE_BILLING"
    assert failure["added_development_spend_usd"] == 0.0
    assert failure["prior_conservative_development_spend_usd"] == pytest.approx(0.070296)
    assert [pod["desired_status"] for pod in failure["provider_status_after_failure"]] == [
        "EXITED", "EXITED", "EXITED"
    ]
    assert not failure["pherc1218_v2_opened"]
    assert not failure["scientific_endpoints_inspected"]
    assert not failure["confirmation_outputs_inspected"]


def test_replacement_reservation_preserves_budget_provider_and_egress_gate():
    plan = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json")
    retry = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_MULTI_POD_RETRY.json")
    reservation = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_REPLACEMENT_RESERVATION.json")
    assert reservation["runpod_development_plan_payload_sha256"] == plan["payload_sha256"]
    assert reservation["runpod_multi_pod_retry_payload_sha256"] == retry["payload_sha256"]
    assert reservation["provider_contract"]["total_gpu_count"] == 7
    assert reservation["provider_contract"]["aggregate_hourly_ceiling_usd"] == 2.38
    assert reservation["authorization"]["private_bundle_egress_to_replacements_permitted"] is False
    assert reservation["budget"]["prior_conservative_development_spend_usd"] == pytest.approx(0.070296)
    assert reservation["budget"]["remaining_development_cutoff_usd"] == pytest.approx(11.929704)
    assert not reservation["sealed_gates"]["pherc1218_v2_opened"]
    assert not reservation["sealed_gates"]["confirmation_outputs_inspected"]


def test_ordered_hardware_substitution_is_uniform_capped_and_still_sealed():
    plan = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json")
    substitute = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_HARDWARE_SUBSTITUTION.json")
    assert substitute["runpod_development_plan_payload_sha256"] == plan["payload_sha256"]
    assert substitute["egress"]["private_bundle_egress_to_replacements_permitted"] is False
    assert substitute["numerical_reproducibility"][
        "all_12_jobs_must_use_one_uniform_selected_hardware_type"
    ]
    assert substitute["numerical_reproducibility"]["cross_hardware_partial_mix_permitted"] is False
    assert substitute["numerical_reproducibility"]["partial_scoring_or_selection_permitted"] is False
    assert [item["gpu_type_id"] for item in substitute["allowed_hardware_in_order"]] == [
        "NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA RTX A6000"
    ]
    for item in substitute["allowed_hardware_in_order"]:
        assert item["total_gpu_count"] == 7
        assert item["gpu_memory_gb"] >= 24
        assert item["aggregate_hourly_ceiling_usd"] <= 2.38
    assert not substitute["sealed_gates"]["pherc1218_v2_opened"]
    assert not substitute["sealed_gates"]["scientific_endpoints_scored"]


def test_substitute_gpu_validation_remains_exact_and_rate_capped():
    plan = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json")
    pod = {
        "id": "replacement",
        "name": "replacement",
        "gpuCount": 2,
        "machine": {"gpuDisplayName": "RTX A5000"},
        "imageName": plan["execution"]["container_image"],
        "costPerHr": 0.32,
        "desiredStatus": "RUNNING",
    }
    ORCH.validate_pod(
        pod,
        plan,
        require_exact_reused_pod=False,
        expected_gpu_count=2,
        expected_gpu_display="RTX A5000",
        maximum_rate_usd=0.32,
    )
    pod["machine"]["gpuDisplayName"] = "RTX A6000"
    with pytest.raises(RuntimeError, match="GPU type mismatch"):
        ORCH.validate_pod(
            pod,
            plan,
            require_exact_reused_pod=False,
            expected_gpu_count=2,
            expected_gpu_display="RTX A5000",
            maximum_rate_usd=0.32,
        )


def test_substitute_reservation_receipt_is_stopped_uniform_and_not_egress_approved():
    receipt = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_SUBSTITUTE_RESERVATION_RECEIPT.json")
    assert receipt["status"] == "STOPPED_SUBSTITUTE_PODS_PUBLICLY_FROZEN_AWAITING_EXACT_EGRESS_APPROVAL"
    assert receipt["selected_hardware"] == {
        "gpu_name": "RTX 3090", "uniform_across_all_12_jobs": True
    }
    assert [pod["id"] for pod in receipt["pods"]] == ["68e48azozqvnhb", "b9outlq8lhhsnk"]
    assert [pod["desired_status"] for pod in receipt["pods"]] == ["EXITED", "EXITED"]
    assert sum(pod["gpu_count"] for pod in receipt["pods"]) == 7
    assert receipt["billing"]["aggregate_hourly_rate_usd"] == pytest.approx(1.54)
    assert receipt["billing"]["conservative_development_spend_usd"] == pytest.approx(0.083282)
    assert receipt["bundle_egress"]["permitted"] is False
    assert receipt["bundle_egress"]["explicit_approval_for_exact_replacement_ids_received"] is False
    assert not receipt["pherc1218_v2_opened"]
    assert not receipt["scientific_endpoints_inspected"]
    assert not receipt["confirmation_outputs_inspected"]


def test_exact_substitute_egress_approval_is_bound_capped_and_excludes_holdout():
    approval = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_SUBSTITUTE_EGRESS_RESUME.json")
    receipt = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_SUBSTITUTE_RESERVATION_RECEIPT.json")
    assert approval["substitute_reservation_receipt_payload_sha256"] == receipt["payload_sha256"]
    assert [pod["id"] for pod in approval["pods"]] == ["68e48azozqvnhb", "b9outlq8lhhsnk"]
    assert approval["billing"]["aggregate_hourly_rate_usd"] == pytest.approx(1.54)
    assert approval["billing"]["prior_conservative_development_spend_usd"] == pytest.approx(0.083282)
    assert approval["billing"]["remaining_development_cutoff_usd"] == pytest.approx(11.916718)
    assert approval["egress"]["pherc1218_included"] is False
    assert approval["egress"]["confirmation_seeds_500_504_included"] is False
    assert approval["egress"]["scientific_outputs_included"] is False
    assert not approval["sealed_gates"]["pherc1218_v2_opened"]
    assert not approval["sealed_gates"]["scientific_endpoints_scored"]


def test_dynamic_substitute_egress_is_bounded_and_requires_id_recording():
    approval = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_DYNAMIC_SUBSTITUTE_EGRESS.json")
    substitute = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_HARDWARE_SUBSTITUTION.json")
    assert approval["hardware_substitution_payload_sha256"] == substitute["payload_sha256"]
    assert approval["destination_contract"]["allowed_gpu_type_ids"] == [
        item["gpu_type_id"] for item in substitute["allowed_hardware_in_order"]
    ]
    assert approval["destination_contract"]["maximum_total_gpu_count"] == 7
    assert approval["destination_contract"]["maximum_aggregate_hourly_rate_usd"] == pytest.approx(2.38)
    assert approval["destination_contract"]["exact_ids_recorded_in_provider_receipt_before_upload"]
    assert approval["billing"]["prior_conservative_development_spend_usd"] == pytest.approx(0.083282)
    assert approval["egress_exclusions"]["pherc1218_included"] is False
    assert approval["egress_exclusions"]["confirmation_seeds_500_504_included"] is False
    assert not approval["sealed_gates"]["scientific_endpoints_scored"]


def test_ssh_injection_retry_is_public_key_only_and_accounts_failed_attempt():
    retry = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_SSH_INJECTION_RETRY.json")
    assert retry["failed_deployment"]["bundle_bytes_uploaded"] == 0
    assert retry["failed_deployment"]["error"] == "Permission denied (publickey,password)."
    assert retry["retry"]["inject_public_key_environment_variable"] == "SSH_PUBLIC_KEY"
    assert retry["retry"]["private_key_leaves_local_mac"] is False
    assert retry["retry"]["public_key_material_published_in_repo"] is False
    assert retry["billing"]["prior_conservative_development_spend_usd"] == pytest.approx(0.122661)
    assert retry["billing"]["remaining_development_cutoff_usd"] == pytest.approx(11.877339)
    assert not retry["sealed_gates"]["pherc1218_v2_opened"]
    assert not retry["sealed_gates"]["scientific_endpoints_scored"]


def test_ssh_public_key_fingerprint_matches_frozen_local_key():
    retry = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_SSH_INJECTION_RETRY.json")
    public_key = Path("/Users/mazalcohen/.ssh/id_ed25519.pub").read_text()
    assert ORCH.public_key_fingerprint(public_key) == retry["retry"]["public_key_fingerprint"]


def test_account_ssh_key_retry_is_result_blind_bounded_and_preallocation():
    retry = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_ACCOUNT_SSH_KEY_RETRY.json")
    ssh_retry = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_SSH_INJECTION_RETRY.json")
    assert retry["ssh_injection_retry_payload_sha256"] == ssh_retry["payload_sha256"]
    assert retry["failed_deployment"]["bundle_bytes_uploaded"] == 0
    assert retry["failed_deployment"]["pods_stopped"] == [
        "h3rlmwe7dydtpg", "dyx0gchkguob6y"
    ]
    assert retry["account_key_precheck"]["target_public_key_already_registered"] is False
    assert retry["account_key_registration"]["registration_required_before_new_pod_creation"]
    assert retry["account_key_registration"]["private_key_leaves_local_mac"] is False
    assert retry["account_key_registration"]["credit_card_charge_permitted"] is False
    assert retry["billing"]["prior_conservative_development_spend_usd"] == pytest.approx(0.159383)
    assert retry["billing"]["remaining_development_cutoff_usd"] == pytest.approx(11.840617)
    assert not retry["sealed_gates"]["pherc1218_v2_opened"]
    assert not retry["sealed_gates"]["scientific_endpoints_scored"]


def test_account_public_key_match_ignores_comments_and_unrelated_lines():
    expected = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest expected-comment"
    registered = [
        "malformed",
        "ssh-rsa AAAAB3NzaUnrelated old-key",
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest account-comment",
    ]
    assert ORCH.account_has_public_key(registered, expected)
    assert not ORCH.account_has_public_key(registered, "ssh-ed25519 AAAADifferent")
