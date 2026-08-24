"""Projection checks for the result-blind verified parallel scorer records."""

from pathlib import Path

import stage_heldout_for_scoring as stager


HERE = Path(__file__).resolve().parent


def test_verified_parallel_records_are_operational_projection_metadata() -> None:
    assert {
        "result_blind_verified_parallel_real_scoring",
        "result_blind_public_commit_binding_correction",
        "result_blind_verified_parallel_projection_correction",
    } <= stager.OPERATIONAL_PLAN_FIELDS


def test_projection_drops_only_verified_parallel_operational_records() -> None:
    plan = {
        "schema_version": "1.0",
        "model": {"frozen": True},
        "result_blind_verified_parallel_real_scoring": {"parallel_workers": 32},
        "result_blind_public_commit_binding_correction": {"result_blind": True},
        "result_blind_verified_parallel_projection_correction": {"result_blind": True},
    }
    assert stager.scientific_projection(plan) == {
        "schema_version": "1.0",
        "model": {"frozen": True},
    }
