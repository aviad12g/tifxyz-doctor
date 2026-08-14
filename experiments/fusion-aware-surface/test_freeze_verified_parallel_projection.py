"""Checks for the result-blind verified parallel projection freeze."""

from pathlib import Path

import freeze_verified_parallel_projection as freezer


HERE = Path(__file__).resolve().parent


def test_projection_correction_binds_current_stager_without_science() -> None:
    record = freezer.correction(HERE / "stage_heldout_for_scoring.py")
    assert record["held_out_result_opened_or_used"] is False
    assert record["scientific_contract_changed"] is False
    assert record["corrected_scoring_stager"]["sha256"] == freezer.sha256_file(
        HERE / "stage_heldout_for_scoring.py"
    )
