"""Checks for result-blind correction of locally predicted Git bindings."""

from pathlib import Path

import correct_verified_parallel_commit_bindings as correction


HERE = Path(__file__).resolve().parent


def test_actual_equivalence_commit_exists_and_contains_exact_report() -> None:
    assert correction.ACTUAL_EQUIVALENCE_COMMIT == (
        "85b0190613ea35b73f16079a67ae52d9eaa9cf8e"
    )
    assert correction.sha256_file(
        HERE / "PARALLEL_REAL_SCORER_EQUIVALENCE.json"
    ) == "b46fa4c7514f15fe0c75a761ac1c0857d9ec301ab79664ffab398b3f99a38960"


def test_correction_is_operational_and_result_blind() -> None:
    record = correction.correction_record()
    assert record["scientific_outputs_opened_or_used"] is False
    assert record["scientific_contract_changed"] is False
    assert record["erroneous_unpublished_equivalence_commit"] != record[
        "actual_equivalence_commit"
    ]
