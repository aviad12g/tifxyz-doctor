from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_cpu_exact_launcher_uses_original_scorer_without_parallel_arguments() -> None:
    source = (HERE / "one_shot_scoring_launcher_cpu_exact.py").read_text(encoding="utf-8")
    assert '"score_real_test.py": "3529b8213237a60d392ffec04efca602988b3242f6af8cadc87423e8e224bb79"' in source
    assert '"--parallel-workers"' not in source
    assert "REAL_PARALLEL_WORKERS" not in source


def test_exact_scorer_freezer_refuses_to_assume_equivalence() -> None:
    source = (HERE / "freeze_runpod_cpu_exact_scorer_correction.py").read_text(
        encoding="utf-8"
    )
    assert '"parallel_scorer_scientific_equivalence_assumed": False' in source
    assert '"parallel_scorer_used_for_retry": False' in source
    assert '"exact_original_real_scorer_restored": ORIGINAL_REAL_SCORER' in source
    assert '"scientific_result_created_or_opened": False' in source
