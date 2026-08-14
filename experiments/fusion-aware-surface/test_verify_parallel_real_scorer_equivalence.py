"""Structural checks for the public official-metric equivalence verifier."""

from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_equivalence_verifier_is_result_blind_and_fail_closed() -> None:
    source = (HERE / "verify_parallel_real_scorer_equivalence.py").read_text(
        encoding="utf-8"
    )
    assert "held_out_cache_used\": False" in source
    assert "for workers in (4, 16, 32)" in source
    assert "observed == sequential" in source
    assert "result_sha256\"] == sequential_sha256" in source
    assert "np.load" not in source


def test_parallel_scorer_only_changes_independent_scheduling() -> None:
    exact = (HERE / "score_real_test.py").read_text(encoding="utf-8")
    parallel = (HERE / "score_real_test_parallel.py").read_text(encoding="utf-8")
    assert "ThreadPoolExecutor" not in exact
    assert "ThreadPoolExecutor" in parallel
    assert "executor.map(scorer, caches)" in parallel
    for frozen in (
        "BOOTSTRAP_SEED = 20260808",
        "BOOTSTRAP_REPLICATES = 10_000",
        'METRICS = ("blend", "toposcore", "surface_dice", "voi_score")',
        "SENSITIVITY_THRESHOLDS = (0.4, 0.5, 0.6)",
        '"real_blend_noninferiority": pooled["blend"]["mean"] >= -0.005',
        '"real_toposcore_improvement": pooled["toposcore"]["mean"] > 0.0',
    ):
        assert frozen in exact
        assert frozen in parallel
