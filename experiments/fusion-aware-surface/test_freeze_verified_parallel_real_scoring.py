"""Tests for the result-blind verified parallel scorer freeze."""

from pathlib import Path

import freeze_verified_parallel_real_scoring as freezer


HERE = Path(__file__).resolve().parent


def test_public_equivalence_report_is_exact_and_complete() -> None:
    observed = freezer.equivalence_identity(
        HERE / "PARALLEL_REAL_SCORER_EQUIVALENCE.json"
    )
    assert observed == {
        "commit": "85b0190613ea35b73f16079a67ae52d9eaa9cf8e",
        "file": "PARALLEL_REAL_SCORER_EQUIVALENCE.json",
        "bytes": 1838,
        "sha256": "b46fa4c7514f15fe0c75a761ac1c0857d9ec301ab79664ffab398b3f99a38960",
        "payload_sha256": "d373745060c54a956191af64f0fd49fdea75aaade60ee61abe11cf555ac64e67",
    }


def test_launcher_uses_only_frozen_parallel_worker_count() -> None:
    source = (HERE / "one_shot_scoring_launcher.py").read_text(encoding="utf-8")
    assert "REAL_PARALLEL_WORKERS = 32" in source
    assert 'command.extend(["--parallel-workers", str(REAL_PARALLEL_WORKERS)])' in source
    assert '"score_real_test_parallel.py"' in source
    assert '"score_real_test.py"' in source
