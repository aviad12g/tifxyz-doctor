from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_reviewed_patch_benchmark.py"
SPEC = importlib.util.spec_from_file_location(
    "run_reviewed_patch_benchmark",
    RUNNER_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load benchmark runner: {RUNNER_PATH}")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class ReviewedBenchmarkRunnerTests(unittest.TestCase):
    def test_component_bootstrap_is_deterministic(self) -> None:
        observations = [
            {"overlap_component_id": "a", "detected": True},
            {"overlap_component_id": "a", "detected": False},
            {"overlap_component_id": "b", "detected": True},
        ]

        first = RUNNER._component_bootstrap_rate(
            observations,
            "detected",
            iterations=250,
        )
        second = RUNNER._component_bootstrap_rate(
            observations,
            "detected",
            iterations=250,
        )

        self.assertEqual(first, second)
        self.assertIsNotNone(first)
        assert first is not None
        self.assertGreaterEqual(first[0], 0.0)
        self.assertLessEqual(first[1], 1.0)

    def test_repository_paths_are_portable(self) -> None:
        path = PROJECT_ROOT / "benchmarks" / "example.json"

        self.assertEqual(
            RUNNER._portable_path(path),
            "benchmarks/example.json",
        )


if __name__ == "__main__":
    unittest.main()
