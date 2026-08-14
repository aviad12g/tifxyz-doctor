import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("resume_parallel_runpod_real_scoring_transfer.py")
SPEC = importlib.util.spec_from_file_location("parallel_transfer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ParallelTransferLayoutTest(unittest.TestCase):
    def test_exact_seven_job_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runpod_real_input_manifest.json").write_text("{}")
            jobs = root / "jobs"
            jobs.mkdir()
            for name in MODULE.EXPECTED_JOBS:
                (jobs / name).mkdir()
            self.assertEqual(
                tuple(path.name for path in MODULE.validate_input_layout(root)),
                MODULE.EXPECTED_JOBS,
            )

    def test_rejects_extra_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runpod_real_input_manifest.json").write_text("{}")
            jobs = root / "jobs"
            jobs.mkdir()
            for name in MODULE.EXPECTED_JOBS + ("extra",):
                (jobs / name).mkdir()
            with self.assertRaises(RuntimeError):
                MODULE.validate_input_layout(root)


if __name__ == "__main__":
    unittest.main()
