from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


controller = load_module("orchestrate_runpod_primary.py", "runpod_controller")
executor = load_module("runpod_execute_primary.py", "runpod_executor")


class FakeRunPod:
    def __init__(self, pod: dict):
        self.pod = pod
        self.stopped = []

    def get_pod(self, pod_id: str) -> dict:
        assert pod_id == self.pod["id"]
        return self.pod

    def stop_pod(self, pod_id: str) -> None:
        self.stopped.append(pod_id)


class RunPodReplacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan_path = HERE / "runpod_synthetic_replacement_plan.json"
        self.plan = controller.load_plan(self.plan_path)

    def test_primary_replication_and_budget_contract(self) -> None:
        self.assertTrue(self.plan["execution"]["runpod_is_authoritative_primary"])
        self.assertEqual(
            self.plan["kaggle_replication"]["role"],
            "secondary cross-platform replication",
        )
        self.assertFalse(
            self.plan["kaggle_replication"]["selection_between_platforms_permitted"]
        )
        self.assertEqual(self.plan["budget"]["hard_total_cap_usd"], 20.0)
        self.assertLessEqual(self.plan["budget"]["billing_cutoff_usd"], 18.5)
        self.assertEqual(len(self.plan["jobs"]), 7)

    def test_price_gate_rejects_expensive_gpu(self) -> None:
        pod = {
            "costPerHr": 3.01,
            "gpuCount": 7,
            "id": "expensive",
            "imageName": self.plan["container"]["image"],
            "machine": {"gpuDisplayName": "NVIDIA GeForce RTX 4090"},
        }
        with self.assertRaisesRegex(RuntimeError, "exceeds frozen ceiling"):
            controller.validate_pod(pod, self.plan, 7)

    def test_scipy_operational_dependency_is_exactly_pinned(self) -> None:
        self.assertEqual(self.plan["frozen_runtime"]["scipy"], "1.16.3")
        executor_source = (HERE / "runpod_execute_primary.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"scipy==1.16.3"', executor_source)

    def test_guarded_budget_gate_stops_before_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "receipt.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "billing_cutoff_usd": 18.5,
                        "plan_payload_sha256": self.plan["payload_sha256"],
                        "pods": [{"cost_per_hour_usd": 2.8, "id": "pod-1"}],
                    }
                ),
                encoding="utf-8",
            )
            fake = FakeRunPod(
                {
                    "costPerHr": 2.8,
                    "desiredStatus": "RUNNING",
                    "id": "pod-1",
                    "uptimeSeconds": 23_500,
                }
            )
            args = argparse.Namespace(
                enforce=True,
                guard_seconds=300,
                interval=1,
                receipt=receipt_path,
            )
            with mock.patch.object(controller, "load_runpod", return_value=fake):
                controller.status(args, self.plan, watch=False)
            self.assertEqual(fake.stopped, ["pod-1"])
            self.assertEqual(
                json.loads(receipt_path.read_text())["status"],
                "BUDGET_CUTOFF_STOPPED",
            )

    def test_bundle_verifier_rejects_unexpected_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.txt"
            payload.write_text("sealed input\n", encoding="utf-8")
            record = {
                "bytes": payload.stat().st_size,
                "path": "payload.txt",
                "sha256": executor.sha256_file(payload),
            }
            manifest = {
                "files": [record],
                "job_ids": [self.plan["jobs"][0]["job_id"]],
                "plan_payload_sha256": self.plan["payload_sha256"],
            }
            manifest["payload_sha256"] = executor.sha256_bytes(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            )
            manifest_path = root / "bundle_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            executor.verify_bundle(root, manifest_path, self.plan)
            (root / "unexpected.txt").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "file-set mismatch"):
                executor.verify_bundle(root, manifest_path, self.plan)


if __name__ == "__main__":
    unittest.main()
