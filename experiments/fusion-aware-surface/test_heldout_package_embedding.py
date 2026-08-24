from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import generate_heldout_job_packages as packages


def _identity(path: Path) -> dict:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": packages.sha256_file(path),
    }


def _job(mode: str, run: str) -> dict:
    prefix = "real-test" if mode == "real_test_cache" else "synthetic"
    suffix = "" if mode == "real_test_cache" else "-all-shards"
    return {
        "job_id": f"{prefix}-{run.replace('_', '-')}{suffix}",
        "mode": mode,
        "run": run,
        "kernel_id": None,
        "kernel_version": None,
    }


def test_generated_kernel_embeds_config_and_preserves_base_launcher(
    tmp_path: Path, monkeypatch
) -> None:
    root = Path(__file__).resolve().parent
    launcher = root / "heldout_cache_launcher.py"
    generator = root / "generate_heldout_job_packages.py"
    runs = (
        "baseline",
        "control_seed11",
        "control_seed23",
        "control_seed47",
        "gap8_seed11",
        "gap8_seed23",
        "gap8_seed47",
    )
    plan = {
        "schema_version": "1.0",
        "status": "held-out execution plan frozen before held-out inference",
        "heldout_cache_launcher": _identity(launcher),
        "heldout_package_generator": _identity(generator),
        "threshold_binding": {
            "kernel_id": "aviadcohen1/vesuvius-fusion-aware-freeze-thresholds",
            "kernel_version": 6,
        },
        "real_test_jobs": [_job("real_test_cache", run) for run in runs],
        "synthetic_ray_jobs": [_job("synthetic_ray_cache", run) for run in runs],
    }
    plan["payload_sha256"] = packages.canonical_sha256(plan)
    plan_path = tmp_path / "heldout_execution_plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output = tmp_path / "packages"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_heldout_job_packages.py",
            "--plan",
            str(plan_path),
            "--public-plan-commit",
            "a" * 40,
            "--out",
            str(output),
        ],
    )
    assert packages.main() == 0

    assert not list(output.rglob("heldout_job.json"))
    index = json.loads((output / "generated_packages_index.json").read_text())
    assert index["launcher"] == _identity(launcher)
    for record in index["packages"]:
        wrapper = output / record["directory"] / "heldout_cache_launcher.py"
        compile(wrapper.read_bytes(), str(wrapper), "exec")
        module_name = "heldout_fixture_" + record["job_id"].replace("-", "_")
        spec = importlib.util.spec_from_file_location(module_name, wrapper)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.load_job_config()
        assert config["job_id"] == record["job_id"]
        assert config["public_plan_commit"] == "a" * 40
        assert module.launcher_source_identity() == _identity(launcher)
        raw = module.embedded_job_config_bytes()
        assert record["embedded_job_config"] == {
            "encoding": "hex in first source comment",
            "bytes": len(raw),
            "sha256": packages.hashlib.sha256(raw).hexdigest(),
            "payload_sha256": config["payload_sha256"],
        }
