from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import collect_scoring_results as result_collector
import generate_scoring_job_packages as generator
import orchestrate_scoring_pair as pair
import pytest
import validate_final_results as validator

RUNS = (
    "baseline",
    "control_seed11",
    "control_seed23",
    "control_seed47",
    "gap8_seed11",
    "gap8_seed23",
    "gap8_seed47",
)


def test_scoring_download_suppresses_automatic_kernel_log(tmp_path: Path, monkeypatch) -> None:
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(result_collector.subprocess, "run", fake_run)
    destination = tmp_path / "selected"
    result_collector.download_selected("kaggle", "owner/kernel", destination)
    command = observed["command"]
    assert command[command.index("--page-size") + 1] == "100"
    assert command[command.index("--page-token") + 1] == ""
    assert "--force" not in command


def _identity(path: Path) -> dict:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": generator.sha256_file(path),
    }


def _canonical(payload: dict) -> dict:
    result = dict(payload)
    result["payload_sha256"] = generator.canonical_sha256(result)
    return result


def _write_public_inputs(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    root = Path(__file__).resolve().parent
    launcher = root / "one_shot_scoring_launcher.py"
    stager = root / "stage_heldout_for_scoring.py"
    metric = root / "prepare_metric_runtime.py"
    panel_renderer = root / "render_real_panels.py"
    package_generator = root / "generate_scoring_job_packages.py"
    real_jobs = [
        {
            "job_id": f"real-test-{run.replace('_', '-')}",
            "mode": "real_test_cache",
            "run": run,
        }
        for run in RUNS
    ]
    synthetic_jobs = [
        {
            "job_id": f"synthetic-{run.replace('_', '-')}-all-shards",
            "mode": "synthetic_ray_cache",
            "run": run,
        }
        for run in RUNS
    ]
    threshold = {
        "kernel_id": "aviadcohen1/vesuvius-fusion-aware-freeze-thresholds",
        "kernel_version": 6,
    }
    plan = _canonical(
        {
            "schema_version": "1.0",
            "status": "held-out execution plan frozen before held-out inference",
            "threshold_binding": threshold,
            "pre_inference_protocol_clarification": {
                "file": "PROTOCOL_CLARIFICATION.md",
                "bytes": 2036,
                "sha256": "c849d7a268465dd17b8c5d0c6774e7caab5caa343eb2dacb0cf4d862db3166dd",
                "first_public_commit": "1e0ced2c928fa0842aa8c601f39dbcd49272247e",
                "scope": "synthetic primary-gate pooling only",
            },
            "real_test_jobs": real_jobs,
            "synthetic_ray_jobs": synthetic_jobs,
            "one_shot_scoring_launcher": _identity(launcher),
            "one_shot_scoring_stager": _identity(stager),
            "metric_runtime_preparer": _identity(metric),
            "real_panel_renderer": _identity(panel_renderer),
            "scoring_package_generator": _identity(package_generator),
            "scoring_pair_controller": _identity(root / "orchestrate_scoring_pair.py"),
            "scoring_result_collector": _identity(root / "collect_scoring_results.py"),
            "one_shot_scorers": {
                "real": {
                    "script": {
                        "file": "score_real_test_parallel.py",
                        "sha256": generator.sha256_file(
                            root / "score_real_test_parallel.py"
                        ),
                    }
                },
                "synthetic": {
                    "script": {
                        "file": "score_synthetic_test_v2.py",
                        "sha256": "d594cea7d58b08bbeccab5ec65f0a3d64191a70d07e9423314cd607d9fe53d05",
                    }
                },
            },
            "result_blind_verified_parallel_real_scoring": {
                "schema_version": "1.0",
                "status": "result-blind verified parallel real scoring frozen before retry",
                "parallel_workers": 32,
                "equivalence_report": {
                    "commit": "85b0190541e402c1c20e6f7870da5669ed756521",
                    "file": "PARALLEL_REAL_SCORER_EQUIVALENCE.json",
                    "bytes": 1838,
                    "sha256": "b46fa4c7514f15fe0c75a761ac1c0857d9ec301ab79664ffab398b3f99a38960",
                    "payload_sha256": "d373745060c54a956191af64f0fd49fdea75aaade60ee61abe11cf555ac64e67",
                },
                "exact_predecessor_scorer": {
                    "file": "score_real_test.py",
                    "sha256": "3529b8213237a60d392ffec04efca602988b3242f6af8cadc87423e8e224bb79",
                },
                "scientific_contract": {
                    "per_cache_metric_subprocess_changed": False,
                    "cache_threshold_or_input_changed": False,
                    "aggregation_bootstrap_or_serialization_changed": False,
                    "frozen_cache_order_preserved": True,
                    "test_time_tuning_permitted": False,
                    "only_independent_subprocess_scheduling_changed": True,
                },
            },
        }
    )
    plan_path = tmp_path / "heldout_execution_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    jobs = real_jobs + synthetic_jobs
    delivery = _canonical(
        {
            "schema_version": "1.0",
            "status": "all 14 publicly planned held-out caches sealed before one-shot scoring",
            "public_execution_plan": {
                "commit": "a" * 40,
                "file": plan_path.name,
                "bytes": plan_path.stat().st_size,
                "sha256": generator.sha256_file(plan_path),
                "payload_sha256": plan["payload_sha256"],
            },
            "threshold_binding": threshold,
            "job_order": [job["job_id"] for job in jobs],
            "jobs": [
                {
                    "job_id": job["job_id"],
                    "mode": job["mode"],
                    "run": job["run"],
                    "kernel_id": f"aviadcohen1/vesuvius-fusion-{job['job_id']}",
                    "kernel_version": 1,
                }
                for job in jobs
            ],
            "counts": {
                "jobs": 14,
                "real_jobs": 7,
                "synthetic_jobs": 7,
                "real_probability_caches": 266,
                "synthetic_ray_caches": 700,
            },
            "scientific_gate": {
                "all_cache_jobs_completed": True,
                "thresholds_publicly_frozen_before_cache_inference": True,
                "scientific_endpoints_scored": False,
                "scientific_endpoints_printed": False,
                "cache_npz_payloads_opened_or_inspected_by_delivery_freezer": False,
                "one_shot_scoring_permitted_after_this_public_freeze": True,
            },
        }
    )
    delivery_path = tmp_path / "heldout_cache_delivery_manifest.json"
    delivery_path.write_text(
        json.dumps(delivery, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return plan_path, delivery_path, plan, delivery


def test_scoring_packages_are_single_file_and_fetch_hash_bound_helpers(
    tmp_path: Path, monkeypatch
) -> None:
    plan_path, delivery_path, plan, delivery = _write_public_inputs(tmp_path)
    output = tmp_path / "scoring-packages"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_scoring_job_packages.py",
            "--plan",
            str(plan_path),
            "--delivery",
            str(delivery_path),
            "--public-plan-commit",
            "a" * 40,
            "--public-delivery-commit",
            "b" * 40,
            "--out",
            str(output),
        ],
    )
    assert generator.main() == 0
    assert not list(output.rglob("scoring_job.json"))
    assert not list(output.rglob("stage_heldout_for_scoring.py"))
    assert not list(output.rglob("prepare_metric_runtime.py"))

    index = generator.load_hashed(output / "generated_scoring_packages_index.json")
    assert index["package_count"] == 2
    for record in index["packages"]:
        package = output / record["directory"]
        assert {path.name for path in package.iterdir()} == {
            "one_shot_scoring_launcher.py",
            "kernel-metadata.json",
            "KERNEL_SHA256SUMS",
        }
        wrapper = package / "one_shot_scoring_launcher.py"
        compile(wrapper.read_bytes(), str(wrapper), "exec")
        spec = importlib.util.spec_from_file_location("scoring_fixture_" + record["mode"], wrapper)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.load_config()
        assert config["mode"] == record["mode"]
        assert module.launcher_source_identity() == plan["one_shot_scoring_launcher"]
        raw = module.embedded_scoring_config_bytes()
        assert record["embedded_job_config"] == {
            "encoding": "hex in first source comment",
            "bytes": len(raw),
            "sha256": generator.hashlib.sha256(raw).hexdigest(),
            "payload_sha256": config["payload_sha256"],
        }

        helpers = {
            plan["one_shot_scoring_stager"]["file"]: (
                Path(__file__).resolve().with_name("stage_heldout_for_scoring.py")
            ).read_bytes(),
            plan["metric_runtime_preparer"]["file"]: (
                Path(__file__).resolve().with_name("prepare_metric_runtime.py")
            ).read_bytes(),
            plan["real_panel_renderer"]["file"]: (
                Path(__file__).resolve().with_name("render_real_panels.py")
            ).read_bytes(),
            "score_synthetic_test_v2.py": (
                Path(__file__).resolve().with_name("score_synthetic_test_v2.py")
            ).read_bytes(),
            "PARALLEL_REAL_SCORER_EQUIVALENCE.json": (
                Path(__file__).resolve().with_name(
                    "PARALLEL_REAL_SCORER_EQUIVALENCE.json"
                )
            ).read_bytes(),
            "score_real_test_parallel.py": (
                Path(__file__).resolve().with_name("score_real_test_parallel.py")
            ).read_bytes(),
        }
        monkeypatch.setattr(
            module,
            "fetch_public",
            lambda _commit, name, _sha, helpers=helpers: helpers[name],
        )
        scratch = tmp_path / ("helpers-" + record["mode"])
        scratch.mkdir()
        stager = module.materialize_public_helper(
            "a" * 40, plan["one_shot_scoring_stager"], scratch
        )
        metric = module.materialize_public_helper(
            "a" * 40, plan["metric_runtime_preparer"], scratch
        )
        panel_renderer = module.materialize_public_helper(
            "a" * 40, plan["real_panel_renderer"], scratch
        )
        module.verify_public_contract(
            plan, delivery, record["mode"], stager, metric, panel_renderer
        )
        scorer = module.materialize_public_scorer(record["mode"], "a" * 40, scratch)
        expected_scorer = (
            "score_real_test_parallel.py"
            if record["mode"] == "real"
            else "score_synthetic_test_v2.py"
        )
        assert module.sha256_file(scorer) == module.PROJECT_HASHES[expected_scorer]


def test_pair_controller_accepts_both_before_result_collection(tmp_path: Path, monkeypatch) -> None:
    plan_path, delivery_path, _, _ = _write_public_inputs(tmp_path)
    packages_root = tmp_path / "scoring-packages"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_scoring_job_packages.py",
            "--plan",
            str(plan_path),
            "--delivery",
            str(delivery_path),
            "--public-plan-commit",
            "a" * 40,
            "--public-delivery-commit",
            "b" * 40,
            "--out",
            str(packages_root),
        ],
    )
    assert generator.main() == 0
    index_path = packages_root / "generated_scoring_packages_index.json"
    index = generator.load_hashed(index_path)
    statuses = {record["kaggle_kernel_id"]: None for record in index["packages"]}
    monkeypatch.setattr(
        pair.queue,
        "kernel_status",
        lambda _kaggle, kernel_id: statuses[kernel_id],
    )

    def fake_push(_kaggle: str, _root: Path, record: dict) -> int:
        statuses[record["kaggle_kernel_id"]] = "PENDING"
        return 1

    monkeypatch.setattr(pair, "push_package", fake_push)
    receipt_path = tmp_path / "scoring_pair_receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "orchestrate_scoring_pair.py",
            "--packages",
            str(packages_root),
            "--index",
            str(index_path),
            "--receipt",
            str(receipt_path),
        ],
    )
    assert pair.main() == 0
    receipt = pair.queue.load_canonical(receipt_path)
    assert [record["mode"] for record in receipt["accepted"]] == [
        "real",
        "synthetic",
    ]

    for kernel_id in statuses:
        statuses[kernel_id] = "COMPLETE"
    package_by_kernel = {record["kaggle_kernel_id"]: record for record in index["packages"]}

    def fake_current(kernel_id: str) -> tuple[int, bytes]:
        package = package_by_kernel[kernel_id]
        source = (
            packages_root / package["directory"] / "one_shot_scoring_launcher.py"
        ).read_bytes()
        return 1, source

    def fake_download(_kaggle: str, kernel_id: str, destination: Path) -> None:
        destination.mkdir()
        mode = package_by_kernel[kernel_id]["mode"]
        (destination / result_collector.RESULT_NAMES[mode]).write_text("{}\n", encoding="utf-8")
        (destination / "scoring_run_manifest.json").write_text("{}\n", encoding="utf-8")
        if mode == "real":
            (destination / "real_panel_render_manifest.json").write_text("{}\n", encoding="utf-8")
            for index in range(1, 5):
                (destination / f"real_panel_{index:02d}.png").write_bytes(b"png")

    monkeypatch.setattr(
        result_collector.heldout_collector,
        "current_kernel_version_and_source",
        fake_current,
    )
    monkeypatch.setattr(result_collector, "download_selected", fake_download)
    output = tmp_path / "scoring-results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_scoring_results.py",
            "--packages",
            str(packages_root),
            "--package-index",
            str(index_path),
            "--pair-receipt",
            str(receipt_path),
            "--out",
            str(output),
        ],
    )
    assert result_collector.main() == 0
    sources = pair.queue.load_canonical(output / "scoring_result_sources.json")
    assert set(sources["artifacts"]) == {"real", "synthetic"}
    assert sources["scientific_gate"]["both_scorers_completed_before_first_result_download"] is True
    result_paths = {
        mode: output / "downloads" / mode / result_collector.RESULT_NAMES[mode]
        for mode in pair.MODES
    }
    run_paths = {
        mode: output / "downloads" / mode / "scoring_run_manifest.json" for mode in pair.MODES
    }
    delivery = generator.load_hashed(delivery_path)
    assert (
        validator.validate_scoring_result_collection(
            plan=generator.load_hashed(plan_path),
            delivery=delivery,
            plan_path=plan_path,
            delivery_path=delivery_path,
            result_sources_path=output / "scoring_result_sources.json",
            package_index_path=index_path,
            pair_receipt_path=receipt_path,
            real_run_path=run_paths["real"],
            synthetic_run_path=run_paths["synthetic"],
            real_result_path=result_paths["real"],
            synthetic_result_path=result_paths["synthetic"],
            real_panel_manifest_path=(
                output / "downloads" / "real" / "real_panel_render_manifest.json"
            ),
            real_panel_image_paths=[
                output / "downloads" / "real" / f"real_panel_{index:02d}.png"
                for index in range(1, 5)
            ],
        )
        == "b" * 40
    )

    tampered = dict(sources)
    tampered["artifacts"] = {mode: dict(record) for mode, record in sources["artifacts"].items()}
    tampered["artifacts"]["real"]["source_sha256"] = "0" * 64
    tampered.pop("payload_sha256")
    tampered["payload_sha256"] = generator.canonical_sha256(tampered)
    tampered_path = tmp_path / "tampered_scoring_result_sources.json"
    tampered_path.write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="artifact provenance mismatch"):
        validator.validate_scoring_result_collection(
            plan=generator.load_hashed(plan_path),
            delivery=delivery,
            plan_path=plan_path,
            delivery_path=delivery_path,
            result_sources_path=tampered_path,
            package_index_path=index_path,
            pair_receipt_path=receipt_path,
            real_run_path=run_paths["real"],
            synthetic_run_path=run_paths["synthetic"],
            real_result_path=result_paths["real"],
            synthetic_result_path=result_paths["synthetic"],
            real_panel_manifest_path=(
                output / "downloads" / "real" / "real_panel_render_manifest.json"
            ),
            real_panel_image_paths=[
                output / "downloads" / "real" / f"real_panel_{index:02d}.png"
                for index in range(1, 5)
            ],
        )
