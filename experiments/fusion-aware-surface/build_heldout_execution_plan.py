#!/usr/bin/env python3
"""Freeze the exact held-out job matrix without opening held-out outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

PREREGISTRATION_COMMIT = "5ca0444fb31863c8e02466316bf9e560cf567876"
PUBLIC_CHECKPOINT_FREEZE_COMMIT = "5d98eb683d692c7d4fe7b8f9e4dd0a285e8bf4ff"
PUBLIC_CHECKPOINT_MANIFEST_SHA256 = (
    "b8dec8f2c21d2e3686e3435a136de747b3e95f47820b2eb8a41b7044808f8564"
)
PUBLIC_INPUT_FREEZE_COMMIT = "62521d98a1771f736cf48dc1e9609a04a5779bb2"
PUBLIC_INPUT_MANIFEST_SHA256 = (
    "9e0e70b1e98d9d879ac0b27ba902cbfdc960f4577580146488f076efebda9fad"
)
SOURCE_CHECKPOINT_SHA256 = (
    "f1990a02ac91889c1f989522ae0e45421a91cb666320448aaf579d42b081636f"
)
REAL_SPLIT_MANIFEST_SHA256 = (
    "dedc881134d9de2ed2605162f82dfb219b52c6e05629c68223100b148b15d4fe"
)
REAL_SPLIT_RECORDS_SHA256 = (
    "20c600d6061bf8715ada20423a05c2f03a9bc14827b2c79bac7b7e1b3cc0499d"
)
REAL_PANEL_MANIFEST_SHA256 = (
    "e1799845c7b8bd428e6bd374f745d6b42399673a5b8c2c11d9282324b3e72779"
)
REAL_PANEL_PAYLOAD_SHA256 = (
    "65dbd84d13aac0fe2e27baf877e5b836d58f2202f9c5e9bcf84bb41694500844"
)
PROTOCOL_CLARIFICATION_SHA256 = (
    "c849d7a268465dd17b8c5d0c6774e7caab5caa343eb2dacb0cf4d862db3166dd"
)
PROTOCOL_CLARIFICATION_COMMIT = "1e0ced2c928fa0842aa8c601f39dbcd49272247e"
VALIDATION_CACHE_KERNEL_ID = "aviadcohen1/vesuvius-fusion-validation-cache-all"
VALIDATION_CACHE_KERNEL_VERSION = 4
VALIDATION_CACHE_LAUNCHER_SHA256 = (
    "b741d2a031f88fe41c27ba218ebc3dfb7f274ac1d6efaa808ab3c345e38fe6bb"
)
VALIDATION_CACHE_PAYLOAD_SHA256 = (
    "921a47f2a4ed182e95863ae4cec2ccbd4484013c14b444070783fe2bebe53faf"
)
THRESHOLD_KERNEL_ID = "aviadcohen1/vesuvius-fusion-aware-freeze-thresholds"
THRESHOLD_LAUNCHER_SHA256 = (
    "115c8f27ca5f0a8dee9fe9e56883f9ecbfa4e76dd22742a0f8ba39d5b95a3af9"
)
METRIC_DATASET_ID = "sohier/vesuvius-metric-resources"
METRIC_DATASET_VERSION = 1
METRIC_RUNTIME_DATASET_ID = "aviadcohen1/vesuvius-metric-runtime-cp312"
METRIC_RUNTIME_DATASET_VERSION = 1
METRIC_RUNTIME_MANIFEST_SHA256 = (
    "e4a78330328700f6a873d417f82073a454fff75e909c23757b325e9f195deb01"
)
METRIC_RUNTIME_PAYLOAD_SHA256 = (
    "f768df2789f680efcb6f412cce64567cc38b8916e128e6edb6b3eff6c802d3ad"
)
METRIC_RUNTIME_REQUIREMENTS_SHA256 = (
    "2950f16b007540f8c0233af2d338cdbabaf367531e4d3b94dd94fb8c2af04f64"
)
METRIC_RUNTIME_WHEEL_LEDGER_SHA256 = (
    "569720cd792c2b8b73e9d703405b83ea61b9d587122253c355be9fdc9f5c83bf"
)
METRIC_RUNTIME_WHEEL_BYTES = 139_954_739
RUNTIME_PACKAGES = {
    "absl-py": "2.3.1",
    "cmake": "3.31.6",
    "connected-components-3d": "3.26.0",
    "imagecodecs": "2025.8.2",
    "imageio": "2.37.0",
    "lazy-loader": "0.4",
    "networkx": "3.5",
    "numpy": "1.26.4",
    "packaging": "25.0",
    "pillow": "12.0.0",
    "pybind11": "2.13.6",
    "pybind11-global": "2.13.6",
    "scikit-image": "0.25.2",
    "scipy": "1.15.3",
    "surface-distance": "0.1",
    "tifffile": "2025.6.11",
}

RUN_ORDER = (
    "baseline",
    "control_seed11",
    "control_seed23",
    "control_seed47",
    "gap8_seed11",
    "gap8_seed23",
    "gap8_seed47",
)
THRESHOLD_GRID = tuple(round(0.30 + 0.05 * index, 2) for index in range(9))
SYNTHETIC_SHARDS = 10
SCORER_SOURCE_HASHES = {
    "real": (
        "score_real_test.py",
        "3529b8213237a60d392ffec04efca602988b3242f6af8cadc87423e8e224bb79",
    ),
    "synthetic": (
        "score_synthetic_test_v2.py",
        "d594cea7d58b08bbeccab5ec65f0a3d64191a70d07e9423314cd607d9fe53d05",
    ),
}

TRAINED = {
    "control_seed11": {
        "kernel_id": "aviadcohen1/vesuvius-fusion-aware-control-seed-11",
        "kernel_version": 4,
        "model_state_sha256": "7a2e6168f32b3a3389bdc2b43b39a6654568a6da467e71948f523e7cbef6248f",
        "training_run_manifest_bytes": 2481,
        "training_run_manifest_sha256": "18b52d4fa2895307fa282a3b26e9cdeeeafd4204991c8ed4636fe366065a6f82",
    },
    "control_seed23": {
        "kernel_id": "aviadcohen1/vesuvius-fusion-aware-control-seed-23",
        "kernel_version": 4,
        "model_state_sha256": "d40c4b856c9a65cc2de127e98d37d08633e03e6b7b5e375bec8991b1bf9487b3",
        "training_run_manifest_bytes": 2481,
        "training_run_manifest_sha256": "9ad56528323f821e60c98faff48265c327a4ede351148e0596a14737aa493f2d",
    },
    "control_seed47": {
        "kernel_id": "aviadcohen1/vesuvius-fusion-aware-control-seed-47",
        "kernel_version": 1,
        "model_state_sha256": "fd86b40b3b25f45f86f2ba43668997d4351fec7d3ebd35fa41856e4c0d505e44",
        "training_run_manifest_bytes": 2481,
        "training_run_manifest_sha256": "2eede7befa96f01aed362040705da55643fffe287b11c9759596e5197f99ecb2",
    },
    "gap8_seed11": {
        "kernel_id": "aviadcohen1/vesuvius-fusion-aware-gap8-seed-11",
        "kernel_version": 1,
        "model_state_sha256": "d3061db89cb48a3a619910bd57aebd5c8f2ae3439074874c9ff8a84c71b6c2f3",
        "training_run_manifest_bytes": 2463,
        "training_run_manifest_sha256": "cb590391f03ce81f8f3d350490a367ec9618e59f689baccec2277b9b79d9680a",
    },
    "gap8_seed23": {
        "kernel_id": "aviadcohen1/vesuvius-fusion-aware-gap8-seed-23",
        "kernel_version": 1,
        "model_state_sha256": "1b8df06140ebbc58733648f938452afc42757f9c184593a2b1dcb4e8fa7e4dde",
        "training_run_manifest_bytes": 2463,
        "training_run_manifest_sha256": "963ef73888cb70066f0773bb2f75b1a7d11ab4c9b0568453bff005536c3e0021",
    },
    "gap8_seed47": {
        "kernel_id": "aviadcohen1/vesuvius-fusion-aware-gap8-seed-47",
        "kernel_version": 1,
        "model_state_sha256": "022f3921b241f1f59af655d7be3434a59025dc3f4af226779398998c2e3acf6a",
        "training_run_manifest_bytes": 2463,
        "training_run_manifest_sha256": "f472604140dcc1f72ba9749edb0603b043bcd50d0efe0f69ceed06086c75d2a2",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def load_hashed_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    observed = payload.get("payload_sha256")
    without_hash = dict(payload)
    without_hash.pop("payload_sha256", None)
    expected = canonical_sha256(without_hash)
    if observed != expected:
        raise RuntimeError(f"embedded payload SHA-256 mismatch: {path}")
    return payload


def validate_thresholds(path: Path) -> dict:
    payload = load_hashed_json(path)
    checks = {
        "schema_version": "1.0",
        "status": "thresholds frozen from Scroll-1 validation before test inference",
        "candidate_thresholds": list(THRESHOLD_GRID),
        "tie_break": "maximum mean official blend; nearest 0.5; lower threshold",
        "source_split_records_sha256": REAL_SPLIT_RECORDS_SHA256,
        "validation_patch_count": 24,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"threshold manifest mismatch for {key}")
    runs = payload.get("runs")
    if (
        not isinstance(runs, dict)
        or tuple(runs) != RUN_ORDER
        or set(runs) != set(RUN_ORDER)
    ):
        raise RuntimeError("threshold run order mismatch")
    threshold_keys = {str(threshold) for threshold in THRESHOLD_GRID}
    metric_keys = {"blend", "toposcore", "surface_dice", "voi_score"}
    for run in RUN_ORDER:
        record = runs[run]
        if not isinstance(record, dict) or set(record) != {
            "selected_threshold",
            "mean_blend",
            "per_patch_official",
            "validation_files",
        }:
            raise RuntimeError(f"{run}: threshold-result schema mismatch")
        selected = record.get("selected_threshold")
        if selected not in THRESHOLD_GRID:
            raise RuntimeError(f"{run}: threshold outside frozen grid")
        means = record.get("mean_blend")
        rows_by_threshold = record.get("per_patch_official")
        if not isinstance(means, dict) or set(means) != threshold_keys:
            raise RuntimeError(f"{run}: mean-blend grid mismatch")
        if (
            not isinstance(rows_by_threshold, dict)
            or set(rows_by_threshold) != threshold_keys
        ):
            raise RuntimeError(f"{run}: per-patch metric grid mismatch")
        recomputed_means = {}
        for threshold in THRESHOLD_GRID:
            key = str(threshold)
            rows = rows_by_threshold[key]
            if not isinstance(rows, list) or len(rows) != 24:
                raise RuntimeError(f"{run}@{threshold}: expected 24 metric rows")
            blends = []
            for row in rows:
                if not isinstance(row, dict) or set(row) != metric_keys:
                    raise RuntimeError(f"{run}@{threshold}: metric schema mismatch")
                for metric, raw_value in row.items():
                    if isinstance(raw_value, bool):
                        raise TypeError(f"{run}@{threshold}: boolean {metric}")
                    value = float(raw_value)
                    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise RuntimeError(f"{run}@{threshold}: invalid {metric} value")
                blends.append(float(row["blend"]))
            recomputed = statistics.fmean(blends)
            recorded = means[key]
            if isinstance(recorded, bool) or not math.isclose(
                float(recorded), recomputed, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise RuntimeError(f"{run}@{threshold}: mean-blend mismatch")
            recomputed_means[threshold] = float(recorded)
        expected_selected = min(
            THRESHOLD_GRID,
            key=lambda threshold: (
                -recomputed_means[threshold],
                abs(threshold - 0.5),
                threshold,
            ),
        )
        if selected != expected_selected:
            raise RuntimeError(f"{run}: frozen threshold violates the tie-break")
        validation_files = record.get("validation_files")
        if not isinstance(validation_files, list) or len(validation_files) != 24:
            raise RuntimeError(f"{run}: threshold source is not 24 validation files")
        if any(
            not isinstance(item, dict)
            or set(item) != {"file", "bytes", "sha256"}
            or not isinstance(item["file"], str)
            or not item["file"].endswith(".npz")
            or not isinstance(item["bytes"], int)
            or item["bytes"] <= 0
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            for item in validation_files
        ):
            raise RuntimeError(f"{run}: validation-file identity schema mismatch")
        if len({item["file"] for item in validation_files}) != 24:
            raise RuntimeError(f"{run}: duplicate validation-file identity")
    return payload


def validate_threshold_run_manifest(
    path: Path, thresholds_path: Path, thresholds: dict
) -> dict:
    payload = load_hashed_json(path)
    checks = {
        "schema_version": "1.0",
        "status": "thresholds frozen from verified Scroll-1 validation caches before test inference",
        "public_checkpoint_freeze_commit": PUBLIC_CHECKPOINT_FREEZE_COMMIT,
        "public_checkpoint_freeze_manifest_sha256": PUBLIC_CHECKPOINT_MANIFEST_SHA256,
        "public_evaluation_input_freeze_commit": PUBLIC_INPUT_FREEZE_COMMIT,
        "public_evaluation_input_freeze_manifest_sha256": PUBLIC_INPUT_MANIFEST_SHA256,
        "source_split_manifest_sha256": REAL_SPLIT_MANIFEST_SHA256,
        "source_split_records_sha256": REAL_SPLIT_RECORDS_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"threshold run manifest mismatch for {key}")
    cache = payload.get("validation_cache", {})
    cache_checks = {
        "kernel_id": VALIDATION_CACHE_KERNEL_ID,
        "kernel_version": VALIDATION_CACHE_KERNEL_VERSION,
        "launcher_sha256": VALIDATION_CACHE_LAUNCHER_SHA256,
        "run_order": list(RUN_ORDER),
        "validation_patch_count_per_run": 24,
    }
    for key, expected in cache_checks.items():
        if cache.get(key) != expected:
            raise RuntimeError(f"threshold validation-cache binding mismatch for {key}")
    if (
        cache.get("aggregate_index", {}).get("payload_sha256")
        != VALIDATION_CACHE_PAYLOAD_SHA256
    ):
        raise RuntimeError(
            "threshold run manifest points to another validation-cache payload"
        )
    launcher = payload.get("threshold_launcher", {})
    if launcher.get("sha256") != THRESHOLD_LAUNCHER_SHA256:
        raise RuntimeError("threshold launcher SHA-256 mismatch")
    gate = payload.get("scientific_gate", {})
    expected_gate = {
        "held_out_test_or_synthetic_inference_or_scoring_executed": False,
        "held_out_test_or_synthetic_outputs_opened_or_inspected": False,
        "threshold_selector_invocations": 1,
        "thresholds_selected_from": "Scroll-1 validation only",
    }
    for key, expected in expected_gate.items():
        if gate.get(key) != expected:
            raise RuntimeError(f"threshold blind gate mismatch for {key}")
    metric = payload.get("metric", {})
    if (
        metric.get("dataset_id") != METRIC_DATASET_ID
        or metric.get("dataset_version") != METRIC_DATASET_VERSION
        or metric.get("source_root") != "topological-metrics-kaggle"
        or metric.get("build", {}).get("cmake_version") != "cmake version 3.31.6"
    ):
        raise RuntimeError("threshold metric identity mismatch")
    smoke = metric.get("identity_smoke", {}).get("values", {})
    if set(smoke) != {"blend", "surface_dice", "toposcore", "voi_score"} or any(
        not math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=2e-15)
        for value in smoke.values()
    ):
        raise RuntimeError("threshold metric identity smoke mismatch")
    metric_runtime = payload.get("metric_runtime", {})
    runtime_checks = {
        "dataset_id": METRIC_RUNTIME_DATASET_ID,
        "dataset_version": METRIC_RUNTIME_DATASET_VERSION,
        "wheel_count": 16,
        "wheel_bytes": METRIC_RUNTIME_WHEEL_BYTES,
    }
    for key, expected in runtime_checks.items():
        if metric_runtime.get(key) != expected:
            raise RuntimeError(f"threshold metric-runtime mismatch for {key}")
    if (
        metric_runtime.get("manifest", {}).get("sha256")
        != METRIC_RUNTIME_MANIFEST_SHA256
        or metric_runtime.get("manifest", {}).get("payload_sha256")
        != METRIC_RUNTIME_PAYLOAD_SHA256
        or metric_runtime.get("requirements", {}).get("sha256")
        != METRIC_RUNTIME_REQUIREMENTS_SHA256
        or metric_runtime.get("wheel_ledger", {}).get("sha256")
        != METRIC_RUNTIME_WHEEL_LEDGER_SHA256
        or metric_runtime.get("wheel_ledger", {}).get("record_count") != 16
    ):
        raise RuntimeError("threshold metric-runtime ledger identity mismatch")
    if payload.get("runtime", {}).get("packages") != RUNTIME_PACKAGES:
        raise RuntimeError("threshold runtime-package identity mismatch")
    if payload.get("invocation", {}).get("returncode") != 0:
        raise RuntimeError("threshold selector invocation was not successful")
    output = payload.get("outputs", {}).get("frozen_thresholds", {})
    if output.get("sha256") != sha256_file(thresholds_path):
        raise RuntimeError("threshold run manifest file identity mismatch")
    if output.get("bytes") != thresholds_path.stat().st_size:
        raise RuntimeError("threshold run manifest byte-size mismatch")
    if output.get("payload_sha256") != thresholds["payload_sha256"]:
        raise RuntimeError("threshold run manifest payload identity mismatch")
    return payload


def model_record(run: str) -> dict:
    if run == "baseline":
        return {
            "kernel_id": None,
            "kernel_version": None,
            "model_state_sha256": None,
        }
    return dict(TRAINED[run])


def build_plan(
    *,
    thresholds_path: Path,
    threshold_run_manifest_path: Path,
    panel_manifest_path: Path,
    threshold_kernel_version: int,
    public_threshold_commit: str,
) -> dict:
    if threshold_kernel_version <= 0:
        raise ValueError("threshold kernel version must be positive")
    if len(public_threshold_commit) != 40 or any(
        character not in "0123456789abcdef" for character in public_threshold_commit
    ):
        raise ValueError(
            "public threshold commit must be 40 lowercase hexadecimal characters"
        )
    thresholds = validate_thresholds(thresholds_path)
    threshold_run = validate_threshold_run_manifest(
        threshold_run_manifest_path, thresholds_path, thresholds
    )
    panel_manifest = load_hashed_json(panel_manifest_path)
    if (
        sha256_file(panel_manifest_path) != REAL_PANEL_MANIFEST_SHA256
        or panel_manifest.get("payload_sha256") != REAL_PANEL_PAYLOAD_SHA256
        or panel_manifest.get("status")
        != "model-blind; selected from held-out labels before prediction"
        or len(panel_manifest.get("selected", [])) != 4
    ):
        raise RuntimeError("frozen real-panel manifest identity mismatch")
    generator_path = Path(__file__).resolve()
    clarification_path = generator_path.with_name("PROTOCOL_CLARIFICATION.md")
    if (
        not clarification_path.is_file()
        or sha256_file(clarification_path) != PROTOCOL_CLARIFICATION_SHA256
    ):
        raise RuntimeError("pre-inference protocol clarification identity mismatch")
    heldout_launcher_path = generator_path.with_name("heldout_cache_launcher.py")
    if not heldout_launcher_path.is_file():
        raise RuntimeError(
            f"held-out cache launcher is absent: {heldout_launcher_path}"
        )
    heldout_package_generator_path = generator_path.with_name(
        "generate_heldout_job_packages.py"
    )
    if not heldout_package_generator_path.is_file():
        raise RuntimeError(
            f"held-out package generator is absent: {heldout_package_generator_path}"
        )
    heldout_queue_controller_path = generator_path.with_name(
        "orchestrate_heldout_queue.py"
    )
    if not heldout_queue_controller_path.is_file():
        raise RuntimeError(
            f"held-out queue controller is absent: {heldout_queue_controller_path}"
        )
    heldout_delivery_collector_path = generator_path.with_name(
        "collect_heldout_delivery_inputs.py"
    )
    if not heldout_delivery_collector_path.is_file():
        raise RuntimeError(
            f"held-out delivery collector is absent: {heldout_delivery_collector_path}"
        )
    delivery_freezer_path = generator_path.with_name("freeze_heldout_cache_delivery.py")
    if not delivery_freezer_path.is_file():
        raise RuntimeError(
            f"held-out delivery freezer is absent: {delivery_freezer_path}"
        )
    result_validator_path = generator_path.with_name("validate_final_results.py")
    if not result_validator_path.is_file():
        raise RuntimeError(f"final-result validator is absent: {result_validator_path}")
    panel_renderer_path = generator_path.with_name("render_real_panels.py")
    if not panel_renderer_path.is_file():
        raise RuntimeError(f"real-panel renderer is absent: {panel_renderer_path}")
    scoring_stager_path = generator_path.with_name("stage_heldout_for_scoring.py")
    if not scoring_stager_path.is_file():
        raise RuntimeError(f"one-shot scoring stager is absent: {scoring_stager_path}")
    evidence_renderer_path = generator_path.with_name("render_final_evidence.py")
    if not evidence_renderer_path.is_file():
        raise RuntimeError(
            f"final evidence renderer is absent: {evidence_renderer_path}"
        )
    scoring_launcher_path = generator_path.with_name("one_shot_scoring_launcher.py")
    if not scoring_launcher_path.is_file():
        raise RuntimeError(
            f"one-shot scoring launcher is absent: {scoring_launcher_path}"
        )
    metric_preparer_path = generator_path.with_name("prepare_metric_runtime.py")
    if not metric_preparer_path.is_file():
        raise RuntimeError(f"metric-runtime preparer is absent: {metric_preparer_path}")
    scoring_package_generator_path = generator_path.with_name(
        "generate_scoring_job_packages.py"
    )
    if not scoring_package_generator_path.is_file():
        raise RuntimeError(
            f"scoring package generator is absent: {scoring_package_generator_path}"
        )
    scoring_pair_controller_path = generator_path.with_name(
        "orchestrate_scoring_pair.py"
    )
    if not scoring_pair_controller_path.is_file():
        raise RuntimeError(
            f"scoring-pair controller is absent: {scoring_pair_controller_path}"
        )
    scoring_result_collector_path = generator_path.with_name(
        "collect_scoring_results.py"
    )
    if not scoring_result_collector_path.is_file():
        raise RuntimeError(
            f"scoring-result collector is absent: {scoring_result_collector_path}"
        )
    visual_assessment_recorder_path = generator_path.with_name(
        "record_visual_assessment.py"
    )
    if not visual_assessment_recorder_path.is_file():
        raise RuntimeError(
            f"visual-assessment recorder is absent: {visual_assessment_recorder_path}"
        )
    threshold_binding = {
        "kernel_id": THRESHOLD_KERNEL_ID,
        "kernel_version": threshold_kernel_version,
        "launcher_sha256": THRESHOLD_LAUNCHER_SHA256,
        "frozen_thresholds": {
            "file": thresholds_path.name,
            "bytes": thresholds_path.stat().st_size,
            "sha256": sha256_file(thresholds_path),
            "payload_sha256": thresholds["payload_sha256"],
        },
        "threshold_run_manifest": {
            "file": threshold_run_manifest_path.name,
            "bytes": threshold_run_manifest_path.stat().st_size,
            "sha256": sha256_file(threshold_run_manifest_path),
            "payload_sha256": threshold_run["payload_sha256"],
        },
        "public_freeze_commit": public_threshold_commit,
    }
    real_jobs = []
    synthetic_jobs = []
    for run in RUN_ORDER:
        model = model_record(run)
        selected = thresholds["runs"][run]["selected_threshold"]
        common = {
            "run": run,
            "selected_threshold": selected,
            **model,
        }
        real_jobs.append(
            {
                "job_id": f"real-test-{run.replace('_', '-')}",
                "mode": "real_test_cache",
                **common,
                "test_patch_count": 38,
                "scroll_patch_counts": {"s4": 36, "s5": 2},
                "expected_cache_manifest": f"cache_manifest_{run}_test.json",
            }
        )
        synthetic_jobs.append(
            {
                "job_id": f"synthetic-{run.replace('_', '-')}-all-shards",
                "mode": "synthetic_ray_cache",
                **common,
                "shard_indices": list(range(SYNTHETIC_SHARDS)),
                "shard_count": SYNTHETIC_SHARDS,
                "cell_count": 100,
                "expected_cache_manifests": [
                    f"synthetic_manifest_{run}_shard{shard:02d}.json"
                    for shard in range(SYNTHETIC_SHARDS)
                ],
            }
        )
    payload = {
        "schema_version": "1.0",
        "status": "held-out execution plan frozen before held-out inference",
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "public_checkpoint_freeze_commit": PUBLIC_CHECKPOINT_FREEZE_COMMIT,
        "public_checkpoint_freeze_manifest_sha256": PUBLIC_CHECKPOINT_MANIFEST_SHA256,
        "public_evaluation_input_freeze_commit": PUBLIC_INPUT_FREEZE_COMMIT,
        "public_evaluation_input_freeze_manifest_sha256": PUBLIC_INPUT_MANIFEST_SHA256,
        "source_split_manifest_sha256": REAL_SPLIT_MANIFEST_SHA256,
        "source_split_records_sha256": REAL_SPLIT_RECORDS_SHA256,
        "real_panel_manifest": {
            "file": panel_manifest_path.name,
            "bytes": panel_manifest_path.stat().st_size,
            "sha256": REAL_PANEL_MANIFEST_SHA256,
            "payload_sha256": REAL_PANEL_PAYLOAD_SHA256,
        },
        "pre_inference_protocol_clarification": {
            "file": clarification_path.name,
            "bytes": clarification_path.stat().st_size,
            "sha256": PROTOCOL_CLARIFICATION_SHA256,
            "first_public_commit": PROTOCOL_CLARIFICATION_COMMIT,
            "scope": "synthetic primary-gate pooling only",
        },
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "threshold_binding": threshold_binding,
        "plan_generator": {
            "file": generator_path.name,
            "bytes": generator_path.stat().st_size,
            "sha256": sha256_file(generator_path),
        },
        "heldout_cache_launcher": {
            "file": heldout_launcher_path.name,
            "bytes": heldout_launcher_path.stat().st_size,
            "sha256": sha256_file(heldout_launcher_path),
        },
        "heldout_package_generator": {
            "file": heldout_package_generator_path.name,
            "bytes": heldout_package_generator_path.stat().st_size,
            "sha256": sha256_file(heldout_package_generator_path),
        },
        "heldout_queue_controller": {
            "file": heldout_queue_controller_path.name,
            "bytes": heldout_queue_controller_path.stat().st_size,
            "sha256": sha256_file(heldout_queue_controller_path),
            "max_active_gpu_jobs": 2,
        },
        "heldout_delivery_collector": {
            "file": heldout_delivery_collector_path.name,
            "bytes": heldout_delivery_collector_path.stat().st_size,
            "sha256": sha256_file(heldout_delivery_collector_path),
        },
        "heldout_delivery_freezer": {
            "file": delivery_freezer_path.name,
            "bytes": delivery_freezer_path.stat().st_size,
            "sha256": sha256_file(delivery_freezer_path),
        },
        "final_result_validator": {
            "file": result_validator_path.name,
            "bytes": result_validator_path.stat().st_size,
            "sha256": sha256_file(result_validator_path),
        },
        "real_panel_renderer": {
            "file": panel_renderer_path.name,
            "bytes": panel_renderer_path.stat().st_size,
            "sha256": sha256_file(panel_renderer_path),
        },
        "one_shot_scoring_stager": {
            "file": scoring_stager_path.name,
            "bytes": scoring_stager_path.stat().st_size,
            "sha256": sha256_file(scoring_stager_path),
        },
        "final_evidence_renderer": {
            "file": evidence_renderer_path.name,
            "bytes": evidence_renderer_path.stat().st_size,
            "sha256": sha256_file(evidence_renderer_path),
        },
        "one_shot_scoring_launcher": {
            "file": scoring_launcher_path.name,
            "bytes": scoring_launcher_path.stat().st_size,
            "sha256": sha256_file(scoring_launcher_path),
        },
        "metric_runtime_preparer": {
            "file": metric_preparer_path.name,
            "bytes": metric_preparer_path.stat().st_size,
            "sha256": sha256_file(metric_preparer_path),
        },
        "scoring_package_generator": {
            "file": scoring_package_generator_path.name,
            "bytes": scoring_package_generator_path.stat().st_size,
            "sha256": sha256_file(scoring_package_generator_path),
        },
        "scoring_pair_controller": {
            "file": scoring_pair_controller_path.name,
            "bytes": scoring_pair_controller_path.stat().st_size,
            "sha256": sha256_file(scoring_pair_controller_path),
        },
        "scoring_result_collector": {
            "file": scoring_result_collector_path.name,
            "bytes": scoring_result_collector_path.stat().st_size,
            "sha256": sha256_file(scoring_result_collector_path),
        },
        "visual_assessment_recorder": {
            "file": visual_assessment_recorder_path.name,
            "bytes": visual_assessment_recorder_path.stat().st_size,
            "sha256": sha256_file(visual_assessment_recorder_path),
        },
        "run_order": list(RUN_ORDER),
        "real_test_jobs": real_jobs,
        "synthetic_ray_jobs": synthetic_jobs,
        "one_shot_scorers": {
            "real": {
                "script": {
                    "file": SCORER_SOURCE_HASHES["real"][0],
                    "sha256": SCORER_SOURCE_HASHES["real"][1],
                },
                "requires_jobs": [record["job_id"] for record in real_jobs],
                "expected_manifests": 7,
                "expected_probability_caches": 266,
                "output": "sealed_real_test_results.json",
            },
            "synthetic": {
                "script": {
                    "file": SCORER_SOURCE_HASHES["synthetic"][0],
                    "sha256": SCORER_SOURCE_HASHES["synthetic"][1],
                },
                "requires_jobs": [record["job_id"] for record in synthetic_jobs],
                "expected_manifests": 70,
                "expected_ray_caches": 700,
                "output": "sealed_synthetic_test_results.json",
            },
        },
        "scientific_gate": {
            "thresholds_publicly_frozen_before_held_out_inference": True,
            "protocol_clarification_publicly_frozen_before_held_out_inference": True,
            "held_out_outputs_inspected_when_plan_frozen": False,
            "manual_threshold_entry_permitted": False,
            "test_time_tuning_permitted": False,
            "scientific_run_order_or_shard_membership_changes_permitted": False,
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--threshold-run-manifest", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--threshold-kernel-version", type=int, required=True)
    parser.add_argument("--public-threshold-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"output must start absent: {args.out}")
    payload = build_plan(
        thresholds_path=args.thresholds,
        threshold_run_manifest_path=args.threshold_run_manifest,
        panel_manifest_path=args.panel_manifest,
        threshold_kernel_version=args.threshold_kernel_version,
        public_threshold_commit=args.public_threshold_commit,
    )
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("real held-out jobs:", len(payload["real_test_jobs"]))
    print("synthetic sealed-ray jobs:", len(payload["synthetic_ray_jobs"]))
    print("held-out execution-plan payload SHA-256:", payload["payload_sha256"])
    print("HELDOUT_EXECUTION_PLAN_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
