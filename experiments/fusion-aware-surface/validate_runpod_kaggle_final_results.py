#!/usr/bin/env python3
"""Validate jointly collected RunPod-real and Kaggle-synthetic results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from validate_final_results import (
    METRICS,
    PROTOCOL_CLARIFICATION,
    SCORING_IMPORT_RUNTIME,
    expected_scoring_job_config_identity,
    load_hashed,
    local_identity,
    require_hex,
    sha256_file,
    validate_file_identity,
    validate_real,
    validate_real_panel_artifacts,
    validate_sources,
    validate_synthetic,
    validate_visual_assessment,
)


COLLECTION_GATE = {
    "runpod_real_complete_copied_hashed_and_billing_terminated": True,
    "synthetic_v4_complete_and_server_source_verified_before_collection": True,
    "both_sources_verified_before_output_directory_creation": True,
    "synthetic_scorer_repeated": False,
    "kernel_logs_opened_or_read": False,
    "results_or_manifests_opened_or_parsed_by_collector": False,
    "panel_images_opened_or_read_by_collector": False,
    "independent_validation_required_next": True,
}


def identity(path: Path, payload: dict | None = None) -> dict:
    record = local_identity(path)
    if payload is not None:
        record["payload_sha256"] = payload["payload_sha256"]
    return record


def validate_bound_source(plan: dict, label: str, path: Path, payload: dict | None = None) -> None:
    if plan["sources"].get(label) != identity(path, payload):
        raise RuntimeError(f"validation plan points to another {label}")


def validate_artifact_record(record: dict, path: Path, expected_suffix: str) -> None:
    if set(record) != {"file", "bytes", "sha256"}:
        raise RuntimeError(f"artifact record schema mismatch: {expected_suffix}")
    if not record["file"].endswith(expected_suffix):
        raise RuntimeError(f"artifact path mismatch: {expected_suffix}")
    if record["bytes"] != path.stat().st_size or record["sha256"] != sha256_file(path):
        raise RuntimeError(f"artifact identity mismatch: {expected_suffix}")


def validate_collection(
    *,
    validation_plan: dict,
    collection_plan_path: Path,
    result_sources_path: Path,
    package_index_path: Path,
    pair_receipt_path: Path,
    real_result_path: Path,
    synthetic_result_path: Path,
    real_run_path: Path,
    synthetic_run_path: Path,
    panel_manifest_path: Path,
    panel_images: list[Path],
) -> None:
    collection_plan = load_hashed(collection_plan_path)
    sources = load_hashed(result_sources_path)
    package_index = load_hashed(package_index_path)
    pair_receipt = load_hashed(pair_receipt_path)
    validate_bound_source(validation_plan, "collection_plan", collection_plan_path, collection_plan)
    validate_bound_source(validation_plan, "result_sources", result_sources_path, sources)
    validate_bound_source(validation_plan, "package_index", package_index_path, package_index)
    validate_bound_source(validation_plan, "pair_receipt", pair_receipt_path, pair_receipt)
    if sources.get("status") != "sealed RunPod real and Kaggle synthetic v4 artifacts jointly collected":
        raise RuntimeError("wrong mixed-result source status")
    if sources.get("scientific_gate") != COLLECTION_GATE:
        raise RuntimeError("mixed-result scientific gate mismatch")
    if sources.get("collection_plan") != identity(collection_plan_path, collection_plan):
        raise RuntimeError("mixed-result receipt points to another collection plan")
    if sources.get("collector") != collection_plan.get("collector"):
        raise RuntimeError("mixed-result collector identity mismatch")

    real_records = sources.get("sources", {}).get("runpod_real", {}).get("artifacts")
    expected_real = [
        (real_result_path, "downloads/real/sealed_real_test_results.json"),
        (real_run_path, "downloads/real/scoring_run_manifest.json"),
        (panel_manifest_path, "downloads/real/real-panels/real_panel_render_manifest.json"),
        *[
            (path, f"downloads/real/real-panels/real_panel_{index:02d}.png")
            for index, path in enumerate(sorted(panel_images), start=1)
        ],
    ]
    if not isinstance(real_records, list) or len(real_records) != len(expected_real):
        raise RuntimeError("mixed-result real artifact count mismatch")
    for record, (path, suffix) in zip(real_records, expected_real, strict=True):
        validate_artifact_record(record, path, suffix)

    synthetic_source = sources.get("sources", {}).get("kaggle_synthetic_v4", {})
    validate_artifact_record(
        synthetic_source.get("result", {}),
        synthetic_result_path,
        "downloads/synthetic/fusion-one-shot-synthetic/sealed_synthetic_test_results.json",
    )
    validate_artifact_record(
        synthetic_source.get("run_manifest", {}),
        synthetic_run_path,
        "downloads/synthetic/fusion-one-shot-synthetic/scoring_run_manifest.json",
    )
    if synthetic_source.get("kernel_version") != 4:
        raise RuntimeError("synthetic source is not frozen version 4")
    if synthetic_source.get("package_index") != identity(package_index_path, package_index):
        raise RuntimeError("synthetic source package-index mismatch")
    if synthetic_source.get("acceptance_receipt") != identity(pair_receipt_path, pair_receipt):
        raise RuntimeError("synthetic source acceptance-receipt mismatch")
    packages = [record for record in package_index.get("packages", []) if record.get("mode") == "synthetic"]
    if len(packages) != 1:
        raise RuntimeError("synthetic package is not unique")
    expected_outer_source = packages[0]["files"]["one_shot_scoring_launcher.py"]["sha256"]
    if synthetic_source.get("source_sha256") != expected_outer_source:
        raise RuntimeError("synthetic server source mismatch")


def validate_run_manifest(
    *,
    mode: str,
    run_path: Path,
    result_path: Path,
    expected: dict,
    threshold_binding: dict,
    panel_manifest_path: Path,
    panel_images: list[Path],
) -> None:
    run = load_hashed(run_path)
    base_keys = {
        "schema_version", "status", "mode", "public_execution_plan",
        "public_cache_delivery", "threshold_binding", "launcher", "job_config",
        "staging", "project_source_hashes", "import_runtime", "metric_runtime",
        "scorer_invocation", "panel_invocation", "scientific_gate", "payload_sha256",
    }
    expected_keys = base_keys | ({"scorer_public_imports", "scorer_public_metric_worker"} if mode == "real" else set())
    if set(run) != expected_keys or run.get("schema_version") != "1.0":
        raise RuntimeError(f"{mode}: scoring run-manifest schema mismatch")
    if run.get("mode") != mode or run.get("status") != f"{mode} held-out result scored exactly once after public cache-delivery freeze":
        raise RuntimeError(f"{mode}: scoring run-manifest identity mismatch")
    if run.get("public_execution_plan") != expected["public_execution_plan"]:
        raise RuntimeError(f"{mode}: public execution-plan mismatch")
    if run.get("public_cache_delivery") != expected["public_cache_delivery"]:
        raise RuntimeError(f"{mode}: public cache-delivery mismatch")
    if run.get("threshold_binding") != threshold_binding:
        raise RuntimeError(f"{mode}: frozen threshold binding mismatch")
    if run.get("launcher") != expected["launcher"]:
        raise RuntimeError(f"{mode}: launcher identity mismatch")
    public_plan = expected["public_execution_plan"]
    public_delivery = expected["public_cache_delivery"]
    if run.get("job_config") != expected_scoring_job_config_identity(
        mode=mode,
        public_plan_commit=require_hex(public_plan["commit"], 40, f"{mode} plan commit"),
        public_plan_file_sha256=public_plan["file_sha256"],
        public_delivery_commit=require_hex(public_delivery["commit"], 40, f"{mode} delivery commit"),
        public_delivery_file_sha256=public_delivery["file_sha256"],
    ):
        raise RuntimeError(f"{mode}: job-config identity mismatch")
    if run.get("import_runtime") != SCORING_IMPORT_RUNTIME:
        raise RuntimeError(f"{mode}: import-runtime mismatch")
    if mode == "real":
        if run.get("scorer_public_imports") != expected["public_imports"]:
            raise RuntimeError("real: public import dependency mismatch")
        if run.get("scorer_public_metric_worker") != expected["public_metric_worker"]:
            raise RuntimeError("real: public metric-worker mismatch")
        metric = run.get("metric_runtime")
        values = metric.get("identity_smoke", {}).get("values", {}) if isinstance(metric, dict) else {}
        if set(values) != set(METRICS) or any(
            not math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=2e-15)
            for value in values.values()
        ):
            raise RuntimeError("real: official metric identity smoke mismatch")
    elif run.get("metric_runtime") is not None:
        raise RuntimeError("synthetic: unexpected official metric runtime")
    invocation = run.get("scorer_invocation", {})
    if invocation.get("script") != expected["scorer"] or invocation.get("returncode") != 0:
        raise RuntimeError(f"{mode}: scorer invocation mismatch")
    for stream in ("stdout", "stderr"):
        if not isinstance(invocation.get(f"{stream}_bytes"), int):
            raise RuntimeError(f"{mode}: invalid {stream} length")
        require_hex(invocation.get(f"{stream}_sha256"), 64, f"{mode} {stream} SHA-256")
    validate_file_identity(invocation.get("result", {}), result_path, mode)
    panel = run.get("panel_invocation")
    if mode == "synthetic":
        if panel is not None:
            raise RuntimeError("synthetic: unexpected panel invocation")
    else:
        if not isinstance(panel, dict) or panel.get("renderer") != expected["panel_renderer"] or panel.get("returncode") != 0:
            raise RuntimeError("real: panel invocation mismatch")
        if panel.get("manifest") != local_identity(panel_manifest_path):
            raise RuntimeError("real: panel manifest identity mismatch")
        if panel.get("images") != [local_identity(path) for path in sorted(panel_images)]:
            raise RuntimeError("real: panel image identities mismatch")
    if run.get("scientific_gate") != {
        "all_14_cache_jobs_publicly_frozen_before_scoring": True,
        "scorer_invocations": 1,
        "manual_threshold_override_used": False,
        "test_time_tuning_permitted": False,
        "scientific_stdout_echoed_by_launcher": False,
        "sealed_result_opened_or_parsed_by_launcher": False,
        "fixed_real_panels_rendered": mode == "real",
        "panel_render_manifest_opened_or_parsed_by_launcher": False,
        "panels_opened_or_visually_assessed_by_launcher": False,
    }:
        raise RuntimeError(f"{mode}: scoring scientific gate mismatch")


def validate_all(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    validation_plan = load_hashed(args.validation_plan)
    if validation_plan.get("status") != "RunPod-real and Kaggle-synthetic final validation frozen before scientific result access":
        raise RuntimeError("wrong final-validation plan status")
    if validation_plan.get("validator") != local_identity(Path(__file__).resolve()):
        raise RuntimeError("running final validator differs from frozen plan")
    scientific_plan = load_hashed(args.scientific_plan)
    delivery = load_hashed(args.delivery)
    runpod_plan = load_hashed(args.runpod_plan)
    thresholds = load_hashed(args.thresholds)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    panel_source = load_hashed(args.real_panel_source_manifest)
    for label, path, payload in (
        ("scientific_plan", args.scientific_plan, scientific_plan),
        ("delivery", args.delivery, delivery),
        ("runpod_plan", args.runpod_plan, runpod_plan),
        ("thresholds", args.thresholds, thresholds),
        ("split", args.split, None),
        ("panel_source", args.real_panel_source_manifest, panel_source),
        ("visual_recorder", args.visual_recorder, None),
        ("operational_audit", args.operational_audit, None),
    ):
        validate_bound_source(validation_plan, label, path, payload)
    if scientific_plan.get("pre_inference_protocol_clarification") != PROTOCOL_CLARIFICATION:
        raise RuntimeError("scientific plan lacks frozen protocol clarification")
    if delivery.get("status") != "all 14 publicly planned held-out caches sealed before one-shot scoring":
        raise RuntimeError("wrong held-out delivery status")
    if validation_plan.get("scientific_gate") != {
        "paired_collection_completed_before_validation_freeze": True,
        "real_and_synthetic_results_opened_or_parsed_by_freezer": False,
        "panels_opened_or_inspected_by_freezer": False,
        "all_seven_preregistered_gates_must_be_reported": True,
        "all_four_panels_and_twelve_comparisons_must_be_reported": True,
        "adverse_null_and_failed_outcomes_must_be_published": True,
        "result_dependent_tuning_or_selection_permitted": False,
        "official_form_submission_permitted_without_aviad_review": False,
    }:
        raise RuntimeError("final-validation scientific gate mismatch")
    validate_collection(
        validation_plan=validation_plan,
        collection_plan_path=args.collection_plan,
        result_sources_path=args.result_sources,
        package_index_path=args.scoring_package_index,
        pair_receipt_path=args.scoring_pair_receipt,
        real_result_path=args.real,
        synthetic_result_path=args.synthetic,
        real_run_path=args.real_run_manifest,
        synthetic_run_path=args.synthetic_run_manifest,
        panel_manifest_path=args.real_panel_render_manifest,
        panel_images=args.real_panel_image,
    )
    expected = validation_plan["expected_run_provenance"]
    validate_run_manifest(
        mode="real", run_path=args.real_run_manifest, result_path=args.real,
        expected=expected["real"] | {"panel_renderer": expected["panel_renderer"]},
        threshold_binding=expected["threshold_binding"],
        panel_manifest_path=args.real_panel_render_manifest,
        panel_images=args.real_panel_image,
    )
    validate_run_manifest(
        mode="synthetic", run_path=args.synthetic_run_manifest,
        result_path=args.synthetic, expected=expected["synthetic"],
        threshold_binding=expected["threshold_binding"],
        panel_manifest_path=args.real_panel_render_manifest,
        panel_images=args.real_panel_image,
    )
    validate_sources(thresholds, split)
    validate_real_panel_artifacts(
        plan=scientific_plan,
        thresholds=thresholds,
        source_panel_path=args.real_panel_source_manifest,
        render_manifest_path=args.real_panel_render_manifest,
        image_paths=args.real_panel_image,
    )
    visual = validate_visual_assessment(
        plan=scientific_plan,
        assessment_path=args.visual_assessment,
        observations_path=args.visual_observations,
        render_manifest_path=args.real_panel_render_manifest,
        image_paths=args.real_panel_image,
    )
    real = load_hashed(args.real)
    synthetic = load_hashed(args.synthetic)
    validate_real(real, thresholds, split)
    validate_synthetic(synthetic, thresholds, split)
    return real, synthetic, visual, validation_plan


def add_arguments(parser: argparse.ArgumentParser) -> None:
    for name in ("real", "synthetic", "thresholds", "split", "scientific_plan", "delivery", "runpod_plan", "collection_plan", "result_sources", "scoring_package_index", "scoring_pair_receipt", "real_run_manifest", "synthetic_run_manifest", "real_panel_render_manifest", "real_panel_source_manifest", "visual_assessment", "visual_observations", "visual_recorder", "operational_audit", "validation_plan"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--real-panel-image", type=Path, action="append", required=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    validate_all(args)
    print("real result file SHA-256:", sha256_file(args.real))
    print("synthetic result file SHA-256:", sha256_file(args.synthetic))
    print("ALL_RUNPOD_KAGGLE_FINAL_RESULTS_INDEPENDENTLY_RECOMPUTED_AND_VALIDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
