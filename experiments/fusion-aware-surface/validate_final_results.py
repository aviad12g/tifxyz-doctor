#!/usr/bin/env python3
"""Independently recompute and validate both sealed final-result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

RUN_ORDER = (
    "baseline",
    "control_seed11",
    "control_seed23",
    "control_seed47",
    "gap8_seed11",
    "gap8_seed23",
    "gap8_seed47",
)
METRICS = ("blend", "toposcore", "surface_dice", "voi_score")
SENSITIVITY_THRESHOLDS = (0.4, 0.5, 0.6)
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_REPLICATES = 10_000
THRESHOLD_GRID = tuple(round(0.30 + 0.05 * index, 2) for index in range(9))
COUNT_FIELDS = (
    "neighbour_sites",
    "detected_neighbour_sites",
    "fused_detected_sites",
    "control_sites",
    "false_split_sites",
)
RATE_FIELDS = (
    "site_center_detection_rate",
    "conditional_fusion_rate",
    "false_split_rate",
)
SCORING_IMPORT_RUNTIME = {
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "numpy": "1.26.4",
    "scipy": "1.15.3",
    "tifffile": "2025.2.18",
    "timm": "1.0.27",
    "einops": "0.8.1",
}
VISUAL_CRITERION = (
    "at least one preregistered real compressed-region panel shows a visually "
    "verifiable gap8-versus-matched-control separation improvement without a "
    "new nearby break"
)
PROTOCOL_CLARIFICATION = {
    "file": "PROTOCOL_CLARIFICATION.md",
    "bytes": 2036,
    "sha256": "c849d7a268465dd17b8c5d0c6774e7caab5caa343eb2dacb0cf4d862db3166dd",
    "first_public_commit": "1e0ced2c928fa0842aa8c601f39dbcd49272247e",
    "scope": "synthetic primary-gate pooling only",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_identity(path: Path) -> dict:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    observed = payload.get("payload_sha256")
    content = dict(payload)
    content.pop("payload_sha256", None)
    if observed != canonical_sha256(content):
        raise RuntimeError(f"embedded payload SHA-256 mismatch: {path}")
    return payload


def finite_unit(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label}: boolean is not a metric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RuntimeError(f"{label}: metric outside [0,1]")
    return number


def close(actual: object, expected: object, label: str) -> None:
    if isinstance(actual, bool) or not math.isclose(
        float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RuntimeError(f"{label}: recomputation mismatch")


def require_hex(value: object, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"invalid {label}")
    return value


def validate_file_identity(record: dict, path: Path, label: str) -> None:
    expected = {"file", "bytes", "sha256"}
    if set(record) != expected:
        raise RuntimeError(f"{label}: result identity schema mismatch")
    if record["file"] != path.name or record["bytes"] != path.stat().st_size:
        raise RuntimeError(f"{label}: result file name/size mismatch")
    if record["sha256"] != sha256_file(path):
        raise RuntimeError(f"{label}: result file SHA-256 mismatch")
    load_hashed(path)


def expected_scoring_job_config_identity(
    *,
    mode: str,
    public_plan_commit: str,
    public_plan_file_sha256: str,
    public_delivery_commit: str,
    public_delivery_file_sha256: str,
) -> dict:
    config = {
        "schema_version": "1.0",
        "mode": mode,
        "public_plan_commit": public_plan_commit,
        "public_plan_file_sha256": public_plan_file_sha256,
        "public_delivery_commit": public_delivery_commit,
        "public_delivery_file_sha256": public_delivery_file_sha256,
    }
    config["payload_sha256"] = canonical_sha256(config)
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {
        "encoding": "hex in first source comment",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload_sha256": config["payload_sha256"],
    }


def validate_scoring_result_collection(
    *,
    plan: dict,
    delivery: dict,
    plan_path: Path,
    delivery_path: Path,
    result_sources_path: Path,
    package_index_path: Path,
    pair_receipt_path: Path,
    real_run_path: Path,
    synthetic_run_path: Path,
    real_result_path: Path,
    synthetic_result_path: Path,
    real_panel_manifest_path: Path,
    real_panel_image_paths: list[Path],
) -> str:
    package_index = load_hashed(package_index_path)
    pair_receipt = load_hashed(pair_receipt_path)
    result_sources = load_hashed(result_sources_path)

    expected_index_top = {
        "schema_version",
        "status",
        "public_execution_plan",
        "public_cache_delivery",
        "generator",
        "launcher",
        "pair_controller",
        "result_collector",
        "package_count",
        "packages",
        "scientific_gate",
        "payload_sha256",
    }
    if (
        set(package_index) != expected_index_top
        or package_index.get("schema_version") != "1.0"
    ):
        raise RuntimeError("scoring-package index schema mismatch")
    if package_index.get("status") != (
        "two one-shot scorer packages generated after public cache-delivery freeze"
    ):
        raise RuntimeError("wrong scoring-package index status")
    plan_commit = require_hex(
        delivery.get("public_execution_plan", {}).get("commit"),
        40,
        "public plan commit",
    )
    if package_index.get("public_execution_plan") != {
        "commit": plan_commit,
        "file_sha256": sha256_file(plan_path),
        "payload_sha256": plan["payload_sha256"],
    }:
        raise RuntimeError("scoring-package index plan binding mismatch")
    public_delivery = package_index.get("public_cache_delivery", {})
    delivery_commit = require_hex(
        public_delivery.get("commit"), 40, "public delivery commit"
    )
    if public_delivery != {
        "commit": delivery_commit,
        "file_sha256": sha256_file(delivery_path),
        "payload_sha256": delivery["payload_sha256"],
    }:
        raise RuntimeError("scoring-package index delivery binding mismatch")
    identity_bindings = {
        "generator": "scoring_package_generator",
        "launcher": "one_shot_scoring_launcher",
        "pair_controller": "scoring_pair_controller",
        "result_collector": "scoring_result_collector",
    }
    for index_key, plan_key in identity_bindings.items():
        if package_index.get(index_key) != plan.get(plan_key):
            raise RuntimeError(f"scoring-package {index_key} identity mismatch")
    if package_index.get("scientific_gate") != {
        "all_14_cache_jobs_publicly_frozen": True,
        "scientific_results_opened_or_inspected": False,
        "both_scorer_packages_generated_together": True,
        "both_scorers_must_be_launched_before_either_result_is_opened": True,
    }:
        raise RuntimeError("scoring-package scientific gate mismatch")

    packages = package_index.get("packages")
    if (
        not isinstance(packages, list)
        or package_index.get("package_count") != 2
        or [record.get("mode") for record in packages] != ["real", "synthetic"]
    ):
        raise RuntimeError("scoring-package index does not contain the fixed pair")
    packages_by_mode: dict[str, dict] = {}
    for record in packages:
        mode = record["mode"]
        if set(record) != {
            "mode",
            "kaggle_kernel_id",
            "directory",
            "required_cache_kernel_count",
            "threshold_kernel",
            "files",
            "embedded_job_config",
        }:
            raise RuntimeError(f"{mode}: scoring-package record schema mismatch")
        if (
            record.get("kaggle_kernel_id")
            != f"aviadcohen1/vesuvius-fusion-one-shot-{mode}-scoring"
            or record.get("directory") != mode
            or record.get("required_cache_kernel_count") != 7
            or record.get("threshold_kernel")
            != (
                f"{plan['threshold_binding']['kernel_id']}/"
                f"{plan['threshold_binding']['kernel_version']}"
            )
        ):
            raise RuntimeError(f"{mode}: scoring-package identity mismatch")
        files = record.get("files")
        if not isinstance(files, dict) or set(files) != {
            "KERNEL_SHA256SUMS",
            "kernel-metadata.json",
            "one_shot_scoring_launcher.py",
        }:
            raise RuntimeError(f"{mode}: scoring-package file set mismatch")
        for name, identity in files.items():
            if not isinstance(identity, dict) or set(identity) != {"bytes", "sha256"}:
                raise RuntimeError(f"{mode}: invalid package identity for {name}")
            if not isinstance(identity["bytes"], int) or identity["bytes"] <= 0:
                raise RuntimeError(f"{mode}: invalid package byte size for {name}")
            require_hex(identity["sha256"], 64, f"{mode} {name} SHA-256")
        if record.get("embedded_job_config") != expected_scoring_job_config_identity(
            mode=mode,
            public_plan_commit=plan_commit,
            public_plan_file_sha256=sha256_file(plan_path),
            public_delivery_commit=delivery_commit,
            public_delivery_file_sha256=sha256_file(delivery_path),
        ):
            raise RuntimeError(f"{mode}: package embedded config mismatch")
        packages_by_mode[mode] = record

    expected_package_index_identity = {
        "file": package_index_path.name,
        "bytes": package_index_path.stat().st_size,
        "sha256": sha256_file(package_index_path),
        "payload_sha256": package_index["payload_sha256"],
    }
    if (
        set(pair_receipt)
        != {
            "schema_version",
            "status",
            "scoring_package_index",
            "accepted",
            "payload_sha256",
        }
        or pair_receipt.get("schema_version") != "1.0"
    ):
        raise RuntimeError("scoring-pair receipt schema mismatch")
    if pair_receipt.get("status") != "paired one-shot scorer acceptance receipt":
        raise RuntimeError("wrong scoring-pair receipt status")
    if pair_receipt.get("scoring_package_index") != expected_package_index_identity:
        raise RuntimeError("scoring-pair receipt points to another package index")
    accepted = pair_receipt.get("accepted")
    if not isinstance(accepted, list) or [
        record.get("mode") for record in accepted
    ] != [
        "real",
        "synthetic",
    ]:
        raise RuntimeError("scoring-pair receipt does not contain the fixed pair")
    accepted_by_mode: dict[str, dict] = {}
    for record in accepted:
        mode = record["mode"]
        if set(record) != {
            "mode",
            "kernel_id",
            "kernel_version",
            "package_ledger_sha256",
            "accepted_at_utc",
        }:
            raise RuntimeError(f"{mode}: scoring-pair receipt record schema mismatch")
        package = packages_by_mode[mode]
        if (
            record.get("kernel_id") != package["kaggle_kernel_id"]
            or not isinstance(record.get("kernel_version"), int)
            or record["kernel_version"] <= 0
            or record.get("package_ledger_sha256")
            != package["files"]["KERNEL_SHA256SUMS"]["sha256"]
            or not isinstance(record.get("accepted_at_utc"), str)
            or not record["accepted_at_utc"].endswith("Z")
        ):
            raise RuntimeError(f"{mode}: scoring-pair receipt identity mismatch")
        accepted_by_mode[mode] = record

    if (
        set(result_sources)
        != {
            "schema_version",
            "status",
            "scoring_package_index",
            "pair_receipt",
            "artifacts",
            "collector",
            "scientific_gate",
            "payload_sha256",
        }
        or result_sources.get("schema_version") != "1.0"
    ):
        raise RuntimeError("scoring-result source schema mismatch")
    if result_sources.get("status") != (
        "both sealed scoring artifacts collected after both scorers completed"
    ):
        raise RuntimeError("wrong scoring-result source status")
    if result_sources.get("scoring_package_index") != expected_package_index_identity:
        raise RuntimeError("scoring-result sources point to another package index")
    if result_sources.get("pair_receipt") != {
        "file": pair_receipt_path.name,
        "bytes": pair_receipt_path.stat().st_size,
        "sha256": sha256_file(pair_receipt_path),
        "payload_sha256": pair_receipt["payload_sha256"],
    }:
        raise RuntimeError("scoring-result sources point to another pair receipt")
    if result_sources.get("collector") != plan.get("scoring_result_collector"):
        raise RuntimeError("scoring-result collector identity mismatch")
    if result_sources.get("scientific_gate") != {
        "both_scorers_accepted_before_result_access": True,
        "both_scorers_completed_before_first_result_download": True,
        "latest_kernel_versions_match_pair_receipt": True,
        "server_sources_match_accepted_packages": True,
        "kernel_logs_opened_or_read": False,
        "results_opened_or_read_by_collector": False,
        "panel_images_opened_or_read_by_collector": False,
        "independent_validation_required_next": True,
    }:
        raise RuntimeError("scoring-result collection gate mismatch")
    paths_by_mode = {
        "real": (real_result_path.resolve(), real_run_path.resolve()),
        "synthetic": (synthetic_result_path.resolve(), synthetic_run_path.resolve()),
    }
    resolved_panel_images = sorted(path.resolve() for path in real_panel_image_paths)
    if [path.name for path in resolved_panel_images] != [
        f"real_panel_{index:02d}.png" for index in range(1, 5)
    ]:
        raise RuntimeError("real-panel image argument set mismatch")
    artifacts = result_sources.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"real", "synthetic"}:
        raise RuntimeError("scoring-result artifact set mismatch")
    for mode, (result_path, run_path) in paths_by_mode.items():
        accepted_record = accepted_by_mode[mode]
        package = packages_by_mode[mode]
        if artifacts[mode] != {
            "kernel_id": accepted_record["kernel_id"],
            "kernel_version": accepted_record["kernel_version"],
            "source_sha256": package["files"]["one_shot_scoring_launcher.py"]["sha256"],
            "result": str(result_path),
            "run_manifest": str(run_path),
            "panel_manifest": (
                str(real_panel_manifest_path.resolve()) if mode == "real" else None
            ),
            "panel_images": (
                [str(path) for path in resolved_panel_images] if mode == "real" else []
            ),
        }:
            raise RuntimeError(f"{mode}: scoring-result artifact provenance mismatch")
    return delivery_commit


def validate_real_panel_artifacts(
    *,
    plan: dict,
    thresholds: dict,
    source_panel_path: Path,
    render_manifest_path: Path,
    image_paths: list[Path],
) -> dict:
    source_panels = load_hashed(source_panel_path)
    rendered = load_hashed(render_manifest_path)
    if plan.get("real_panel_manifest") != local_identity(source_panel_path) | {
        "payload_sha256": source_panels["payload_sha256"]
    }:
        raise RuntimeError("source panel manifest differs from public plan")
    if source_panels.get("status") != (
        "model-blind; selected from held-out labels before prediction"
    ):
        raise RuntimeError("source real panels were not frozen model-blind")
    selected = source_panels.get("selected")
    if not isinstance(selected, list) or len(selected) != 4:
        raise RuntimeError("source real-panel selection count mismatch")
    if (
        set(rendered)
        != {
            "schema_version",
            "status",
            "source_thresholds",
            "source_panel_manifest",
            "rendering",
            "panels",
            "scientific_gate",
            "renderer",
            "payload_sha256",
        }
        or rendered.get("schema_version") != "1.0"
    ):
        raise RuntimeError("real-panel render-manifest schema mismatch")
    if rendered.get("status") != (
        "four model-blind real panels rendered after sealed test delivery"
    ):
        raise RuntimeError("wrong real-panel render status")
    threshold_record = plan["threshold_binding"]["frozen_thresholds"]
    if rendered.get("source_thresholds") != threshold_record or threshold_record.get(
        "payload_sha256"
    ) != thresholds.get("payload_sha256"):
        raise RuntimeError("real-panel threshold binding mismatch")
    if rendered.get("source_panel_manifest") != plan.get("real_panel_manifest"):
        raise RuntimeError("real-panel source-manifest binding mismatch")
    if rendered.get("renderer") != plan.get("real_panel_renderer"):
        raise RuntimeError("real-panel renderer identity mismatch")
    if rendered.get("scientific_gate") != {
        "panel_locations_selected_before_predictions": True,
        "selected_thresholds_used_unchanged": True,
        "panel_locations_or_plane_orientation_tuned_after_predictions": False,
        "visual_gate_assessed_by_renderer": False,
    }:
        raise RuntimeError("real-panel scientific gate mismatch")
    records = rendered.get("panels")
    resolved_images = sorted(path.resolve() for path in image_paths)
    if not isinstance(records, list) or len(records) != 4 or len(resolved_images) != 4:
        raise RuntimeError("real-panel output count mismatch")
    for index, (record, source, image_path) in enumerate(
        zip(records, selected, resolved_images, strict=True), start=1
    ):
        if (
            record.get("panel_index") != index
            or record.get("image") != source.get("image")
            or record.get("center_zyx") != source.get("center_zyx")
        ):
            raise RuntimeError(f"real panel {index}: frozen location mismatch")
        output = record.get("output")
        if not isinstance(output, dict) or output != {
            "file": image_path.name,
            "bytes": image_path.stat().st_size,
            "sha256": sha256_file(image_path),
            "width": 700,
            "height": 2349,
        }:
            raise RuntimeError(f"real panel {index}: image identity mismatch")
        with Image.open(image_path) as image:
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or image.size != (output["width"], output["height"])
            ):
                raise RuntimeError(f"real panel {index}: invalid PNG payload")
        caches = record.get("source_caches")
        if not isinstance(caches, dict) or tuple(caches) != RUN_ORDER:
            raise RuntimeError(f"real panel {index}: cache run order mismatch")
        for run, identity in caches.items():
            if (
                not isinstance(identity, dict)
                or set(identity) != {"file", "bytes", "sha256"}
                or identity["file"] != record.get("cache_file")
                or not isinstance(identity["bytes"], int)
                or identity["bytes"] <= 0
            ):
                raise RuntimeError(f"real panel {index}: {run} cache identity invalid")
            require_hex(identity["sha256"], 64, f"real panel {index} {run} SHA-256")
    return rendered


def validate_visual_assessment(
    *,
    plan: dict,
    assessment_path: Path,
    observations_path: Path,
    render_manifest_path: Path,
    image_paths: list[Path],
) -> dict:
    assessment = load_hashed(assessment_path)
    render = load_hashed(render_manifest_path)
    if (
        set(assessment)
        != {
            "schema_version",
            "status",
            "criterion",
            "assessor",
            "assessment_method",
            "source_observations",
            "source_panel_render_manifest",
            "source_panel_images",
            "comparisons",
            "visual_gate_pass",
            "scientific_gate",
            "recorder",
            "payload_sha256",
        }
        or assessment.get("schema_version") != "1.0"
    ):
        raise RuntimeError("visual-assessment schema mismatch")
    if (
        assessment.get("status")
        != "all four preregistered real panels assessed without omission"
        or assessment.get("criterion") != VISUAL_CRITERION
        or not isinstance(assessment.get("assessor"), str)
        or not assessment["assessor"].strip()
        or not isinstance(assessment.get("assessment_method"), str)
        or not assessment["assessment_method"].strip()
        or assessment.get("recorder") != plan.get("visual_assessment_recorder")
    ):
        raise RuntimeError("visual-assessment identity mismatch")
    if assessment.get("source_panel_render_manifest") != local_identity(
        render_manifest_path
    ) | {"payload_sha256": render["payload_sha256"]}:
        raise RuntimeError("visual assessment points to another panel render")
    resolved_images = sorted(path.resolve() for path in image_paths)
    if assessment.get("source_panel_images") != [
        local_identity(path) for path in resolved_images
    ]:
        raise RuntimeError("visual assessment points to another panel image set")
    if assessment.get("source_observations") != local_identity(observations_path):
        raise RuntimeError("visual assessment points to another observation record")
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    if not isinstance(observations, dict) or set(observations) != {
        "schema_version",
        "status",
        "assessor",
        "assessment_method",
        "criterion",
        "comparisons",
    }:
        raise RuntimeError("visual-observation schema mismatch")
    if (
        observations["schema_version"] != "1.0"
        or observations["status"] != "all fixed real-panel comparisons assessed"
        or observations["assessor"] != assessment["assessor"]
        or observations["assessment_method"] != assessment["assessment_method"]
        or observations["criterion"] != VISUAL_CRITERION
    ):
        raise RuntimeError("visual-observation identity mismatch")
    comparisons = assessment.get("comparisons")
    expected_order = [
        (panel_index, seed) for panel_index in (1, 2, 3, 4) for seed in (11, 23, 47)
    ]
    if (
        not isinstance(comparisons, list)
        or [(record.get("panel_index"), record.get("seed")) for record in comparisons]
        != expected_order
    ):
        raise RuntimeError("visual assessment does not cover the fixed 4x3 order")
    raw_comparisons = observations.get("comparisons")
    if not isinstance(raw_comparisons, list) or len(raw_comparisons) != 12:
        raise RuntimeError("visual observations do not contain all comparisons")
    passes = []
    for record, raw_record in zip(comparisons, raw_comparisons, strict=True):
        if set(record) != {
            "panel_index",
            "seed",
            "separation_improvement",
            "new_nearby_break",
            "notes",
            "comparison_pass",
        }:
            raise RuntimeError("visual-assessment comparison schema mismatch")
        if not isinstance(record["separation_improvement"], bool) or not isinstance(
            record["new_nearby_break"], bool
        ):
            raise TypeError("visual-assessment decisions must be Boolean")
        if not isinstance(record["notes"], str) or not record["notes"].strip():
            raise RuntimeError("visual-assessment comparison note is absent")
        if record != raw_record | {"comparison_pass": record["comparison_pass"]}:
            raise RuntimeError("visual assessment differs from raw observations")
        expected_pass = (
            record["separation_improvement"] and not record["new_nearby_break"]
        )
        if record["comparison_pass"] is not expected_pass:
            raise RuntimeError("visual-assessment comparison gate mismatch")
        passes.append(expected_pass)
    if assessment.get("visual_gate_pass") is not any(passes):
        raise RuntimeError("visual-assessment primary gate mismatch")
    if assessment.get("scientific_gate") != {
        "all_four_panels_assessed": True,
        "all_three_matched_seeds_assessed_per_panel": True,
        "panels_or_comparisons_omitted": False,
        "renderer_made_visual_decision": False,
        "assessment_recorded_after_fixed_render": True,
    }:
        raise RuntimeError("visual-assessment completeness gate mismatch")
    return assessment


def validate_scoring_provenance(
    *,
    plan_path: Path,
    delivery_path: Path,
    real_run_path: Path,
    synthetic_run_path: Path,
    real_result_path: Path,
    synthetic_result_path: Path,
    result_sources_path: Path,
    package_index_path: Path,
    pair_receipt_path: Path,
    real_panel_manifest_path: Path,
    real_panel_image_paths: list[Path],
    real_panel_source_path: Path,
    thresholds: dict,
) -> None:
    plan = load_hashed(plan_path)
    delivery = load_hashed(delivery_path)
    if plan.get("final_result_validator") != local_identity(Path(__file__).resolve()):
        raise RuntimeError("running final-result validator differs from public plan")
    if plan.get("status") != "held-out execution plan frozen before held-out inference":
        raise RuntimeError("wrong held-out execution-plan status")
    if plan.get("pre_inference_protocol_clarification") != PROTOCOL_CLARIFICATION:
        raise RuntimeError("execution plan lacks the frozen protocol clarification")
    if delivery.get("status") != (
        "all 14 publicly planned held-out caches sealed before one-shot scoring"
    ):
        raise RuntimeError("wrong held-out cache-delivery status")
    public_plan = delivery.get("public_execution_plan", {})
    plan_commit = require_hex(public_plan.get("commit"), 40, "public plan commit")
    if public_plan != {
        "commit": plan_commit,
        "file": plan_path.name,
        "bytes": plan_path.stat().st_size,
        "sha256": sha256_file(plan_path),
        "payload_sha256": plan["payload_sha256"],
    }:
        raise RuntimeError("cache delivery points to another execution plan")
    if delivery.get("threshold_binding") != plan.get("threshold_binding"):
        raise RuntimeError("cache delivery threshold binding mismatch")
    if plan.get("threshold_binding", {}).get("frozen_thresholds", {}).get(
        "payload_sha256"
    ) != thresholds.get("payload_sha256"):
        raise RuntimeError("execution plan points to another threshold payload")
    if (
        delivery.get("scientific_gate", {}).get(
            "one_shot_scoring_permitted_after_this_public_freeze"
        )
        is not True
    ):
        raise RuntimeError("cache delivery does not permit one-shot scoring")

    expected_delivery_commit = validate_scoring_result_collection(
        plan=plan,
        delivery=delivery,
        plan_path=plan_path,
        delivery_path=delivery_path,
        result_sources_path=result_sources_path,
        package_index_path=package_index_path,
        pair_receipt_path=pair_receipt_path,
        real_run_path=real_run_path,
        synthetic_run_path=synthetic_run_path,
        real_result_path=real_result_path,
        synthetic_result_path=synthetic_result_path,
        real_panel_manifest_path=real_panel_manifest_path,
        real_panel_image_paths=real_panel_image_paths,
    )
    validate_real_panel_artifacts(
        plan=plan,
        thresholds=thresholds,
        source_panel_path=real_panel_source_path,
        render_manifest_path=real_panel_manifest_path,
        image_paths=real_panel_image_paths,
    )

    manifests = {
        "real": (load_hashed(real_run_path), real_result_path),
        "synthetic": (load_hashed(synthetic_run_path), synthetic_result_path),
    }
    delivery_commits = set()
    for mode, (run, result_path) in manifests.items():
        expected_top = {
            "schema_version",
            "status",
            "mode",
            "public_execution_plan",
            "public_cache_delivery",
            "threshold_binding",
            "launcher",
            "job_config",
            "staging",
            "project_source_hashes",
            "import_runtime",
            "metric_runtime",
            "scorer_invocation",
            "panel_invocation",
            "scientific_gate",
            "payload_sha256",
        }
        if set(run) != expected_top or run.get("schema_version") != "1.0":
            raise RuntimeError(f"{mode}: scoring run-manifest schema mismatch")
        if (
            run.get("status")
            != (
                f"{mode} held-out result scored exactly once after public cache-delivery freeze"
            )
            or run.get("mode") != mode
        ):
            raise RuntimeError(f"{mode}: scoring run-manifest identity mismatch")
        if run.get("public_execution_plan") != {
            "commit": plan_commit,
            "file_sha256": sha256_file(plan_path),
            "payload_sha256": plan["payload_sha256"],
        }:
            raise RuntimeError(f"{mode}: scoring plan binding mismatch")
        public_delivery = run.get("public_cache_delivery", {})
        delivery_commit = require_hex(
            public_delivery.get("commit"), 40, f"{mode} public delivery commit"
        )
        delivery_commits.add(delivery_commit)
        if public_delivery != {
            "commit": delivery_commit,
            "file_sha256": sha256_file(delivery_path),
            "payload_sha256": delivery["payload_sha256"],
        }:
            raise RuntimeError(f"{mode}: scoring delivery binding mismatch")
        if run.get("threshold_binding") != plan.get("threshold_binding"):
            raise RuntimeError(f"{mode}: scoring threshold binding mismatch")
        if run.get("launcher") != plan.get("one_shot_scoring_launcher"):
            raise RuntimeError(f"{mode}: scoring launcher identity mismatch")
        if run.get("job_config") != expected_scoring_job_config_identity(
            mode=mode,
            public_plan_commit=plan_commit,
            public_plan_file_sha256=sha256_file(plan_path),
            public_delivery_commit=delivery_commit,
            public_delivery_file_sha256=sha256_file(delivery_path),
        ):
            raise RuntimeError(f"{mode}: embedded scoring config identity mismatch")
        if run.get("import_runtime") != SCORING_IMPORT_RUNTIME:
            raise RuntimeError(f"{mode}: scoring import-runtime mismatch")
        metric = run.get("metric_runtime")
        if mode == "synthetic" and metric is not None:
            raise RuntimeError("synthetic scorer unexpectedly prepared official metric")
        if mode == "real":
            if not isinstance(metric, dict):
                raise TypeError("real scorer metric runtime is absent")
            values = metric.get("identity_smoke", {}).get("values", {})
            if set(values) != set(METRICS) or any(
                not math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=2e-15)
                for value in values.values()
            ):
                raise RuntimeError("real scorer metric identity smoke mismatch")
        invocation = run.get("scorer_invocation", {})
        expected_script = plan.get("one_shot_scorers", {}).get(mode, {}).get("script")
        if (
            invocation.get("script") != expected_script
            or invocation.get("returncode") != 0
        ):
            raise RuntimeError(f"{mode}: scorer invocation identity mismatch")
        for stream in ("stdout", "stderr"):
            if (
                not isinstance(invocation.get(f"{stream}_bytes"), int)
                or invocation[f"{stream}_bytes"] < 0
            ):
                raise RuntimeError(f"{mode}: scorer {stream} byte size invalid")
            require_hex(
                invocation.get(f"{stream}_sha256"),
                64,
                f"{mode} scorer {stream} SHA-256",
            )
        validate_file_identity(invocation.get("result", {}), result_path, mode)
        panel_invocation = run.get("panel_invocation")
        if mode == "synthetic":
            if panel_invocation is not None:
                raise RuntimeError("synthetic scorer unexpectedly rendered real panels")
        else:
            if not isinstance(panel_invocation, dict) or set(panel_invocation) != {
                "renderer",
                "command",
                "returncode",
                "stdout_bytes",
                "stdout_sha256",
                "stderr_bytes",
                "stderr_sha256",
                "manifest",
                "images",
            }:
                raise RuntimeError("real-panel invocation schema mismatch")
            if (
                panel_invocation.get("renderer") != plan.get("real_panel_renderer")
                or panel_invocation.get("returncode") != 0
            ):
                raise RuntimeError("real-panel invocation identity mismatch")
            command = panel_invocation.get("command")
            if (
                not isinstance(command, list)
                or len(command) != 20
                or Path(command[1]).name != "render_real_panels.py"
                or command[2::2]
                != [
                    "--test-root",
                    "--thresholds",
                    "--panel-manifest",
                    "--plan",
                    "--delivery",
                    "--score-input-index",
                    "--public-plan-commit",
                    "--public-delivery-commit",
                    "--out",
                ]
            ):
                raise RuntimeError("real-panel command contract mismatch")
            command_values = dict(zip(command[2::2], command[3::2], strict=True))
            if (
                command_values["--public-plan-commit"] != plan_commit
                or command_values["--public-delivery-commit"] != delivery_commit
                or Path(command_values["--thresholds"]).name
                != plan["threshold_binding"]["frozen_thresholds"]["file"]
                or Path(command_values["--panel-manifest"]).name
                != plan["real_panel_manifest"]["file"]
                or Path(command_values["--score-input-index"]).parent
                != Path(command_values["--test-root"])
                or Path(command_values["--out"]).name != "real-panels"
            ):
                raise RuntimeError("real-panel command provenance mismatch")
            for stream in ("stdout", "stderr"):
                if (
                    not isinstance(panel_invocation.get(f"{stream}_bytes"), int)
                    or panel_invocation[f"{stream}_bytes"] < 0
                ):
                    raise RuntimeError(f"real-panel {stream} byte size invalid")
                require_hex(
                    panel_invocation.get(f"{stream}_sha256"),
                    64,
                    f"real-panel {stream} SHA-256",
                )
            expected_panel_manifest = local_identity(real_panel_manifest_path)
            if panel_invocation.get("manifest") != expected_panel_manifest:
                raise RuntimeError("real-panel invocation manifest mismatch")
            if panel_invocation.get("images") != [
                local_identity(path) for path in sorted(real_panel_image_paths)
            ]:
                raise RuntimeError("real-panel invocation image set mismatch")
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
    if delivery_commits != {expected_delivery_commit}:
        raise RuntimeError("real and synthetic scorers used different delivery commits")


def bootstrap(values: np.ndarray, seed: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise RuntimeError("invalid bootstrap input")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "patches": len(values),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
    }


def compare_bootstrap(observed: dict, expected: dict, label: str) -> None:
    if set(observed) != set(expected):
        raise RuntimeError(f"{label}: bootstrap schema mismatch")
    for key, value in expected.items():
        if isinstance(value, int):
            if observed.get(key) != value:
                raise RuntimeError(f"{label}: bootstrap {key} mismatch")
        else:
            close(observed.get(key), value, f"{label}.{key}")


def expected_test_files(split: dict) -> list[str]:
    records = [
        record for record in split.get("records", []) if record.get("split") == "test"
    ]
    counts = {
        scroll: sum(record.get("scroll") == scroll for record in records)
        for scroll in ("s4", "s5")
    }
    if counts != {"s4": 36, "s5": 2} or len(records) != 38:
        raise RuntimeError(f"frozen test split mismatch: {counts}")
    return sorted(Path(record["image"]).with_suffix(".npz").name for record in records)


def validate_sources(thresholds: dict, split: dict) -> None:
    if thresholds.get("schema_version") != "1.0" or thresholds.get("status") != (
        "thresholds frozen from Scroll-1 validation before test inference"
    ):
        raise RuntimeError("threshold-freeze identity mismatch")
    if thresholds.get("candidate_thresholds") != list(THRESHOLD_GRID):
        raise RuntimeError("threshold candidate grid changed")
    if thresholds.get("tie_break") != (
        "maximum mean official blend; nearest 0.5; lower threshold"
    ):
        raise RuntimeError("threshold tie-break changed")
    runs = thresholds.get("runs")
    if not isinstance(runs, dict) or tuple(runs) != RUN_ORDER:
        raise RuntimeError("threshold run order mismatch")
    if any(
        runs[run].get("selected_threshold") not in THRESHOLD_GRID for run in RUN_ORDER
    ):
        raise RuntimeError("selected threshold lies outside the frozen grid")
    records = split.get("records")
    if not isinstance(records, list):
        raise TypeError("split records are absent")
    records_sha256 = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if split.get("records_sha256") != records_sha256:
        raise RuntimeError("split records SHA-256 mismatch")
    if thresholds.get("source_split_records_sha256") != records_sha256:
        raise RuntimeError("thresholds and split records disagree")


def validate_real(real: dict, thresholds: dict, split: dict) -> None:
    if set(real) != {
        "schema_version",
        "status",
        "source_split_records_sha256",
        "source_threshold_payload_sha256",
        "test_patch_count",
        "sensitivity_thresholds",
        "bootstrap",
        "runs",
        "paired_gap8_minus_control",
        "pooled_seed_mean_gap8_minus_control",
        "preregistered_real_gates",
        "payload_sha256",
    }:
        raise RuntimeError("real-result top-level schema mismatch")
    if real.get("schema_version") != "1.0" or real.get("status") != (
        "sealed Scroll-4/5 test scored at frozen thresholds"
    ):
        raise RuntimeError("real-result identity mismatch")
    if real.get("source_split_records_sha256") != split.get("records_sha256"):
        raise RuntimeError("real result belongs to another split")
    if real.get("source_threshold_payload_sha256") != thresholds.get("payload_sha256"):
        raise RuntimeError("real result belongs to another threshold freeze")
    expected_files = expected_test_files(split)
    if real.get("test_patch_count") != 38:
        raise RuntimeError("real-result test-patch count mismatch")
    if real.get("sensitivity_thresholds") != list(SENSITIVITY_THRESHOLDS):
        raise RuntimeError("real-result sensitivity thresholds changed")
    if real.get("bootstrap") != {
        "unit": "held-out real patch",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "interval": "percentile 95% descriptive interval",
    }:
        raise RuntimeError("real-result bootstrap contract mismatch")
    runs = real.get("runs")
    if not isinstance(runs, dict) or tuple(runs) != RUN_ORDER:
        raise RuntimeError("real-result run order mismatch")

    selected_rows: dict[str, dict[str, dict[str, float]]] = {}
    for run in RUN_ORDER:
        selected = thresholds["runs"][run]["selected_threshold"]
        record = runs.get(run)
        if not isinstance(record, dict) or set(record) != {
            "selected_threshold",
            "thresholds",
        }:
            raise RuntimeError(f"{run}: real-result run schema mismatch")
        if record["selected_threshold"] != selected:
            raise RuntimeError(f"{run}: selected threshold mismatch")
        expected_keys = {
            str(value) for value in sorted({*SENSITIVITY_THRESHOLDS, selected})
        }
        threshold_records = record["thresholds"]
        if (
            not isinstance(threshold_records, dict)
            or set(threshold_records) != expected_keys
        ):
            raise RuntimeError(f"{run}: real threshold set mismatch")
        for threshold_key, threshold_record in threshold_records.items():
            if set(threshold_record) != {"means", "per_patch"}:
                raise RuntimeError(
                    f"{run}@{threshold_key}: real threshold schema mismatch"
                )
            rows = threshold_record["per_patch"]
            means = threshold_record["means"]
            if set(rows) != set(expected_files) or set(means) != set(METRICS):
                raise RuntimeError(
                    f"{run}@{threshold_key}: real patch/metric set mismatch"
                )
            for name, row in rows.items():
                if set(row) != set(METRICS):
                    raise RuntimeError(
                        f"{run}@{threshold_key}/{name}: metric schema mismatch"
                    )
                for metric in METRICS:
                    finite_unit(row[metric], f"{run}@{threshold_key}/{name}.{metric}")
            for metric in METRICS:
                expected_mean = np.mean(
                    [float(rows[name][metric]) for name in expected_files]
                )
                close(means[metric], expected_mean, f"{run}@{threshold_key}.{metric}")
        selected_rows[run] = threshold_records[str(selected)]["per_patch"]

    expected_paired = {}
    pooled_by_metric: dict[str, list[np.ndarray]] = {metric: [] for metric in METRICS}
    for seed in (11, 23, 47):
        control = selected_rows[f"control_seed{seed}"]
        gap = selected_rows[f"gap8_seed{seed}"]
        seed_record = {}
        for metric in METRICS:
            delta = np.asarray(
                [
                    float(gap[name][metric]) - float(control[name][metric])
                    for name in expected_files
                ]
            )
            seed_record[metric] = bootstrap(delta, BOOTSTRAP_SEED + seed)
            pooled_by_metric[metric].append(delta)
        expected_paired[str(seed)] = seed_record
    paired = real.get("paired_gap8_minus_control")
    if not isinstance(paired, dict) or set(paired) != set(expected_paired):
        raise RuntimeError("real paired-result seed set mismatch")
    for seed, metrics in expected_paired.items():
        if set(paired[seed]) != set(METRICS):
            raise RuntimeError(
                f"real paired-result metric set mismatch for seed {seed}"
            )
        for metric, expected in metrics.items():
            compare_bootstrap(
                paired[seed][metric], expected, f"real.paired.{seed}.{metric}"
            )

    expected_pooled = {}
    for metric in METRICS:
        per_patch_seed_mean = np.stack(pooled_by_metric[metric], axis=0).mean(axis=0)
        expected_pooled[metric] = bootstrap(per_patch_seed_mean, BOOTSTRAP_SEED)
    pooled = real.get("pooled_seed_mean_gap8_minus_control")
    if not isinstance(pooled, dict) or set(pooled) != set(METRICS):
        raise RuntimeError("real pooled-result metric set mismatch")
    for metric, expected in expected_pooled.items():
        compare_bootstrap(pooled[metric], expected, f"real.pooled.{metric}")
    expected_gates = {
        "real_blend_noninferiority": expected_pooled["blend"]["mean"] >= -0.005,
        "real_toposcore_improvement": expected_pooled["toposcore"]["mean"] > 0.0,
    }
    if real.get("preregistered_real_gates") != expected_gates:
        raise RuntimeError("real preregistered gates do not match recomputed values")


def synthetic_cells() -> dict[str, str]:
    cells = {}
    for seed in (300, 301, 302, 303, 304):
        for pitch in (170, 200, 230, 260):
            for papyrus in (35, 50, 65, 90):
                cells[f"primary_seed{seed}_pitch{pitch}_pap{papyrus}"] = "primary"
        for papyrus in (35, 50, 65, 90):
            cells[f"control_seed{seed}_pitch700_pap{papyrus}"] = "single_sheet_control"
    if len(cells) != 100:
        raise AssertionError("independent synthetic-cell generator changed")
    return cells


def recompute_rates(record: dict, label: str) -> dict[str, int | float]:
    if set(record) != set(COUNT_FIELDS) | set(RATE_FIELDS):
        raise RuntimeError(f"{label}: ray-score schema mismatch")
    counts = {}
    for field in COUNT_FIELDS:
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"{label}: invalid {field}")
        counts[field] = value
    expected = {
        **counts,
        "site_center_detection_rate": counts["detected_neighbour_sites"]
        / max(counts["neighbour_sites"], 1),
        "conditional_fusion_rate": counts["fused_detected_sites"]
        / max(counts["detected_neighbour_sites"], 1),
        "false_split_rate": counts["false_split_sites"]
        / max(counts["control_sites"], 1),
    }
    for field in RATE_FIELDS:
        close(record[field], expected[field], f"{label}.{field}")
    return expected


def pool(rows: list[dict], label: str) -> dict[str, int | float]:
    totals = {field: sum(int(row[field]) for row in rows) for field in COUNT_FIELDS}
    return recompute_rates(
        {
            **totals,
            "site_center_detection_rate": totals["detected_neighbour_sites"]
            / max(totals["neighbour_sites"], 1),
            "conditional_fusion_rate": totals["fused_detected_sites"]
            / max(totals["detected_neighbour_sites"], 1),
            "false_split_rate": totals["false_split_sites"]
            / max(totals["control_sites"], 1),
        },
        label,
    )


def compare_rates(observed: dict, expected: dict, label: str) -> None:
    if set(observed) != set(expected):
        raise RuntimeError(f"{label}: pooled schema mismatch")
    for field in COUNT_FIELDS:
        if observed[field] != expected[field]:
            raise RuntimeError(f"{label}: pooled {field} mismatch")
    for field in RATE_FIELDS:
        close(observed[field], expected[field], f"{label}.{field}")


def validate_synthetic(synthetic: dict, thresholds: dict, split: dict) -> None:
    if set(synthetic) != {
        "schema_version",
        "status",
        "gate_contract",
        "source_split_records_sha256",
        "source_threshold_payload_sha256",
        "cell_count",
        "runs",
        "paired_gap8_minus_control",
        "pooled_gap8_minus_control",
        "preregistered_synthetic_gates",
        "payload_sha256",
    }:
        raise RuntimeError("synthetic-result top-level schema mismatch")
    if synthetic.get("schema_version") != "1.1" or synthetic.get("status") != (
        "sealed synthetic test scored once from complete ray caches"
    ):
        raise RuntimeError("synthetic-result identity mismatch")
    if synthetic.get("gate_contract") != (
        "pre-inference clarification: pooled raw counts across matched seeds"
    ):
        raise RuntimeError("synthetic-result gate contract mismatch")
    if synthetic.get("source_split_records_sha256") != split.get("records_sha256"):
        raise RuntimeError("synthetic result belongs to another split")
    if synthetic.get("source_threshold_payload_sha256") != thresholds.get(
        "payload_sha256"
    ):
        raise RuntimeError("synthetic result belongs to another threshold freeze")
    cells = synthetic_cells()
    if synthetic.get("cell_count") != 100:
        raise RuntimeError("synthetic cell count mismatch")
    runs = synthetic.get("runs")
    if not isinstance(runs, dict) or tuple(runs) != RUN_ORDER:
        raise RuntimeError("synthetic-result run order mismatch")
    selected_pooled = {}
    for run in RUN_ORDER:
        selected = thresholds["runs"][run]["selected_threshold"]
        record = runs[run]
        if (
            set(record) != {"selected_threshold", "thresholds"}
            or record["selected_threshold"] != selected
        ):
            raise RuntimeError(f"{run}: synthetic run identity mismatch")
        expected_keys = {
            str(value) for value in sorted({*SENSITIVITY_THRESHOLDS, selected})
        }
        threshold_records = record["thresholds"]
        if set(threshold_records) != expected_keys:
            raise RuntimeError(f"{run}: synthetic threshold set mismatch")
        for threshold_key, threshold_record in threshold_records.items():
            if set(threshold_record) != {
                "primary_pooled",
                "single_sheet_control_pooled",
                "per_cell",
            }:
                raise RuntimeError(f"{run}@{threshold_key}: synthetic schema mismatch")
            rows = threshold_record["per_cell"]
            if set(rows) != set(cells):
                raise RuntimeError(
                    f"{run}@{threshold_key}: synthetic cell set mismatch"
                )
            checked = {
                name: recompute_rates(row, f"{run}@{threshold_key}/{name}")
                for name, row in rows.items()
            }
            primary = pool(
                [checked[name] for name, kind in cells.items() if kind == "primary"],
                f"{run}@{threshold_key}.primary",
            )
            controls = pool(
                [
                    checked[name]
                    for name, kind in cells.items()
                    if kind == "single_sheet_control"
                ],
                f"{run}@{threshold_key}.control",
            )
            compare_rates(threshold_record["primary_pooled"], primary, f"{run}.primary")
            compare_rates(
                threshold_record["single_sheet_control_pooled"],
                controls,
                f"{run}.control",
            )
        selected_pooled[run] = threshold_records[str(selected)]

    expected_paired = {}
    for seed in (11, 23, 47):
        control = selected_pooled[f"control_seed{seed}"]
        gap = selected_pooled[f"gap8_seed{seed}"]
        fusion_delta = float(gap["primary_pooled"]["conditional_fusion_rate"]) - float(
            control["primary_pooled"]["conditional_fusion_rate"]
        )
        detection_delta = float(
            gap["primary_pooled"]["site_center_detection_rate"]
        ) - float(control["primary_pooled"]["site_center_detection_rate"])
        false_split_delta = float(
            gap["single_sheet_control_pooled"]["false_split_rate"]
        ) - float(control["single_sheet_control_pooled"]["false_split_rate"])
        expected_paired[str(seed)] = {
            "conditional_fusion_delta": fusion_delta,
            "site_center_detection_delta": detection_delta,
            "false_split_delta": false_split_delta,
            "secondary_per_seed_diagnostics": {
                "fusion_reduction_at_least_10pp": fusion_delta <= -0.10,
                "detection_drop_no_more_than_2pp": detection_delta >= -0.02,
                "false_split_increase_no_more_than_2pp": false_split_delta <= 0.02,
            },
        }
    paired = synthetic.get("paired_gap8_minus_control")
    if not isinstance(paired, dict) or set(paired) != set(expected_paired):
        raise RuntimeError("synthetic paired-result seed set mismatch")
    for seed, expected in expected_paired.items():
        if set(paired[seed]) != set(expected):
            raise RuntimeError(f"synthetic paired schema mismatch for seed {seed}")
        for field in (
            "conditional_fusion_delta",
            "site_center_detection_delta",
            "false_split_delta",
        ):
            close(paired[seed][field], expected[field], f"synthetic.{seed}.{field}")
        if (
            paired[seed]["secondary_per_seed_diagnostics"]
            != expected["secondary_per_seed_diagnostics"]
        ):
            raise RuntimeError(
                f"synthetic paired secondary diagnostics mismatch for seed {seed}"
            )

    control_primary = pool(
        [
            selected_pooled[f"control_seed{seed}"]["primary_pooled"]
            for seed in (11, 23, 47)
        ],
        "synthetic.pooled.control_primary",
    )
    gap_primary = pool(
        [
            selected_pooled[f"gap8_seed{seed}"]["primary_pooled"]
            for seed in (11, 23, 47)
        ],
        "synthetic.pooled.gap8_primary",
    )
    control_single = pool(
        [
            selected_pooled[f"control_seed{seed}"]["single_sheet_control_pooled"]
            for seed in (11, 23, 47)
        ],
        "synthetic.pooled.control_single",
    )
    gap_single = pool(
        [
            selected_pooled[f"gap8_seed{seed}"]["single_sheet_control_pooled"]
            for seed in (11, 23, 47)
        ],
        "synthetic.pooled.gap8_single",
    )
    pooled_deltas = {
        "conditional_fusion_delta": gap_primary["conditional_fusion_rate"]
        - control_primary["conditional_fusion_rate"],
        "site_center_detection_delta": gap_primary["site_center_detection_rate"]
        - control_primary["site_center_detection_rate"],
        "false_split_delta": gap_single["false_split_rate"]
        - control_single["false_split_rate"],
    }
    expected_pooled = {
        "control_primary": control_primary,
        "gap8_primary": gap_primary,
        "control_single_sheet_control": control_single,
        "gap8_single_sheet_control": gap_single,
        "deltas": pooled_deltas,
    }
    observed_pooled = synthetic.get("pooled_gap8_minus_control")
    if not isinstance(observed_pooled, dict) or set(observed_pooled) != set(
        expected_pooled
    ):
        raise RuntimeError("synthetic pooled-result schema mismatch")
    for key in (
        "control_primary",
        "gap8_primary",
        "control_single_sheet_control",
        "gap8_single_sheet_control",
    ):
        compare_rates(observed_pooled[key], expected_pooled[key], f"synthetic.{key}")
    if set(observed_pooled["deltas"]) != set(pooled_deltas):
        raise RuntimeError("synthetic pooled-delta schema mismatch")
    for field, expected in pooled_deltas.items():
        close(observed_pooled["deltas"][field], expected, f"synthetic.pooled.{field}")

    expected_gates = {
        "pooled_conditional_fusion_reduction_at_least_10pp": (
            pooled_deltas["conditional_fusion_delta"] <= -0.10
        ),
        "negative_conditional_fusion_delta_all_three_seeds": all(
            record["conditional_fusion_delta"] < 0.0
            for record in expected_paired.values()
        ),
        "pooled_detection_drop_no_more_than_2pp": (
            pooled_deltas["site_center_detection_delta"] >= -0.02
        ),
        "pooled_false_split_increase_no_more_than_2pp": (
            pooled_deltas["false_split_delta"] <= 0.02
        ),
    }
    expected_gates["synthetic_gate_pass"] = all(expected_gates.values())
    if synthetic.get("preregistered_synthetic_gates") != expected_gates:
        raise RuntimeError(
            "synthetic preregistered gates do not match recomputed values"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--real-run-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-run-manifest", type=Path, required=True)
    parser.add_argument("--result-sources", type=Path, required=True)
    parser.add_argument("--scoring-package-index", type=Path, required=True)
    parser.add_argument("--scoring-pair-receipt", type=Path, required=True)
    parser.add_argument("--real-panel-render-manifest", type=Path, required=True)
    parser.add_argument("--real-panel-source-manifest", type=Path, required=True)
    parser.add_argument("--real-panel-image", type=Path, action="append", required=True)
    parser.add_argument("--visual-assessment", type=Path, required=True)
    parser.add_argument("--visual-observations", type=Path, required=True)
    args = parser.parse_args()
    thresholds = load_hashed(args.thresholds)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    real = load_hashed(args.real)
    synthetic = load_hashed(args.synthetic)
    validate_sources(thresholds, split)
    validate_scoring_provenance(
        plan_path=args.plan,
        delivery_path=args.delivery,
        real_run_path=args.real_run_manifest,
        synthetic_run_path=args.synthetic_run_manifest,
        real_result_path=args.real,
        synthetic_result_path=args.synthetic,
        result_sources_path=args.result_sources,
        package_index_path=args.scoring_package_index,
        pair_receipt_path=args.scoring_pair_receipt,
        real_panel_manifest_path=args.real_panel_render_manifest,
        real_panel_image_paths=args.real_panel_image,
        real_panel_source_path=args.real_panel_source_manifest,
        thresholds=thresholds,
    )
    validate_visual_assessment(
        plan=load_hashed(args.plan),
        assessment_path=args.visual_assessment,
        observations_path=args.visual_observations,
        render_manifest_path=args.real_panel_render_manifest,
        image_paths=args.real_panel_image,
    )
    validate_real(real, thresholds, split)
    validate_synthetic(synthetic, thresholds, split)
    print("real result file SHA-256:", sha256_file(args.real))
    print("synthetic result file SHA-256:", sha256_file(args.synthetic))
    print("ALL_FINAL_RESULTS_INDEPENDENTLY_RECOMPUTED_AND_VALIDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
