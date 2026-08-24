#!/usr/bin/env python3
"""Render complete evidence after RunPod/Kaggle paired validation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from render_final_evidence import render_report, source_record
from validate_final_results import load_hashed
from validate_runpod_kaggle_final_results import add_arguments, validate_all


def main() -> int:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--out-markdown", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.out_markdown.exists() or args.out_manifest.exists():
        raise RuntimeError("evidence outputs must start absent")
    if args.out_markdown.parent.resolve() != args.out_manifest.parent.resolve():
        raise RuntimeError("evidence outputs must share one directory")
    real, synthetic, visual, validation_plan = validate_all(args)
    if validation_plan.get("renderer") != source_record(Path(__file__).resolve()):
        raise RuntimeError("running evidence renderer differs from frozen plan")
    thresholds = load_hashed(args.thresholds)
    scientific_plan = load_hashed(args.scientific_plan)
    sources = {
        "thresholds": source_record(args.thresholds, thresholds),
        "split": source_record(args.split),
        "plan": source_record(args.scientific_plan, scientific_plan),
        "delivery": source_record(args.delivery, load_hashed(args.delivery)),
        "real_run_manifest": source_record(args.real_run_manifest, load_hashed(args.real_run_manifest)),
        "synthetic_run_manifest": source_record(args.synthetic_run_manifest, load_hashed(args.synthetic_run_manifest)),
        "result_sources": source_record(args.result_sources, load_hashed(args.result_sources)),
        "scoring_package_index": source_record(args.scoring_package_index, load_hashed(args.scoring_package_index)),
        "scoring_pair_receipt": source_record(args.scoring_pair_receipt, load_hashed(args.scoring_pair_receipt)),
        "real_panel_source_manifest": source_record(args.real_panel_source_manifest, load_hashed(args.real_panel_source_manifest)),
        "real_panel_render_manifest": source_record(args.real_panel_render_manifest, load_hashed(args.real_panel_render_manifest)),
        "real_panel_images": [source_record(path) for path in sorted(args.real_panel_image)],
        "visual_observations": source_record(args.visual_observations),
        "visual_assessment": source_record(args.visual_assessment, visual),
        "real": source_record(args.real, real),
        "synthetic": source_record(args.synthetic, synthetic),
    }
    args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
    panel_assets = []
    for source in sorted(path.resolve() for path in args.real_panel_image):
        destination = args.out_markdown.parent / source.name
        if destination.exists():
            raise RuntimeError(f"panel report asset must start absent: {destination}")
        shutil.copyfile(source, destination)
        panel_assets.append(destination)
    audit_destination = args.out_markdown.parent / args.operational_audit.name
    if audit_destination.exists():
        raise RuntimeError("operational audit report asset must start absent")
    shutil.copyfile(args.operational_audit, audit_destination)
    report = render_report(real, synthetic, visual, sources, scientific_plan)
    report += (
        "\n## Mixed execution provenance\n\n"
        "The real result was scored once on the completed CPU-only RunPod path; "
        "the synthetic result is the unchanged completed Kaggle version 4. Both "
        "sources were hash-verified and jointly collected before either scientific "
        "result or panel was opened. The full result-blind operational history is "
        f"published in `{audit_destination.name}`.\n"
    )
    args.out_markdown.write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "status": "complete RunPod-real Kaggle-synthetic evidence rendered after independent paired validation",
        "renderer": source_record(Path(__file__).resolve()),
        "validation_plan": source_record(args.validation_plan, validation_plan),
        "sources": sources,
        "runpod_plan": source_record(args.runpod_plan, load_hashed(args.runpod_plan)),
        "collection_plan": source_record(args.collection_plan, load_hashed(args.collection_plan)),
        "operational_audit": source_record(audit_destination),
        "report_panel_assets": [source_record(path) for path in panel_assets],
        "report": source_record(args.out_markdown),
        "all_seven_preregistered_gates_reported": True,
        "all_four_panels_reported": True,
        "all_twelve_visual_comparisons_reported": True,
        "adverse_null_or_failed_rows_omitted": False,
        "manual_claim_selection_permitted": False,
        "official_form_submission_permitted_without_aviad_review": False,
    }
    from freeze_runpod_kaggle_final_validation import canonical_sha256
    manifest["payload_sha256"] = canonical_sha256(manifest)
    args.out_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("report SHA-256:", manifest["report"]["sha256"])
    print("report-manifest payload SHA-256:", manifest["payload_sha256"])
    print("COMPLETE_RUNPOD_KAGGLE_EVIDENCE_RENDERED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
