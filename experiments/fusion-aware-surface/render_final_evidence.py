#!/usr/bin/env python3
"""Render a complete, result-neutral report from independently validated outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from validate_final_results import (
    METRICS,
    RUN_ORDER,
    load_hashed,
    sha256_file,
    validate_real,
    validate_scoring_provenance,
    validate_sources,
    validate_synthetic,
    validate_visual_assessment,
)


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_record(path: Path, payload: dict | None = None) -> dict:
    record = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if payload is not None and "payload_sha256" in payload:
        record["payload_sha256"] = payload["payload_sha256"]
    return record


def fmt(value: object) -> str:
    return f"{float(value):.6f}"


def gate(value: object) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    raise TypeError("gate must be Boolean")


def yes_no(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    raise TypeError("value must be Boolean")


def selected_real_table(real: dict) -> list[str]:
    lines = [
        "| run | threshold | blend | TopoScore | surface Dice | VOI score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in RUN_ORDER:
        selected = real["runs"][run]["selected_threshold"]
        means = real["runs"][run]["thresholds"][str(selected)]["means"]
        lines.append(
            "| "
            + " | ".join(
                [run, fmt(selected), *(fmt(means[metric]) for metric in METRICS)]
            )
            + " |"
        )
    return lines


def real_sensitivity_table(real: dict) -> list[str]:
    lines = [
        "| run | threshold | selected | blend | TopoScore | surface Dice | VOI score |",
        "|---|---:|:---:|---:|---:|---:|---:|",
    ]
    for run in RUN_ORDER:
        selected = real["runs"][run]["selected_threshold"]
        threshold_records = real["runs"][run]["thresholds"]
        for threshold_key in sorted(threshold_records, key=float):
            record = threshold_records[threshold_key]
            means = record["means"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        run,
                        fmt(threshold_key),
                        "yes" if float(threshold_key) == float(selected) else "no",
                        *(fmt(means[metric]) for metric in METRICS),
                    ]
                )
                + " |"
            )
    return lines


def real_delta_table(real: dict) -> list[str]:
    lines = [
        "| comparison | metric | mean gap8-control | 95% descriptive interval | patches |",
        "|---|---|---:|---:|---:|",
    ]
    for seed in ("11", "23", "47"):
        for metric in METRICS:
            record = real["paired_gap8_minus_control"][seed][metric]
            lines.append(
                f"| seed {seed} | {metric} | {fmt(record['mean'])} | "
                f"[{fmt(record['ci95_low'])}, {fmt(record['ci95_high'])}] | "
                f"{record['patches']} |"
            )
    for metric in METRICS:
        record = real["pooled_seed_mean_gap8_minus_control"][metric]
        lines.append(
            f"| pooled seed mean | {metric} | {fmt(record['mean'])} | "
            f"[{fmt(record['ci95_low'])}, {fmt(record['ci95_high'])}] | "
            f"{record['patches']} |"
        )
    return lines


def synthetic_selected_table(synthetic: dict) -> list[str]:
    lines = [
        "| run | threshold | primary detection | primary conditional fusion | control false split |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in RUN_ORDER:
        selected = synthetic["runs"][run]["selected_threshold"]
        record = synthetic["runs"][run]["thresholds"][str(selected)]
        primary = record["primary_pooled"]
        control = record["single_sheet_control_pooled"]
        lines.append(
            f"| {run} | {fmt(selected)} | "
            f"{fmt(primary['site_center_detection_rate'])} | "
            f"{fmt(primary['conditional_fusion_rate'])} | "
            f"{fmt(control['false_split_rate'])} |"
        )
    return lines


def synthetic_sensitivity_table(synthetic: dict) -> list[str]:
    lines = [
        "| run | threshold | selected | primary detection | primary conditional fusion | control false split |",
        "|---|---:|:---:|---:|---:|---:|",
    ]
    for run in RUN_ORDER:
        selected = synthetic["runs"][run]["selected_threshold"]
        threshold_records = synthetic["runs"][run]["thresholds"]
        for threshold_key in sorted(threshold_records, key=float):
            record = threshold_records[threshold_key]
            primary = record["primary_pooled"]
            control = record["single_sheet_control_pooled"]
            lines.append(
                f"| {run} | {fmt(threshold_key)} | "
                f"{'yes' if float(threshold_key) == float(selected) else 'no'} | "
                f"{fmt(primary['site_center_detection_rate'])} | "
                f"{fmt(primary['conditional_fusion_rate'])} | "
                f"{fmt(control['false_split_rate'])} |"
            )
    return lines


def synthetic_delta_table(synthetic: dict) -> list[str]:
    lines = [
        "| seed | fusion delta | detection delta | false-split delta | >=10 pp secondary | detection secondary | false-split secondary |",
        "|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for seed in ("11", "23", "47"):
        record = synthetic["paired_gap8_minus_control"][seed]
        gates = record["secondary_per_seed_diagnostics"]
        lines.append(
            f"| {seed} | {fmt(record['conditional_fusion_delta'])} | "
            f"{fmt(record['site_center_detection_delta'])} | "
            f"{fmt(record['false_split_delta'])} | "
            f"{gate(gates['fusion_reduction_at_least_10pp'])} | "
            f"{gate(gates['detection_drop_no_more_than_2pp'])} | "
            f"{gate(gates['false_split_increase_no_more_than_2pp'])} |"
        )
    return lines


def synthetic_pooled_table(synthetic: dict) -> list[str]:
    pooled = synthetic["pooled_gap8_minus_control"]
    control_primary = pooled["control_primary"]
    gap_primary = pooled["gap8_primary"]
    control_single = pooled["control_single_sheet_control"]
    gap_single = pooled["gap8_single_sheet_control"]
    deltas = pooled["deltas"]
    return [
        "| arm | neighbour sites | detected neighbour sites | fused detected sites | control sites | false-split sites | detection rate | conditional fusion | false-split rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| control | {control_primary['neighbour_sites']} | {control_primary['detected_neighbour_sites']} | {control_primary['fused_detected_sites']} | {control_single['control_sites']} | {control_single['false_split_sites']} | {fmt(control_primary['site_center_detection_rate'])} | {fmt(control_primary['conditional_fusion_rate'])} | {fmt(control_single['false_split_rate'])} |",
        f"| gap8 | {gap_primary['neighbour_sites']} | {gap_primary['detected_neighbour_sites']} | {gap_primary['fused_detected_sites']} | {gap_single['control_sites']} | {gap_single['false_split_sites']} | {fmt(gap_primary['site_center_detection_rate'])} | {fmt(gap_primary['conditional_fusion_rate'])} | {fmt(gap_single['false_split_rate'])} |",
        f"| gap8-control | — | — | — | — | — | {fmt(deltas['site_center_detection_delta'])} | {fmt(deltas['conditional_fusion_delta'])} | {fmt(deltas['false_split_delta'])} |",
    ]


def visual_assessment_table(visual: dict) -> list[str]:
    lines = [
        "| panel | seed | separation improvement | new nearby break | comparison pass | notes |",
        "|---:|---:|:---:|:---:|:---:|---|",
    ]
    for record in visual["comparisons"]:
        notes = record["notes"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {record['panel_index']} | {record['seed']} | "
            f"{yes_no(record['separation_improvement'])} | "
            f"{yes_no(record['new_nearby_break'])} | "
            f"{gate(record['comparison_pass'])} | {notes} |"
        )
    return lines


def render_report(
    real: dict, synthetic: dict, visual: dict, sources: dict, plan: dict
) -> str:
    real_gates = real["preregistered_real_gates"]
    synthetic_gates = synthetic["preregistered_synthetic_gates"]
    clarification = plan["pre_inference_protocol_clarification"]
    lines = [
        "# Fusion-aware surface training: sealed held-out evidence",
        "",
        "This report is generated mechanically from the sealed result artifacts after independent recomputation. It reports every preregistered gate and the complete fixed-threshold sensitivity summaries; values are rounded to six decimals only for display.",
        "",
        "The intervention is the preregistered gap8 supervision term. The matched control and gap8 arms share architecture, data, initialization seeds, optimizer schedule, and evaluation protocol. These tests evaluate surface topology and segmentation; they do not by themselves establish recovered text or a complete-scroll reading.",
        "",
        (
            "The synthetic primary-gate pooling implementation was clarified "
            "before held-out inference, without viewing any held-out endpoint. "
            "The immutable clarification first appeared at commit "
            f"`{clarification['first_public_commit']}` with file SHA-256 "
            f"`{clarification['sha256']}`. The original stricter per-seed "
            "numeric checks are reported below as secondary diagnostics."
        ),
        "",
        "## Artifact identities",
        "",
        "| artifact | bytes | SHA-256 | embedded payload SHA-256 |",
        "|---|---:|---|---|",
    ]
    for label in (
        "thresholds",
        "split",
        "plan",
        "delivery",
        "real_run_manifest",
        "synthetic_run_manifest",
        "result_sources",
        "scoring_package_index",
        "scoring_pair_receipt",
        "real_panel_source_manifest",
        "real_panel_render_manifest",
        "visual_observations",
        "visual_assessment",
        "real",
        "synthetic",
    ):
        record = sources[label]
        lines.append(
            f"| {record['file']} | {record['bytes']} | `{record['sha256']}` | "
            f"`{record.get('payload_sha256', 'n/a')}` |"
        )
    for record in sources["real_panel_images"]:
        lines.append(
            f"| {record['file']} | {record['bytes']} | `{record['sha256']}` | n/a |"
        )
    lines.extend(
        [
            "",
            "## Preregistered gate readout",
            "",
            "| domain | gate | criterion | result |",
            "|---|---|---|:---:|",
            f"| real Scroll-4/5 | blend non-inferiority | pooled mean gap8-control >= -0.005 | {gate(real_gates['real_blend_noninferiority'])} |",
            f"| real Scroll-4/5 | TopoScore improvement | pooled mean gap8-control > 0 | {gate(real_gates['real_toposcore_improvement'])} |",
            f"| sealed synthetic | pooled fusion reduction | pooled raw-count delta <= -0.10 | {gate(synthetic_gates['pooled_conditional_fusion_reduction_at_least_10pp'])} |",
            f"| sealed synthetic | seed consistency | fusion delta < 0 in all three matched seeds | {gate(synthetic_gates['negative_conditional_fusion_delta_all_three_seeds'])} |",
            f"| sealed synthetic | pooled detection preservation | pooled raw-count delta >= -0.02 | {gate(synthetic_gates['pooled_detection_drop_no_more_than_2pp'])} |",
            f"| sealed synthetic | pooled false-split control | pooled raw-count delta <= +0.02 | {gate(synthetic_gates['pooled_false_split_increase_no_more_than_2pp'])} |",
            f"| fixed real panels | visual separation | at least one gap8/matched-control comparison improves separation without a new nearby break | {gate(visual['visual_gate_pass'])} |",
            "",
            "A failed gate remains evidence and is not removed or relabeled. Descriptive intervals are not substituted for the preregistered Boolean criteria.",
            "",
            "## Real Scroll-4/5 selected-threshold results",
            "",
            *selected_real_table(real),
            "",
            "## Real matched-seed deltas",
            "",
            *real_delta_table(real),
            "",
            "## Real fixed sensitivity readout",
            "",
            *real_sensitivity_table(real),
            "",
            "## Sealed synthetic selected-threshold results",
            "",
            *synthetic_selected_table(synthetic),
            "",
            "## Sealed synthetic pooled primary readout",
            "",
            *synthetic_pooled_table(synthetic),
            "",
            "## Sealed synthetic matched-seed deltas and stricter secondary diagnostics",
            "",
            *synthetic_delta_table(synthetic),
            "",
            "## Sealed synthetic fixed sensitivity readout",
            "",
            *synthetic_sensitivity_table(synthetic),
            "",
            "## Scope and limitations",
            "",
            "- Scroll-4/5 results are held-out real-data segmentation metrics at thresholds selected once on Scroll-1 validation.",
            "- Synthetic ray results isolate a known fusion geometry and test the directional mechanism; they are not a direct estimate of real-scroll fusion prevalence.",
            "- The four preregistered real panels are a separate fixed-location qualitative check and must be published with their hashes; this report does not convert visual judgment into an automatic pass.",
            "- The submission must retain null, mixed, or adverse results and must not claim text recovery unless a separate text-reading evaluation supports it.",
            "",
            "## Four preregistered fixed real panels",
            "",
            "These panels are rendered mechanically from the four model-blind coordinates. All four are shown; no panel is selected or omitted after inference.",
            "",
            *[
                f"![Fixed real panel {index}](real_panel_{index:02d}.png)"
                for index in range(1, 5)
            ],
            "",
            "## Complete fixed-panel visual assessment",
            "",
            f"Assessor: {visual['assessor']}. Method: {visual['assessment_method']}.",
            "",
            *visual_assessment_table(visual),
            "",
        ]
    )
    return "\n".join(lines)


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
    parser.add_argument("--out-markdown", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.out_markdown.exists() or args.out_manifest.exists():
        raise RuntimeError("report outputs must start absent")
    if args.out_markdown.parent.resolve() != args.out_manifest.parent.resolve():
        raise RuntimeError("report and manifest must share one output directory")

    thresholds = load_hashed(args.thresholds)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    plan = load_hashed(args.plan)
    renderer_path = Path(__file__).resolve()
    if plan.get("final_evidence_renderer") != source_record(renderer_path):
        raise RuntimeError("running final-evidence renderer differs from public plan")
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
    validate_real(real, thresholds, split)
    validate_synthetic(synthetic, thresholds, split)
    visual = validate_visual_assessment(
        plan=plan,
        assessment_path=args.visual_assessment,
        observations_path=args.visual_observations,
        render_manifest_path=args.real_panel_render_manifest,
        image_paths=args.real_panel_image,
    )
    args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
    panel_assets = []
    for source in sorted(path.resolve() for path in args.real_panel_image):
        destination = args.out_markdown.parent / source.name
        if destination.resolve() != source:
            if destination.exists():
                raise RuntimeError(
                    f"report panel asset must start absent: {destination}"
                )
            shutil.copyfile(source, destination)
        panel_assets.append(destination)

    sources = {
        "thresholds": source_record(args.thresholds, thresholds),
        "split": source_record(args.split, split),
        "plan": source_record(args.plan, plan),
        "delivery": source_record(args.delivery, load_hashed(args.delivery)),
        "real_run_manifest": source_record(
            args.real_run_manifest, load_hashed(args.real_run_manifest)
        ),
        "synthetic_run_manifest": source_record(
            args.synthetic_run_manifest, load_hashed(args.synthetic_run_manifest)
        ),
        "result_sources": source_record(
            args.result_sources, load_hashed(args.result_sources)
        ),
        "scoring_package_index": source_record(
            args.scoring_package_index, load_hashed(args.scoring_package_index)
        ),
        "scoring_pair_receipt": source_record(
            args.scoring_pair_receipt, load_hashed(args.scoring_pair_receipt)
        ),
        "real_panel_render_manifest": source_record(
            args.real_panel_render_manifest,
            load_hashed(args.real_panel_render_manifest),
        ),
        "real_panel_source_manifest": source_record(
            args.real_panel_source_manifest,
            load_hashed(args.real_panel_source_manifest),
        ),
        "real_panel_images": [
            source_record(path) for path in sorted(args.real_panel_image)
        ],
        "visual_assessment": source_record(args.visual_assessment, visual),
        "visual_observations": source_record(args.visual_observations),
        "real": source_record(args.real, real),
        "synthetic": source_record(args.synthetic, synthetic),
    }
    report = render_report(real, synthetic, visual, sources, plan)
    args.out_markdown.write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "status": "complete result-neutral evidence report rendered from independently validated sealed outputs",
        "renderer": source_record(renderer_path),
        "sources": sources,
        "pre_inference_protocol_clarification": plan[
            "pre_inference_protocol_clarification"
        ],
        "report_panel_assets": [source_record(path) for path in panel_assets],
        "report": source_record(args.out_markdown),
        "all_preregistered_gates_reported": True,
        "fixed_sensitivity_readouts_reported": True,
        "result_rows_omitted": False,
        "manual_claim_selection_permitted": False,
    }
    manifest["payload_sha256"] = canonical_sha256(manifest)
    args.out_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("report SHA-256:", manifest["report"]["sha256"])
    print("report-manifest payload SHA-256:", manifest["payload_sha256"])
    print("COMPLETE_RESULT_NEUTRAL_EVIDENCE_RENDERED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
