#!/usr/bin/env python3
"""Record a complete human assessment of all four frozen real panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SEEDS = (11, 23, 47)
PANEL_INDICES = (1, 2, 3, 4)
CRITERION = (
    "at least one preregistered real compressed-region panel shows a visually "
    "verifiable gap8-versus-matched-control separation improvement without a "
    "new nearby break"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    content = dict(payload)
    observed = content.pop("payload_sha256", None)
    if observed != canonical_sha256(content):
        raise RuntimeError(f"embedded payload SHA-256 mismatch: {path}")
    return payload


def file_identity(path: Path, *, payload: dict | None = None) -> dict:
    record = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if payload is not None:
        record["payload_sha256"] = payload["payload_sha256"]
    return record


def validate_observations(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "status",
        "assessor",
        "assessment_method",
        "criterion",
        "comparisons",
    }:
        raise RuntimeError("visual-observation schema mismatch")
    if (
        payload["schema_version"] != "1.0"
        or payload["status"] != "all fixed real-panel comparisons assessed"
        or not isinstance(payload["assessor"], str)
        or not payload["assessor"].strip()
        or not isinstance(payload["assessment_method"], str)
        or not payload["assessment_method"].strip()
        or payload["criterion"] != CRITERION
    ):
        raise RuntimeError("visual-observation identity mismatch")
    comparisons = payload["comparisons"]
    expected_order = [
        (panel_index, seed) for panel_index in PANEL_INDICES for seed in SEEDS
    ]
    if (
        not isinstance(comparisons, list)
        or [(record.get("panel_index"), record.get("seed")) for record in comparisons]
        != expected_order
    ):
        raise RuntimeError("visual observations do not cover the fixed 4x3 order")
    for record in comparisons:
        if set(record) != {
            "panel_index",
            "seed",
            "separation_improvement",
            "new_nearby_break",
            "notes",
        }:
            raise RuntimeError("visual-comparison schema mismatch")
        if not isinstance(record["separation_improvement"], bool) or not isinstance(
            record["new_nearby_break"], bool
        ):
            raise TypeError("visual-comparison decisions must be Boolean")
        if not isinstance(record["notes"], str) or not record["notes"].strip():
            raise RuntimeError("every visual comparison requires a note")
    return payload, comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-render-manifest", type=Path, required=True)
    parser.add_argument("--panel-image", type=Path, action="append", required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("visual-assessment output must start absent")
    render = load_hashed(args.panel_render_manifest)
    if render.get("status") != (
        "four model-blind real panels rendered after sealed test delivery"
    ):
        raise RuntimeError("wrong real-panel render status")
    images = sorted(path.resolve() for path in args.panel_image)
    records = render.get("panels")
    if not isinstance(records, list) or len(records) != 4 or len(images) != 4:
        raise RuntimeError("expected exactly four fixed panel images")
    for index, (record, image) in enumerate(zip(records, images, strict=True), start=1):
        output = record.get("output", {})
        if (
            image.name != f"real_panel_{index:02d}.png"
            or output.get("file") != image.name
            or output.get("bytes") != image.stat().st_size
            or output.get("sha256") != sha256_file(image)
        ):
            raise RuntimeError(f"real panel {index}: image identity mismatch")
    observations, comparisons = validate_observations(args.observations)
    assessed = [
        record
        | {
            "comparison_pass": record["separation_improvement"]
            and not record["new_nearby_break"]
        }
        for record in comparisons
    ]
    payload = {
        "schema_version": "1.0",
        "status": "all four preregistered real panels assessed without omission",
        "criterion": CRITERION,
        "assessor": observations["assessor"],
        "assessment_method": observations["assessment_method"],
        "source_observations": file_identity(args.observations),
        "source_panel_render_manifest": file_identity(
            args.panel_render_manifest, payload=render
        ),
        "source_panel_images": [file_identity(path) for path in images],
        "comparisons": assessed,
        "visual_gate_pass": any(record["comparison_pass"] for record in assessed),
        "scientific_gate": {
            "all_four_panels_assessed": True,
            "all_three_matched_seeds_assessed_per_panel": True,
            "panels_or_comparisons_omitted": False,
            "renderer_made_visual_decision": False,
            "assessment_recorded_after_fixed_render": True,
        },
        "recorder": file_identity(Path(__file__).resolve()),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("visual gate pass:", payload["visual_gate_pass"])
    print("visual-assessment payload SHA-256:", payload["payload_sha256"])
    print("ALL_FIXED_REAL_PANEL_COMPARISONS_RECORDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
