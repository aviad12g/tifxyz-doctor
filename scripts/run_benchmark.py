#!/usr/bin/env python3
"""Run the real-data smoke benchmark and emit a compact deterministic snapshot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tifxyz_doctor import __version__  # noqa: E402
from tifxyz_doctor.audit import AuditConfig, audit_mesh  # noqa: E402
from tifxyz_doctor.integrity import audit_tifxyz_integrity  # noqa: E402
from tifxyz_doctor.io import load_tifxyz  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "benchmarks" / "realdata-smoke.json"
DEFAULT_DATA = PROJECT_ROOT / "benchmark-data"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "realdata-results-v0.1.0.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _summary(case: dict, path: Path, config: AuditConfig) -> dict:
    contract = audit_tifxyz_integrity(path)
    report = audit_mesh(load_tifxyz(path), config)
    geometry = report["geometry"]
    return {
        "id": case["id"],
        "role": case["role"],
        "contract_status": contract["status"],
        "contract_errors": contract["summary"]["errors"],
        "contract_warnings": contract["summary"]["warnings"],
        "shape_hw": contract["coordinates"]["shape"],
        "portable_valid_faces": contract["geometry"]["portable_valid_face_count"],
        "review_cues": [finding["code"] for finding in report["findings"]],
        "facts": {
            "valid_face_components": report["topology"]["valid_quad_components"],
            "enclosed_face_holes": report["topology"]["enclosed_invalid_regions"],
            "enclosed_face_hole_cells": sum(
                report["topology"]["enclosed_invalid_region_sizes"]
            ),
            "long_edges": geometry["long_edges"],
            "short_edges": geometry["short_edges"],
            "degenerate_triangles": geometry["degenerate_triangles"],
            "official_split_folded_quads": geometry["folded_quads"],
            "normal_jumps": geometry["normal_jumps"]["jumps_above_threshold"],
            "symmetric_stretch_cells": geometry["high_symmetric_stretch_cells"],
            "area_distortion_cells": geometry["high_area_distortion_cells"],
            "high_shear_cells": geometry["high_shear_cells"],
            "sampled_nonlocal_vertex_pairs": report["nonlocal_proximity"]["pair_count"],
        },
        "distributions": {
            "condition_number_p95": geometry["condition_number"]["p95"],
            "condition_number_max": geometry["condition_number"]["max"],
            "symmetric_stretch_p95": geometry["symmetric_stretch"]["p95"],
            "symmetric_stretch_max": geometry["symmetric_stretch"]["max"],
            "normal_jump_degrees_p95": geometry["normal_jumps"]["angle_degrees"]["p95"],
            "normal_jump_degrees_max": geometry["normal_jumps"]["angle_degrees"]["max"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    config = AuditConfig()
    observations = [
        _summary(case, args.data / case["id"], config)
        for case in manifest["cases"]
    ]
    snapshot = {
        "schema_version": "1.0.0",
        "tool": {"name": "tifxyz-doctor", "version": __version__},
        "benchmark": manifest["name"],
        "benchmark_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "configuration": asdict(config),
        "interpretation": (
            "Contract statuses are objective checks. Review cues are thresholded "
            "observations, not ground-truth quality labels."
        ),
        "source_data_license": manifest["dataset_license"],
        "observations": observations,
    }
    encoded = json.dumps(snapshot, allow_nan=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(f"wrote {args.output} ({len(observations)} cases)")
    return 2 if any(item["contract_errors"] for item in observations) else 0


if __name__ == "__main__":
    raise SystemExit(main())
