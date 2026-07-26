#!/usr/bin/env python3
"""Verify detections against exact public zero-valid TIFXYZ artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tifxyz_doctor import __version__  # noqa: E402
from tifxyz_doctor.integrity import audit_tifxyz_integrity  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "benchmarks" / "public-empty-regressions.json"
DEFAULT_DATA = PROJECT_ROOT / "benchmark-regression-data"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "public-empty-results-v0.1.0.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _observation(case: dict, path: Path) -> dict:
    report = audit_tifxyz_integrity(path)
    expected = case["expected"]
    actual_codes = [finding["code"] for finding in report["findings"]]
    checks = {
        "status": report["status"] == expected["status"],
        "shape_hw": report["coordinates"]["shape"] == expected["shape_hw"],
        "canonical_sentinel_vertices": (
            report["coordinates"]["canonical_sentinel_vertex_count"]
            == expected["canonical_sentinel_vertices"]
        ),
        "portable_valid_vertices": (
            report["validity"]["portable_valid_vertex_count"]
            == expected["portable_valid_vertices"]
        ),
        "portable_valid_faces": (
            report["geometry"]["portable_valid_face_count"]
            == expected["portable_valid_faces"]
        ),
        "required_finding_codes": set(expected["required_finding_codes"]).issubset(
            actual_codes
        ),
    }
    return {
        "id": case["id"],
        "expectation_pass": all(checks.values()),
        "checks": checks,
        "actual": {
            "status": report["status"],
            "shape_hw": report["coordinates"]["shape"],
            "canonical_sentinel_vertices": report["coordinates"][
                "canonical_sentinel_vertex_count"
            ],
            "portable_valid_vertices": report["validity"][
                "portable_valid_vertex_count"
            ],
            "portable_valid_faces": report["geometry"][
                "portable_valid_face_count"
            ],
            "finding_codes": actual_codes,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    observations = [
        _observation(case, args.data / case["id"]) for case in manifest["cases"]
    ]
    snapshot = {
        "schema_version": "1.0.0",
        "tool": {"name": "tifxyz-doctor", "version": __version__},
        "benchmark": manifest["name"],
        "benchmark_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "interpretation": (
            "A passing regression means the tool reproduced the expected "
            "semantic-empty-artifact detections; the source artifacts themselves "
            "are expected to have error status."
        ),
        "source_data_license": manifest["dataset_license"],
        "all_expectations_pass": all(
            observation["expectation_pass"] for observation in observations
        ),
        "observations": observations,
    }
    encoded = json.dumps(snapshot, allow_nan=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(
        f"wrote {args.output} "
        f"({sum(item['expectation_pass'] for item in observations)}/"
        f"{len(observations)} expectations passed)"
    )
    return 0 if snapshot["all_expectations_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
