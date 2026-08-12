#!/usr/bin/env python3
"""Collect only the two approved threshold-freeze JSON artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import collect_heldout_delivery_inputs as heldout_collector
import orchestrate_heldout_queue as queue

THRESHOLD_KERNEL_ID = "aviadcohen1/vesuvius-fusion-aware-freeze-thresholds"
THRESHOLD_KERNEL_VERSION = 6
THRESHOLD_LAUNCHER_SHA256 = (
    "115c8f27ca5f0a8dee9fe9e56883f9ecbfa4e76dd22742a0f8ba39d5b95a3af9"
)
OUTPUT_PATTERN = (
    r"(^|/)(frozen_thresholds[.]json|threshold_run_manifest[.]json)$"
)
EXPECTED_NAMES = {"frozen_thresholds.json", "threshold_run_manifest.json"}


def download_selected(kaggle: str, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"threshold download destination must start absent: {destination}")
    result = subprocess.run(
        [
            kaggle,
            "kernels",
            "output",
            THRESHOLD_KERNEL_ID,
            "-p",
            str(destination),
            "--file-pattern",
            OUTPUT_PATTERN,
            "--page-size",
            "100",
            "--page-token",
            "",
            "--quiet",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        combined = "\n".join(
            part for part in (result.stdout, result.stderr) if part
        ).strip()
        raise RuntimeError(f"threshold artifact download failed: {combined}")


def validate_download(destination: Path) -> list[Path]:
    files = sorted(path.resolve() for path in destination.rglob("*") if path.is_file())
    if len(files) != 2 or {path.name for path in files} != EXPECTED_NAMES:
        raise RuntimeError("threshold download is not the exact two-artifact set")
    if any(path.suffix == ".log" for path in files):
        raise RuntimeError("threshold downloader unexpectedly emitted a log")
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kaggle", default="kaggle")
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"threshold collector output must start absent: {args.out}")
    expected_source = args.expected_source.read_bytes()
    if queue.sha256_bytes(expected_source) != THRESHOLD_LAUNCHER_SHA256:
        raise RuntimeError("local threshold launcher SHA-256 mismatch")
    status = queue.kernel_status(args.kaggle, THRESHOLD_KERNEL_ID)
    if status != "COMPLETE":
        raise RuntimeError(f"threshold kernel must be complete, observed {status}")
    version, server_source = heldout_collector.current_kernel_version_and_source(
        THRESHOLD_KERNEL_ID
    )
    if version != THRESHOLD_KERNEL_VERSION or server_source != expected_source:
        raise RuntimeError("threshold kernel version/source differs from frozen launcher")

    args.out.mkdir(parents=True)
    artifact_root = args.out / "artifacts"
    download_selected(args.kaggle, artifact_root)
    files = validate_download(artifact_root)
    collector_path = Path(__file__).resolve()
    payload = {
        "schema_version": "1.0",
        "status": "exact threshold-freeze artifact pair collected after completion",
        "kernel_id": THRESHOLD_KERNEL_ID,
        "kernel_version": version,
        "server_source_sha256": queue.sha256_bytes(server_source),
        "artifacts": [
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": queue.sha256_file(path),
                "relative_path": path.relative_to(args.out.resolve()).as_posix(),
            }
            for path in files
        ],
        "collector": {
            "file": collector_path.name,
            "bytes": collector_path.stat().st_size,
            "sha256": queue.sha256_file(collector_path),
        },
        "scientific_gate": {
            "kernel_status_complete_before_download": True,
            "latest_version_matches_frozen_version": True,
            "server_source_matches_frozen_launcher": True,
            "kernel_log_downloaded_or_opened": False,
            "validation_probability_caches_downloaded_or_opened": False,
            "held_out_test_or_synthetic_outputs_downloaded_or_opened": False,
            "selected_threshold_values_reported_by_collector": False,
        },
    }
    payload["payload_sha256"] = queue.canonical_sha256(payload)
    record = args.out / "threshold_artifact_sources.json"
    record.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("threshold artifact-source payload SHA-256:", payload["payload_sha256"])
    print("EXACT_THRESHOLD_FREEZE_ARTIFACT_PAIR_COLLECTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
