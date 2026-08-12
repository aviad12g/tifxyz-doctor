"""Download both sealed scorer artifacts only after both scorers complete."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import collect_heldout_delivery_inputs as heldout_collector
import orchestrate_heldout_queue as queue
import orchestrate_scoring_pair as pair

OUTPUT_PATTERN = (
    r"(^|/)(sealed_real_test_results[.]json|"
    r"sealed_synthetic_test_results[.]json|scoring_run_manifest[.]json|"
    r"real_panel_render_manifest[.]json|real_panel_0[1-4][.]png)$"
)
RESULT_NAMES = {
    "real": "sealed_real_test_results.json",
    "synthetic": "sealed_synthetic_test_results.json",
}


def download_selected(kaggle: str, kernel_id: str, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(
            f"scoring download destination must start absent: {destination}"
        )
    result = subprocess.run(
        [
            kaggle,
            "kernels",
            "output",
            kernel_id,
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
        raise RuntimeError(
            f"scoring artifact download failed for {kernel_id}: {combined}"
        )


def validate_download(
    destination: Path, mode: str
) -> tuple[Path, Path, Path | None, list[Path]]:
    result_paths = [
        path.resolve()
        for path in destination.rglob(RESULT_NAMES[mode])
        if path.is_file()
    ]
    run_paths = [
        path.resolve()
        for path in destination.rglob("scoring_run_manifest.json")
        if path.is_file()
    ]
    if len(result_paths) != 1 or len(run_paths) != 1:
        raise RuntimeError(f"{mode}: expected one sealed result and run manifest")
    panel_manifest_paths = [
        path.resolve()
        for path in destination.rglob("real_panel_render_manifest.json")
        if path.is_file()
    ]
    panel_image_paths = sorted(
        path.resolve()
        for path in destination.rglob("real_panel_*.png")
        if path.is_file()
    )
    expected_panel_count = 4 if mode == "real" else 0
    if (
        len(panel_manifest_paths) != (1 if mode == "real" else 0)
        or len(panel_image_paths) != expected_panel_count
    ):
        raise RuntimeError(f"{mode}: fixed real-panel artifact set mismatch")
    logs = [path.resolve() for path in destination.rglob("*.log") if path.is_file()]
    if logs:
        raise RuntimeError(f"{mode}: downloader unexpectedly emitted a log")
    allowed = {
        result_paths[0],
        run_paths[0],
        *panel_manifest_paths,
        *panel_image_paths,
    }
    observed = {path.resolve() for path in destination.rglob("*") if path.is_file()}
    if observed != allowed:
        raise RuntimeError(f"{mode}: unexpected downloaded scoring file set")
    panel_manifest = panel_manifest_paths[0] if panel_manifest_paths else None
    return result_paths[0], run_paths[0], panel_manifest, panel_image_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--package-index", type=Path, required=True)
    parser.add_argument("--pair-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kaggle", default="kaggle")
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"scoring collector output must start absent: {args.out}")
    index, packages = pair.validate_packages(args.packages, args.package_index)
    local_collector = Path(__file__).resolve()
    if index["result_collector"] != {
        "file": local_collector.name,
        "bytes": local_collector.stat().st_size,
        "sha256": queue.sha256_file(local_collector),
    }:
        raise RuntimeError("running result collector differs from public plan")
    receipt = pair.load_receipt(args.pair_receipt, args.package_index, index)
    accepted = pair.receipt_map(receipt, packages)
    if set(accepted) != set(pair.MODES):
        raise RuntimeError("both scorer kernels must be accepted before collection")

    statuses = {
        record["mode"]: queue.kernel_status(
            args.kaggle, accepted[record["mode"]]["kernel_id"]
        )
        for record in packages
    }
    if any(status != "COMPLETE" for status in statuses.values()):
        raise RuntimeError(
            f"both scorers must complete before either download: {statuses}"
        )

    packages_by_mode = {record["mode"]: record for record in packages}
    server_sources = {}
    for mode in pair.MODES:
        record = accepted[mode]
        version, source = heldout_collector.current_kernel_version_and_source(
            record["kernel_id"]
        )
        if version != record["kernel_version"]:
            raise RuntimeError(f"{mode}: latest Kaggle scorer version changed")
        package = packages_by_mode[mode]
        package_source = (
            args.packages / package["directory"] / index["launcher"]["file"]
        ).read_bytes()
        if source != package_source:
            raise RuntimeError(f"{mode}: Kaggle scorer source differs from package")
        server_sources[mode] = {
            "kernel_id": record["kernel_id"],
            "kernel_version": version,
            "source_sha256": queue.sha256_bytes(source),
        }

    args.out.mkdir(parents=True)
    downloads = args.out / "downloads"
    downloads.mkdir()
    artifacts = {}
    for mode in pair.MODES:
        destination = downloads / mode
        download_selected(args.kaggle, accepted[mode]["kernel_id"], destination)
        result_path, run_path, panel_manifest, panel_images = validate_download(
            destination, mode
        )
        artifacts[mode] = {
            **server_sources[mode],
            "result": str(result_path),
            "run_manifest": str(run_path),
            "panel_manifest": (
                str(panel_manifest) if panel_manifest is not None else None
            ),
            "panel_images": [str(path) for path in panel_images],
        }
        print(f"SEALED_SCORING_ARTIFACTS_COLLECTED mode={mode}")

    payload = {
        "schema_version": "1.0",
        "status": "both sealed scoring artifacts collected after both scorers completed",
        "scoring_package_index": {
            "file": args.package_index.name,
            "bytes": args.package_index.stat().st_size,
            "sha256": queue.sha256_file(args.package_index),
            "payload_sha256": index["payload_sha256"],
        },
        "pair_receipt": {
            "file": args.pair_receipt.name,
            "bytes": args.pair_receipt.stat().st_size,
            "sha256": queue.sha256_file(args.pair_receipt),
            "payload_sha256": receipt["payload_sha256"],
        },
        "artifacts": artifacts,
        "collector": {
            "file": local_collector.name,
            "bytes": local_collector.stat().st_size,
            "sha256": queue.sha256_file(local_collector),
        },
        "scientific_gate": {
            "both_scorers_accepted_before_result_access": True,
            "both_scorers_completed_before_first_result_download": True,
            "latest_kernel_versions_match_pair_receipt": True,
            "server_sources_match_accepted_packages": True,
            "kernel_logs_opened_or_read": False,
            "results_opened_or_read_by_collector": False,
            "panel_images_opened_or_read_by_collector": False,
            "independent_validation_required_next": True,
        },
    }
    payload["payload_sha256"] = queue.canonical_sha256(payload)
    output = args.out / "scoring_result_sources.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("scoring-result-source payload SHA-256:", payload["payload_sha256"])
    print("BOTH_SEALED_RESULTS_READY_FOR_INDEPENDENT_VALIDATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
