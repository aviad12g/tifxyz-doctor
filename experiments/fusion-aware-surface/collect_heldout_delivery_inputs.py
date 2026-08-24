"""Collect only sealed held-out indexes/manifests for the delivery freeze.

The collector proves that every latest Kaggle kernel version is the version in
the result-blind queue receipt and that its source equals the accepted package.
It downloads no NPZ cache and never reads the automatically emitted kernel log.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import orchestrate_heldout_queue as queue

OUTPUT_PATTERN = (
    r"(^|/)(heldout_job_index[.]json|cache_manifest_[^/]+[.]json|"
    r"synthetic_manifest_[^/]+[.]json)$"
)


def expected_jobs(plan: dict) -> list[dict]:
    real = plan.get("real_test_jobs")
    synthetic = plan.get("synthetic_ray_jobs")
    if not isinstance(real, list) or len(real) != 7:
        raise RuntimeError("public plan does not contain seven real jobs")
    if not isinstance(synthetic, list) or len(synthetic) != 7:
        raise RuntimeError("public plan does not contain seven synthetic jobs")
    jobs = real + synthetic
    if len({record.get("job_id") for record in jobs}) != 14:
        raise RuntimeError("public plan held-out job identities are not unique")
    return jobs


def current_kernel_version_and_source(kernel_id: str) -> tuple[int, bytes]:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    owner, slug = kernel_id.split("/", 1)
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name = owner
        request.kernel_slug = slug
        response = client.kernels.kernels_api_client.get_kernel(request)
    if response.metadata is None or response.blob is None:
        raise RuntimeError(f"Kaggle returned incomplete kernel metadata: {kernel_id}")
    version = response.metadata.current_version_number
    if not isinstance(version, int) or version <= 0:
        raise RuntimeError(f"Kaggle returned an invalid kernel version: {kernel_id}")
    return version, response.blob.source.encode("utf-8")


def download_selected(kaggle: str, kernel_id: str, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"download destination must start absent: {destination}")
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
            f"selected-output download failed for {kernel_id}: {combined}"
        )


def validate_download(destination: Path, job: dict) -> Path:
    if list(destination.rglob("*.npz")):
        raise RuntimeError(f"{job['job_id']}: NPZ cache was unexpectedly downloaded")
    indexes = [
        path.resolve()
        for path in destination.rglob("heldout_job_index.json")
        if path.is_file()
    ]
    if len(indexes) != 1:
        raise RuntimeError(f"{job['job_id']}: expected one held-out job index")
    expected_manifests = (
        {job["expected_cache_manifest"]}
        if job["mode"] == "real_test_cache"
        else set(job["expected_cache_manifests"])
    )
    manifests = [
        path
        for path in destination.rglob("*.json")
        if path.name != "heldout_job_index.json"
    ]
    if {path.name for path in manifests} != expected_manifests or len(manifests) != len(
        expected_manifests
    ):
        raise RuntimeError(f"{job['job_id']}: downloaded manifest set mismatch")
    logs = [path for path in destination.rglob("*.log") if path.is_file()]
    if logs:
        raise RuntimeError(f"{job['job_id']}: downloader unexpectedly emitted a log")
    allowed = {
        indexes[0],
        *(path.resolve() for path in manifests),
    }
    observed = {path.resolve() for path in destination.rglob("*") if path.is_file()}
    if observed != allowed:
        raise RuntimeError(f"{job['job_id']}: unexpected downloaded file set")
    return indexes[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--public-plan-commit", required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--package-index", type=Path, required=True)
    parser.add_argument("--queue-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kaggle", default="kaggle")
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"collector output must start absent: {args.out}")
    if len(args.public_plan_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.public_plan_commit
    ):
        raise ValueError("public plan commit must be 40 lowercase hexadecimal")

    plan = queue.load_canonical(args.plan)
    if plan.get("status") != "held-out execution plan frozen before held-out inference":
        raise RuntimeError("wrong public held-out plan status")
    package_index, packages = queue.validate_packages(args.packages, args.package_index)
    if package_index.get("public_plan_commit") != args.public_plan_commit:
        raise RuntimeError("package index points to another public plan commit")
    if package_index.get("public_plan_file_sha256") != queue.sha256_file(args.plan):
        raise RuntimeError("package index points to another public plan file")
    if package_index.get("public_plan_payload_sha256") != plan["payload_sha256"]:
        raise RuntimeError("package index points to another public plan payload")
    receipt = queue.load_receipt(args.queue_receipt, args.package_index, package_index)
    accepted = queue.receipt_map(receipt, packages)
    jobs = expected_jobs(plan)
    if list(accepted) != [job["job_id"] for job in jobs]:
        raise RuntimeError("queue receipt does not contain all 14 jobs in plan order")

    args.out.mkdir(parents=True)
    downloads = args.out / "downloads"
    downloads.mkdir()
    package_by_job = {record["job_id"]: record for record in packages}
    sources = []
    for job in jobs:
        record = accepted[job["job_id"]]
        package = package_by_job[job["job_id"]]
        status = queue.kernel_status(args.kaggle, record["kernel_id"])
        if status != "COMPLETE":
            raise RuntimeError(
                f"{job['job_id']}: kernel is not complete (status={status})"
            )
        version, source = current_kernel_version_and_source(record["kernel_id"])
        if version != record["kernel_version"]:
            raise RuntimeError(f"{job['job_id']}: latest Kaggle version changed")
        package_source = (
            args.packages
            / package["directory"]
            / plan["heldout_cache_launcher"]["file"]
        )
        if source != package_source.read_bytes():
            raise RuntimeError(f"{job['job_id']}: Kaggle source differs from package")
        destination = downloads / job["job_id"]
        download_selected(args.kaggle, record["kernel_id"], destination)
        index_path = validate_download(destination, job)
        sources.append(
            {
                "job_id": job["job_id"],
                "kernel_id": record["kernel_id"],
                "kernel_version": version,
                "job_index": str(index_path),
            }
        )
        print(f"HELDOUT_DELIVERY_INPUT_COLLECTED job={job['job_id']}")

    payload = {
        "schema_version": "1.0",
        "status": "all held-out delivery indexes and manifests collected without NPZ",
        "public_plan": {
            "commit": args.public_plan_commit,
            "file": args.plan.name,
            "bytes": args.plan.stat().st_size,
            "sha256": queue.sha256_file(args.plan),
            "payload_sha256": plan["payload_sha256"],
        },
        "package_index": {
            "file": args.package_index.name,
            "bytes": args.package_index.stat().st_size,
            "sha256": queue.sha256_file(args.package_index),
            "payload_sha256": package_index["payload_sha256"],
        },
        "queue_receipt": {
            "file": args.queue_receipt.name,
            "bytes": args.queue_receipt.stat().st_size,
            "sha256": queue.sha256_file(args.queue_receipt),
            "payload_sha256": receipt["payload_sha256"],
        },
        "jobs": sources,
        "collector": {
            "file": Path(__file__).resolve().name,
            "bytes": Path(__file__).resolve().stat().st_size,
            "sha256": queue.sha256_file(Path(__file__).resolve()),
        },
        "scientific_gate": {
            "all_latest_kernel_versions_match_acceptance_receipt": True,
            "all_kernel_sources_match_accepted_packages": True,
            "npz_probability_caches_downloaded": False,
            "kernel_logs_opened_or_read": False,
            "scientific_endpoints_opened_or_read": False,
        },
    }
    payload["payload_sha256"] = queue.canonical_sha256(payload)
    output = args.out / "delivery_source_records.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("delivery-source payload SHA-256:", payload["payload_sha256"])
    print("ALL_HELDOUT_DELIVERY_INPUTS_COLLECTED_WITHOUT_NPZ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
