#!/usr/bin/env python3
"""Jointly collect sealed RunPod-real and Kaggle-synthetic scorer artifacts.

The collector verifies both sources before creating its output directory. It
hashes but never parses scientific result JSON, scoring manifests, or panels.
Only operational status/receipt/plan JSON is parsed before collection.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import collect_heldout_delivery_inputs as heldout
import collect_scoring_results as kaggle_results
import orchestrate_heldout_queue as queue


SYNTHETIC_KERNEL = "aviadcohen1/vesuvius-fusion-one-shot-synthetic-scoring"
REAL_ARTIFACTS = (
    "fusion-one-shot-real/sealed_real_test_results.json",
    "fusion-one-shot-real/scoring_run_manifest.json",
    "fusion-one-shot-real/real-panels/real_panel_render_manifest.json",
    "fusion-one-shot-real/real-panels/real_panel_01.png",
    "fusion-one-shot-real/real-panels/real_panel_02.png",
    "fusion-one-shot-real/real-panels/real_panel_03.png",
    "fusion-one-shot-real/real-panels/real_panel_04.png",
)
REAL_OPERATIONAL = (
    "real-scoring-pipeline-status-v12/REAL_SCORING_COMPLETE",
    "real-scoring-pipeline-status-v12/pipeline-operational.log",
    "real-scoring-pipeline-status-v12/status.json",
    "real-scoring-status-v12/REAL_SCORING_COMPLETE",
    "real-scoring-status-v12/real-scoring-operational.log",
    "real-scoring-status-v12/status.json",
    "real-transport-status-v12/PRIVATE_TRANSPORT_VERIFIED",
    "real-transport-status-v12/transport-status.json",
)
REAL_EXPECTED = frozenset((*REAL_ARTIFACTS, *REAL_OPERATIONAL))


def identity(path: Path, *, relative_to: Path | None = None) -> dict:
    return {
        "file": (
            path.relative_to(relative_to).as_posix()
            if relative_to is not None
            else path.name
        ),
        "bytes": path.stat().st_size,
        "sha256": queue.sha256_file(path),
    }


def read_sha256s(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not name.startswith("./")
        ):
            raise RuntimeError("invalid real-output SHA256SUMS record")
        relative = name[2:]
        if relative in records:
            raise RuntimeError("duplicate real-output SHA256SUMS record")
        records[relative] = digest
    return records


def load_operational_status(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"operational status is not an object: {path}")
    return payload


def validate_real_source(
    root: Path, *, expected_plan_payload: str, expected_ledger: dict[str, str]
) -> dict:
    ledger_path = root / "SHA256SUMS"
    if not ledger_path.is_file():
        raise RuntimeError("real-output SHA256SUMS is absent")
    ledger = read_sha256s(ledger_path)
    if set(ledger) != REAL_EXPECTED or ledger != expected_ledger:
        raise RuntimeError("real-output checksum ledger differs from frozen plan")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != ledger_path
    }
    if observed != REAL_EXPECTED:
        raise RuntimeError("real-output file set differs from frozen plan")
    for relative, digest in ledger.items():
        if queue.sha256_file(root / relative) != digest:
            raise RuntimeError(f"real-output hash mismatch: {relative}")

    complete = load_operational_status(
        root / "real-scoring-pipeline-status-v12/status.json"
    )
    scoring = load_operational_status(root / "real-scoring-status-v12/status.json")
    transport = load_operational_status(
        root / "real-transport-status-v12/transport-status.json"
    )
    for name, payload in (("pipeline", complete), ("scoring", scoring)):
        if (
            payload.get("state") != "COMPLETE"
            or payload.get("returncode") != 0
            or payload.get("plan_payload_sha256") != expected_plan_payload
            or payload.get("scientific_outputs_inspected") is not False
        ):
            raise RuntimeError(f"real {name} completion status mismatch")
    if (
        transport.get("state") != "PRIVATE_TRANSPORT_VERIFIED"
        or transport.get("plan_payload_sha256") != expected_plan_payload
        or transport.get("scientific_outputs_inspected") is not False
        or transport.get("credentials_removed") is not True
    ):
        raise RuntimeError("real transport status mismatch")
    return {
        "checksum_ledger": identity(ledger_path, relative_to=root),
        "artifacts": [identity(root / relative, relative_to=root) for relative in REAL_ARTIFACTS],
        "operational_statuses": [
            identity(root / relative, relative_to=root) for relative in REAL_OPERATIONAL
        ],
    }


def synthetic_package(index: dict) -> dict:
    packages = index.get("packages")
    if not isinstance(packages, list):
        raise RuntimeError("synthetic package index has no package list")
    matches = [record for record in packages if record.get("mode") == "synthetic"]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one synthetic scorer package")
    return matches[0]


def validate_synthetic_source(
    packages_root: Path, index_path: Path, receipt_path: Path
) -> dict:
    index = queue.load_canonical(index_path)
    if index.get("status") != (
        "two one-shot scorer packages generated after public cache-delivery freeze"
    ):
        raise RuntimeError("wrong synthetic scoring-package index status")
    package = synthetic_package(index)
    if package.get("kaggle_kernel_id") != SYNTHETIC_KERNEL:
        raise RuntimeError("synthetic kernel identity mismatch")
    directory = packages_root / package["directory"]
    files = package.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("synthetic package file ledger is invalid")
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    if observed != set(files):
        raise RuntimeError("synthetic package file set mismatch")
    for name, record in files.items():
        path = directory / name
        if (
            path.stat().st_size != record.get("bytes")
            or queue.sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(f"synthetic package identity mismatch: {name}")

    receipt = queue.load_canonical(receipt_path)
    if receipt.get("status") != "paired one-shot scorer acceptance receipt":
        raise RuntimeError("wrong synthetic v4 acceptance-receipt status")
    expected_index = {
        "file": index_path.name,
        "bytes": index_path.stat().st_size,
        "sha256": queue.sha256_file(index_path),
        "payload_sha256": index["payload_sha256"],
    }
    if receipt.get("scoring_package_index") != expected_index:
        raise RuntimeError("synthetic v4 receipt points to another package index")
    accepted = receipt.get("accepted")
    if not isinstance(accepted, list):
        raise RuntimeError("synthetic v4 receipt accepted list is invalid")
    matches = [record for record in accepted if record.get("mode") == "synthetic"]
    if len(matches) != 1:
        raise RuntimeError("synthetic v4 receipt has no unique synthetic record")
    record = matches[0]
    if (
        record.get("kernel_id") != SYNTHETIC_KERNEL
        or record.get("kernel_version") != 4
        or record.get("package_ledger_sha256")
        != files["KERNEL_SHA256SUMS"]["sha256"]
    ):
        raise RuntimeError("synthetic v4 acceptance binding mismatch")
    launcher = directory / index["launcher"]["file"]
    return {
        "index": index,
        "receipt": receipt,
        "package": package,
        "launcher": launcher,
        "index_identity": {**identity(index_path), "payload_sha256": index["payload_sha256"]},
        "receipt_identity": {
            **identity(receipt_path),
            "payload_sha256": receipt["payload_sha256"],
        },
        "launcher_identity": identity(launcher),
    }


def verify_synthetic_server(kaggle: str, source: dict) -> dict:
    status = queue.kernel_status(kaggle, SYNTHETIC_KERNEL)
    if status != "COMPLETE":
        raise RuntimeError(f"synthetic v4 scorer is not COMPLETE ({status})")
    version, server_source = heldout.current_kernel_version_and_source(
        SYNTHETIC_KERNEL
    )
    if version != 4:
        raise RuntimeError("latest synthetic scorer version is not accepted v4")
    if server_source != source["launcher"].read_bytes():
        raise RuntimeError("synthetic v4 server source differs from accepted package")
    return {
        "kernel_id": SYNTHETIC_KERNEL,
        "kernel_version": 4,
        "source_sha256": queue.sha256_bytes(server_source),
    }


def validate_plan_inputs(args: argparse.Namespace, plan: dict) -> tuple[dict, dict]:
    if plan.get("status") != (
        "RunPod real and Kaggle synthetic v4 mixed collection frozen before result access"
    ):
        raise RuntimeError("wrong mixed-collection plan status")
    collector = Path(__file__).resolve()
    if plan.get("collector") != identity(collector):
        raise RuntimeError("running mixed collector differs from frozen plan")
    real_plan = queue.load_canonical(args.real_plan)
    real_receipt = queue.load_canonical(args.real_receipt)
    if plan.get("runpod_real", {}).get("execution_plan") != {
        **identity(args.real_plan),
        "payload_sha256": real_plan["payload_sha256"],
    }:
        raise RuntimeError("mixed plan points to another RunPod real plan")
    if plan.get("runpod_real", {}).get("provider_receipt") != {
        **identity(args.real_receipt),
        "payload_sha256": real_receipt["payload_sha256"],
    }:
        raise RuntimeError("mixed plan points to another RunPod receipt")
    if plan.get("runpod_real", {}).get("artifact_checksum_ledger") != identity(
        args.real_output / "SHA256SUMS"
    ):
        raise RuntimeError("mixed plan points to another real checksum ledger")
    if (
        real_receipt.get("status")
        != "RunPod CPU real-scoring allocation terminated after sealed copy"
        or real_receipt.get("terminated_at") is None
        or real_receipt.get("scientific_outputs_inspected") is not False
        or real_receipt.get("plan_payload_sha256") != real_plan["payload_sha256"]
    ):
        raise RuntimeError("RunPod real receipt is not a sealed completed termination")
    real = validate_real_source(
        args.real_output,
        expected_plan_payload=real_plan["payload_sha256"],
        expected_ledger=plan["runpod_real"]["artifact_sha256s"],
    )
    synthetic = validate_synthetic_source(
        args.synthetic_packages, args.synthetic_index, args.synthetic_receipt
    )
    if plan.get("kaggle_synthetic_v4", {}).get("package_index") != synthetic[
        "index_identity"
    ] or plan.get("kaggle_synthetic_v4", {}).get("acceptance_receipt") != synthetic[
        "receipt_identity"
    ]:
        raise RuntimeError("mixed plan points to another synthetic v4 source")
    if plan["kaggle_synthetic_v4"].get("launcher") != synthetic[
        "launcher_identity"
    ]:
        raise RuntimeError("mixed plan points to another synthetic v4 launcher")
    return real, synthetic


def copy_real_artifacts(source_root: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for relative in REAL_ARTIFACTS:
        source = source_root / relative
        target = destination / Path(relative).relative_to("fusion-one-shot-real")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if queue.sha256_file(target) != queue.sha256_file(source):
            raise RuntimeError(f"copied real artifact hash mismatch: {relative}")
        copied.append(target)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--real-plan", type=Path, required=True)
    parser.add_argument("--real-receipt", type=Path, required=True)
    parser.add_argument("--real-output", type=Path, required=True)
    parser.add_argument("--synthetic-packages", type=Path, required=True)
    parser.add_argument("--synthetic-index", type=Path, required=True)
    parser.add_argument("--synthetic-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kaggle", default="kaggle")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("mixed collection output must start absent")
    plan = queue.load_canonical(args.plan)
    real, synthetic = validate_plan_inputs(args, plan)
    if args.dry_run:
        print("RUNPOD_KAGGLE_MIXED_COLLECTION_PREFLIGHT_OK")
        return 0
    server = verify_synthetic_server(args.kaggle, synthetic)

    args.out.mkdir(parents=True)
    downloads = args.out / "downloads"
    real_destination = downloads / "real"
    synthetic_destination = downloads / "synthetic"
    copied_real = copy_real_artifacts(args.real_output, real_destination)
    kaggle_results.download_selected(
        args.kaggle, SYNTHETIC_KERNEL, synthetic_destination
    )
    synthetic_result, synthetic_run, panel_manifest, panels = (
        kaggle_results.validate_download(synthetic_destination, "synthetic")
    )
    if panel_manifest is not None or panels:
        raise RuntimeError("synthetic download unexpectedly contained panels")

    payload = {
        "schema_version": "1.0",
        "status": "sealed RunPod real and Kaggle synthetic v4 artifacts jointly collected",
        "collection_plan": {
            **identity(args.plan),
            "payload_sha256": plan["payload_sha256"],
        },
        "sources": {
            "runpod_real": {
                "provider_receipt": plan["runpod_real"]["provider_receipt"],
                "verified_source": real,
                "artifacts": [identity(path, relative_to=args.out) for path in copied_real],
            },
            "kaggle_synthetic_v4": {
                **server,
                "package_index": synthetic["index_identity"],
                "acceptance_receipt": synthetic["receipt_identity"],
                "result": identity(synthetic_result, relative_to=args.out),
                "run_manifest": identity(synthetic_run, relative_to=args.out),
            },
        },
        "collector": identity(Path(__file__).resolve()),
        "scientific_gate": {
            "runpod_real_complete_copied_hashed_and_billing_terminated": True,
            "synthetic_v4_complete_and_server_source_verified_before_collection": True,
            "both_sources_verified_before_output_directory_creation": True,
            "synthetic_scorer_repeated": False,
            "kernel_logs_opened_or_read": False,
            "results_or_manifests_opened_or_parsed_by_collector": False,
            "panel_images_opened_or_read_by_collector": False,
            "independent_validation_required_next": True,
        },
    }
    content = dict(payload)
    payload["payload_sha256"] = queue.canonical_sha256(content)
    output = args.out / "mixed_scoring_result_sources.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("mixed scoring source payload SHA-256:", payload["payload_sha256"])
    print("BOTH_SEALED_RESULTS_READY_FOR_INDEPENDENT_VALIDATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
