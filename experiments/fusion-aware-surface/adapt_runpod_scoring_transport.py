#!/usr/bin/env python3
"""Replace synthetic cache kernel mounts with one sealed RunPod dataset mount."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import orchestrate_scoring_pair as pair


DATASET_PATTERN = re.compile(r"^[a-z0-9_-]+/[a-z0-9_-]+/[1-9][0-9]*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    content = dict(payload)
    content.pop("payload_sha256", None)
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("payload_sha256") != canonical_sha256(payload):
        raise RuntimeError(f"invalid hashed JSON: {path}")
    return payload


def local_identity(path: Path) -> dict:
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def adapt_metadata(
    metadata: dict,
    *,
    threshold_kernel: str,
    expected_synthetic_kernel_sources: list[str],
    dataset_source: str,
) -> dict:
    if metadata.get("kernel_sources") != [
        threshold_kernel,
        *expected_synthetic_kernel_sources,
    ]:
        raise RuntimeError("synthetic scoring metadata source order mismatch")
    datasets = metadata.get("dataset_sources")
    if not isinstance(datasets, list) or dataset_source in datasets:
        raise RuntimeError("synthetic scoring dataset source state is invalid")
    adapted = dict(metadata)
    adapted["kernel_sources"] = [threshold_kernel]
    adapted["dataset_sources"] = [*datasets, dataset_source]
    return adapted


def rewrite_package_ledger(directory: Path) -> dict[str, dict]:
    ledger = directory / "KERNEL_SHA256SUMS"
    names = sorted(path.name for path in directory.iterdir() if path.is_file() and path != ledger)
    identities = {
        name: {"bytes": (directory / name).stat().st_size, "sha256": sha256_file(directory / name)}
        for name in names
    }
    ledger.write_text(
        "".join(f"{identities[name]['sha256']}  {name}\n" for name in names),
        encoding="utf-8",
    )
    identities[ledger.name] = {
        "bytes": ledger.stat().st_size,
        "sha256": sha256_file(ledger),
    }
    return identities


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--synthetic-dataset-source", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("adapted package output must start absent")
    if not DATASET_PATTERN.fullmatch(args.synthetic_dataset_source):
        raise RuntimeError("invalid versioned Kaggle dataset source")
    index, packages = pair.validate_packages(args.packages, args.index)
    delivery = load_hashed(args.delivery)
    authority = delivery.get("authority")
    if authority != {
        "real_provider": "Kaggle",
        "synthetic_primary_provider": "RunPod",
        "kaggle_synthetic_role": "sealed secondary cross-platform replication",
        "selection_between_synthetic_platforms_permitted": False,
        "synthetic_kernel_locator_fields_are_non_authoritative_transport_placeholders": True,
    }:
        raise RuntimeError("delivery does not carry the frozen RunPod authority rule")
    synthetic_delivery = [
        record for record in delivery.get("jobs", []) if record.get("mode") == "synthetic_ray_cache"
    ]
    if len(synthetic_delivery) != 7 or any(
        record.get("execution_origin", {}).get("provider") != "RunPod"
        for record in synthetic_delivery
    ):
        raise RuntimeError("delivery does not contain seven RunPod synthetic jobs")
    expected_sources = [
        f"{record['kernel_id']}/{record['kernel_version']}" for record in synthetic_delivery
    ]
    shutil.copytree(args.packages, args.out, symlinks=False)
    copied_index_path = args.out / args.index.name
    copied_index_path.unlink(missing_ok=True)
    packages_by_mode = {record["mode"]: record for record in packages}
    synthetic_record = packages_by_mode["synthetic"]
    synthetic_dir = args.out / synthetic_record["directory"]
    metadata_path = synthetic_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    adapted_metadata = adapt_metadata(
        metadata,
        threshold_kernel=synthetic_record["threshold_kernel"],
        expected_synthetic_kernel_sources=expected_sources,
        dataset_source=args.synthetic_dataset_source,
    )
    metadata_path.write_text(json.dumps(adapted_metadata, indent=2) + "\n", encoding="utf-8")
    synthetic_record["files"] = rewrite_package_ledger(synthetic_dir)
    adapter = Path(__file__).resolve()
    adaptation = {
        "schema_version": "1.0",
        "status": "RunPod primary caches mounted by one private hash-bound Kaggle dataset",
        "input_scoring_package_index": {
            "file": args.index.name,
            "bytes": args.index.stat().st_size,
            "sha256": sha256_file(args.index),
            "payload_sha256": index["payload_sha256"],
        },
        "public_cache_delivery": {
            "file": args.delivery.name,
            "bytes": args.delivery.stat().st_size,
            "sha256": sha256_file(args.delivery),
            "payload_sha256": delivery["payload_sha256"],
        },
        "synthetic_dataset_source": args.synthetic_dataset_source,
        "removed_non_authoritative_kernel_sources": expected_sources,
        "retained_threshold_kernel": synthetic_record["threshold_kernel"],
        "scorer_source_changed": False,
        "scientific_inputs_or_settings_changed": False,
        "runpod_authority_changed": False,
        "adapter": local_identity(adapter),
    }
    adaptation["payload_sha256"] = canonical_sha256(adaptation)
    adaptation_path = args.out / "synthetic_transport_adaptation.json"
    adaptation_path.write_text(
        json.dumps(adaptation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    updated = dict(index)
    updated.pop("payload_sha256", None)
    updated["packages"] = packages
    updated["transport_adapter"] = local_identity(adapter)
    updated["synthetic_transport_adaptation"] = {
        **local_identity(adaptation_path),
        "payload_sha256": adaptation["payload_sha256"],
    }
    gate = dict(updated["scientific_gate"])
    gate.update(
        {
            "runpod_primary_mounted_via_private_hash_bound_dataset": True,
            "secondary_synthetic_kernel_sources_removed_before_launch": True,
            "scorer_source_changed_by_transport_adapter": False,
        }
    )
    updated["scientific_gate"] = gate
    updated["payload_sha256"] = canonical_sha256(updated)
    copied_index_path.write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pair.validate_packages(args.out, copied_index_path)
    print("RUNPOD_SYNTHETIC_SCORING_TRANSPORT_ADAPTED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
