#!/usr/bin/env python3
"""Remove the scientifically unused GPU request from the accelerated real scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import orchestrate_scoring_pair as pair


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


def identity(path: Path) -> dict:
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def rewrite_ledger(directory: Path) -> dict[str, dict]:
    ledger = directory / "KERNEL_SHA256SUMS"
    names = sorted(path.name for path in directory.iterdir() if path.is_file() and path != ledger)
    records = {
        name: {"bytes": (directory / name).stat().st_size, "sha256": sha256_file(directory / name)}
        for name in names
    }
    ledger.write_text(
        "".join(f"{records[name]['sha256']}  {name}\n" for name in names),
        encoding="utf-8",
    )
    records[ledger.name] = identity(ledger) | {"file": ledger.name}
    records[ledger.name].pop("file")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("CPU-adapted package output must start absent")
    index, packages = pair.validate_packages(args.packages, args.index)
    shutil.copytree(args.packages, args.out, symlinks=False)
    copied_index = args.out / args.index.name
    copied_index.unlink()
    records = {record["mode"]: record for record in packages}
    real = records["real"]
    real_root = args.out / real["directory"]
    metadata_path = real_root / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("enable_gpu") is not True or metadata.get("machine_shape") != "Gpu":
        raise RuntimeError("real scorer source package is not the frozen GPU-requesting package")
    metadata["enable_gpu"] = False
    metadata["machine_shape"] = ""
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    real["files"] = rewrite_ledger(real_root)
    adapter = Path(__file__).resolve()
    adaptation = {
        "schema_version": "1.0",
        "status": "real scorer provider metadata corrected to CPU-only after rejected GPU-quota push",
        "input_package_index": identity(args.index) | {"payload_sha256": index["payload_sha256"]},
        "rejected_push": {
            "provider": "Kaggle",
            "kernel_id": real["kaggle_kernel_id"],
            "version_created": False,
            "message": "Maximum weekly GPU quota of 30.00 hours reached",
        },
        "correction": {
            "enable_gpu": False,
            "machine_shape": "",
            "scorer_source_changed": False,
            "scientific_input_or_setting_changed": False,
            "parallel_cpu_workers": 4,
        },
        "adapter": identity(adapter),
    }
    adaptation["payload_sha256"] = canonical_sha256(adaptation)
    adaptation_path = args.out / "real_cpu_metadata_adaptation.json"
    adaptation_path.write_text(
        json.dumps(adaptation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    updated = dict(index)
    updated.pop("payload_sha256", None)
    updated["packages"] = packages
    updated["real_cpu_metadata_adaptation"] = identity(adaptation_path) | {
        "payload_sha256": adaptation["payload_sha256"]
    }
    updated["real_cpu_metadata_adapter"] = identity(adapter)
    gate = dict(updated["scientific_gate"])
    gate.update(
        {
            "real_scorer_requests_cpu_only": True,
            "provider_metadata_changed_scientific_contract": False,
            "rejected_gpu_quota_push_created_version": False,
        }
    )
    updated["scientific_gate"] = gate
    updated["payload_sha256"] = canonical_sha256(updated)
    copied_index.write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pair.validate_packages(args.out, copied_index)
    print("REAL_SCORER_CPU_ONLY_METADATA_ADAPTED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
