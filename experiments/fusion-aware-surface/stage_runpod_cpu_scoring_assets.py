#!/usr/bin/env python3
"""Stage the public, scoring-only subset of the frozen asset ledger.

This deliberately excludes model checkpoints, training archives, labels, and
diagnostic research inputs.  Every staged byte remains bound to the original
immutable SOURCE_SHA256SUMS ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath


PROJECT_FILES = (
    "cache_real_predictions.py",
    "freeze_thresholds.py",
    "official_metric.py",
    "real_panel_manifest.json",
    "real_split_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload_sha256(payload: dict) -> str:
    content = dict(payload)
    content.pop("payload_sha256", None)
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise RuntimeError(f"unsafe ledger path: {relative!r}")
    path = (root / Path(*pure.parts)).resolve()
    if root.resolve() not in path.parents:
        raise RuntimeError(f"ledger path escapes root: {relative!r}")
    return path


def load_ledger(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in records:
            raise RuntimeError(f"duplicate ledger path: {relative}")
        records[relative] = digest
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"output must start absent: {out}")
    source_ledger = source / "SOURCE_SHA256SUMS"
    records = load_ledger(source_ledger)
    selected = sorted(
        relative
        for relative in records
        if relative in {f"project/{name}" for name in PROJECT_FILES}
    )
    expected_project = {f"project/{name}" for name in PROJECT_FILES}
    if expected_project - set(selected):
        raise RuntimeError(
            f"required project files missing from parent ledger: {sorted(expected_project - set(selected))}"
        )
    out.mkdir(parents=True)
    total_bytes = 0
    for relative in selected:
        src = safe_path(source, relative)
        if not src.is_file() or sha256_file(src) != records[relative]:
            raise RuntimeError(f"parent-ledger source mismatch: {relative}")
        dst = out / Path(*PurePosixPath(relative).parts)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        total_bytes += dst.stat().st_size

    minimal_ledger = "".join(
        f"{records[relative]}  {relative}\n" for relative in selected
    )
    (out / "SOURCE_SHA256SUMS").write_text(minimal_ledger, encoding="utf-8")
    shutil.copy2(source_ledger, out / "PARENT_SOURCE_SHA256SUMS")
    manifest = {
        "schema_version": "1.0",
        "status": "result-blind public scoring-only asset subset",
        "parent_ledger": {
            "file": "PARENT_SOURCE_SHA256SUMS",
            "records": len(records),
            "sha256": sha256_file(source_ledger),
        },
        "scoring_ledger": {
            "file": "SOURCE_SHA256SUMS",
            "records": len(selected),
            "sha256": sha256_file(out / "SOURCE_SHA256SUMS"),
        },
        "files": len(selected),
        "bytes": total_bytes,
        "excluded_private_classes": [
            "model checkpoints",
            "training images and labels",
            "synthetic painters and diagnostic research inputs",
            "unused network and training source files",
        ],
        "scientific_outputs_inspected": False,
    }
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    (out / "scoring_asset_subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
