#!/usr/bin/env python3
"""Verify the frozen August TIFXYZ reliability evidence package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/reliability-toolkit-final-evidence/evidence_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    if manifest["schema_version"] != "tifxyz-reliability-august-evidence-v1":
        raise SystemExit("unexpected evidence schema")

    current = manifest["current_contribution"]
    if current["pull_request"] != 1299:
        raise SystemExit("August contribution must remain bound to Villa PR #1299")
    if not str(current["created_at"]).startswith("2026-08-"):
        raise SystemExit("current contribution is not an August contribution")

    historical = manifest["historical_references"]
    if not historical or any(item["novelty_class"] != "historical_context" for item in historical):
        raise SystemExit("historical work must not be presented as a new August contribution")

    verified = []
    for item in manifest["local_artifacts"]:
        path = ROOT / item["path"]
        if not path.is_file():
            raise SystemExit(f"missing evidence artifact: {item['path']}")
        actual_size = path.stat().st_size
        actual_sha = sha256(path)
        if actual_size != item["bytes"]:
            raise SystemExit(f"size mismatch: {item['path']}")
        if actual_sha != item["sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {item['path']}")
        verified.append(item["path"])

    claims = manifest["claims"]
    if claims["original_roots_scanned"] + claims["normalized_roots_scanned"] != claims["registry_roots_scanned"]:
        raise SystemExit("registry denominator mismatch")
    if claims["normalized_roots_with_uuid_output_tifxyz"] + claims["original_roots_with_uuid_output_tifxyz"] != claims["roots_with_uuid_output_tifxyz"]:
        raise SystemExit("UUID finding denominator mismatch")

    return {
        "status": "verified",
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "local_artifacts_verified": len(verified),
        "current_pull_request": current["pull_request"],
        "historical_references": len(historical),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
