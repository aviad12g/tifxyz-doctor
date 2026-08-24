#!/usr/bin/env python3
"""Create and verify the compact release's byte-level file manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "SHA256SUMS"
INDEX_NAME = "RELEASE_INDEX.json"
IGNORED_PARTS = {".pytest_cache", "__pycache__"}
FORBIDDEN_TEXT = (
    "/Users/" + "mazalcohen",
    "BEGIN OPENSSH" + " PRIVATE KEY",
    "RUNPOD" + "_API_KEY=",
    "HF" + "_TOKEN=",
    "HUGGING_FACE_HUB" + "_TOKEN=",
)
TEXT_SUFFIXES = {".cff", ".css", ".html", ".json", ".md", ".py", ".sql", ".txt"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files(*, include_index: bool) -> list[Path]:
    files: list[Path] = []
    for path in RELEASE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(RELEASE_ROOT)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.suffix == ".pyc" or relative.name == MANIFEST_NAME:
            continue
        if not include_index and relative.name == INDEX_NAME:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(RELEASE_ROOT).as_posix())


def audit_text(files: list[Path]) -> None:
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_TEXT:
            if forbidden in text:
                relative = path.relative_to(RELEASE_ROOT)
                raise RuntimeError(f"forbidden release text {forbidden!r} in {relative}")


def main() -> int:
    content_files = release_files(include_index=False)
    audit_text(content_files)
    index = {
        "schema_version": 1,
        "release": "fail-closed-surface-to-ink-v1",
        "generated_date": "2026-08-15",
        "content_file_count": len(content_files),
        "content_bytes": sum(path.stat().st_size for path in content_files),
        "manifest_algorithm": "SHA-256",
        "manifest_path": MANIFEST_NAME,
    }
    (RELEASE_ROOT / INDEX_NAME).write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    files = release_files(include_index=True)
    audit_text(files)
    lines = [
        f"{sha256_file(path)}  {path.relative_to(RELEASE_ROOT).as_posix()}"
        for path in files
    ]
    (RELEASE_ROOT / MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "release_root": str(RELEASE_ROOT),
                "manifest_entries": len(files),
                "payload_bytes": sum(path.stat().st_size for path in files),
                "sha256sums_sha256": sha256_file(RELEASE_ROOT / MANIFEST_NAME),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
