#!/usr/bin/env python3
"""Fetch the small, externally licensed real-data smoke benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "benchmarks" / "realdata-smoke.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark-data"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Fetch only this case ID; repeat to select multiple cases.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, specification: dict) -> None:
    expected_bytes = specification.get("bytes")
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise RuntimeError(
            f"{path}: expected {expected_bytes} bytes, got {path.stat().st_size}"
        )
    expected_sha256 = specification.get("sha256")
    if expected_sha256 is not None:
        actual = _sha256(path)
        if actual != expected_sha256:
            raise RuntimeError(f"{path}: SHA-256 mismatch ({actual})")


def _download(url: str, destination: Path, specification: dict, *, force: bool) -> None:
    if not url.startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS benchmark URL: {url}")
    if destination.is_file() and not force:
        _verify(destination, specification)
        print(f"verified {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "tifxyz-doctor-benchmark/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        _verify(temporary, specification)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"downloaded {destination}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = set(args.case_ids or ())
    known = {case["id"] for case in manifest["cases"]}
    unknown = selected - known
    if unknown:
        raise SystemExit(f"Unknown case ID(s): {', '.join(sorted(unknown))}")

    print(
        "Data license:",
        manifest["dataset_license"]["spdx"],
        manifest["dataset_license"]["url"],
        file=sys.stderr,
    )
    cases = [
        case
        for case in manifest["cases"]
        if not selected or case["id"] in selected
    ]
    for case in cases:
        case_directory = args.output / case["id"]
        for filename, specification in case["files"].items():
            _download(
                case["base_url"] + filename,
                case_directory / filename,
                specification,
                force=args.force,
            )
    print(f"ready: {len(cases)} case(s) in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
