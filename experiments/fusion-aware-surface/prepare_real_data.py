#!/usr/bin/env python3
"""Download and verify the public Dataset059 patch release used for replay/eval."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path


RELEASE = "https://github.com/Jinhojeong/vesuvius-surface-geometry-diagnostic/releases/download/patches-v2"
ASSETS = {
    "images_s1.tar": "cce01d96c77cc7966a41a805b2690db3e7705ce92cbc52d35d0a83a5c7ba35a5",
    "images_s4_s5.tar": "960a152238df8fc60d108a13302af9676c096019c21f204447aa3bf7d9dce4ae",
    "labels.tar": "6e01e4d5f0591796a060bda0a1357ed8cc8b801912a7d3e569ecb246764741a6",
}
EXPECTED_SIZES = {
    "images_s1.tar": 1_478_256_640,
    "images_s4_s5.tar": 444_723_200,
    "labels.tar": 83_087_360,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url) as source, partial.open("wb") as target:
        while chunk := source.read(8 << 20):
            target.write(chunk)
    partial.replace(destination)


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as handle:
        root = destination.resolve()
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        handle.extractall(destination, filter="data")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archives = args.out / "archives"
    extracted = args.out / "extracted"
    archives.mkdir(exist_ok=True)
    extracted.mkdir(exist_ok=True)

    for name, expected in ASSETS.items():
        path = archives / name
        if not path.exists() or sha256_file(path) != expected:
            print(f"DOWNLOADING {name}", flush=True)
            download(f"{RELEASE}/{name}", path)
        actual_size = path.stat().st_size
        if actual_size != EXPECTED_SIZES[name]:
            raise RuntimeError(
                f"size mismatch for {name}: expected {EXPECTED_SIZES[name]}, "
                f"received {actual_size}"
            )
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual}")
        marker = extracted / f".{name}.done"
        if not marker.exists():
            print(f"EXTRACTING {name}", flush=True)
            safe_extract(path, extracted)
            marker.write_text(expected + "\n", encoding="utf-8")

    images = list(extracted.rglob("*_0000.tif"))
    labels = [p for p in extracted.rglob("*.tif") if not p.name.endswith("_0000.tif")]
    if not images or not labels:
        raise RuntimeError("release extraction did not produce image/label TIFFs")
    print(f"REAL_DATA_READY images={len(images)} labels={len(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
