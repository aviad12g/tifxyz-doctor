#!/usr/bin/env python3
"""Validate and freeze the filename-level real-data split before training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VALIDATION_COUNT = 24


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict:
    images = sorted(root.rglob("*_0000.tif"))
    labels = {
        path.name: path
        for path in root.rglob("*.tif")
        if not path.name.endswith("_0000.tif")
    }
    image_records = []
    for image in images:
        label_name = image.name.replace("_0000.tif", ".tif")
        label = labels.get(label_name)
        if label is None:
            raise RuntimeError(f"missing label for {image.name}")
        scroll = image.name.split("_", 1)[0]
        if scroll not in {"s1", "s4", "s5"}:
            raise RuntimeError(f"unexpected scroll prefix in {image.name}")
        image_records.append(
            {
                "image": image.name,
                "label": label.name,
                "scroll": scroll,
                "image_sha256": sha256_file(image),
                "label_sha256": sha256_file(label),
            }
        )

    s1 = [record for record in image_records if record["scroll"] == "s1"]
    if len(s1) <= VALIDATION_COUNT:
        raise RuntimeError(f"insufficient Scroll-1 patches: {len(s1)}")
    validation_names = {
        record["image"]
        for record in sorted(
            s1,
            key=lambda record: (
                hashlib.sha256(record["image"].encode()).hexdigest(),
                record["image"],
            ),
        )[:VALIDATION_COUNT]
    }
    records = []
    for record in image_records:
        if record["scroll"] == "s1":
            split = "validation" if record["image"] in validation_names else "train"
        else:
            split = "test"
        records.append({**record, "split": split})

    names = [record["image"] for record in records]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate image basenames make the split ambiguous")
    counts = {}
    for record in records:
        key = f'{record["scroll"]}:{record["split"]}'
        counts[key] = counts.get(key, 0) + 1
    if counts.get("s1:train", 0) < 100 or counts.get("s1:validation", 0) != VALIDATION_COUNT:
        raise RuntimeError(f"insufficient Scroll-1 split: {counts}")
    if counts.get("s4:test", 0) + counts.get("s5:test", 0) < 20:
        raise RuntimeError(f"insufficient cross-scroll test set: {counts}")

    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": "1.0",
        "rule": (
            "exactly 24 s1 validation patches with lexicographically lowest "
            "SHA256(filename), filename tie-break; remaining s1 train; s4/s5 test"
        ),
        "counts": counts,
        "records_sha256": hashlib.sha256(canonical).hexdigest(),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("counts:", manifest["counts"])
    print("records SHA-256:", manifest["records_sha256"])
    print("REAL_SPLIT_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
