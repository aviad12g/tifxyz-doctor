from __future__ import annotations

import hashlib
import json
from pathlib import Path

import heldout_cache_launcher as launcher
import pytest


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    images = tmp_path / "archives" / "images_s4_s5" / "imagesTr"
    labels = tmp_path / "archives" / "labels" / "labelsTr"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    records = []
    for index in range(38):
        scroll = "s4" if index < 36 else "s5"
        image_name = f"{scroll}_z{index}_y0_x0_0000.tif"
        label_name = f"{scroll}_z{index}_y0_x0.tif"
        image_payload = f"image-{index}\n".encode()
        label_payload = f"label-{index}\n".encode()
        (images / image_name).write_bytes(image_payload)
        (labels / label_name).write_bytes(label_payload)
        records.append(
            {
                "image": image_name,
                "label": label_name,
                "scroll": scroll,
                "image_sha256": _sha256(image_payload),
                "label_sha256": _sha256(label_payload),
                "split": "test",
            }
        )
    for split_name, count in (("train", 138), ("validation", 24)):
        for index in range(count):
            records.append(
                {
                    "image": f"s1_{split_name}_{index}_0000.tif",
                    "label": f"s1_{split_name}_{index}.tif",
                    "scroll": "s1",
                    "image_sha256": "0" * 64,
                    "label_sha256": "1" * 64,
                    "split": split_name,
                }
            )
    split = {"records": records}
    split["records_sha256"] = launcher.sha256_bytes(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    )
    split_path = tmp_path / "project" / "real_split_manifest.json"
    split_path.parent.mkdir()
    split_path.write_text(json.dumps(split))
    monkeypatch.setattr(launcher, "REAL_SPLIT_RECORDS_SHA256", split["records_sha256"])
    return images


def test_expanded_mount_is_copied_with_split_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path, monkeypatch)
    destination = tmp_path / "scratch"
    launcher.prepare_test_only(tmp_path, destination)
    assert len(list((destination / "imagesTr").glob("*.tif"))) == 38
    assert len(list((destination / "labelsTr").glob("*.tif"))) == 38


def test_expanded_mount_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = _fixture(tmp_path, monkeypatch)
    next(images.glob("*.tif")).write_bytes(b"tampered\n")
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        launcher.prepare_test_only(tmp_path, tmp_path / "scratch")
