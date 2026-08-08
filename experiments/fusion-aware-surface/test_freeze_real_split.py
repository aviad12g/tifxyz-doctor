from pathlib import Path

import pytest

from freeze_real_split import build_manifest


def make_pairs(root: Path, scroll: str, count: int) -> None:
    images = root / "images"
    labels = root / "labels"
    images.mkdir(exist_ok=True)
    labels.mkdir(exist_ok=True)
    for index in range(count):
        stem = f"{scroll}_z{index}_y0_x0"
        (images / f"{stem}_0000.tif").write_bytes(f"image-{stem}".encode())
        (labels / f"{stem}.tif").write_bytes(f"label-{stem}".encode())


def test_manifest_freezes_disjoint_scroll_roles(tmp_path: Path) -> None:
    make_pairs(tmp_path, "s1", 162)
    make_pairs(tmp_path, "s4", 20)
    manifest = build_manifest(tmp_path)
    assert manifest["counts"]["s1:train"] == 138
    assert manifest["counts"]["s1:validation"] == 24
    assert manifest["counts"]["s4:test"] == 20
    assert len(manifest["records_sha256"]) == 64
    assert all(
        record["split"] == "test"
        for record in manifest["records"]
        if record["scroll"] == "s4"
    )
    second = build_manifest(tmp_path)
    assert second["records_sha256"] == manifest["records_sha256"]
    assert [
        record["image"]
        for record in second["records"]
        if record["split"] == "validation"
    ] == [
        record["image"]
        for record in manifest["records"]
        if record["split"] == "validation"
    ]


def test_missing_label_fails_closed(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "s1_z0_y0_x0_0000.tif").write_bytes(b"image")
    with pytest.raises(RuntimeError, match="missing label"):
        build_manifest(tmp_path)
