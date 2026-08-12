from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "stage_runpod_primary_dataset", HERE / "stage_runpod_primary_dataset.py"
)
stager = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stager)


def write_hashed(path: Path, payload: dict) -> dict:
    value = dict(payload)
    value["payload_sha256"] = stager.canonical_sha256(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return value


def identity(path: Path, payload: dict) -> dict:
    return {
        "bytes": path.stat().st_size,
        "sha256": stager.sha256_file(path),
        "payload_sha256": payload["payload_sha256"],
    }


def test_stage_job_rehashes_and_hardlinks_only_declared_files(tmp_path: Path) -> None:
    root = tmp_path / "source"
    run = root / "synthetic-baseline"
    run.mkdir(parents=True)
    cache = run / "cache.npz"
    cache.write_bytes(b"opaque-npz-bytes")
    manifest_path = root / "cache-manifest.json"
    manifest = write_hashed(manifest_path, {"kind": "cache-manifest"})
    index_path = root / "heldout_job_index.json"
    index = write_hashed(
        index_path,
        {
            "job": {"run": "synthetic-baseline"},
            "cache_manifests": [
                {
                    "file": manifest_path.name,
                    "bytes": manifest_path.stat().st_size,
                    "sha256": stager.sha256_file(manifest_path),
                    "payload_sha256": manifest["payload_sha256"],
                }
            ],
            "sealed_cache_files": [
                {
                    "file": cache.name,
                    "bytes": cache.stat().st_size,
                    "sha256": stager.sha256_file(cache),
                }
            ],
        },
    )
    destination = tmp_path / "dataset"
    manifests, caches, staged = stager.stage_job(
        {
            "job_id": "synthetic-baseline-all-shards",
            "job_index": str(index_path),
            "job_index_identity": identity(index_path, index),
        },
        destination,
    )
    assert (manifests, caches, len(staged)) == (1, 1, 3)
    staged_cache = (
        destination
        / "primary"
        / "synthetic-baseline-all-shards"
        / "synthetic-baseline"
        / cache.name
    )
    assert staged_cache.stat().st_ino == cache.stat().st_ino
    assert stager.sha256_file(staged_cache) == stager.sha256_file(cache)


def test_safe_link_refuses_symlink_and_collision(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"sealed")
    link = tmp_path / "source-link"
    link.symlink_to(source)
    with pytest.raises(RuntimeError, match="regular file"):
        stager.safe_link(link, tmp_path / "destination")
    destination = tmp_path / "destination"
    destination.write_bytes(b"existing")
    with pytest.raises(RuntimeError, match="collision"):
        stager.safe_link(source, destination)


@pytest.mark.parametrize("value", ["../escape", "/absolute", "a/../b", "./file"])
def test_safe_relative_rejects_escape_paths(value: str) -> None:
    with pytest.raises(RuntimeError, match="unsafe"):
        stager.safe_relative(value, "fixture path")


def test_stager_never_opens_scientific_payloads() -> None:
    source = (HERE / "stage_runpod_primary_dataset.py").read_text(encoding="utf-8")
    assert "numpy" not in source
    assert "np.load" not in source
    assert "zipfile" not in source
