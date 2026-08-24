import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_demo_is_deterministic_and_self_hashing(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(HERE / "demo_gap8.py"), "--out", str(tmp_path)],
        check=True,
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    summary = json.loads((tmp_path / "demo_summary.json").read_text())
    manifest = json.loads((tmp_path / "demo_manifest.json").read_text())

    assert summary["shape_yx"] == [96, 96]
    assert summary["gap_weight"] == 8
    assert summary["gap_voxels"] == 423
    assert summary["normalized_gap_gradient_multiplier_vs_control"] == 6.054693274
    assert set(manifest["files"]) == {"gap8_demo.png", "demo_summary.json"}
    for name, identity in manifest["files"].items():
        artifact = tmp_path / name
        assert artifact.stat().st_size == identity["bytes"]
        assert sha256(artifact) == identity["sha256"]

    assert manifest["files"]["gap8_demo.png"] == {
        "bytes": 3545,
        "sha256": "f1ff3c088f2e6e255beb325a3a2477dfb0a2d19c7d51de051f8bd5066033fc2b",
    }
    assert manifest["files"]["demo_summary.json"] == {
        "bytes": 442,
        "sha256": "85720feb8e371720f3e1f6711f3204c8e83b3992ecf88c7b0cea8f8a53bbbd13",
    }
