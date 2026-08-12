#!/usr/bin/env python3
"""Verify and prepare the pinned official-metric runtime for one-shot scoring."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

METRIC_DATASET_ID = "sohier/vesuvius-metric-resources"
METRIC_RUNTIME_DATASET_ID = "aviadcohen1/vesuvius-metric-runtime-cp312"
METRIC_RUNTIME_MANIFEST_SHA256 = (
    "e4a78330328700f6a873d417f82073a454fff75e909c23757b325e9f195deb01"
)
METRIC_RUNTIME_PAYLOAD_SHA256 = (
    "f768df2789f680efcb6f412cce64567cc38b8916e128e6edb6b3eff6c802d3ad"
)
METRIC_RUNTIME_REQUIREMENTS_SHA256 = (
    "2950f16b007540f8c0233af2d338cdbabaf367531e4d3b94dd94fb8c2af04f64"
)
METRIC_RUNTIME_WHEEL_LEDGER_SHA256 = (
    "569720cd792c2b8b73e9d703405b83ea61b9d587122253c355be9fdc9f5c83bf"
)
METRIC_RUNTIME_WHEEL_COUNT = 16
METRIC_RUNTIME_WHEEL_BYTES = 139_954_739
OFFICIAL_METRIC_SHA256 = (
    "da5236e67117c1ca6a634c38656dad5cad7579d97017e27bd1c3854f6bcec0fb"
)
VERIFY_METRIC_SHA256 = (
    "09ba89028aa48405a3fc96390b76b455bd0b26d07c908b976f8d6fdfd1aa4e00"
)
METRIC_SOURCE_HASHES = {
    "src/topometrics/__init__.py": "9f1f546d4404e10619fd2da78de6b3d7dbf91a078d4de764476bf21b7c37b69e",
    "src/topometrics/_bm_loader.py": "c2abf08c60341f8da685c14d89a15f9f4bcfa4bc331139e3c78d5351e8a679eb",
    "src/topometrics/leaderboard.py": "f0db94436eea4464a30f252ebc7c35553e539da5e4832e4efaf523ed664cd811",
    "src/topometrics/toposcore.py": "ba09b873d40f7bc8681fcab9c0d5260d117ceb79bc2870f6d65cd18a4c5017c0",
    "src/topometrics/voi.py": "1c8e5131c6102baa2b5e01539b6dde813a35bfc737d157c759c76d598732dc47",
    "external/Betti-Matching-3D/CMakeLists.txt": "1dd442109ca75cd64b2399acab82721e4b7d3d9b8240b7b7afe289bf5fdc81a2",
    "external/Betti-Matching-3D/src/_BettiMatching.cpp": "7b8c41dc4f9154abfdeaa392ae8e928f20f1967f3cb4405c7ef1b159a227fe2e",
}
RUNTIME_PACKAGES = {
    "absl-py": "2.3.1",
    "cmake": "3.31.6",
    "connected-components-3d": "3.26.0",
    "imagecodecs": "2025.8.2",
    "imageio": "2.37.0",
    "lazy-loader": "0.4",
    "networkx": "3.5",
    "numpy": "1.26.4",
    "packaging": "25.0",
    "pillow": "12.0.0",
    "pybind11": "2.13.6",
    "pybind11-global": "2.13.6",
    "scikit-image": "0.25.2",
    "scipy": "1.15.3",
    "surface-distance": "0.1",
    "tifffile": "2025.6.11",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def canonical_payload_sha256(payload: dict) -> str:
    content = dict(payload)
    observed = content.pop("payload_sha256", None)
    expected = sha256_bytes(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    )
    if observed != expected:
        raise RuntimeError("metric-runtime embedded payload SHA-256 mismatch")
    return expected


def require_file(path: Path, digest: str, *, expected_bytes: int | None = None) -> None:
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError(f"file identity mismatch: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(f"file byte-size mismatch: {path}")


def verify_metric_source(root: Path) -> None:
    for relative, expected in METRIC_SOURCE_HASHES.items():
        require_file(root / relative, expected)


def find_metric_source(input_root: Path) -> Path:
    matches = []
    for candidate in input_root.rglob("topological-metrics-kaggle"):
        if not candidate.is_dir():
            continue
        try:
            verify_metric_source(candidate)
        except RuntimeError:
            continue
        matches.append(candidate.resolve())
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one pinned {METRIC_DATASET_ID} source tree; found {matches}"
        )
    return matches[0]


def copy_metric_source(source: Path, destination: Path) -> Path:
    if destination.exists():
        raise RuntimeError("metric scratch must start absent")
    verify_metric_source(source)
    shutil.copytree(source, destination, symlinks=False)
    verify_metric_source(destination)
    return destination


def find_metric_runtime(input_root: Path) -> tuple[Path, dict, dict[str, dict]]:
    manifests = [
        path.resolve()
        for path in input_root.rglob("runtime_manifest.json")
        if path.is_file() and sha256_file(path) == METRIC_RUNTIME_MANIFEST_SHA256
    ]
    if len(manifests) != 1:
        raise RuntimeError(
            f"expected one pinned {METRIC_RUNTIME_DATASET_ID} mount; found {manifests}"
        )
    root = manifests[0].parent
    manifest = load_json(manifests[0])
    if canonical_payload_sha256(manifest) != METRIC_RUNTIME_PAYLOAD_SHA256:
        raise RuntimeError("metric-runtime manifest payload mismatch")
    expected_manifest = {
        "schema_version": "1.0",
        "status": "exact CPython 3.12 Linux metric-runtime wheels frozen before threshold selection",
        "target": {
            "implementation": "cp",
            "python_version": "3.12",
            "abi": "cp312",
            "platforms": ["manylinux_2_28_x86_64", "manylinux2014_x86_64"],
        },
        "source_index": "https://pypi.org/simple",
        "requirements_file": {
            "file": "requirements.txt",
            "sha256": METRIC_RUNTIME_REQUIREMENTS_SHA256,
        },
        "wheel_ledger": {
            "file": "WHEEL_SHA256SUMS",
            "sha256": METRIC_RUNTIME_WHEEL_LEDGER_SHA256,
            "record_count": METRIC_RUNTIME_WHEEL_COUNT,
        },
        "wheel_count": METRIC_RUNTIME_WHEEL_COUNT,
        "wheel_bytes": METRIC_RUNTIME_WHEEL_BYTES,
        "threshold_or_scientific_outputs_inspected": False,
        "payload_sha256": METRIC_RUNTIME_PAYLOAD_SHA256,
    }
    if manifest != expected_manifest:
        raise RuntimeError("metric-runtime manifest content mismatch")
    requirements = root / "requirements.txt"
    ledger = root / "WHEEL_SHA256SUMS"
    require_file(requirements, METRIC_RUNTIME_REQUIREMENTS_SHA256)
    require_file(ledger, METRIC_RUNTIME_WHEEL_LEDGER_SHA256)
    expected_requirements = "".join(
        f"{name}=={version}\n" for name, version in RUNTIME_PACKAGES.items()
    ).encode()
    if requirements.read_bytes() != expected_requirements:
        raise RuntimeError("metric-runtime requirements content mismatch")

    records = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        pure = PurePosixPath(relative)
        if (
            len(pure.parts) != 2
            or pure.parts[0] != "wheels"
            or pure.parts[1] in records
        ):
            raise RuntimeError(f"invalid metric wheel ledger entry: {relative!r}")
        records[pure.parts[1]] = digest
    if len(records) != METRIC_RUNTIME_WHEEL_COUNT:
        raise RuntimeError("metric wheel ledger count mismatch")
    wheels = root / "wheels"
    if {path.name for path in wheels.iterdir()} != set(records) or not all(
        path.is_file() for path in wheels.iterdir()
    ):
        raise RuntimeError("metric wheel directory contents mismatch")
    verified = {}
    for filename, digest in records.items():
        path = wheels / filename
        require_file(path, digest)
        verified[filename] = {"bytes": path.stat().st_size, "sha256": digest}
    if (
        sum(record["bytes"] for record in verified.values())
        != METRIC_RUNTIME_WHEEL_BYTES
    ):
        raise RuntimeError("metric wheel total byte size mismatch")
    return root, manifest, verified


def install_metric_runtime(root: Path) -> dict[str, str]:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"metric wheels require CPython 3.12, found {sys.version}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-index",
            "--find-links",
            str(root / "wheels"),
            "--requirement",
            str(root / "requirements.txt"),
        ],
        check=True,
    )
    observed = {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
    if observed != RUNTIME_PACKAGES:
        raise RuntimeError(f"metric runtime package mismatch: {observed}")
    return observed


def build_betti(metric_root: Path) -> dict:
    import pybind11

    source = metric_root / "external" / "Betti-Matching-3D"
    build = source / "build"
    if build.exists():
        raise RuntimeError("fresh metric source unexpectedly contains a build")
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("pinned CMake executable is unavailable")
    version = subprocess.run(
        [cmake, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    if version != "cmake version 3.31.6":
        raise RuntimeError(f"CMake identity mismatch: {version!r}")
    executable_path = Path(cmake).resolve()
    if Path(sys.prefix).resolve() not in executable_path.parents:
        raise RuntimeError("CMake executable is outside the active Python environment")
    configure = [
        cmake,
        "-S",
        str(source),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DPYBIND11_FINDPYTHON=ON",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DPython_ROOT_DIR={sys.prefix}",
        f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
    ]
    subprocess.run(configure, check=True)
    subprocess.run(
        [cmake, "--build", str(build), "--config", "Release", "--parallel", "2"],
        check=True,
    )
    modules = [path for path in build.glob("betti_matching*.so") if path.is_file()]
    executable = build / "BettiMatching"
    if len(modules) != 1 or not executable.is_file():
        raise RuntimeError("Betti build outputs are incomplete")
    return {
        "cmake_version": version,
        "cmake_executable": str(executable_path),
        "module": {
            "file": modules[0].name,
            "bytes": modules[0].stat().st_size,
            "sha256": sha256_file(modules[0]),
        },
        "executable": {
            "file": executable.name,
            "bytes": executable.stat().st_size,
            "sha256": sha256_file(executable),
        },
    }


def metric_identity_smoke(project: Path, metric_root: Path) -> dict:
    verifier = project / "verify_official_metric.py"
    require_file(project / "official_metric.py", OFFICIAL_METRIC_SHA256)
    require_file(verifier, VERIFY_METRIC_SHA256)
    environment = os.environ.copy()
    environment["FUSION_TOPOMETRICS_ROOT"] = str(metric_root)
    result = subprocess.run(
        [sys.executable, str(verifier)],
        cwd=project,
        env=environment,
        capture_output=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("official metric identity smoke failed")
    lines = [line for line in result.stdout.decode().splitlines() if line.strip()]
    if len(lines) != 2 or lines[1] != "OFFICIAL_METRIC_IDENTITY_CHECK_PASSED":
        raise RuntimeError("official metric identity smoke marker mismatch")
    values = json.loads(lines[0])
    if set(values) != {"blend", "surface_dice", "toposcore", "voi_score"} or any(
        not math.isclose(
            float(value),
            1.0,
            rel_tol=0.0,
            abs_tol=2 * sys.float_info.epsilon,
        )
        for value in values.values()
    ):
        raise RuntimeError("official metric identity smoke values mismatch")
    return {
        "values": values,
        "stdout_bytes": len(result.stdout),
        "stdout_sha256": sha256_bytes(result.stdout),
        "stderr_bytes": len(result.stderr),
        "stderr_sha256": sha256_bytes(result.stderr),
    }


def prepare(input_root: Path, scratch: Path, project: Path) -> tuple[Path, dict]:
    source = find_metric_source(input_root)
    runtime, manifest, wheels = find_metric_runtime(input_root)
    packages = install_metric_runtime(runtime)
    metric_root = copy_metric_source(source, scratch / "topological-metrics-kaggle")
    build = build_betti(metric_root)
    smoke = metric_identity_smoke(project, metric_root)
    return metric_root, {
        "source_hashes": METRIC_SOURCE_HASHES,
        "runtime_manifest_sha256": METRIC_RUNTIME_MANIFEST_SHA256,
        "runtime_payload_sha256": manifest["payload_sha256"],
        "wheel_count": len(wheels),
        "wheel_bytes": sum(record["bytes"] for record in wheels.values()),
        "packages": packages,
        "build": build,
        "identity_smoke": smoke,
    }


if __name__ == "__main__":
    raise SystemExit("import prepare() from one_shot_scoring_launcher.py")
