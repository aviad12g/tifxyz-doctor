#!/usr/bin/env python3
"""Run the hash-pinned VC3D PHerc1447 trace confirmation.

The runner deliberately treats every cross-variant result as an observation.
It asserts only provenance, the preregistered public fixture, successful
execution, output integrity, and three-replicate determinism.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 2
SOURCE_REPOSITORY = "aviad12g/villa"
SOURCE_WORKFLOW_ID = 322868724
SOURCE_WORKFLOW_PATH = ".github/workflows/vc3d-linux.yml"
BUILDER_IMAGE_DIGEST = (
    "sha256:77be7988b71fca1d8a0657e1915368274729400e03e2e7edaacc9bb84ec707a2"
)
BUILD_PRESET = "ci-release-gcc"
BUILD_COMPILER = "GNU 15.2.0"
CHILD_ENVIRONMENT_KEYS = (
    "HOME",
    "LANG",
    "LC_ALL",
    "OMP_DYNAMIC",
    "OMP_NUM_THREADS",
    "PATH",
    "TMPDIR",
    "TZ",
)
HOST_RUNTIME_PACKAGES = (
    "libglvnd0",
    "libglx0",
    "libegl1",
    "libopengl0",
    "libgl1",
    "libgl1-mesa-dri",
    "libglx-mesa0",
    "libx11-6",
    "libx11-xcb1",
    "libxcb1",
    "libice6",
    "libsm6",
    "libfontconfig1",
    "libfreetype6",
    "libharfbuzz0b",
)
TOOLCHAIN_LOG_FRAGMENTS = {
    "runner_version": "Current runner version: '2.336.0'",
    "runner_provisioner_version": "Version: 20260707.563",
    "runner_provisioner_commit": (
        "Commit: 02667638d2b423fbc733a8e32a88b44996a3ba6e"
    ),
    "runner_image": "Image: ubuntu-24.04",
    "runner_image_version": "Version: 20260720.247.2",
    "dockerfile_frontend": (
        "docker-image://docker.io/docker/dockerfile:1.7@"
        "sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e"
    ),
    "builder_image": (
        "FROM ghcr.io/scrollprize/villa/volume-cartographer:"
        f"builder-ubuntu-26.04@{BUILDER_IMAGE_DIGEST}"
    ),
    "c_compiler": "The C compiler identification is GNU 15.2.0",
    "cxx_compiler": "The CXX compiler identification is GNU 15.2.0",
    "fortran_compiler": "The Fortran compiler identification is GNU 15.2.0",
    "generator": "Targeting Ninja",
    "compiler_path": "-- Using /usr/bin/g++ compiler.",
    "configure_preset": f"cmake --preset {BUILD_PRESET}",
    "build_preset": f"cmake --build --preset {BUILD_PRESET}",
    "linuxdeploy": (
        "linuxdeploy version 1-alpha (git commit ID a9f929f), "
        "GitHub actions build 361 built on 2026-07-01 03:57:16 UTC"
    ),
    "linuxdeploy_qt": (
        "linuxdeploy-plugin-qt version 1-alpha (git commit ID 409307d), "
        "GitHub actions build 250 built on 2026-05-16 19:05:46 UTC"
    ),
    "qt": "[qt/stdout] Using Qt version:  6.10.2  ( 6 )",
    "appimagetool": (
        "appimagetool, continuous build (git version 8c8c91f), "
        "build 295 built on 2025-12-04 17:56:36 UTC"
    ),
    "apt_desktop_file_utils": "Setting up desktop-file-utils (0.28-1build1) ...",
    "apt_patchelf": "Setting up patchelf (0.18.0-1.4build1) ...",
    "apt_zstd": "Setting up zstd (1.5.7+dfsg-3) ...",
}
SOURCE_BASE_URL = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHerc1447/segments/"
    "20250502185519-auto_grown_20250502164303733/mesh/intermediate/"
    "tifxyz_original"
)

VARIANTS = {
    "baseline": {
        "run_id": 30443013824,
        "job_id": 90546543662,
        "artifact_id": 8720697165,
        "artifact_size_bytes": 613863853,
        "artifact_archive_sha256": (
            "047cdcbe9a0d4d36deed004cebd05516b557cb4cf9b2c6380338e08b58a31a38"
        ),
        "source_sha": "208faea7d7fec62da03ff75e118e26be0c08a117",
        "run_head_sha": "208faea7d7fec62da03ff75e118e26be0c08a117",
        "run_head_branch": "agent/trace-baseline-208faea",
        "appimage_name": "VC3D-208faea-2026-07-27-linux-x86_64.AppImage",
    },
    "helper": {
        "run_id": 30443054997,
        "job_id": 90546675254,
        "artifact_id": 8720646739,
        "artifact_size_bytes": 613871240,
        "artifact_archive_sha256": (
            "f270d3d8e77a132f1e57690139e393ecaecc0da614546b0b2e69f4c83313bb58"
        ),
        "source_sha": "d5226910817be064d2ae1577f1fae0f850acc8c7",
        "run_head_sha": "d5226910817be064d2ae1577f1fae0f850acc8c7",
        "run_head_branch": "agent/trace-bounds-helper-208faea",
        "appimage_name": "VC3D-d522691-2026-07-29-linux-x86_64.AppImage",
    },
    "full": {
        "run_id": 30443085737,
        "job_id": 90546776777,
        "artifact_id": 8720738643,
        "artifact_size_bytes": 613861881,
        "artifact_archive_sha256": (
            "7659ed946a9dc18bcacf0401a3cf5ba31026a2a1eba8cff98a4bc338fade0b66"
        ),
        "source_sha": "9e15071df41a139ce77ada6aa64367e000387c5d",
        "run_head_sha": "9e15071df41a139ce77ada6aa64367e000387c5d",
        "run_head_branch": "agent/trace-bounds-full-208faea",
        "appimage_name": "VC3D-9e15071-2026-07-29-linux-x86_64.AppImage",
    },
    "pr": {
        "run_id": 30442703266,
        "job_id": 90545712534,
        "artifact_id": 8720610248,
        "artifact_size_bytes": 613913820,
        "artifact_archive_sha256": (
            "914747b9aacaa699cd190f058df62825abffeb1c0ee5c20e8e383ad68986ada4"
        ),
        "source_sha": "6e2bba940d8f93b53b0d1bcf7f66af19c5b81295",
        # This run dispatched the workflow from main and supplied source_sha
        # through vc3d-linux.yml's commit_sha input. The immutable AppImage
        # filename below proves the checked-out/build revision.
        "run_head_sha": "d992fecf59a0fd9188376caad27d49c0d3f11858",
        "run_head_branch": "main",
        "appimage_name": "VC3D-6e2bba9-2026-07-27-linux-x86_64.AppImage",
    },
}

SOURCE_HASHES = {
    "x.tif": "179da10e706424a8479681555a03a68dee80e91c8ad241877ea62b4044039651",
    "y.tif": "dae0c8d118cd8bfa40b8e38c04682801191402a6cc66b14544d649eb7fba8153",
    "z.tif": "f7a028ea6121c492bf6c15c7174bcd379b3b00a780ef8361054f1c3266728b4f",
    "meta.json": "158dc7ab72251dffb95df471f1789eb4b020e0bef70bd275714399f72b142634",
}

SOURCE_SHAPE_ROWS_COLS = [160, 136]
SOURCE_VALID_VERTICES = 16_316
SOURCE_VALID_QUADS = 15_702
BOUNDARY_CELLS_ROW_COL = [
    [51, 134],
    [52, 134],
    [53, 134],
    [158, 88],
    [158, 89],
    [158, 90],
]

FAKE_VOLUME_META_TEXT = """{
  "format": "zarr",
  "name": "metadata-only PHerc1447 tracer fixture",
  "voxelsize": 7.91,
  "width": 50000,
  "height": 50000,
  "slices": 50000
}
"""

FAKE_VOLUME_ZARRAY_TEXT = """{
  "zarr_format": 2,
  "shape": [50000, 50000, 50000],
  "chunks": [128, 128, 128],
  "dtype": "<u2",
  "compressor": null,
  "fill_value": 0,
  "order": "C",
  "filters": null
}
"""

TRACE_PARAMS_TEXT = """{
  "resume_growth": true,
  "disable_grid_expansion": true,
  "growth_directions": ["right"],
  "steps": 1,
  "step": 1,
  "src_step": 1,
  "max_width": 256,
  "max_height": 256,
  "global_steps_per_window": 1,
  "debug_images": false,
  "use_cuda": false,
  "consensus_default_th": 2
}
"""

COMPARISON_PAIRS = [
    ("baseline", "helper"),
    ("helper", "full"),
    ("full", "pr"),
    ("baseline", "pr"),
]
SOURCE_COMPARISONS = {
    "baseline_to_helper": {
        "base": VARIANTS["baseline"]["source_sha"],
        "head": VARIANTS["helper"]["source_sha"],
        "files": ["volume-cartographer/core/src/Geometry.cpp"],
    },
    "helper_to_full": {
        "base": VARIANTS["helper"]["source_sha"],
        "head": VARIANTS["full"]["source_sha"],
        "files": ["volume-cartographer/core/src/SurfTrackerData.cpp"],
    },
    "baseline_to_pr": {
        "base": VARIANTS["baseline"]["source_sha"],
        "head": VARIANTS["pr"]["source_sha"],
        "files": [
            "volume-cartographer/core/include/vc/core/util/Geometry.hpp",
            "volume-cartographer/core/src/Geometry.cpp",
            "volume-cartographer/core/src/QuadSurface.cpp",
            "volume-cartographer/core/src/SurfTrackerData.cpp",
            "volume-cartographer/core/test/test_geometry.cpp",
            "volume-cartographer/core/test/test_quadsurface_basics.cpp",
            "volume-cartographer/core/test/test_surf_tracker_data.cpp",
        ],
    },
}

APPIMAGE_NAME_RE = re.compile(
    r"^VC3D-(?P<revision>[0-9a-fA-F]{7,40})"
    r"(?:-(?P<commit_date>\d{4}-\d{2}-\d{2}))?"
    r"-linux-x86_64\.AppImage$"
)
LOG_PATTERNS = {
    "resume": re.compile(
        r"resume_growth initialized (\d+) low-res points, fringe (\d+).*?"
        r"used_area \[(\d+) x (\d+) from \((\d+), (\d+)\)\]"
    ),
    "generation": re.compile(
        r"gen 0 processing (\d+) fringe cands .*? area (\d+) vx\^2"
    ),
    "area": re.compile(r"area est: (\d+) vx\^2 \(([^)]+) cm\^2\)"),
}


class EvidenceError(RuntimeError):
    """Raised when a pinned input or evidence invariant is violated."""


@dataclasses.dataclass(frozen=True)
class Surface:
    directory: Path
    x: Any
    y: Any
    z: Any
    valid: Any
    grid_offset_col_row: tuple[int, int]
    meta: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(item) for item in self.x.shape)


def minimal_child_environment(
    home: Path,
    temporary: Path,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return a deliberately secret-free environment for artifact execution."""
    home.mkdir(parents=True, exist_ok=False)
    temporary.mkdir(parents=True, exist_ok=False)
    environment = {
        "HOME": str(home),
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "TMPDIR": str(temporary),
    }
    if extra:
        environment.update(extra)
    return environment


def purge_runner_credentials() -> list[str]:
    """Remove credential-bearing runner variables before native code executes."""
    exact = {
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "ACTIONS_RUNTIME_TOKEN",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "SYSTEM_ACCESSTOKEN",
    }
    removed: list[str] = []
    for key in list(os.environ):
        if key in exact or key.endswith("_TOKEN") or key.endswith("_PASSWORD"):
            os.environ.pop(key, None)
            removed.append(key)
    return sorted(removed)


def installed_host_package_versions() -> dict[str, str]:
    result = subprocess.run(
        [
            "dpkg-query",
            "-W",
            "-f=${Package}\t${Version}\n",
            *HOST_RUNTIME_PACKAGES,
        ],
        env={
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    require(
        result.returncode == 0,
        f"could not record host runtime packages: {tail_text(result.stderr)}",
    )
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        package, separator, version = line.partition("\t")
        require(bool(separator and package and version), "invalid dpkg-query output")
        parsed[package] = version
    require(
        set(parsed) == set(HOST_RUNTIME_PACKAGES),
        "dpkg-query did not return the exact runtime package set",
    )
    return dict(sorted(parsed.items()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def require_regular_nonempty_file(path: Path) -> None:
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
        f"expected a regular nonempty file: {path}",
    )


def dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import tifffile
    except ImportError as exc:
        raise EvidenceError(
            "numpy and tifffile are required; install the workflow's pinned "
            "Python dependencies"
        ) from exc
    return np, tifffile


def github_api_json(
    url: str, token: str | None, *, attempts: int = 3
) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vc3d-trace-confirmation/1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=30
            ) as response:
                require(response.status == 200, f"GitHub API returned {response.status}")
                value = json.load(response)
                require(isinstance(value, dict), "GitHub API response is not an object")
                return value
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt != attempts:
                time.sleep(attempt)
    raise EvidenceError(f"GitHub API request failed for {url}: {last_error}")


def github_api_text(
    url: str, token: str | None, *, attempts: int = 3
) -> str:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vc3d-trace-confirmation/1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(
            self,
            req: Any,
            fp: Any,
            code: int,
            msg: str,
            response_headers: Any,
            newurl: str,
        ) -> None:
            return None

    opener = urllib.request.build_opener(NoRedirect)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            redirect_url: str | None = None
            try:
                response = opener.open(
                    urllib.request.Request(url, headers=headers), timeout=60
                )
            except urllib.error.HTTPError as exc:
                if exc.code not in (301, 302, 303, 307, 308):
                    raise
                redirect_url = exc.headers.get("Location")
                exc.close()
                require(bool(redirect_url), "GitHub log redirect lacks Location")
                parsed = urllib.parse.urlsplit(str(redirect_url))
                require(
                    parsed.scheme == "https" and bool(parsed.netloc),
                    "GitHub log redirect is not an absolute HTTPS URL",
                )
                response = urllib.request.urlopen(
                    urllib.request.Request(
                        str(redirect_url),
                        headers={"User-Agent": headers["User-Agent"]},
                    ),
                    timeout=60,
                )
            with response:
                require(response.status == 200, f"GitHub API returned {response.status}")
                payload = response.read(5 * 1024 * 1024 + 1)
                require(
                    len(payload) <= 5 * 1024 * 1024,
                    "GitHub job log exceeds the evaluator size limit",
                )
                return payload.decode("utf-8", errors="strict")
        except (OSError, UnicodeError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt != attempts:
                time.sleep(attempt)
    raise EvidenceError(f"GitHub API request failed for {url}: {last_error}")


def check_build_log_provenance(
    repository: str,
    variant: str,
    expected: dict[str, Any],
    token: str | None,
) -> dict[str, Any]:
    job_id = int(expected["job_id"])
    log = github_api_text(
        f"https://api.github.com/repos/{repository}/actions/jobs/{job_id}/logs",
        token,
    )
    source_sha = str(expected["source_sha"])
    required_fragments = {
        **TOOLCHAIN_LOG_FRAGMENTS,
        "exact_source_revision": f'-DVC_GIT_SHA1="{source_sha}"',
    }
    for label, fragment in required_fragments.items():
        require(
            fragment in log,
            f"{variant}: build log lacks {label} evidence {fragment!r}",
        )
    return {
        "job_id": job_id,
        "job_html_url": (
            f"https://github.com/{repository}/actions/runs/"
            f"{expected['run_id']}/job/{job_id}"
        ),
        "job_log_sha256": hashlib.sha256(log.encode()).hexdigest(),
        "source_sha": source_sha,
        "builder_image_digest": BUILDER_IMAGE_DIGEST,
        "cmake_preset": BUILD_PRESET,
        "cxx_compiler": BUILD_COMPILER,
        "runner_image": "ubuntu-24.04",
        "shared_toolchain_fingerprint": TOOLCHAIN_LOG_FRAGMENTS,
        "required_fragments_checked": sorted(required_fragments),
        "interpretation_guard": (
            "The logs establish identical resolved versions and configuration "
            "across these four builds; they do not make live APT or continuous "
            "packaging downloads bit-for-bit hermetic."
        ),
    }


def check_run_provenance(
    repository: str, token: str | None
) -> dict[str, dict[str, Any]]:
    checked: dict[str, dict[str, Any]] = {}
    for variant, expected in VARIANTS.items():
        run_id = int(expected["run_id"])
        run = github_api_json(
            f"https://api.github.com/repos/{repository}/actions/runs/{run_id}",
            token,
        )
        head_sha = str(run.get("head_sha", "")).lower()
        actual_repository = str(
            (run.get("repository") or {}).get("full_name", "")
        )
        require(int(run.get("id", 0)) == run_id, f"{variant}: run ID mismatch")
        require(
            actual_repository.lower() == repository.lower(),
            f"{variant}: API repository {actual_repository!r} != {repository!r}",
        )
        require(
            head_sha == str(expected["run_head_sha"]),
            f"{variant}: run head {head_sha!r} != "
            f"{expected['run_head_sha']}",
        )
        require(
            run.get("head_branch") == expected["run_head_branch"],
            f"{variant}: run branch {run.get('head_branch')!r} != "
            f"{expected['run_head_branch']!r}",
        )
        require(
            run.get("status") == "completed" and run.get("conclusion") == "success",
            f"{variant}: run is not completed/successful",
        )
        require(
            run.get("event") == "workflow_dispatch",
            f"{variant}: unexpected workflow event {run.get('event')!r}",
        )
        require(
            run.get("path") == SOURCE_WORKFLOW_PATH,
            f"{variant}: unexpected workflow path {run.get('path')!r}",
        )
        require(
            run.get("workflow_id") == SOURCE_WORKFLOW_ID,
            f"{variant}: workflow ID {run.get('workflow_id')!r} != "
            f"{SOURCE_WORKFLOW_ID}",
        )
        require(
            run.get("run_attempt") == 1,
            f"{variant}: expected original build attempt 1, got "
            f"{run.get('run_attempt')!r}",
        )
        artifacts = github_api_json(
            "https://api.github.com/repos/"
            f"{repository}/actions/runs/{run_id}/artifacts"
            "?name=VC3D-linux-appimage",
            token,
        )
        artifact_items = artifacts.get("artifacts")
        require(
            artifacts.get("total_count") == 1
            and isinstance(artifact_items, list)
            and len(artifact_items) == 1,
            f"{variant}: expected one named workflow artifact",
        )
        artifact = artifact_items[0]
        require(
            artifact.get("name") == "VC3D-linux-appimage"
            and artifact.get("expired") is False,
            f"{variant}: artifact is misnamed or expired",
        )
        expected_archive_digest = (
            f"sha256:{expected['artifact_archive_sha256']}"
        )
        require(
            artifact.get("id") == expected["artifact_id"]
            and artifact.get("digest") == expected_archive_digest
            and artifact.get("size_in_bytes") == expected["artifact_size_bytes"],
            f"{variant}: artifact ID/size/digest does not match the pinned archive",
        )
        build_log = check_build_log_provenance(
            repository, variant, expected, token
        )
        checked[variant] = {
            "run_id": run_id,
            "source_sha": expected["source_sha"],
            "expected_run_head_sha": expected["run_head_sha"],
            "expected_appimage_name": expected["appimage_name"],
            "head_sha": head_sha,
            "head_branch": run.get("head_branch"),
            "repository": actual_repository,
            "workflow_id": run.get("workflow_id"),
            "workflow_path": run.get("path"),
            "run_attempt": run.get("run_attempt"),
            "event": run.get("event"),
            "status": run.get("status"),
            "conclusion": run.get("conclusion"),
            "created_at": run.get("created_at"),
            "updated_at": run.get("updated_at"),
            "html_url": run.get("html_url"),
            "build_log": build_log,
            "artifact": {
                key: artifact.get(key)
                for key in (
                    "id",
                    "name",
                    "size_in_bytes",
                    "digest",
                    "created_at",
                    "expires_at",
                    "expired",
                )
            },
        }
    return checked


def check_source_comparisons(
    repository: str,
    token: str | None,
) -> dict[str, dict[str, Any]]:
    checked: dict[str, dict[str, Any]] = {}
    for name, expected in SOURCE_COMPARISONS.items():
        base = str(expected["base"])
        head = str(expected["head"])
        comparison = github_api_json(
            f"https://api.github.com/repos/{repository}/compare/{base}...{head}",
            token,
        )
        files = comparison.get("files")
        observed_files = sorted(
            str(item.get("filename"))
            for item in files
            if isinstance(item, dict)
        ) if isinstance(files, list) else []
        expected_files = sorted(str(item) for item in expected["files"])
        require(
            comparison.get("status") == "ahead"
            and comparison.get("ahead_by") == 1
            and comparison.get("total_commits") == 1,
            f"{name}: expected a one-commit direct source comparison",
        )
        require(
            str((comparison.get("merge_base_commit") or {}).get("sha", ""))
            == base,
            f"{name}: comparison merge base does not equal the pinned base",
        )
        require(
            str((comparison.get("commits") or [{}])[-1].get("sha", ""))
            == head,
            f"{name}: comparison head does not equal the pinned head",
        )
        require(
            observed_files == expected_files,
            f"{name}: changed files {observed_files!r} != {expected_files!r}",
        )
        checked[name] = {
            "base_sha": base,
            "head_sha": head,
            "status": comparison.get("status"),
            "ahead_by": comparison.get("ahead_by"),
            "total_commits": comparison.get("total_commits"),
            "changed_files": observed_files,
            "html_url": comparison.get("html_url"),
        }
    return checked


def load_archive_verification(artifact_root: Path) -> dict[str, dict[str, Any]]:
    path = artifact_root / "archive-verification.json"
    require(path.is_file(), f"missing verified archive record {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(
        isinstance(value, dict) and value.get("schema_version") == SCHEMA_VERSION,
        "archive verification has an unexpected schema",
    )
    records = value.get("verified_archives")
    require(
        isinstance(records, dict) and set(records) == set(VARIANTS),
        "archive verification does not contain exactly the four variants",
    )
    checked: dict[str, dict[str, Any]] = {}
    for variant, expected in VARIANTS.items():
        record = records[variant]
        require(isinstance(record, dict), f"{variant}: archive record is not an object")
        require(
            record.get("artifact_id") == expected["artifact_id"],
            f"{variant}: downloaded artifact ID is not pinned",
        )
        require(
            record.get("archive_sha256")
            == expected["artifact_archive_sha256"],
            f"{variant}: downloaded archive SHA-256 is not pinned",
        )
        require(
            record.get("archive_size_bytes")
            == expected["artifact_size_bytes"],
            f"{variant}: downloaded archive byte size is not pinned",
        )
        require(
            record.get("appimage_filename") == expected["appimage_name"],
            f"{variant}: archive record names an unexpected AppImage",
        )
        appimage_sha = record.get("appimage_sha256")
        require(
            isinstance(appimage_sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", appimage_sha) is not None,
            f"{variant}: archive record lacks an AppImage SHA-256",
        )
        checked[variant] = {
            "artifact_id": record["artifact_id"],
            "archive_size_bytes": record["archive_size_bytes"],
            "archive_sha256": record["archive_sha256"],
            "appimage_filename": record["appimage_filename"],
            "appimage_sha256": appimage_sha,
        }
    return checked


def find_and_validate_appimages(
    artifact_root: Path,
    archive_verification: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for variant, expected in VARIANTS.items():
        root = artifact_root / variant
        require(root.is_dir(), f"missing downloaded artifact directory {root}")
        candidates = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.name.endswith(".AppImage")
        )
        require(
            len(candidates) == 1,
            f"{variant}: expected exactly one AppImage, found {len(candidates)}",
        )
        appimage = candidates[0]
        require(
            appimage.name == expected["appimage_name"],
            f"{variant}: AppImage {appimage.name!r} != "
            f"{expected['appimage_name']!r}",
        )
        match = APPIMAGE_NAME_RE.fullmatch(appimage.name)
        require(bool(match), f"{variant}: unexpected AppImage name {appimage.name!r}")
        revision = match.group("revision").lower()
        require(
            str(expected["source_sha"]).startswith(revision),
            f"{variant}: filename revision {revision!r} does not identify "
            f"{expected['source_sha']}",
        )
        appimage.chmod(appimage.stat().st_mode | 0o111)
        observed_sha256 = sha256_file(appimage)
        require(
            observed_sha256 == archive_verification[variant]["appimage_sha256"],
            f"{variant}: AppImage changed after its verified archive was extracted",
        )
        records[variant] = {
            "path": appimage,
            "filename": appimage.name,
            "filename_revision": revision,
            "filename_commit_date": match.group("commit_date"),
            "sha256": observed_sha256,
            "size_bytes": appimage.stat().st_size,
            "verified_archive": archive_verification[variant],
        }
    return records


def download_pinned_file(
    url: str,
    target: Path,
    expected_sha256: str,
    *,
    max_bytes: int = 16 * 1024 * 1024,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "vc3d-trace-confirmation/1"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                require(response.status == 200, f"{url}: HTTP {response.status}")
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    require(
                        int(content_length) <= max_bytes,
                        f"{target.name}: declared size exceeds {max_bytes} bytes",
                    )
                received = 0
                with partial.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        received += len(chunk)
                        require(
                            received <= max_bytes,
                            f"{target.name}: download exceeds {max_bytes} bytes",
                        )
                        handle.write(chunk)
            observed = sha256_file(partial)
            require(
                observed == expected_sha256,
                f"{target.name}: SHA-256 {observed} != {expected_sha256}",
            )
            partial.replace(target)
            return
        except (OSError, urllib.error.URLError, EvidenceError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt != 3:
                time.sleep(attempt)
    raise EvidenceError(f"could not download pinned {url}: {last_error}")


def valid_quad_mask(valid: Any) -> Any:
    require(valid.ndim == 2, "validity mask must be two-dimensional")
    if valid.shape[0] < 2 or valid.shape[1] < 2:
        return valid[:0, :0]
    return (
        valid[:-1, :-1]
        & valid[:-1, 1:]
        & valid[1:, :-1]
        & valid[1:, 1:]
    )


def preregister_fixture(source: Path) -> dict[str, Any]:
    np, tifffile = dependencies()
    for axis in ("x", "y", "z"):
        require_regular_nonempty_file(source / f"{axis}.tif")
    arrays = {
        axis: tifffile.imread(source / f"{axis}.tif") for axis in ("x", "y", "z")
    }
    for axis, array in arrays.items():
        require(
            list(array.shape) == SOURCE_SHAPE_ROWS_COLS,
            f"source {axis}.tif shape {list(array.shape)} != "
            f"{SOURCE_SHAPE_ROWS_COLS}",
        )
        require(array.ndim == 2, f"source {axis}.tif is not a 2-D image")
        require(
            str(array.dtype) == "float32",
            f"source {axis}.tif dtype {array.dtype} != float32",
        )
        require(
            bool(np.all(np.isfinite(array))),
            f"source {axis}.tif contains non-finite values",
        )
    masks = {axis: array != -1 for axis, array in arrays.items()}
    require(
        np.array_equal(masks["x"], masks["y"])
        and np.array_equal(masks["x"], masks["z"]),
        "source XYZ sentinel masks disagree",
    )
    valid = masks["x"]
    quads = valid_quad_mask(valid)
    valid_vertices = int(np.count_nonzero(valid))
    valid_quads = int(np.count_nonzero(quads))
    boundary: list[list[int]] = []
    for row, col in np.argwhere(quads):
        row_int, col_int = int(row), int(col)
        if row_int + 2 == valid.shape[0] or col_int + 2 == valid.shape[1]:
            boundary.append([row_int, col_int])
    require(
        valid_vertices == SOURCE_VALID_VERTICES,
        f"source valid vertices {valid_vertices} != {SOURCE_VALID_VERTICES}",
    )
    require(
        valid_quads == SOURCE_VALID_QUADS,
        f"source valid quads {valid_quads} != {SOURCE_VALID_QUADS}",
    )
    require(
        boundary == BOUNDARY_CELLS_ROW_COL,
        f"source boundary cells {boundary!r} != preregistration",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "fixture_preregistered",
        "source": {
            "description": "Public PHerc1447 unverified auto-grown hard candidate",
            "base_url": SOURCE_BASE_URL,
            "shape_rows_cols": SOURCE_SHAPE_ROWS_COLS,
            "sha256": SOURCE_HASHES,
        },
        "definitions": {
            "valid_vertex": "all XYZ coordinates differ from the canonical -1 sentinel",
            "valid_quad": "all four neighboring vertices are valid",
            "boundary_cell": (
                "a valid quad whose origin is on final legal row rows-2 "
                "or final legal column cols-2"
            ),
        },
        "observed_before_trace": {
            "valid_vertices": valid_vertices,
            "valid_quads": valid_quads,
            "boundary_cell_count": len(boundary),
            "boundary_cells_row_col": boundary,
        },
    }


def create_fixture(work: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    source = work / "input-parent" / "pherc1447-source"
    for filename, expected_hash in SOURCE_HASHES.items():
        download_pinned_file(
            f"{SOURCE_BASE_URL}/{filename}", source / filename, expected_hash
        )

    volume = work / "fake-volume-pherc1447"
    params = work / "resume-one-generation.json"
    write_text(volume / "meta.json", FAKE_VOLUME_META_TEXT)
    write_text(volume / "0" / ".zarray", FAKE_VOLUME_ZARRAY_TEXT)
    write_text(params, TRACE_PARAMS_TEXT)

    provenance = fixture_hashes(source, volume, params)
    return source, volume, params, provenance


def fixture_hashes(source: Path, volume: Path, params: Path) -> dict[str, Any]:
    fixture_root = source.parent.parent
    require(
        volume.parent == fixture_root and params.parent == fixture_root,
        "fixture paths do not share the expected isolated root",
    )
    complete_tree: dict[str, dict[str, Any]] = {}
    for path in sorted(fixture_root.rglob("*")):
        relative = path.relative_to(fixture_root).as_posix()
        require(not path.is_symlink(), f"fixture contains symlink {relative}")
        if path.is_dir():
            complete_tree[relative] = {"type": "directory"}
        elif path.is_file():
            complete_tree[relative] = {
                "type": "file",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        else:
            raise EvidenceError(f"fixture contains non-file entry {relative}")
    return {
        "source": {
            name: sha256_file(source / name) for name in SOURCE_HASHES
        },
        "fake_volume": {
            "meta.json": sha256_file(volume / "meta.json"),
            "0/.zarray": sha256_file(volume / "0" / ".zarray"),
        },
        "params": sha256_file(params),
        "complete_tree": complete_tree,
    }


def make_fixture_read_only(source: Path, volume: Path, params: Path) -> None:
    files = [
        *(source / name for name in SOURCE_HASHES),
        volume / "meta.json",
        volume / "0" / ".zarray",
        params,
    ]
    for path in files:
        path.chmod(0o444)


def copy_fixture_for_replicate(
    source: Path,
    volume: Path,
    params: Path,
    destination: Path,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    copied_source = destination / "input-parent" / source.name
    copied_volume = destination / volume.name
    copied_params = destination / params.name
    shutil.copytree(source, copied_source)
    shutil.copytree(volume, copied_volume)
    copied_params.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(params, copied_params)
    make_fixture_read_only(copied_source, copied_volume, copied_params)
    return (
        copied_source,
        copied_volume,
        copied_params,
        fixture_hashes(copied_source, copied_volume, copied_params),
    )


def extract_appimage_once(
    variant: str, appimage: Path, extraction_parent: Path
) -> tuple[Path, dict[str, Any]]:
    destination = extraction_parent / variant
    destination.mkdir(parents=True, exist_ok=False)
    result = subprocess.run(
        [str(appimage), "--appimage-extract"],
        cwd=destination,
        env=minimal_child_environment(
            destination / "isolated-home",
            destination / "isolated-tmp",
        ),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise EvidenceError(
            f"{variant}: AppImage extraction failed ({result.returncode})\n"
            f"stdout tail:\n{tail_text(result.stdout)}\n"
            f"stderr tail:\n{tail_text(result.stderr)}"
        )
    root = destination / "squashfs-root"
    app_run = root / "AppRun"
    require(app_run.is_file(), f"{variant}: extracted AppRun is missing")
    require(os.access(app_run, os.X_OK), f"{variant}: AppRun is not executable")
    dispatch = root / "vc_grow_seg_from_segments"
    dispatch.symlink_to("AppRun")
    require(dispatch.is_symlink(), f"{variant}: dispatch symlink was not created")
    return dispatch, {
        "extraction_stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "extraction_stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "dispatch": "named symlink vc_grow_seg_from_segments -> AppRun",
    }


def tail_text(value: str, line_count: int = 40) -> str:
    return "\n".join(value.splitlines()[-line_count:])


def parse_trace_log(stdout: str) -> dict[str, Any]:
    matches = {name: pattern.search(stdout) for name, pattern in LOG_PATTERNS.items()}
    missing = [name for name, match in matches.items() if match is None]
    require(not missing, f"tracer stdout is missing parseable fields: {missing}")
    resume = matches["resume"]
    generation = matches["generation"]
    area = matches["area"]
    assert resume is not None and generation is not None and area is not None
    selected_lines = [
        line.strip()
        for line in stdout.splitlines()
        if (
            "resume_growth initialized" in line
            or "gen 0 processing" in line
            or "area est:" in line
        )
    ]
    return {
        "resume_initialized_points": int(resume.group(1)),
        "resume_fringe_points": int(resume.group(2)),
        "used_area": {
            "width": int(resume.group(3)),
            "height": int(resume.group(4)),
            "x": int(resume.group(5)),
            "y": int(resume.group(6)),
        },
        "generation_zero_processed_fringe": int(generation.group(1)),
        "generation_zero_area_vx2": int(generation.group(2)),
        "final_area_estimate_vx2": int(area.group(1)),
        "final_area_estimate_cm2": float(area.group(2)),
        "selected_lines": selected_lines[:12],
    }


def locate_trace_directory(target: Path) -> Path:
    matches = sorted(target.glob("auto_trace_*"))
    require(
        len(matches) == 1,
        f"{target}: expected one auto_trace output, found {len(matches)}",
    )
    return matches[0]


def load_surface(directory: Path) -> Surface:
    np, tifffile = dependencies()
    for axis in ("x", "y", "z"):
        require_regular_nonempty_file(directory / f"{axis}.tif")
    arrays = {
        axis: tifffile.imread(directory / f"{axis}.tif")
        for axis in ("x", "y", "z")
    }
    shape = arrays["x"].shape
    require(
        len(shape) == 2
        and 0 < shape[0] <= 256
        and 0 < shape[1] <= 256,
        f"{directory}: output shape {shape} is not within 1..256 per axis",
    )
    require(
        arrays["y"].shape == shape and arrays["z"].shape == shape,
        f"{directory}: XYZ shapes disagree",
    )
    for axis, array in arrays.items():
        require(
            str(array.dtype) == "float32",
            f"{directory}: {axis}.tif dtype {array.dtype} != float32",
        )
        require(
            bool(np.all(np.isfinite(array))),
            f"{directory}: {axis}.tif contains non-finite values",
        )
    masks = {axis: array != -1 for axis, array in arrays.items()}
    valid = masks["x"]
    for axis, array in arrays.items():
        require(
            bool(np.all(np.isfinite(array[valid]))),
            f"{directory}: {axis}.tif has non-finite values at x-valid vertices",
        )
    require_regular_nonempty_file(directory / "meta.json")
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    require(isinstance(meta, dict), f"{directory}: meta.json is not an object")
    require(meta.get("format") == "tifxyz", f"{directory}: output format is not tifxyz")
    require(
        meta.get("source") == "vc_grow_seg_from_segments",
        f"{directory}: unexpected output producer {meta.get('source')!r}",
    )
    offset = meta.get("grid_offset")
    require(
        isinstance(offset, list)
        and len(offset) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in offset),
        f"{directory}: grid_offset is not [col, row] integer metadata",
    )
    return Surface(
        directory=directory,
        x=arrays["x"],
        y=arrays["y"],
        z=arrays["z"],
        valid=valid,
        grid_offset_col_row=(offset[0], offset[1]),
        meta=meta,
    )


def surface_observations(surface: Surface) -> dict[str, Any]:
    np, _ = dependencies()
    quads = valid_quad_mask(surface.valid)
    packed_mask = np.packbits(surface.valid, bitorder="little").tobytes()
    mask_header = json.dumps(
        {
            "shape_rows_cols": list(surface.shape),
            "grid_offset_col_row": list(surface.grid_offset_col_row),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    array_hashes = {
        axis: hashlib.sha256(
            np.ascontiguousarray(getattr(surface, axis)).tobytes()
        ).hexdigest()
        for axis in ("x", "y", "z")
    }
    masks = {
        axis: getattr(surface, axis) != -1 for axis in ("x", "y", "z")
    }
    return {
        "shape_rows_cols": list(surface.shape),
        "grid_offset_col_row": list(surface.grid_offset_col_row),
        "valid_vertices": int(np.count_nonzero(surface.valid)),
        "valid_quads": int(np.count_nonzero(quads)),
        "valid_mask_sha256": hashlib.sha256(mask_header + packed_mask).hexdigest(),
        "array_sha256": array_hashes,
        "sentinel_mask_disagreement": {
            "x_vs_y": int(np.count_nonzero(masks["x"] != masks["y"])),
            "x_vs_z": int(np.count_nonzero(masks["x"] != masks["z"])),
        },
        "meta_observations": {
            key: surface.meta.get(key)
            for key in ("area_vx2", "area_cm2", "grid_offset")
        },
    }


def output_file_hashes(trace: Path) -> dict[str, str]:
    names = ("x.tif", "y.tif", "z.tif", "generations.tif")
    for name in names:
        require_regular_nonempty_file(trace / name)
    return {name: sha256_file(trace / name) for name in names}


def validate_trace_integrity(
    trace: Path,
    surface: Surface,
    log_observations: dict[str, Any],
) -> dict[str, Any]:
    np, tifffile = dependencies()
    require_regular_nonempty_file(trace / "generations.tif")
    generations = tifffile.imread(trace / "generations.tif")
    require(
        generations.shape == surface.x.shape,
        f"{trace}: generations.tif shape differs from XYZ",
    )
    require(
        str(generations.dtype) == "uint16",
        f"{trace}: generations.tif dtype {generations.dtype} != uint16",
    )
    area_vx2 = surface.meta.get("area_vx2")
    area_cm2 = surface.meta.get("area_cm2")
    for name, value in (("area_vx2", area_vx2), ("area_cm2", area_cm2)):
        require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0,
            f"{trace}: metadata {name} is not a finite non-negative number",
        )
    require(
        math.isclose(
            float(area_vx2),
            float(log_observations["final_area_estimate_vx2"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        f"{trace}: metadata area_vx2 disagrees with the parsed final log area",
    )
    return {
        "xyz_dtype": "float32",
        "generations_dtype": "uint16",
        "generations_shape_rows_cols": [int(item) for item in generations.shape],
        "valid_xyz_all_finite": True,
        "metadata_area_is_finite_nonnegative": True,
        "metadata_area_vx2_matches_final_log": True,
        "generation_min": int(np.min(generations)) if generations.size else None,
        "generation_max": int(np.max(generations)) if generations.size else None,
    }


def determinism_key(record: dict[str, Any]) -> str:
    stable = {
        "output_file_sha256": record["output_file_sha256"],
        "surface": record["surface"],
        "log_observations": record["log"]["observations"],
    }
    return sha256_json(stable)


def run_replicates(
    variant: str,
    dispatch: Path,
    source: Path,
    volume: Path,
    params: Path,
    run_root: Path,
) -> tuple[list[dict[str, Any]], Surface]:
    records: list[dict[str, Any]] = []
    first_surface: Surface | None = None
    for replicate in range(1, 4):
        replicate_root = run_root / variant / f"replicate-{replicate}"
        target = replicate_root / "output"
        target.mkdir(parents=True, exist_ok=False)
        (
            replicate_source,
            replicate_volume,
            replicate_params,
            fixture_before,
        ) = copy_fixture_for_replicate(
            source,
            volume,
            params,
            replicate_root / "fixture",
        )
        command = [
            str(dispatch),
            "--volume",
            str(replicate_volume),
            "--src-dir",
            str(replicate_source.parent),
            "--target-dir",
            str(target),
            "--params",
            str(replicate_params),
            "--src-segment",
            str(replicate_source),
        ]
        env = minimal_child_environment(
            replicate_root / "isolated-home",
            replicate_root / "isolated-tmp",
            {"OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE"},
        )
        started = time.monotonic()
        try:
            result = subprocess.run(
                command,
                cwd=replicate_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            fixture_after_timeout = fixture_hashes(
                replicate_source, replicate_volume, replicate_params
            )
            raise EvidenceError(
                f"{variant} replicate {replicate} timed out; "
                f"fixture_unchanged={fixture_after_timeout == fixture_before}"
            ) from exc
        duration = time.monotonic() - started
        fixture_after = fixture_hashes(
            replicate_source, replicate_volume, replicate_params
        )
        if result.returncode != 0:
            raise EvidenceError(
                f"{variant} replicate {replicate} failed ({result.returncode})\n"
                f"fixture_unchanged={fixture_after == fixture_before}\n"
                f"stdout tail:\n{tail_text(result.stdout)}\n"
                f"stderr tail:\n{tail_text(result.stderr)}"
            )
        require(
            fixture_after == fixture_before,
            f"{variant} replicate {replicate}: fixture mutated during execution",
        )
        trace = locate_trace_directory(target)
        log_observations = parse_trace_log(result.stdout)
        surface = load_surface(trace)
        record = {
            "replicate": replicate,
            "duration_seconds": round(duration, 6),
            "environment": {
                "OMP_NUM_THREADS": "1",
                "OMP_DYNAMIC": "FALSE",
            },
            "fixture_integrity": {
                "before": fixture_before,
                "after": fixture_after,
                "unchanged": True,
                "isolated_copy": True,
            },
            "output_file_sha256": output_file_hashes(trace),
            "surface": surface_observations(surface),
            "output_integrity": validate_trace_integrity(
                trace, surface, log_observations
            ),
            "log": {
                "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
                "observations": log_observations,
            },
        }
        records.append(record)
        if first_surface is None:
            first_surface = surface
    keys = [determinism_key(record) for record in records]
    require(
        len(set(keys)) == 1,
        f"{variant}: three replicates are not deterministic: {keys}",
    )
    assert first_surface is not None
    return records, first_surface


def global_values(surface: Surface) -> dict[tuple[int, int], tuple[float, float, float]]:
    np, _ = dependencies()
    values: dict[tuple[int, int], tuple[float, float, float]] = {}
    for row, col in np.argwhere(surface.valid):
        row_int, col_int = int(row), int(col)
        values[global_row_col(row_int, col_int, surface.grid_offset_col_row)] = (
            float(surface.x[row_int, col_int]),
            float(surface.y[row_int, col_int]),
            float(surface.z[row_int, col_int]),
        )
    return values


def global_row_col(
    local_row: int,
    local_col: int,
    grid_offset_col_row: tuple[int, int],
) -> tuple[int, int]:
    col_offset, row_offset = grid_offset_col_row
    # grid_offset records where source [0, 0] appears in the saved local grid:
    # local = source + offset. Convert saved indices back to source coordinates.
    return local_row - row_offset, local_col - col_offset


def sampled_locations(locations: Iterable[tuple[int, int]]) -> dict[str, Any]:
    ordered = sorted(locations)
    return {
        "count": len(ordered),
        "locations_row_col": [list(item) for item in ordered[:64]],
        "locations_truncated": len(ordered) > 64,
        "locations_sha256": sha256_json([list(item) for item in ordered]),
    }


def json_float(value: float) -> float | str:
    if value != value:
        return "nan"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return value


def sampled_xyz(
    values: dict[tuple[int, int], tuple[float, float, float]],
    locations: Iterable[tuple[int, int]],
) -> dict[str, Any]:
    ordered = sorted(locations)
    samples = [
        {
            "location_row_col": list(location),
            "xyz": [json_float(value) for value in values[location]],
        }
        for location in ordered[:64]
    ]
    return {
        "count": len(ordered),
        "samples": samples,
        "samples_truncated": len(ordered) > 64,
        "full_values_sha256": sha256_json(
            [
                {
                    "location_row_col": list(location),
                    "xyz": [json_float(value) for value in values[location]],
                }
                for location in ordered
            ]
        ),
    }


def compare_surfaces(
    before_name: str,
    before: Surface,
    after_name: str,
    after: Surface,
) -> dict[str, Any]:
    np, _ = dependencies()
    before_values = global_values(before)
    after_values = global_values(after)
    before_locations = set(before_values)
    after_locations = set(after_values)
    common = sorted(before_locations & after_locations)
    before_only = before_locations - after_locations
    after_only = after_locations - before_locations

    if common:
        before_xyz = np.asarray([before_values[item] for item in common], dtype=np.float64)
        after_xyz = np.asarray([after_values[item] for item in common], dtype=np.float64)
        delta = after_xyz - before_xyz
        changed_mask = np.any(before_xyz != after_xyz, axis=1)
        displacement = np.linalg.norm(delta, axis=1)
        finite_mask = np.isfinite(displacement)
        changed_finite = changed_mask & finite_mask
        changed_count = int(np.count_nonzero(changed_mask))
        nonfinite_count = int(np.count_nonzero(changed_mask & ~finite_mask))
        if np.any(changed_finite):
            changed_displacement = displacement[changed_finite]
            finite_indices = np.flatnonzero(changed_finite)
            maximum_relative_index = int(np.argmax(changed_displacement))
            maximum_index = int(finite_indices[maximum_relative_index])
            mean_changed = float(np.mean(changed_displacement))
            maximum = float(changed_displacement[maximum_relative_index])
            maximum_location: list[int] | None = list(common[maximum_index])
        else:
            mean_changed = 0.0
            maximum = 0.0
            maximum_location = None
        all_finite = displacement[finite_mask]
        mean_all_common = float(np.mean(all_finite)) if all_finite.size else 0.0
    else:
        changed_count = 0
        nonfinite_count = 0
        mean_changed = 0.0
        maximum = 0.0
        maximum_location = None
        mean_all_common = 0.0

    return {
        "before": before_name,
        "after": after_name,
        "alignment": {
            "coordinate_system": "global [row, col]",
            "rule": (
                "source_col=local_col-grid_offset[0]; "
                "source_row=local_row-grid_offset[1]"
            ),
            "before_grid_offset_col_row": list(before.grid_offset_col_row),
            "after_grid_offset_col_row": list(after.grid_offset_col_row),
        },
        "valid_vertices": {
            "before": len(before_locations),
            "after": len(after_locations),
            "common": len(common),
        },
        "valid_quads": {
            "before": surface_observations(before)["valid_quads"],
            "after": surface_observations(after)["valid_quads"],
        },
        "valid_mask_symmetric_difference": {
            "count": len(before_only) + len(after_only),
            "before_only": {
                **sampled_locations(before_only),
                "xyz": sampled_xyz(before_values, before_only),
            },
            "after_only": {
                **sampled_locations(after_only),
                "xyz": sampled_xyz(after_values, after_only),
            },
        },
        "common_valid_xyz_displacement": {
            "changed_vertices": changed_count,
            "nonfinite_changed_displacements": nonfinite_count,
            "mean_over_changed_finite": mean_changed,
            "mean_over_all_common_finite": mean_all_common,
            "maximum_finite": maximum,
            "maximum_location_row_col": maximum_location,
        },
    }


def compact_variant_summary(
    records: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    return {
        variant: {
            "deterministic_replicates": 3,
            "surface": variant_records[0]["surface"],
            "log_observations": variant_records[0]["log"]["observations"],
        }
        for variant, variant_records in records.items()
    }


def write_evidence_index(output: Path) -> None:
    files = sorted(path for path in output.glob("*.json") if path.name != "index.json")
    write_json(
        output / "index.json",
        {
            "schema_version": SCHEMA_VERSION,
            "files": {
                path.name: {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in files
            },
        },
    )


def evaluator_provenance(
    workflow_file: Path,
    repository: str,
    token: str | None,
) -> dict[str, Any]:
    runner_file = Path(__file__).resolve()
    workflow_file = workflow_file.resolve()
    require(workflow_file.is_file(), f"workflow file does not exist: {workflow_file}")
    environment_fields = (
        "EVIDENCE_RUN_ID",
        "EVIDENCE_RUN_ATTEMPT",
        "EVIDENCE_RUN_SHA",
        "EVIDENCE_RUN_REF",
        "RUNNER_OS_NAME",
        "RUNNER_ARCH_NAME",
        "RUNNER_IMAGE_OS",
        "RUNNER_IMAGE_VERSION",
    )
    confirmation_environment = {
        key.lower(): os.environ.get(key) for key in environment_fields
    }
    run_id_text = confirmation_environment["evidence_run_id"]
    run_attempt_text = confirmation_environment["evidence_run_attempt"]
    run_sha = confirmation_environment["evidence_run_sha"]
    require(
        isinstance(run_id_text, str) and run_id_text.isdigit(),
        "evaluator run ID is missing",
    )
    require(
        isinstance(run_attempt_text, str) and run_attempt_text.isdigit(),
        "evaluator run attempt is missing",
    )
    require(
        isinstance(run_sha, str)
        and re.fullmatch(r"[0-9a-f]{40}", run_sha) is not None,
        "evaluator run SHA is missing",
    )
    run = github_api_json(
        f"https://api.github.com/repos/{repository}/actions/runs/{run_id_text}",
        token,
    )
    require(
        str((run.get("repository") or {}).get("full_name", "")).lower()
        == repository.lower(),
        "evaluator run repository mismatch",
    )
    require(
        run.get("event") == "workflow_dispatch"
        and run.get("path") == ".github/workflows/vc3d-trace-confirm.yml"
        and run.get("head_sha") == run_sha
        and run.get("run_attempt") == int(run_attempt_text),
        "evaluator run does not match its workflow/SHA/attempt environment",
    )
    return {
        "runner": {
            "filename": runner_file.name,
            "sha256": sha256_file(runner_file),
        },
        "workflow": {
            "filename": workflow_file.name,
            "sha256": sha256_file(workflow_file),
        },
        "confirmation_run": confirmation_environment
        | {
            "repository": repository,
            "event": run.get("event"),
            "workflow_id": run.get("workflow_id"),
            "workflow_path": run.get("path"),
            "head_branch": run.get("head_branch"),
            "status_during_evaluation": run.get("status"),
            "html_url": run.get("html_url"),
            "api_provenance_checked": True,
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
        },
    }


def execute(args: argparse.Namespace) -> None:
    artifact_root = args.artifact_root.resolve()
    require(
        not args.output.is_symlink(),
        f"refusing symlink evidence directory {args.output}",
    )
    output = args.output.resolve()
    require(artifact_root.is_dir(), f"artifact root does not exist: {artifact_root}")
    if output.exists():
        require(
            output.is_dir() and not any(output.iterdir()),
            f"refusing non-empty evidence directory {output}",
        )
    output.mkdir(parents=True, exist_ok=True)

    token = os.environ.pop("GITHUB_TOKEN", None)
    require(bool(token), "GITHUB_TOKEN is required for source build-log provenance")
    provenance = check_run_provenance(args.repository, token)
    source_comparisons = check_source_comparisons(args.repository, token)
    evaluator = evaluator_provenance(
        args.workflow_file, args.repository, token
    )
    token = None
    purged_credentials = purge_runner_credentials()
    host_runtime_packages = installed_host_package_versions()
    archive_verification = load_archive_verification(artifact_root)
    appimages = find_and_validate_appimages(
        artifact_root, archive_verification
    )

    with tempfile.TemporaryDirectory(prefix="vc3d-trace-confirm-") as temporary:
        work = Path(temporary)
        source, volume, params, fixture_files = create_fixture(work)
        preregistration = preregister_fixture(source)
        make_fixture_read_only(source, volume, params)
        write_json(output / "preregistration.json", preregistration)

        extraction_root = work / "extracted"
        extraction_root.mkdir()
        extraction_records: dict[str, dict[str, Any]] = {}
        run_root = work / "runs"
        run_records: dict[str, list[dict[str, Any]]] = {}
        surfaces: dict[str, Surface] = {}
        for variant in VARIANTS:
            dispatch, extraction_records[variant] = extract_appimage_once(
                variant, appimages[variant]["path"], extraction_root
            )
            run_records[variant], surfaces[variant] = run_replicates(
                variant,
                dispatch,
                source,
                volume,
                params,
                run_root,
            )
            shutil.rmtree(extraction_root / variant)

        comparisons = {
            f"{before}_to_{after}": compare_surfaces(
                before, surfaces[before], after, surfaces[after]
            )
            for before, after in COMPARISON_PAIRS
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "trace_confirmation_complete",
                "interpretation_guard": (
                    "Cross-variant outcomes are recorded, not asserted. Passing means "
                "the artifacts/input were pinned and each variant's TIFF arrays, "
                "selected metadata, and parsed scientific log observations were "
                "deterministic; generated UUID/path metadata is excluded."
            ),
            "runtime": {
                "python": sys.version,
                "numpy": dependencies()[0].__version__,
                "tifffile": dependencies()[1].__version__,
                "host_runtime_packages": host_runtime_packages,
                "secret_free_child_environment_keys": list(
                    CHILD_ENVIRONMENT_KEYS
                ),
                "inherited_parent_environment": False,
                "credential_variable_names_purged": purged_credentials,
            },
            "evaluator": evaluator,
            "repository": args.repository,
            "runs": provenance,
            "source_comparisons": source_comparisons,
            "appimages": {
                variant: {
                    key: value
                    for key, value in record.items()
                    if key != "path"
                }
                | extraction_records[variant]
                for variant, record in appimages.items()
            },
            "fixture_files": fixture_files,
            "archive_verification": archive_verification,
            "preregistration_sha256": sha256_file(
                output / "preregistration.json"
            ),
            "variants": run_records,
            "comparisons": comparisons,
        }
        write_json(output / "manifest.json", manifest)
        write_json(
            output / "summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "trace_confirmation_complete",
                "preregistered_boundary_cells_row_col": BOUNDARY_CELLS_ROW_COL,
                "provenance": provenance,
                "source_comparisons": source_comparisons,
                "evaluator": evaluator,
                "variants": compact_variant_summary(run_records),
                "comparisons": comparisons,
                "assertions": {
                    "run_api_provenance_checked": True,
                    "build_logs_and_toolchain_checked": True,
                    "source_ablation_changed_files_checked": True,
                    "artifact_id_and_archive_sha256_checked": True,
                    "appimage_sha256_linked_to_verified_archive": True,
                    "appimage_exact_filenames_checked": True,
                    "public_source_hashes_checked": True,
                    "isolated_fixture_hashes_checked_before_and_after": True,
                    "native_subprocesses_received_secret_free_environment": True,
                    "evaluator_runner_and_workflow_hashes_recorded": True,
                    "fixture_boundary_cells_preregistered": 6,
                    "replicates_per_variant": 3,
                    "scientifically_relevant_outputs_deterministic": True,
                    "cross_variant_expected_outcome_asserted": False,
                },
            },
        )
    write_evidence_index(output)
    print(output / "summary.json")


def self_test() -> None:
    example_log = (
        "resume_growth initialized 15702 low-res points, fringe 12 "
        "used_area [159 x 135 from (0, 0)]\n"
        "gen 0 processing 12 fringe cands done area 15702 vx^2\n"
        "area est: 15702 vx^2 (0.981 cm^2)\n"
    )
    parsed = parse_trace_log(example_log)
    require(parsed["resume_initialized_points"] == 15702, "log parser count")
    require(parsed["used_area"] == {"width": 159, "height": 135, "x": 0, "y": 0},
            "log parser used_area")
    require(parsed["final_area_estimate_cm2"] == 0.981, "log parser float")
    require(len(VARIANTS) == 4, "variant count")
    require(len(COMPARISON_PAIRS) == 4, "comparison count")
    require(len(SOURCE_COMPARISONS) == 3, "source comparison count")
    require(len(BOUNDARY_CELLS_ROW_COL) == 6, "preregistered boundary count")
    require(
        all(len(value["source_sha"]) == 40 for value in VARIANTS.values()),
        "expected full source SHA length",
    )
    require(
        APPIMAGE_NAME_RE.fullmatch(
            "VC3D-208faea-2026-07-28-linux-x86_64.AppImage"
        )
        is not None,
        "AppImage filename parser",
    )
    require(
        global_row_col(5, 7, (-2, 11)) == (-6, 9),
        "grid_offset [col, row] alignment",
    )
    fixture_text_hashes = {
        "volume_meta": hashlib.sha256(FAKE_VOLUME_META_TEXT.encode()).hexdigest(),
        "volume_zarray": hashlib.sha256(
            FAKE_VOLUME_ZARRAY_TEXT.encode()
        ).hexdigest(),
        "params": hashlib.sha256(TRACE_PARAMS_TEXT.encode()).hexdigest(),
    }
    require(
        fixture_text_hashes
        == {
            "volume_meta": (
                "55039b5c8430c34d579b1625fa94f70a1ed8f502420c21971e634e522273d732"
            ),
            "volume_zarray": (
                "a924d479b459f4b10de540d9e4b7967f2b95cb8f6bcdfd54c9a1b474fbea7f38"
            ),
            "params": (
                "5606943e4f278ad8e2ec9f5522d2d0cad6914ba908f829464594e564ddd31093"
            ),
        },
        "exact fake Zarr and trace-parameter hashes",
    )
    print("self-test: ok")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="directory containing baseline/helper/full/pr artifact downloads",
    )
    parser.add_argument(
        "--output", type=Path, help="new or empty directory for compact JSON evidence"
    )
    parser.add_argument(
        "--workflow-file",
        type=Path,
        help="workflow YAML whose SHA-256 is recorded with the evaluator",
    )
    parser.add_argument(
        "--repository",
        default=SOURCE_REPOSITORY,
        help=f"repository owning the four runs (default: {SOURCE_REPOSITORY})",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="run dependency-free parser checks"
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    if (
        args.artifact_root is None
        or args.output is None
        or args.workflow_file is None
    ):
        parser.error("--artifact-root, --output, and --workflow-file are required")
    require(
        re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository) is not None,
        f"invalid GitHub repository {args.repository!r}",
    )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        self_test()
        return 0
    assert (
        args.artifact_root is not None
        and args.output is not None
        and args.workflow_file is not None
    )
    output_can_receive_failure = (
        not args.output.exists()
        and not args.output.is_symlink()
    ) or (
        args.output.is_dir()
        and not args.output.is_symlink()
        and not any(args.output.iterdir())
    )
    try:
        execute(args)
    except Exception as exc:
        if output_can_receive_failure:
            args.output.mkdir(parents=True, exist_ok=True)
            write_json(
                args.output / "failure.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "trace_confirmation_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
