#!/usr/bin/env python3
"""Execute synthetic TIFXYZ cases through Villa Python and C++ readers.

The Python path loads the exact ``reader.py`` and ``types.py`` files from a
local Villa checkout without importing the rest of the monorepo. The C++ path
uses Villa's packaged ``vc_tifxyz2obj`` executable, which calls
``load_quad_from_tifxyz`` before writing an OBJ. The converter emits only
vertices used by retained faces, so its observable vertex/face signature is
calibrated by the two control cases instead of being mislabeled as a raw
valid-grid count.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PINNED_VILLA_COMMIT = "1162bcab4bc769b12993fc320c69c14fdb4a4fa5"
TESTED_CPP_RELEASE_COMMIT = "05ff9ea39bf1077f8749fc1fe06f43a548350143"
TESTED_CPP_BINARY_SHA256 = (
    "71a32235924499dee93bd24944794c930dffdee8e95450220e2665bd2020e118"
)
TESTED_CPP_ASSET_SHA256 = "d5fca566c8bc843f2350ab60e86e0def25ab88240e3f0ffcbd04662ba522c9a8"
TESTED_CPP_CONVERTER_BLOB_SHA = "eff32a4d0984bd6614308a4484f8dd70cb699d1a"
TESTED_CPP_GEOMETRY_BLOB_SHA = "edfec27becdda604281ad3d2feae26630d4ea62a"


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    z_center: float = 10.0
    mask_kind: str = "absent"
    mask_center_value: int = 255
    expected_python_vertices: int = 25
    expected_doctor_cpp_vertices: int = 25
    expected_cpp_obj_vertices: int = 16
    expected_cpp_obj_faces: int = 18


CASES = (
    Case(
        name="no-mask-all-positive",
        description="Control: no mask and every z value is positive.",
    ),
    Case(
        name="no-mask-nonpositive-z",
        description="Control: without a mask, both readers reject the center z <= 0.",
        z_center=-2.0,
        expected_python_vertices=24,
        expected_doctor_cpp_vertices=24,
        expected_cpp_obj_vertices=12,
        expected_cpp_obj_faces=10,
    ),
    Case(
        name="exact-mask-low-nonzero",
        description="Exact uint8 mask value 1 at center; Python keeps it and C++ rejects it.",
        mask_kind="exact",
        mask_center_value=1,
        expected_python_vertices=25,
        expected_doctor_cpp_vertices=24,
        expected_cpp_obj_vertices=12,
        expected_cpp_obj_faces=10,
    ),
    Case(
        name="exact-mask-revalidates-nonpositive-z",
        description=(
            "Exact all-255 mask with center z <= 0; Python keeps it while the C++ "
            "loader's earlier z gate rejects it."
        ),
        z_center=-2.0,
        mask_kind="exact",
        expected_python_vertices=25,
        expected_doctor_cpp_vertices=24,
        expected_cpp_obj_vertices=12,
        expected_cpp_obj_faces=10,
    ),
    Case(
        name="integer-multiple-mask",
        description=(
            "A 2x higher-resolution mask has one zero sample over the center; "
            "Python ignores the mismatched mask while C++ applies it."
        ),
        mask_kind="double",
        mask_center_value=0,
        expected_python_vertices=25,
        expected_doctor_cpp_vertices=24,
        expected_cpp_obj_vertices=12,
        expected_cpp_obj_faces=10,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_pinned_reader(villa_root: Path) -> tuple[type[Any], dict[str, str]]:
    source_dir = villa_root / "vesuvius" / "src" / "vesuvius" / "tifxyz"
    types_path = source_dir / "types.py"
    reader_path = source_dir / "reader.py"
    for path in (types_path, reader_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    package_name = "_tifxyz_doctor_pinned_villa"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source_dir)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    # The selected reader path does not call OpenCV unless optional label-shape
    # validation is requested. Supplying this narrow stub avoids installing an
    # unrelated GUI/image dependency while executing the real pinned modules.
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.IMREAD_UNCHANGED = -1  # type: ignore[attr-defined]

    def _unexpected_imread(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("The differential harness does not request label image reads")

    cv2_stub.imread = _unexpected_imread  # type: ignore[attr-defined]
    sys.modules.setdefault("cv2", cv2_stub)

    _load_module(f"{package_name}.types", types_path)
    reader_module = _load_module(f"{package_name}.reader", reader_path)
    return reader_module.TifxyzReader, {
        "reader_py_sha256": _sha256(reader_path),
        "types_py_sha256": _sha256(types_path),
    }


def _write_tiff(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values).save(path, format="TIFF")


def _write_case(root: Path, case: Case) -> Path:
    surface = root / case.name
    surface.mkdir()
    rows, cols = np.indices((5, 5), dtype=np.float32)
    x = cols.astype(np.float32)
    y = rows.astype(np.float32)
    z = np.full((5, 5), 10.0, dtype=np.float32)
    z[2, 2] = np.float32(case.z_center)

    for filename, values in (("x.tif", x), ("y.tif", y), ("z.tif", z)):
        _write_tiff(surface / filename, values)

    if case.mask_kind == "exact":
        mask = np.full((5, 5), 255, dtype=np.uint8)
        mask[2, 2] = np.uint8(case.mask_center_value)
        _write_tiff(surface / "mask.tif", mask)
    elif case.mask_kind == "double":
        mask = np.full((10, 10), 255, dtype=np.uint8)
        # Every 2x2 block maps to one stored vertex in the C++ path. One bad
        # sample is sufficient to invalidate the covered center vertex.
        mask[4, 4] = np.uint8(case.mask_center_value)
        _write_tiff(surface / "mask.tif", mask)
    elif case.mask_kind != "absent":
        raise ValueError(f"Unknown mask kind: {case.mask_kind}")

    metadata = {
        "uuid": case.name,
        "format": "tifxyz",
        "type": "seg",
        "scale": [1.0, 1.0],
        "bbox": [[0.0, 0.0, float(np.min(z))], [4.0, 4.0, 10.0]],
    }
    (surface / "meta.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return surface


def _write_all_valid_grid(root: Path, height: int, width: int) -> Path:
    surface = root / f"exporter-probe-{height}x{width}"
    surface.mkdir()
    rows, cols = np.indices((height, width), dtype=np.float32)
    x = cols.astype(np.float32)
    y = rows.astype(np.float32)
    z = np.full((height, width), 10.0, dtype=np.float32)
    for filename, values in (("x.tif", x), ("y.tif", y), ("z.tif", z)):
        _write_tiff(surface / filename, values)
    metadata = {
        "uuid": surface.name,
        "format": "tifxyz",
        "type": "seg",
        "scale": [1.0, 1.0],
        "bbox": [
            [0.0, 0.0, 10.0],
            [float(width - 1), float(height - 1), 10.0],
        ],
    }
    (surface / "meta.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return surface


def _python_vertex_count(reader_type: type[Any], surface: Path) -> int:
    loaded = reader_type(surface).read(
        load_mask=True,
        validate=True,
        discover_label_shapes=False,
    )
    return int(np.asarray(loaded._valid_mask, dtype=bool).sum())


def _obj_counts(cpp_binary: Path, surface: Path, output: Path) -> dict[str, int]:
    process = subprocess.run(
        [str(cpp_binary), str(surface), str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"C++ converter failed for {surface.name} with {process.returncode}: "
            f"{process.stderr.strip() or process.stdout.strip()}"
        )

    dims = re.search(
        r"Point dims:\s*\[(\d+)\s*x\s*(\d+)\]\s*"
        r"cols:\s*(\d+)\s*rows:\s*(\d+)",
        process.stdout,
    )
    if dims is None:
        raise RuntimeError(
            f"C++ converter did not report its loaded grid dimensions for {surface.name}"
        )
    width, height, cols, rows = (int(value) for value in dims.groups())
    if (width, height) != (cols, rows):
        raise RuntimeError(
            "C++ converter reported inconsistent point dimensions: "
            f"[{width} x {height}], cols={cols}, rows={rows}"
        )

    counts = {
        "vertices": 0,
        "uvs": 0,
        "faces": 0,
        "loaded_rows": rows,
        "loaded_cols": cols,
    }
    with output.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                counts["vertices"] += 1
            elif line.startswith("vt "):
                counts["uvs"] += 1
            elif line.startswith("f "):
                counts["faces"] += 1
    return counts


def _git_head(villa_root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=villa_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def _binary_references_cpp_loader(cpp_binary: Path) -> bool:
    """Confirm that the packaged converter links Villa's real loader symbol."""
    process = subprocess.run(
        ["nm", "-u", str(cpp_binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return "load_quad_from_tifxyz" in process.stdout


def _doctor_report(surface: Path) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from tifxyz_doctor.integrity import audit_tifxyz_integrity

    return audit_tifxyz_integrity(surface)


def run(
    *,
    villa_root: Path,
    cpp_binary: Path,
    cpp_asset: Path,
    output: Path,
) -> dict[str, Any]:
    villa_head = _git_head(villa_root)
    if villa_head != PINNED_VILLA_COMMIT:
        raise RuntimeError(
            f"Villa checkout is {villa_head}, expected {PINNED_VILLA_COMMIT}"
        )
    if not cpp_binary.is_file():
        raise FileNotFoundError(cpp_binary)
    if not cpp_asset.is_file():
        raise FileNotFoundError(cpp_asset)
    binary_sha = _sha256(cpp_binary)
    if binary_sha != TESTED_CPP_BINARY_SHA256:
        raise RuntimeError(
            f"C++ executable SHA-256 is {binary_sha}, expected "
            f"{TESTED_CPP_BINARY_SHA256}"
        )
    asset_sha = _sha256(cpp_asset)
    if asset_sha != TESTED_CPP_ASSET_SHA256:
        raise RuntimeError(
            f"C++ release asset SHA-256 is {asset_sha}, expected "
            f"{TESTED_CPP_ASSET_SHA256}"
        )
    loader_symbol_present = _binary_references_cpp_loader(cpp_binary)
    if not loader_symbol_present:
        raise RuntimeError(
            f"{cpp_binary} does not reference load_quad_from_tifxyz"
        )

    reader_type, python_sources = _load_pinned_reader(villa_root)
    case_results: list[dict[str, Any]] = []
    exporter_probe_results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="tifxyz-reader-diff-") as temp:
        root = Path(temp)
        for case in CASES:
            surface = _write_case(root, case)
            python_count = _python_vertex_count(reader_type, surface)
            obj_counts = _obj_counts(
                cpp_binary,
                surface,
                root / f"{case.name}.obj",
            )
            doctor = _doctor_report(surface)
            modeled_python = int(doctor["validity"]["python_valid_vertex_count"])
            modeled_cpp = int(doctor["validity"]["cpp_finite_point_vertex_count"])
            passed = (
                python_count == case.expected_python_vertices == modeled_python
                and modeled_cpp == case.expected_doctor_cpp_vertices
                and obj_counts["vertices"] == case.expected_cpp_obj_vertices
                and obj_counts["faces"] == case.expected_cpp_obj_faces
                and obj_counts["loaded_rows"] == 5
                and obj_counts["loaded_cols"] == 5
            )
            case_results.append(
                {
                    "name": case.name,
                    "description": case.description,
                    "mask_kind": case.mask_kind,
                    "z_center": case.z_center,
                    "expected": {
                        "python_valid_vertices": case.expected_python_vertices,
                        "doctor_modeled_cpp_valid_vertices": (
                            case.expected_doctor_cpp_vertices
                        ),
                        "stable_cpp_obj_vertices": case.expected_cpp_obj_vertices,
                        "stable_cpp_obj_faces": case.expected_cpp_obj_faces,
                    },
                    "actual": {
                        "pinned_python_reader_valid_vertices": python_count,
                        "stable_cpp_obj_vertices": obj_counts["vertices"],
                        "stable_cpp_obj_faces": obj_counts["faces"],
                        "stable_cpp_loaded_grid_shape_hw": [
                            obj_counts["loaded_rows"],
                            obj_counts["loaded_cols"],
                        ],
                        "doctor_modeled_python_valid_vertices": modeled_python,
                        "doctor_modeled_cpp_finite_vertices": modeled_cpp,
                    },
                    "passed": passed,
                }
            )

        probe_shapes = [
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
            (3, 4),
            (4, 7),
            (7, 4),
        ]
        for height, width in probe_shapes:
            surface = _write_all_valid_grid(root, height, width)
            obj_counts = _obj_counts(
                cpp_binary,
                surface,
                root / f"exporter-probe-{height}x{width}.obj",
            )
            expected_vertices = (height - 1) * (width - 1)
            expected_faces = 2 * (height - 2) * (width - 2)
            passed = (
                obj_counts["loaded_rows"] == height
                and obj_counts["loaded_cols"] == width
                and obj_counts["vertices"] == expected_vertices
                and obj_counts["faces"] == expected_faces
            )
            exporter_probe_results.append(
                {
                    "input_grid_shape_hw": [height, width],
                    "expected": {
                        "loaded_grid_shape_hw": [height, width],
                        "obj_vertices": expected_vertices,
                        "obj_faces": expected_faces,
                    },
                    "actual": {
                        "loaded_grid_shape_hw": [
                            obj_counts["loaded_rows"],
                            obj_counts["loaded_cols"],
                        ],
                        "obj_vertices": obj_counts["vertices"],
                        "obj_faces": obj_counts["faces"],
                    },
                    "passed": passed,
                }
            )

    all_cases_pass = all(item["passed"] for item in case_results)
    all_exporter_probes_pass = all(item["passed"] for item in exporter_probe_results)
    result = {
        "schema_version": "tifxyz-reader-differential-v1",
        "status": "pass" if all_cases_pass and all_exporter_probes_pass else "fail",
        "villa_python": {
            "commit": villa_head,
            **python_sources,
        },
        "villa_cpp": {
            "release_commit": TESTED_CPP_RELEASE_COMMIT,
            "binary": cpp_binary.name,
            "binary_sha256": binary_sha,
            "load_quad_from_tifxyz_symbol_present": loader_symbol_present,
            "release_asset_sha256": asset_sha,
            "release_source": {
                "verification_scope": (
                    "The runner records the source blob identities used to explain "
                    "the observable. Their equality at the release and pinned "
                    "repository commits was established by a separate read-only "
                    "repository-history check; the runner does not fetch or compare "
                    "GitHub blobs."
                ),
                "converter": {
                    "path": "volume-cartographer/apps/src/vc_tifxyz2obj.cpp",
                    "blob_sha": TESTED_CPP_CONVERTER_BLOB_SHA,
                },
                "geometry": {
                    "path": "volume-cartographer/core/src/Geometry.cpp",
                    "blob_sha": TESTED_CPP_GEOMETRY_BLOB_SHA,
                },
            },
            "relationship_to_pinned_commit": (
                "At the 2026-07-26 audit date, this test paired the official stable "
                "Apple Silicon release with the Python reader at repository HEAD. "
                "The release commit predates the pinned repository revision by 94 "
                "commits, so this is intentionally a stable-release-versus-repository "
                "comparison, not a same-commit build comparison. Intervening path "
                "history shows no Python-reader edit and no change to the relevant "
                "C++ full-load mask rules."
            ),
            "observable": (
                "vc_tifxyz2obj reports that it loaded the full 5x5 grid. At the "
                "tested release, loc_valid uses a half-open rows-2 by cols-2 "
                "rectangle, so only a 3x3 set of quad origins survives and the "
                "converter emits only their 4x4 corner union: 16 OBJ vertices and "
                "18 faces. Rejecting the center removes four of those nine quads, "
                "leaving the calibrated 12-vertex/10-face signature. This is OBJ "
                "boundary filtering, not loader resizing; disputed cases are "
                "compared with the controls rather than treating OBJ counts as raw "
                "valid-grid counts."
            ),
        },
        "cases": case_results,
        "exporter_boundary_probe": {
            "domain": "all-valid grids with height >= 3 and width >= 3",
            "formula": {
                "obj_vertices": "(height - 1) * (width - 1)",
                "obj_faces": "2 * (height - 2) * (width - 2)",
            },
            "observations": exporter_probe_results,
            "status": "pass" if all_exporter_probes_pass else "fail",
        },
        "summary": {
            "case_count": len(case_results),
            "passed": sum(bool(item["passed"]) for item in case_results),
            "failed": sum(not bool(item["passed"]) for item in case_results),
            "exporter_probe_count": len(exporter_probe_results),
            "exporter_probe_passed": sum(
                bool(item["passed"]) for item in exporter_probe_results
            ),
            "exporter_probe_failed": sum(
                not bool(item["passed"]) for item in exporter_probe_results
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--villa-root", type=Path, required=True)
    parser.add_argument("--cpp-binary", type=Path, required=True)
    parser.add_argument("--cpp-asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run(
        villa_root=args.villa_root.resolve(),
        cpp_binary=args.cpp_binary.resolve(),
        cpp_asset=args.cpp_asset.resolve(),
        output=args.output.resolve(),
    )
    print(
        f"{result['status']}: {result['summary']['passed']}/"
        f"{result['summary']['case_count']} cases; "
        f"{result['summary']['exporter_probe_passed']}/"
        f"{result['summary']['exporter_probe_count']} exporter probe sizes"
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
