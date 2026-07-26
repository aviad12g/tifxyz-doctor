from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tifxyz_doctor.integrity import (
    audit_tifxyz_integrity,
    dumps_integrity_report,
)


def _write_tiff(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path, format="TIFF")


def _make_surface(
    root: Path,
    *,
    height: int = 3,
    width: int = 4,
    coordinate_dtype: np.dtype = np.dtype(np.float32),
    mask: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> tuple[Path, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    surface = root / "surface"
    surface.mkdir()
    rows, cols = np.indices((height, width), dtype=np.float32)
    x = cols.astype(coordinate_dtype)
    y = rows.astype(coordinate_dtype)
    z = np.full((height, width), 10, dtype=coordinate_dtype)
    for name, array in (("x.tif", x), ("y.tif", y), ("z.tif", z)):
        _write_tiff(surface / name, array)

    if metadata is None:
        metadata = {
            "uuid": "synthetic",
            "format": "tifxyz",
            "type": "seg",
            "scale": [1.0, 1.0],
            "bbox": [[0.0, 0.0, 10.0], [float(width - 1), float(height - 1), 10.0]],
        }
    (surface / "meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    if mask is not None:
        _write_tiff(surface / "mask.tif", mask)
    return surface, (x, y, z)


def _codes(report: dict[str, object]) -> set[str]:
    return {
        str(finding["code"])
        for finding in report["findings"]  # type: ignore[index]
    }


class IntegrityTests(unittest.TestCase):
    def test_clean_surface_passes_and_has_strict_deterministic_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            surface, _ = _make_surface(Path(temp))

            first = audit_tifxyz_integrity(surface)
            second = audit_tifxyz_integrity(surface)
            encoded = dumps_integrity_report(first)

        self.assertEqual(first, second)
        self.assertEqual(encoded, dumps_integrity_report(second))
        self.assertEqual(json.loads(encoded), first)
        self.assertEqual(first["status"], "pass")
        self.assertEqual(first["summary"], {"errors": 0, "warnings": 0, "info": 0})
        self.assertEqual(first["coordinates"]["shape"], [3, 4])
        self.assertEqual(first["validity"]["portable_valid_vertex_count"], 12)
        self.assertEqual(first["geometry"]["portable_valid_face_count"], 6)
        self.assertTrue(first["geometry"]["bbox_comparison"]["matches"])

    def test_missing_file_is_reported_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            surface, _ = _make_surface(Path(temp))
            (surface / "y.tif").unlink()

            report = audit_tifxyz_integrity(surface)

        self.assertEqual(report["status"], "error")
        self.assertIn("missing-required-file", _codes(report))
        self.assertFalse(report["files"]["required"]["y.tif"])
        self.assertIsNone(report["coordinates"]["shape"])
        json.dumps(report, allow_nan=False)

    def test_malformed_metadata_and_invalid_scale_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            surface, _ = _make_surface(Path(temp))
            (surface / "meta.json").write_text("{broken", encoding="utf-8")
            malformed = audit_tifxyz_integrity(surface)

        self.assertIn("metadata-json-invalid", _codes(malformed))

        with tempfile.TemporaryDirectory() as temp:
            surface, _ = _make_surface(
                Path(temp),
                metadata={
                    "uuid": "",
                    "format": "other",
                    "type": "mesh",
                    "scale": [0, "x"],
                    "bbox": [[2, 0, 0], [1, 1, 1]],
                    "area": -1,
                },
            )
            invalid = audit_tifxyz_integrity(surface)

        self.assertTrue(
            {
                "metadata-uuid-invalid",
                "metadata-format-nonstandard",
                "metadata-type-nonstandard",
                "metadata-scale-invalid",
                "metadata-bbox-invalid",
                "metadata-area-invalid",
            }.issubset(_codes(invalid))
        )
        json.dumps(invalid, allow_nan=False)

    def test_area_aliases_expose_current_python_schema_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            surface, _ = _make_surface(
                Path(temp),
                metadata={
                    "uuid": "area-aliases",
                    "format": "tifxyz",
                    "type": "seg",
                    "scale": [1.0, 1.0],
                    "bbox": [[0.0, 0.0, 10.0], [3.0, 2.0, 10.0]],
                    "area_vx2": 42.5,
                    "area_cm2": 0.00425,
                },
            )

            report = audit_tifxyz_integrity(surface)

        self.assertEqual(report["status"], "warning")
        self.assertEqual(report["metadata"]["area_vx2"], 42.5)
        self.assertEqual(report["metadata"]["area_cm2"], 0.00425)
        self.assertIsNone(report["metadata"]["area"])
        self.assertIn("metadata-area-schema-divergence", _codes(report))

    def test_non_float_coordinates_are_decoded_but_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            surface, _ = _make_surface(
                Path(temp),
                coordinate_dtype=np.dtype(np.uint16),
            )
            report = audit_tifxyz_integrity(surface)

        findings = [
            item
            for item in report["findings"]
            if item["code"] == "coordinate-dtype-nonstandard"
        ]
        self.assertEqual(len(findings), 3)
        self.assertEqual(report["status"], "warning")
        self.assertEqual(report["validity"]["portable_valid_vertex_count"], 12)

    def test_coordinate_shape_mismatch_stops_dependent_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            surface, _ = _make_surface(Path(temp))
            _write_tiff(surface / "y.tif", np.zeros((2, 4), dtype=np.float32))

            report = audit_tifxyz_integrity(surface)

        self.assertIn("coordinate-shape-mismatch", _codes(report))
        self.assertIsNone(report["coordinates"]["shape"])
        self.assertIsNone(report["geometry"]["portable_valid_face_count"])

    def test_corrupt_tiff_is_reported_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            surface, _ = _make_surface(Path(temp))
            (surface / "x.tif").write_bytes(b"not a tiff")

            report = audit_tifxyz_integrity(surface)

        self.assertIn("tiff-unreadable", _codes(report))
        self.assertFalse(report["tiffs"]["x"]["readable"])
        self.assertIsNone(report["validity"]["portable_valid_vertex_count"])

    def test_partial_sentinel_and_nonfinite_coordinate_are_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            surface, (x, y, z) = _make_surface(Path(temp))
            x[0, 0] = -1
            x[0, 1] = np.nan
            _write_tiff(surface / "x.tif", x)

            report = audit_tifxyz_integrity(surface)

        self.assertEqual(report["coordinates"]["partial_sentinel_vertex_count"], 1)
        self.assertEqual(report["coordinates"]["nonfinite_vertex_count"], 1)
        self.assertIn("partial-sentinel", _codes(report))
        self.assertIn("nonfinite-coordinate", _codes(report))
        self.assertEqual(
            report["validity"]["python_cpp_finite_point_disagreement_count"],
            2,
        )
        json.dumps(report, allow_nan=False)

    def test_full_sentinel_is_counted_and_removes_all_adjacent_faces(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            surface, (x, y, z) = _make_surface(Path(temp), height=2, width=2)
            x[0, 0] = -1
            y[0, 0] = -1
            z[0, 0] = -1
            _write_tiff(surface / "x.tif", x)
            _write_tiff(surface / "y.tif", y)
            _write_tiff(surface / "z.tif", z)

            report = audit_tifxyz_integrity(surface)

        self.assertEqual(report["coordinates"]["canonical_sentinel_vertex_count"], 1)
        self.assertEqual(report["coordinates"]["partial_sentinel_vertex_count"], 0)
        self.assertEqual(report["geometry"]["portable_valid_face_count"], 0)
        self.assertIn("no-valid-face", _codes(report))

    def test_low_nonzero_mask_and_mask_override_expose_reader_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            mask = np.full((3, 4), 255, dtype=np.uint8)
            mask[0, 0] = 1
            surface, (_, _, z) = _make_surface(Path(temp), mask=mask)
            z[0, 1] = -2
            _write_tiff(surface / "z.tif", z)

            report = audit_tifxyz_integrity(surface)

        self.assertEqual(report["validity"]["python_rule"], "exact-size mask != 0")
        self.assertEqual(report["validity"]["python_valid_vertex_count"], 12)
        self.assertEqual(report["validity"]["cpp_valid_mask_vertex_count"], 10)
        self.assertEqual(
            report["validity"]["python_cpp_finite_point_disagreement_count"],
            2,
        )
        self.assertEqual(
            report["validity"]["mask_values"]["low_nonzero_below_255"],
            1,
        )
        self.assertTrue(
            {
                "mask-low-nonzero-interoperability",
                "mask-revalidates-nonpositive-or-nonfinite-z",
                "python-cpp-validity-disagreement",
            }.issubset(_codes(report))
        )

    def test_integer_multiple_mask_models_cpp_while_python_ignores_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            mask = np.full((6, 8), 255, dtype=np.uint8)
            mask[0, 0] = 0
            surface, _ = _make_surface(Path(temp), mask=mask)

            report = audit_tifxyz_integrity(surface)

        self.assertEqual(
            report["validity"]["mask_shape_relation"],
            "integer_multiple",
        )
        self.assertEqual(report["validity"]["mask_scale_factors_yx"], [2, 2])
        self.assertEqual(report["validity"]["python_valid_vertex_count"], 12)
        self.assertEqual(report["validity"]["cpp_valid_mask_vertex_count"], 11)
        self.assertIn("mask-shape-python-incompatible", _codes(report))
        self.assertIn("python-cpp-validity-disagreement", _codes(report))

    def test_bbox_mismatch_and_no_valid_face_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            mask = np.zeros((3, 4), dtype=np.uint8)
            mask[0, 0] = 255
            mask[2, 1] = 255
            surface, _ = _make_surface(Path(temp), mask=mask)

            report = audit_tifxyz_integrity(surface)

        self.assertEqual(report["validity"]["portable_valid_vertex_count"], 2)
        self.assertEqual(report["geometry"]["portable_valid_face_count"], 0)
        self.assertFalse(report["geometry"]["bbox_comparison"]["matches"])
        self.assertIn("no-valid-face", _codes(report))
        self.assertIn("metadata-bbox-mismatch", _codes(report))

    def test_no_portable_valid_vertex_is_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            mask = np.zeros((3, 4), dtype=np.uint8)
            surface, _ = _make_surface(Path(temp), mask=mask)

            report = audit_tifxyz_integrity(surface)
            encoded = dumps_integrity_report(report, indent=None)

        self.assertIn("no-portable-valid-vertex", _codes(report))
        self.assertIn("no-valid-face", _codes(report))
        self.assertIsNone(report["geometry"]["actual_bbox"])
        json.loads(encoded)

    def test_nonfinite_metadata_never_leaks_into_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            surface, _ = _make_surface(
                Path(temp),
                metadata={
                    "uuid": "synthetic",
                    "format": "tifxyz",
                    "type": "seg",
                    "scale": [float("nan"), 1.0],
                    "bbox": [[0.0, 0.0, 10.0], [3.0, float("inf"), 10.0]],
                    "nested": {"bad": float("-inf")},
                },
            )

            report = audit_tifxyz_integrity(surface)
            encoded = dumps_integrity_report(report)

        self.assertIn("metadata-nonfinite-number", _codes(report))
        json.loads(
            encoded,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )


if __name__ == "__main__":
    unittest.main()
