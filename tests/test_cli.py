from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tifxyz_doctor.cli import main


def _surface(path: Path, name: str = "surface") -> Path:
    output = path / name
    output.mkdir()
    rows, cols = np.indices((4, 5), dtype=np.float32)
    for filename, values in (
        ("x.tif", cols),
        ("y.tif", rows),
        ("z.tif", np.full_like(rows, 10.0)),
    ):
        Image.fromarray(values).save(output / filename, format="TIFF")
    (output / "meta.json").write_text(
        json.dumps(
            {
                "uuid": name,
                "format": "tifxyz",
                "type": "seg",
                "scale": [1.0, 1.0],
                "bbox": [[0.0, 0.0, 10.0], [4.0, 3.0, 10.0]],
            }
        ),
        encoding="utf-8",
    )
    return output


class CliTests(unittest.TestCase):
    def test_check_prints_strict_json_and_uses_ci_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            surface = _surface(Path(temp))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                clean_status = main(["check", str(surface)])
            clean = json.loads(stdout.getvalue())

            (surface / "y.tif").unlink()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                broken_status = main(["check", str(surface)])
            broken = json.loads(stdout.getvalue())

        self.assertEqual(clean_status, 0)
        self.assertEqual(clean["status"], "pass")
        self.assertEqual(broken_status, 2)
        self.assertEqual(broken["status"], "error")

    def test_audit_json_embeds_raw_contract_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            surface = _surface(root)
            output = root / "audit.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["audit", str(surface), "--json", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(report["contract"]["status"], "pass")
        self.assertEqual(report["topology"]["valid_quad_count"], 12)

    def test_audit_structural_failure_is_json_and_exit_two(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            surface = _surface(root)
            (surface / "y.tif").unlink()
            output = root / "audit.json"
            stderr = io.StringIO()
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
            ):
                status = main(
                    [
                        "audit",
                        str(surface),
                        "--json",
                        str(output),
                        "--fail-on-integrity",
                    ]
                )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["contract"]["status"], "error")
        self.assertIsNone(report["geometry_audit"])
        self.assertEqual(report["failure"]["code"], "geometry-load-failed")
        self.assertIn("geometry audit could not load", stderr.getvalue())

    def test_audit_ignores_mismatched_mask_but_preserves_contract_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            surface = _surface(root)
            Image.fromarray(np.zeros((8, 10), dtype=np.uint8)).save(
                surface / "mask.tif",
                format="TIFF",
            )
            output = root / "audit.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "audit",
                        str(surface),
                        "--json",
                        str(output),
                        "--fail-on-integrity",
                    ]
                )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(report["contract"]["status"], "error")
        self.assertEqual(report["topology"]["valid_quad_count"], 12)
        self.assertIn(
            "mask-shape-python-incompatible",
            [finding["code"] for finding in report["contract"]["findings"]],
        )

    def test_scan_discovers_surfaces_and_aggregates_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _surface(root, "clean")
            broken = _surface(root, "broken")
            (broken / "x.tif").unlink()
            output = root / "scan.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["scan", str(root), "--json", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(report["surface_count"], 2)
        self.assertEqual(report["status_counts"], {"error": 1, "pass": 1, "warning": 0})

    def test_scan_discovers_nested_surface_with_missing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nested = root / "nested"
            nested.mkdir()
            surface = _surface(nested)
            (surface / "meta.json").unlink()
            output = root / "scan.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["scan", str(root), "--json", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(report["surface_count"], 1)
        self.assertEqual(report["reports"][0]["status"], "error")
        self.assertIn(
            "meta.json",
            [
                finding.get("file")
                for finding in report["reports"][0]["findings"]
                if finding["code"] == "missing-required-file"
            ],
        )

    def test_scan_reports_duplicate_uuids_as_collection_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = _surface(root, "first")
            second = _surface(root, "second")
            second_meta = json.loads((second / "meta.json").read_text(encoding="utf-8"))
            second_meta["uuid"] = "first"
            (second / "meta.json").write_text(json.dumps(second_meta), encoding="utf-8")
            output = root / "scan.json"
            with contextlib.redirect_stdout(io.StringIO()):
                default_status = main(["scan", str(root), "--json", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))
            with contextlib.redirect_stdout(io.StringIO()):
                strict_status = main(
                    ["scan", str(root), "--json", str(output), "--fail-on-warning"]
                )

        self.assertEqual(default_status, 0)
        self.assertEqual(strict_status, 2)
        self.assertEqual(report["collection_findings"][0]["code"], "duplicate-uuid")
        self.assertEqual(report["collection_findings"][0]["paths"], [str(first), str(second)])


if __name__ == "__main__":
    unittest.main()
