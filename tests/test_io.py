from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from tifxyz_doctor.io import _read_tiff, load_tifxyz


class TiffReaderTests(unittest.TestCase):
    def test_falls_back_to_pillow_when_tifffile_decoder_fails(self) -> None:
        expected = np.arange(12, dtype=np.float32).reshape(3, 4)
        broken_tifffile = types.SimpleNamespace(
            imread=mock.Mock(side_effect=ValueError("imagecodecs is required"))
        )

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "values.tif"
            Image.fromarray(expected).save(path, format="TIFF")
            with mock.patch.dict(sys.modules, {"tifffile": broken_tifffile}):
                actual = _read_tiff(path)

        np.testing.assert_array_equal(actual, expected)
        broken_tifffile.imread.assert_called_once_with(path)

    def test_reports_both_reader_failures(self) -> None:
        broken_tifffile = types.SimpleNamespace(
            imread=mock.Mock(side_effect=ValueError("primary decoder failed"))
        )

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "not-a-tiff.tif"
            path.write_bytes(b"not a TIFF")
            with mock.patch.dict(sys.modules, {"tifffile": broken_tifffile}):
                with self.assertRaisesRegex(
                    ValueError,
                    r"no TIFF reader succeeded .*tifffile=ValueError.*Pillow=",
                ):
                    _read_tiff(path)

    def test_mismatched_mask_is_ignored_like_villa_python_reader(self) -> None:
        rows, cols = np.indices((3, 4), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp:
            surface = Path(temp)
            for name, values in (
                ("x.tif", cols),
                ("y.tif", rows),
                ("z.tif", np.full_like(rows, 10.0)),
            ):
                Image.fromarray(values).save(surface / name, format="TIFF")
            Image.fromarray(np.zeros((6, 8), dtype=np.uint8)).save(
                surface / "mask.tif",
                format="TIFF",
            )
            (surface / "meta.json").write_text(
                '{"uuid":"mismatch","scale":[1,1]}',
                encoding="utf-8",
            )

            data = load_tifxyz(surface)

        self.assertIsNone(data.explicit_mask)
        self.assertTrue(data.valid.all())


if __name__ == "__main__":
    unittest.main()
