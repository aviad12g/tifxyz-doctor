from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from sample_raw_ct_seed_cube import (  # noqa: E402
    ZarrV2ArraySpec,
    assemble_roi,
    cube_bounds,
    decode_chunk_payload,
    display_window,
    intersecting_chunk_indices,
    orthogonal_slice,
    render_contact_sheet,
    window_uint8,
)


def make_spec(
    *, shape: tuple[int, int, int] = (24297, 8343, 8343), chunks: tuple[int, int, int] = (128, 128, 128)
) -> ZarrV2ArraySpec:
    return ZarrV2ArraySpec.from_metadata(
        {
            "zarr_format": 2,
            "shape": list(shape),
            "chunks": list(chunks),
            "dtype": "|u1",
            "order": "C",
            "fill_value": 0,
            "filters": None,
            "compressor": None,
            "dimension_separator": "/",
        }
    )


def test_priority_cube_needs_exactly_two_raw_chunks() -> None:
    spec = make_spec()
    lower, upper = cube_bounds((14071, 4191, 3525), 32, spec.shape_zyx)
    np.testing.assert_array_equal(lower, (14039, 4159, 3493))
    np.testing.assert_array_equal(upper, (14104, 4224, 3558))
    assert intersecting_chunk_indices(lower, upper, spec.chunks_zyx) == (
        (109, 32, 27),
        (110, 32, 27),
    )


def test_uncompressed_chunk_decode_is_exact() -> None:
    spec = make_spec(shape=(4, 4, 4), chunks=(4, 4, 4))
    expected = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
    decoded, decoder, raw = decode_chunk_payload(
        expected.tobytes(order="C"), expected.shape, spec
    )
    assert decoder == "zarr-v2-uncompressed"
    assert raw == expected.tobytes(order="C")
    np.testing.assert_array_equal(decoded, expected)


def test_roi_assembly_crosses_all_chunk_axes_without_overlap_error() -> None:
    shape = (7, 6, 5)
    chunks = (4, 3, 3)
    spec = make_spec(shape=shape, chunks=chunks)
    source = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)

    def loader(index: tuple[int, int, int], expected_shape: tuple[int, int, int]):
        start = np.asarray(index) * np.asarray(chunks)
        stop = start + np.asarray(expected_shape)
        slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
        return source[slices].copy(), {"test": True}

    lower = (2, 1, 1)
    upper = (7, 6, 5)
    actual, records = assemble_roi(spec, lower, upper, loader)
    np.testing.assert_array_equal(actual, source[2:7, 1:6, 1:5])
    assert len(records) == 8


def test_orthogonal_slices_preserve_declared_axes() -> None:
    z, y, x = np.mgrid[:5, :5, :5]
    cube = (z * 25 + y * 5 + x).astype(np.uint8)
    np.testing.assert_array_equal(orthogonal_slice(cube, "xy", 2), cube[2, :, :])
    np.testing.assert_array_equal(orthogonal_slice(cube, "xz", 3), cube[:, 3, :])
    np.testing.assert_array_equal(orthogonal_slice(cube, "yz", 1), cube[:, :, 1])


def test_shared_display_window_and_mapping_are_deterministic() -> None:
    cube = np.arange(125, dtype=np.uint8).reshape(5, 5, 5)
    low, high = display_window(cube, 0, 100)
    assert (low, high) == (0.0, 124.0)
    mapped = window_uint8(np.array([0, 62, 124], dtype=np.uint8), low, high)
    np.testing.assert_array_equal(mapped, (0, 128, 255))


def test_contact_sheet_has_three_rows_and_one_column_per_offset() -> None:
    cube = np.arange(65**3, dtype=np.uint32).reshape(65, 65, 65) % 256
    image = render_contact_sheet(
        cube.astype(np.uint8),
        (14071, 4191, 3525),
        (-8, 0, 8),
        0,
        255,
        normal_xyz=(-0.4, 0.8, -0.4),
        scale=1,
    )
    assert image.mode == "RGB"
    assert image.width > 3 * 65
    assert image.height > 3 * 65
