from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from native_surface_sampler import SurfaceGrid  # noqa: E402
from render_tifxyz_volume import build_parser, main as cli_main  # noqa: E402
from tifxyz_render_pipeline import (  # noqa: E402
    AxisNormalizedVolume,
    CalibrationThresholds,
    RenderOptions,
    compare_published_stack,
    load_tifxyz_asset,
    render_surface_to_directory,
    resample_surface_grid,
    sanitize_source,
)


class RecordingArray:
    def __init__(self, data: np.ndarray, *, chunks: tuple[int, ...] | None = None) -> None:
        self.data = data
        self.shape = data.shape
        self.ndim = data.ndim
        self.dtype = data.dtype
        self.chunks = chunks
        self.reads: list[object] = []

    def __getitem__(self, key: object) -> np.ndarray:
        self.reads.append(key)
        return self.data[key]


def test_cli_defaults_to_model_ready_full_surface_resolution(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--tifxyz",
            str(tmp_path / "tifxyz"),
            "--volume",
            str(tmp_path / "volume.npy"),
            "--output",
            str(tmp_path / "output"),
            "--voxel-size-um",
            "8.64",
        ]
    )
    assert args.surface_resolution == "full"


def write_tifxyz(
    directory: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    scale_xy: tuple[float, float] = (1.0, 1.0),
) -> None:
    directory.mkdir(parents=True)
    tifffile.imwrite(directory / "x.tif", x.astype(np.float32))
    tifffile.imwrite(directory / "y.tif", y.astype(np.float32))
    tifffile.imwrite(directory / "z.tif", z.astype(np.float32))
    if mask is not None:
        tifffile.imwrite(directory / "mask.tif", mask.astype(np.uint8) * 255)
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "format": "tifxyz",
                "uuid": "synthetic-segment",
                "scale": list(scale_xy),
                "bbox": [
                    [float(np.nanmin(x)), float(np.nanmin(y)), float(np.nanmin(z))],
                    [float(np.nanmax(x)), float(np.nanmax(y)), float(np.nanmax(z))],
                ],
            }
        ),
        encoding="utf-8",
    )


def plane_surface(
    height: int,
    width: int,
    *,
    z_value: float = 10.0,
    x_origin: float = 4.0,
    y_origin: float = 3.0,
    mask: np.ndarray | None = None,
) -> SurfaceGrid:
    rows, columns = np.mgrid[:height, :width]
    x = columns.astype(np.float64) + x_origin
    y = rows.astype(np.float64) + y_origin
    z = np.full((height, width), z_value, dtype=np.float64)
    return SurfaceGrid.from_tifxyz(x, y, z, mask=mask)


def test_tifxyz_loader_derives_official_z_positive_mask_and_hashes(tmp_path: Path) -> None:
    rows, columns = np.mgrid[:3, :4]
    x = columns.astype(np.float32)
    y = rows.astype(np.float32)
    z = np.full((3, 4), 5.0, dtype=np.float32)
    z[1, 2] = 0.0
    source = tmp_path / "tifxyz"
    write_tifxyz(source, x, y, z, scale_xy=(0.5, 0.5))

    asset = load_tifxyz_asset(source)

    assert asset.stored_shape == (3, 4)
    assert asset.output_shape == (3, 4)
    assert asset.scale_yx == (0.5, 0.5)
    assert asset.mask_source == "derived:z>0-and-finite"
    assert not asset.surface.valid[1, 2]
    assert set(asset.input_sha256) == {"meta.json", "x.tif", "y.tif", "z.tif"}
    assert all(len(digest) == 64 for digest in asset.input_sha256.values())


def test_explicit_tifxyz_mask_controls_zero_z_and_full_resolution(tmp_path: Path) -> None:
    rows, columns = np.mgrid[:3, :4]
    x = columns.astype(np.float32)
    y = rows.astype(np.float32)
    z = np.full((3, 4), 2.0, dtype=np.float32)
    z[0, 0] = 0.0
    mask = np.ones((3, 4), dtype=bool)
    mask[1, 1] = False
    source = tmp_path / "tifxyz"
    write_tifxyz(source, x, y, z, mask=mask, scale_xy=(0.5, 0.5))

    asset = load_tifxyz_asset(source, resolution="full")

    assert asset.output_shape == (6, 8)
    assert asset.mask_source == "mask.tif:nonzero"
    assert asset.surface.valid[0, 0]  # explicit mask, not z > 0, is authoritative
    assert not asset.surface.valid[2, 2]  # source hole remains a strict output hole
    assert asset.interpolation == "linear"


def test_full_resolution_snaps_float32_scale_to_official_integer_shape(
    tmp_path: Path,
) -> None:
    rows, columns = np.mgrid[:3, :4]
    source = tmp_path / "tifxyz"
    float32_step = float(np.float32(0.05))
    write_tifxyz(
        source,
        columns,
        rows,
        np.ones((3, 4)),
        scale_xy=(float32_step, float32_step),
    )

    asset = load_tifxyz_asset(source, resolution="full")

    assert asset.output_shape == (60, 80)


def test_tifxyz_loader_rejects_mismatched_or_color_coordinate_tiffs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bad"
    source.mkdir()
    tifffile.imwrite(source / "x.tif", np.zeros((3, 4), dtype=np.float32))
    tifffile.imwrite(source / "y.tif", np.zeros((3, 5), dtype=np.float32))
    tifffile.imwrite(source / "z.tif", np.zeros((3, 4), dtype=np.float32))
    (source / "meta.json").write_text('{"scale":[1,1]}', encoding="utf-8")
    with pytest.raises(ValueError, match="shapes differ"):
        load_tifxyz_asset(source)

    tifffile.imwrite(
        source / "y.tif",
        np.zeros((3, 4, 3), dtype=np.uint8),
        photometric="rgb",
    )
    with pytest.raises(ValueError, match="2-D grayscale"):
        load_tifxyz_asset(source)


def test_tifxyz_loader_enforces_preallocation_pixel_ceiling(tmp_path: Path) -> None:
    rows, columns = np.mgrid[:3, :4]
    source = tmp_path / "tifxyz"
    write_tifxyz(source, columns, rows, np.ones((3, 4)))
    with pytest.raises(ValueError, match="over safety limit"):
        load_tifxyz_asset(source, maximum_surface_pixels=11)


def test_mask_aware_resampling_does_not_interpolate_through_holes() -> None:
    rows, columns = np.mgrid[:4, :4]
    x = columns.astype(np.float64)
    y = rows.astype(np.float64)
    z = 2.0 + x
    mask = np.ones((4, 4), dtype=bool)
    mask[1, 1] = False
    source = SurfaceGrid.from_tifxyz(x, y, z, mask=mask)

    output = resample_surface_grid(source, target_shape=(8, 8))

    # The expanded region influenced by the hole is masked instead of receiving
    # a coordinate blended across invalid geometry.
    assert not output.valid[2, 2]
    assert not output.valid[3, 3]
    assert output.valid[7, 7]
    np.testing.assert_allclose(output.z[7, 7], 5.0, atol=1e-12)


def test_axis_normalized_volume_transposes_and_reads_source_once() -> None:
    # Source is (t, c, x, y, z), deliberately not spatially ordered.
    source_data = np.arange(2 * 3 * 5 * 6 * 7, dtype=np.uint16).reshape(2, 3, 5, 6, 7)
    source = RecordingArray(source_data, chunks=(1, 1, 2, 3, 4))
    volume = AxisNormalizedVolume(
        source, axes="tcxyz", fixed_indices={"t": 1, "c": 2}
    )

    result = volume[1:4, 2:5, 0:3]

    expected = source_data[1, 2, 0:3, 2:5, 1:4].transpose(2, 1, 0)
    np.testing.assert_array_equal(result, expected)
    assert volume.shape == (7, 6, 5)
    assert volume.chunks == (4, 3, 2)
    assert len(source.reads) == 1


def test_axis_normalized_volume_requires_explicit_nonspatial_indices() -> None:
    source = RecordingArray(np.zeros((2, 3, 4, 5), dtype=np.uint8))
    with pytest.raises(ValueError, match="fixed indices"):
        AxisNormalizedVolume(source, axes="czyx")
    with pytest.raises(ValueError, match="include z, y and x"):
        AxisNormalizedVolume(source, axes="tcxy", fixed_indices={"t": 0, "c": 0})


def test_tiled_pipeline_fetches_one_compact_roi_per_tile_for_both_signs(
    tmp_path: Path,
) -> None:
    z, y, x = np.indices((24, 20, 22), dtype=np.float64)
    raw = (100.0 * z + 10.0 * y + x).astype(np.uint16)
    source = RecordingArray(raw, chunks=(8, 8, 8))
    volume = AxisNormalizedVolume(source, axes="zyx")
    surface = plane_surface(7, 8)
    output = tmp_path / "rendered"
    options = RenderOptions(
        voxel_size_zyx_um=(1.0, 1.0, 1.0),
        offsets_um=(-2.0, 0.0, 2.0),
        signs=("positive", "negative"),
        tile_shape=(3, 4),
        output_dtype="uint16",
        png_mode="all",
    )

    manifest = render_surface_to_directory(
        volume,
        surface,
        output,
        options=options,
        tifxyz_manifest={"uuid": "synthetic"},
        volume_manifest={"axes": "zyx"},
    )

    # ceil(7/3) * ceil(8/4) == 6 tiles. All offsets and both signs share one
    # source read per tile, instead of 3*2 or eight-corner remote reads.
    assert len(source.reads) == 6
    assert manifest["render_statistics"]["tiles_total"] == 6
    assert manifest["render_statistics"]["tiles_fetched"] == 6
    assert manifest["render_statistics"]["max_roi_shape_zyx"] == [5, 3, 4]
    for sign in ("positive", "negative"):
        assert (output / sign / "00.tif").is_file()
        assert (output / sign / "01.tif").is_file()
        assert (output / sign / "02.tif").is_file()
        assert (output / sign / "png" / "01.png").is_file()
        assert np.all(tifffile.imread(output / sign / "valid-all.tif") == 255)

    base = 100.0 * 10 + 10.0 * (np.arange(7)[:, None] + 3) + (
        np.arange(8)[None, :] + 4
    )
    np.testing.assert_array_equal(
        tifffile.imread(output / "positive" / "00.tif"), (base - 200).astype(np.uint16)
    )
    np.testing.assert_array_equal(
        tifffile.imread(output / "positive" / "02.tif"), (base + 200).astype(np.uint16)
    )
    np.testing.assert_array_equal(
        tifffile.imread(output / "negative" / "00.tif"), (base + 200).astype(np.uint16)
    )
    saved_manifest = json.loads((output / "manifest.json").read_text())
    assert saved_manifest["status"] == "complete"
    assert saved_manifest["coordinate_conventions"]["volume_index_order"] == "z,y,x"


def test_pipeline_preserves_surface_holes_and_out_of_bounds_offsets(tmp_path: Path) -> None:
    raw = np.ones((6, 8, 8), dtype=np.uint8) * 100
    source = RecordingArray(raw)
    mask = np.ones((4, 4), dtype=bool)
    mask[2, 2] = False
    surface = plane_surface(4, 4, z_value=1.0, x_origin=2.0, y_origin=2.0, mask=mask)
    output = tmp_path / "rendered"
    options = RenderOptions(
        voxel_size_zyx_um=(1.0, 1.0, 1.0),
        offsets_um=(-3.0, 0.0, 3.0),
        signs=("positive",),
        tile_shape=(2, 2),
        output_dtype="uint8",
        png_mode="none",
    )

    manifest = render_surface_to_directory(source, surface, output, options=options)

    assert tifffile.imread(output / "positive" / "00.tif")[1, 1] == 0
    assert tifffile.imread(output / "positive" / "01.tif")[2, 2] == 0
    assert manifest["outputs_by_normal_sign"]["positive"]["valid_sample_fraction"] < 1.0
    qc = json.loads((output / "geometry-qc.json").read_text())
    warnings = qc["reports_by_normal_sign"]["positive"]["warnings"]
    assert any("leave the volume" in warning for warning in warnings)


def test_published_stack_calibration_passes_exact_render_and_records_ambiguity(
    tmp_path: Path,
) -> None:
    z, y, x = np.indices((20, 16, 18), dtype=np.float64)
    raw = (z + 2 * y + 3 * x).astype(np.uint8)
    output = tmp_path / "rendered"
    options = RenderOptions(
        voxel_size_zyx_um=(1.0, 1.0, 1.0),
        offsets_um=(-2.0, 0.0, 2.0),
        tile_shape=(3, 4),
        output_dtype="uint8",
        png_mode="none",
    )
    render_surface_to_directory(raw, plane_surface(7, 8), output, options=options)
    published = tmp_path / "published"
    published.mkdir()
    for layer in (output / "positive").glob("[0-9][0-9].tif"):
        shutil.copy2(layer, published / layer.name)

    result = compare_published_stack(
        output,
        published,
        thresholds=CalibrationThresholds(erosion_pixels=1),
        expected_orientation="positive-direct",
    )

    assert result["passed"]
    assert result["selected_orientation"] == "positive-direct"
    assert "positive-direct" in result["best_orientation_ties"]
    assert "negative-reversed" in result["best_orientation_ties"]
    assert not result["orientation_resolved"]
    selected = result["candidates"]["positive-direct"]
    assert selected["valid_mask_iou"] == 1.0
    assert selected["median_layer_correlation"] == 1.0
    assert selected["median_absolute_difference"] == 0.0
    assert (output / "calibration.json").is_file()


def test_published_stack_calibration_fails_mask_and_orientation_gates(
    tmp_path: Path,
) -> None:
    z, y, x = np.indices((20, 16, 18), dtype=np.float64)
    raw = (z + y + x + 10).astype(np.uint8)
    output = tmp_path / "rendered"
    render_surface_to_directory(
        raw,
        plane_surface(7, 8),
        output,
        options=RenderOptions(
            voxel_size_zyx_um=(1.0, 1.0, 1.0),
            offsets_um=(-2.0, 0.0, 2.0),
            output_dtype="uint8",
            png_mode="none",
        ),
    )
    published = tmp_path / "published"
    published.mkdir()
    for layer in (output / "positive").glob("[0-9][0-9].tif"):
        array = tifffile.imread(layer)
        array[:, :4] = 0
        tifffile.imwrite(published / layer.name, array)

    result = compare_published_stack(
        output,
        published,
        thresholds=CalibrationThresholds(valid_mask_iou=0.995, erosion_pixels=0),
        expected_orientation="positive-direct",
    )
    assert not result["passed"]
    assert result["candidates"]["positive-direct"]["valid_mask_iou"] == 0.5


def test_cli_offline_npy_smoke_produces_model_ready_layers(tmp_path: Path) -> None:
    rows, columns = np.mgrid[:4, :5]
    tifxyz = tmp_path / "tifxyz"
    write_tifxyz(
        tifxyz,
        columns + 2,
        rows + 2,
        np.full((4, 5), 5.0),
        mask=np.ones((4, 5), dtype=bool),
    )
    z, y, x = np.indices((12, 12, 12), dtype=np.float64)
    volume_path = tmp_path / "volume.npy"
    np.save(volume_path, (z + y + x).astype(np.uint8))
    output = tmp_path / "output"

    result = cli_main(
        [
            "--tifxyz",
            str(tifxyz),
            "--volume",
            str(volume_path),
            "--volume-axes",
            "zyx",
            "--voxel-size-um",
            "1",
            "--frames",
            "3",
            "--spacing-um",
            "1",
            "--normal-sign",
            "positive",
            "--tile-size",
            "2,3",
            "--png-mode",
            "none",
            "--output",
            str(output),
            "--quiet",
        ]
    )

    assert result == 0
    assert [path.name for path in sorted((output / "positive").glob("[0-9]*.tif"))] == [
        "00.tif",
        "01.tif",
        "02.tif",
    ]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["render_options"]["offsets_um"] == [-1.0, 0.0, 1.0]
    assert manifest["volume"]["normalized_axes"] == "zyx"


def test_31_frame_contract_emits_zero_based_layers_00_through_30(tmp_path: Path) -> None:
    volume = np.arange(40, dtype=np.uint8)[:, None, None]
    volume = np.broadcast_to(volume, (40, 6, 6)).copy()
    output = tmp_path / "rendered"
    offsets = tuple(float(value) for value in range(-15, 16))

    render_surface_to_directory(
        volume,
        plane_surface(2, 2, z_value=20.0, x_origin=2.0, y_origin=2.0),
        output,
        options=RenderOptions(
            voxel_size_zyx_um=(1.0, 1.0, 1.0),
            offsets_um=offsets,
            signs=("positive",),
            output_dtype="uint8",
            png_mode="none",
        ),
    )

    names = [path.name for path in sorted((output / "positive").glob("[0-9]*.tif"))]
    assert names == [f"{index:02d}.tif" for index in range(31)]
    assert tifffile.imread(output / "positive" / "00.tif")[0, 0] == 5
    assert tifffile.imread(output / "positive" / "30.tif")[0, 0] == 35


def test_sanitize_source_strips_credentials_and_query() -> None:
    assert (
        sanitize_source("https://user:secret@example.test/path/raw.zarr?token=abc#x")
        == "https://example.test/path/raw.zarr"
    )
