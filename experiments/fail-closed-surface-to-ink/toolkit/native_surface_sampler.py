"""Native-volume rendering for Vesuvius ``tifxyz`` surfaces.

The conventions in this module are intentionally explicit:

* volumes are indexed ``volume[z, y, x]``;
* tifxyz surfaces are supplied as separate ``x``, ``y`` and ``z`` images;
* points and normals use a final ``(x, y, z)`` component axis;
* voxel calibration is supplied in ``(z, y, x)`` order, in micrometres.

Normals are computed once for the complete surface and then sliced for tiles.
That detail is important: independently differentiating adjacent tiles gives
different one-sided derivatives at their shared edge.  ``NativeSurfaceSampler``
therefore makes tiled and untiled rendering numerically identical.

Only NumPy is required.  A volume may be a NumPy array, memmap, or another
array-like object supporting NumPy-style point indexing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, radians
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.floating[Any]]
BoolArray = NDArray[np.bool_]


def _voxel_size_xyz(voxel_size_zyx_um: Sequence[float]) -> NDArray[np.float64]:
    """Validate ``(z, y, x)`` voxel sizes and return ``(x, y, z)``."""

    sizes_zyx = np.asarray(voxel_size_zyx_um, dtype=np.float64)
    if sizes_zyx.shape != (3,):
        raise ValueError(
            "voxel_size_zyx_um must contain exactly three values in (z, y, x) order"
        )
    if not np.isfinite(sizes_zyx).all() or np.any(sizes_zyx <= 0.0):
        raise ValueError("voxel sizes must be finite and strictly positive")
    return sizes_zyx[::-1].copy()


def _validate_volume_shape_zyx(volume_shape_zyx: Sequence[int]) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in volume_shape_zyx)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("volume shape must be three positive integers in (z, y, x) order")
    return shape  # type: ignore[return-value]


@dataclass(frozen=True)
class SurfaceGrid:
    """A masked 2-D tifxyz coordinate grid.

    ``x``, ``y`` and ``z`` are voxel coordinates, not physical distances.
    ``valid`` combines the caller-provided mask with finite-coordinate checks.
    Invalid coordinate values are retained but are never sampled.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    valid: BoolArray

    @classmethod
    def from_tifxyz(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        mask: ArrayLike | None = None,
    ) -> "SurfaceGrid":
        x_array = np.asarray(x, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.float64)
        z_array = np.asarray(z, dtype=np.float64)
        if x_array.ndim != 2:
            raise ValueError(f"tifxyz coordinates must be 2-D; got x.ndim={x_array.ndim}")
        if x_array.shape != y_array.shape or x_array.shape != z_array.shape:
            raise ValueError(
                "tifxyz coordinate shapes must match; "
                f"got x={x_array.shape}, y={y_array.shape}, z={z_array.shape}"
            )
        if mask is None:
            requested_valid = np.ones(x_array.shape, dtype=bool)
        else:
            requested_valid = np.asarray(mask, dtype=bool)
            if requested_valid.shape != x_array.shape:
                raise ValueError(
                    f"surface mask shape {requested_valid.shape} does not match {x_array.shape}"
                )
        finite = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(z_array)
        return cls(x=x_array, y=y_array, z=z_array, valid=requested_valid & finite)

    @property
    def shape(self) -> tuple[int, int]:
        return self.x.shape  # type: ignore[return-value]

    @property
    def points_xyz(self) -> NDArray[np.float64]:
        return np.stack((self.x, self.y, self.z), axis=-1)


@dataclass(frozen=True)
class NormalField:
    """Per-vertex physical unit normals in both orientations.

    The positive orientation is ``cross(dP/dcolumn, dP/drow)`` in xyz space.
    Thus a conventional plane with x increasing across columns and y increasing
    across rows has a positive normal of ``(0, 0, +1)``.  ``negative_xyz`` is
    exactly the negation of ``positive_xyz`` at valid vertices.
    """

    positive_xyz: NDArray[np.float64]
    negative_xyz: NDArray[np.float64]
    valid: BoolArray


def _masked_coordinate_gradient(
    points_xyz: NDArray[np.float64],
    valid: BoolArray,
    *,
    axis: int,
) -> tuple[NDArray[np.float64], BoolArray]:
    """Mask-aware central/one-sided gradient without periodic array rolls."""

    if axis not in (0, 1):
        raise ValueError("surface gradient axis must be 0 (row) or 1 (column)")
    previous = np.zeros_like(points_xyz)
    following = np.zeros_like(points_xyz)
    previous_valid = np.zeros(valid.shape, dtype=bool)
    following_valid = np.zeros(valid.shape, dtype=bool)

    if axis == 0:
        previous[1:, :, :] = points_xyz[:-1, :, :]
        following[:-1, :, :] = points_xyz[1:, :, :]
        previous_valid[1:, :] = valid[:-1, :]
        following_valid[:-1, :] = valid[1:, :]
    else:
        previous[:, 1:, :] = points_xyz[:, :-1, :]
        following[:, :-1, :] = points_xyz[:, 1:, :]
        previous_valid[:, 1:] = valid[:, :-1]
        following_valid[:, :-1] = valid[:, 1:]

    center = valid & previous_valid & following_valid
    forward = valid & ~previous_valid & following_valid
    backward = valid & previous_valid & ~following_valid
    derivative_valid = center | forward | backward

    derivative = np.zeros_like(points_xyz)
    derivative[center] = 0.5 * (following[center] - previous[center])
    derivative[forward] = following[forward] - points_xyz[forward]
    derivative[backward] = points_xyz[backward] - previous[backward]
    return derivative, derivative_valid


def estimate_surface_normals(
    surface: SurfaceGrid,
    *,
    voxel_size_zyx_um: Sequence[float] = (1.0, 1.0, 1.0),
    epsilon_um: float = 1e-12,
) -> NormalField:
    """Estimate mask-aware physical normals from tifxyz coordinate gradients.

    Coordinates are converted to micrometres before differentiation.  The
    resulting unit normals are therefore correct for anisotropic volumes.
    A vertex needs at least one valid row neighbour and one valid column
    neighbour.  Invalid normals are represented by zero vectors plus ``valid=False``.
    """

    if not np.isfinite(epsilon_um) or epsilon_um <= 0.0:
        raise ValueError("epsilon_um must be finite and positive")
    voxel_xyz_um = _voxel_size_xyz(voxel_size_zyx_um)
    physical_points = surface.points_xyz * voxel_xyz_um
    tangent_row, row_valid = _masked_coordinate_gradient(
        physical_points, surface.valid, axis=0
    )
    tangent_column, column_valid = _masked_coordinate_gradient(
        physical_points, surface.valid, axis=1
    )

    # This ordering preserves +z for the conventional x-across/y-down plane.
    raw_normals = np.cross(tangent_column, tangent_row)
    magnitude = np.linalg.norm(raw_normals, axis=-1)
    normal_valid = (
        surface.valid
        & row_valid
        & column_valid
        & np.isfinite(magnitude)
        & (magnitude > float(epsilon_um))
    )
    positive = np.zeros_like(raw_normals)
    positive[normal_valid] = raw_normals[normal_valid] / magnitude[normal_valid, None]
    return NormalField(positive_xyz=positive, negative_xyz=-positive, valid=normal_valid)


@dataclass(frozen=True)
class TrilinearSample:
    """Values and validity returned by :func:`sample_trilinear_zyx`."""

    values: FloatArray
    valid: BoolArray


def sample_trilinear_zyx(
    volume_zyx: Any,
    coordinates_xyz: ArrayLike,
    *,
    point_valid: ArrayLike | None = None,
    volume_mask_zyx: Any | None = None,
    fill_value: float = 0.0,
    output_dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> TrilinearSample:
    """Vectorized trilinear sampling of a scalar ``(z, y, x)`` volume.

    Parameters
    ----------
    volume_zyx:
        Scalar three-dimensional volume indexed in exactly ``(z, y, x)`` order.
    coordinates_xyz:
        Array with final dimension three, storing coordinates as ``(x, y, z)``.
    point_valid:
        Optional Boolean mask matching ``coordinates_xyz.shape[:-1]``.
    volume_mask_zyx:
        Optional mask matching the volume.  Every corner carrying non-zero
        interpolation weight must be valid.  Non-finite source voxels are
        treated the same way.
    fill_value:
        Value written wherever a point or a required source corner is invalid.

    Points on the final voxel center (for example ``x == X - 1``) are valid;
    their upper interpolation index is clamped and has zero fractional weight.
    """

    if getattr(volume_zyx, "ndim", None) != 3:
        raise ValueError("volume_zyx must be a scalar 3-D array in (z, y, x) order")
    volume_shape = _validate_volume_shape_zyx(volume_zyx.shape)
    coordinates = np.asarray(coordinates_xyz, dtype=np.float64)
    if coordinates.ndim < 1 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates_xyz must have a final (x, y, z) axis of length 3")
    point_shape = coordinates.shape[:-1]
    if point_valid is None:
        requested_valid = np.ones(point_shape, dtype=bool)
    else:
        requested_valid = np.asarray(point_valid, dtype=bool)
        if requested_valid.shape != point_shape:
            raise ValueError(
                f"point_valid shape {requested_valid.shape} does not match {point_shape}"
            )
    if volume_mask_zyx is not None and tuple(volume_mask_zyx.shape) != volume_shape:
        raise ValueError(
            f"volume mask shape {volume_mask_zyx.shape} does not match {volume_shape}"
        )
    dtype = np.dtype(output_dtype)
    if not np.issubdtype(dtype, np.floating):
        raise ValueError("output_dtype must be a floating dtype")

    flat_xyz = coordinates.reshape(-1, 3)
    flat_requested = requested_valid.reshape(-1)
    finite = np.isfinite(flat_xyz).all(axis=1)
    z_size, y_size, x_size = volume_shape
    in_bounds = (
        finite
        & (flat_xyz[:, 0] >= 0.0)
        & (flat_xyz[:, 0] <= x_size - 1)
        & (flat_xyz[:, 1] >= 0.0)
        & (flat_xyz[:, 1] <= y_size - 1)
        & (flat_xyz[:, 2] >= 0.0)
        & (flat_xyz[:, 2] <= z_size - 1)
    )

    # Invalid coordinates are replaced before floor/cast to avoid NaN warnings
    # and dangerous sentinel indices.  They are masked out again at the end.
    safe_xyz = np.where(finite[:, None], flat_xyz, 0.0)
    safe_xyz[:, 0] = np.clip(safe_xyz[:, 0], 0.0, x_size - 1)
    safe_xyz[:, 1] = np.clip(safe_xyz[:, 1], 0.0, y_size - 1)
    safe_xyz[:, 2] = np.clip(safe_xyz[:, 2], 0.0, z_size - 1)
    lower_xyz = np.floor(safe_xyz).astype(np.int64)
    upper_xyz = np.minimum(
        lower_xyz + 1, np.asarray((x_size - 1, y_size - 1, z_size - 1))
    )
    fraction_xyz = safe_xyz - lower_xyz

    x0, y0, z0 = lower_xyz.T
    x1, y1, z1 = upper_xyz.T
    fx, fy, fz = fraction_xyz.T
    wx = (1.0 - fx, fx)
    wy = (1.0 - fy, fy)
    wz = (1.0 - fz, fz)
    x_indices = (x0, x1)
    y_indices = (y0, y1)
    z_indices = (z0, z1)

    accumulated = np.zeros(flat_xyz.shape[0], dtype=np.float64)
    sources_valid = np.ones(flat_xyz.shape[0], dtype=bool)
    for z_side in (0, 1):
        for y_side in (0, 1):
            for x_side in (0, 1):
                weight = wz[z_side] * wy[y_side] * wx[x_side]
                values = np.asarray(
                    volume_zyx[
                        z_indices[z_side], y_indices[y_side], x_indices[x_side]
                    ],
                    dtype=np.float64,
                )
                source_ok = np.isfinite(values)
                if volume_mask_zyx is not None:
                    source_ok &= np.asarray(
                        volume_mask_zyx[
                            z_indices[z_side], y_indices[y_side], x_indices[x_side]
                        ],
                        dtype=bool,
                    )
                # A duplicated/clamped corner with zero weight does not affect
                # validity.  Avoid 0 * NaN by replacing invalid values first.
                contributes = weight > 0.0
                sources_valid &= ~contributes | source_ok
                accumulated += weight * np.where(source_ok, values, 0.0)

    final_valid = flat_requested & in_bounds & sources_valid
    result = np.full(flat_xyz.shape[0], fill_value, dtype=dtype)
    result[final_valid] = accumulated[final_valid].astype(dtype, copy=False)
    return TrilinearSample(
        values=result.reshape(point_shape), valid=final_valid.reshape(point_shape)
    )


@dataclass(frozen=True)
class NormalSamplingSpec:
    """Calibrated frame locations along one orientation of a surface normal.

    By default 26 frames are centered on the surface at half-integer multiples
    of ``spacing_um``: ``[-12.5, ..., +12.5] * spacing_um``.  This symmetric
    convention avoids assigning one of an even number of frames special status.
    Use :meth:`from_offsets` when a model requires exact pre-defined locations.
    """

    frame_count: int = 26
    spacing_um: float = 8.0
    center_um: float = 0.0
    normal_sign: int = 1
    explicit_offsets_um: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if int(self.frame_count) != self.frame_count or self.frame_count <= 0:
            raise ValueError("frame_count must be a positive integer")
        if not np.isfinite(self.spacing_um) or self.spacing_um <= 0.0:
            raise ValueError("spacing_um must be finite and positive")
        if not np.isfinite(self.center_um):
            raise ValueError("center_um must be finite")
        if self.normal_sign not in (-1, 1):
            raise ValueError("normal_sign must be +1 or -1")
        if self.explicit_offsets_um is not None:
            offsets = np.asarray(self.explicit_offsets_um, dtype=np.float64)
            if offsets.ndim != 1 or offsets.size == 0 or not np.isfinite(offsets).all():
                raise ValueError("explicit offsets must be a non-empty finite 1-D sequence")
            if offsets.size != self.frame_count:
                raise ValueError(
                    "frame_count must equal the number of explicit normal offsets"
                )

    @classmethod
    def from_offsets(
        cls, offsets_um: Sequence[float], *, normal_sign: int = 1
    ) -> "NormalSamplingSpec":
        offsets = tuple(float(value) for value in offsets_um)
        return cls(
            frame_count=len(offsets),
            spacing_um=1.0,
            center_um=0.0,
            normal_sign=normal_sign,
            explicit_offsets_um=offsets,
        )

    @property
    def offsets_um(self) -> NDArray[np.float64]:
        if self.explicit_offsets_um is not None:
            return np.asarray(self.explicit_offsets_um, dtype=np.float64)
        centered_indices = np.arange(self.frame_count, dtype=np.float64)
        centered_indices -= (self.frame_count - 1) / 2.0
        return self.center_um + centered_indices * self.spacing_um


@dataclass(frozen=True)
class RenderResult:
    """A normal-offset image stack in ``(frame, row, column)`` order."""

    values: FloatArray
    valid: BoolArray
    offsets_um: NDArray[np.float64]
    normal_sign: int


@dataclass(frozen=True)
class RenderedTile:
    row_slice: slice
    column_slice: slice
    result: RenderResult


def _canonical_slice(value: slice, size: int, name: str) -> slice:
    start, stop, step = value.indices(size)
    if step != 1:
        raise ValueError(f"{name} slice step must be 1")
    return slice(start, stop, 1)


class NativeSurfaceSampler:
    """Render a tifxyz surface directly from its native 3-D volume.

    Surface normals are estimated globally during construction.  Every tile
    references that same normal field, so changing tile shape cannot change
    values at a tile boundary.
    """

    def __init__(
        self,
        volume_zyx: Any,
        surface: SurfaceGrid,
        *,
        voxel_size_zyx_um: Sequence[float],
        volume_mask_zyx: Any | None = None,
        fill_value: float = 0.0,
        output_dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
    ) -> None:
        if getattr(volume_zyx, "ndim", None) != 3:
            raise ValueError("volume_zyx must be a scalar 3-D array in (z, y, x) order")
        self.volume_zyx = volume_zyx
        self.volume_shape_zyx = _validate_volume_shape_zyx(volume_zyx.shape)
        self.surface = surface
        self.voxel_size_xyz_um = _voxel_size_xyz(voxel_size_zyx_um)
        self.voxel_size_zyx_um = tuple(float(v) for v in voxel_size_zyx_um)
        if volume_mask_zyx is not None and tuple(volume_mask_zyx.shape) != tuple(
            volume_zyx.shape
        ):
            raise ValueError("volume_mask_zyx must match the volume shape")
        self.volume_mask_zyx = volume_mask_zyx
        self.fill_value = float(fill_value)
        self.output_dtype = np.dtype(output_dtype)
        if not np.issubdtype(self.output_dtype, np.floating):
            raise ValueError("output_dtype must be floating")
        # Global computation is the seam-safety invariant of this class.
        self.normals = estimate_surface_normals(
            surface, voxel_size_zyx_um=self.voxel_size_zyx_um
        )

    def render_tile(
        self,
        row_slice: slice,
        column_slice: slice,
        spec: NormalSamplingSpec | None = None,
    ) -> RenderResult:
        """Render one tile using the globally computed normal field."""

        if spec is None:
            spec = NormalSamplingSpec()
        rows = _canonical_slice(row_slice, self.surface.shape[0], "row")
        columns = _canonical_slice(column_slice, self.surface.shape[1], "column")
        tile_key = (rows, columns)
        base_xyz = self.surface.points_xyz[tile_key]
        positive_normal = self.normals.positive_xyz[tile_key]
        base_valid = self.surface.valid[tile_key] & self.normals.valid[tile_key]
        offsets = spec.offsets_um
        tile_shape = base_xyz.shape[:2]
        values = np.full(
            (offsets.size, *tile_shape), self.fill_value, dtype=self.output_dtype
        )
        valid = np.zeros((offsets.size, *tile_shape), dtype=bool)

        signed_normal_voxels_per_um = (
            spec.normal_sign * positive_normal / self.voxel_size_xyz_um
        )
        for frame_index, offset_um in enumerate(offsets):
            coordinates_xyz = base_xyz + offset_um * signed_normal_voxels_per_um
            sample = sample_trilinear_zyx(
                self.volume_zyx,
                coordinates_xyz,
                point_valid=base_valid,
                volume_mask_zyx=self.volume_mask_zyx,
                fill_value=self.fill_value,
                output_dtype=self.output_dtype,
            )
            values[frame_index] = sample.values
            valid[frame_index] = sample.valid
        return RenderResult(
            values=values,
            valid=valid,
            offsets_um=offsets,
            normal_sign=spec.normal_sign,
        )

    def render_stack(self, spec: NormalSamplingSpec | None = None) -> RenderResult:
        """Render the complete surface as one vectorized tile."""

        return self.render_tile(
            slice(0, self.surface.shape[0]),
            slice(0, self.surface.shape[1]),
            spec,
        )

    def iter_rendered_tiles(
        self,
        *,
        tile_shape: tuple[int, int],
        spec: NormalSamplingSpec | None = None,
    ) -> Iterator[RenderedTile]:
        """Yield non-overlapping, seam-safe tiles in row-major order."""

        tile_rows, tile_columns = (int(value) for value in tile_shape)
        if tile_rows <= 0 or tile_columns <= 0:
            raise ValueError("tile_shape values must be positive")
        height, width = self.surface.shape
        for row_start in range(0, height, tile_rows):
            row_slice = slice(row_start, min(row_start + tile_rows, height), 1)
            for column_start in range(0, width, tile_columns):
                column_slice = slice(
                    column_start, min(column_start + tile_columns, width), 1
                )
                yield RenderedTile(
                    row_slice=row_slice,
                    column_slice=column_slice,
                    result=self.render_tile(row_slice, column_slice, spec),
                )

    def render_stack_tiled(
        self,
        *,
        tile_shape: tuple[int, int],
        spec: NormalSamplingSpec | None = None,
    ) -> RenderResult:
        """Assemble a full stack while bounding temporary coordinate memory."""

        if spec is None:
            spec = NormalSamplingSpec()
        height, width = self.surface.shape
        values = np.full(
            (spec.frame_count, height, width),
            self.fill_value,
            dtype=self.output_dtype,
        )
        valid = np.zeros((spec.frame_count, height, width), dtype=bool)
        for tile in self.iter_rendered_tiles(tile_shape=tile_shape, spec=spec):
            values[:, tile.row_slice, tile.column_slice] = tile.result.values
            valid[:, tile.row_slice, tile.column_slice] = tile.result.valid
        return RenderResult(
            values=values,
            valid=valid,
            offsets_um=spec.offsets_um,
            normal_sign=spec.normal_sign,
        )


@dataclass(frozen=True)
class GeometryValidationReport:
    """Quantitative checks for a tifxyz grid before native-volume rendering."""

    surface_shape: tuple[int, int]
    volume_shape_zyx: tuple[int, int, int]
    coordinate_min_xyz: tuple[float, float, float]
    coordinate_max_xyz: tuple[float, float, float]
    valid_vertex_count: int
    valid_vertex_fraction: float
    in_bounds_vertex_count: int
    in_bounds_vertex_fraction: float
    normal_valid_vertex_count: int
    normal_valid_vertex_fraction: float
    possible_neighbor_edge_count: int
    valid_neighbor_edge_count: int
    valid_neighbor_edge_fraction: float
    median_neighbor_distance_um: float
    p95_neighbor_distance_um: float
    max_neighbor_distance_um: float
    continuity_limit_um: float
    discontinuity_edge_count: int
    discontinuity_edge_fraction: float
    neighbor_wrap_limit_um: float
    neighbor_wrap_risk_edge_count: int
    neighbor_wrap_risk_edge_fraction: float
    valid_quad_count: int
    median_quad_area_um2: float
    degenerate_quad_count: int
    degenerate_quad_fraction: float
    folded_quad_count: int
    folded_quad_fraction: float
    metric_distortion_limit: float
    metric_distortion_p95: float
    distorted_quad_count: int
    distorted_quad_fraction: float
    abrupt_normal_angle_degrees: float
    abrupt_normal_flip_edge_count: int
    abrupt_normal_flip_edge_fraction: float
    offset_sample_in_bounds_fraction: float
    warnings: tuple[str, ...]

    def to_dict(self) -> Mapping[str, Any]:
        """Return JSON-serializable primitive metrics."""

        return asdict(self)


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _safe_percentile(values: NDArray[np.float64], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def _coordinate_bounds(
    points_xyz: NDArray[np.float64], valid: BoolArray
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not np.any(valid):
        nan_triplet = (float("nan"),) * 3
        return nan_triplet, nan_triplet
    selected = points_xyz[valid]
    return (
        tuple(float(value) for value in np.min(selected, axis=0)),
        tuple(float(value) for value in np.max(selected, axis=0)),
    )


def validate_surface_geometry(
    surface: SurfaceGrid,
    *,
    volume_shape_zyx: Sequence[int],
    voxel_size_zyx_um: Sequence[float],
    normal_offsets_um: Sequence[float] | None = None,
    normal_sign: int = 1,
    continuity_factor: float = 4.0,
    neighbor_wrap_factor: float = 8.0,
    metric_distortion_limit: float = 4.0,
    abrupt_normal_angle_degrees: float = 90.0,
    degenerate_area_relative_epsilon: float = 1e-8,
) -> GeometryValidationReport:
    """Validate bounds, mesh continuity and local tifxyz geometry.

    Distance/area metrics are physical (micrometres), while coordinate bounds
    are reported in native voxel coordinates.  A ``neighbor_wrap_risk`` edge is
    a grid-adjacent pair separated by more than ``neighbor_wrap_factor`` times
    the median adjacent distance; no periodic first/last neighbours are ever
    introduced by the implementation.
    """

    volume_shape = _validate_volume_shape_zyx(volume_shape_zyx)
    voxel_xyz_um = _voxel_size_xyz(voxel_size_zyx_um)
    for name, value in (
        ("continuity_factor", continuity_factor),
        ("neighbor_wrap_factor", neighbor_wrap_factor),
        ("metric_distortion_limit", metric_distortion_limit),
        ("degenerate_area_relative_epsilon", degenerate_area_relative_epsilon),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        not np.isfinite(abrupt_normal_angle_degrees)
        or abrupt_normal_angle_degrees <= 0.0
        or abrupt_normal_angle_degrees > 180.0
    ):
        raise ValueError("abrupt_normal_angle_degrees must be in (0, 180]")
    if normal_sign not in (-1, 1):
        raise ValueError("normal_sign must be +1 or -1")

    height, width = surface.shape
    points_xyz = surface.points_xyz
    points_physical = points_xyz * voxel_xyz_um
    valid = surface.valid
    valid_count = int(np.count_nonzero(valid))
    total_vertices = int(height * width)
    coordinate_min, coordinate_max = _coordinate_bounds(points_xyz, valid)

    z_size, y_size, x_size = volume_shape
    in_bounds = valid & (
        (surface.x >= 0.0)
        & (surface.x <= x_size - 1)
        & (surface.y >= 0.0)
        & (surface.y <= y_size - 1)
        & (surface.z >= 0.0)
        & (surface.z <= z_size - 1)
    )
    in_bounds_count = int(np.count_nonzero(in_bounds))

    row_edge_valid = valid[:-1, :] & valid[1:, :]
    column_edge_valid = valid[:, :-1] & valid[:, 1:]
    row_vectors = points_physical[1:, :, :] - points_physical[:-1, :, :]
    column_vectors = points_physical[:, 1:, :] - points_physical[:, :-1, :]
    row_distances = np.linalg.norm(row_vectors, axis=-1)[row_edge_valid]
    column_distances = np.linalg.norm(column_vectors, axis=-1)[column_edge_valid]
    neighbor_distances = np.concatenate((row_distances, column_distances)).astype(
        np.float64, copy=False
    )
    valid_edge_count = int(neighbor_distances.size)
    possible_edge_count = max(height - 1, 0) * width + height * max(width - 1, 0)
    positive_distances = neighbor_distances[
        np.isfinite(neighbor_distances) & (neighbor_distances > 0.0)
    ]
    median_distance = _safe_percentile(positive_distances, 50.0)
    p95_distance = _safe_percentile(positive_distances, 95.0)
    max_distance = (
        float(np.max(positive_distances)) if positive_distances.size else float("nan")
    )
    continuity_limit = median_distance * continuity_factor
    wrap_limit = median_distance * neighbor_wrap_factor
    if np.isfinite(continuity_limit):
        discontinuity_count = int(np.count_nonzero(neighbor_distances > continuity_limit))
    else:
        discontinuity_count = 0
    if np.isfinite(wrap_limit):
        wrap_count = int(np.count_nonzero(neighbor_distances > wrap_limit))
    else:
        wrap_count = 0

    normal_field = estimate_surface_normals(
        surface, voxel_size_zyx_um=voxel_size_zyx_um
    )
    normal_count = int(np.count_nonzero(normal_field.valid))
    normal_row_valid = normal_field.valid[:-1, :] & normal_field.valid[1:, :]
    normal_column_valid = normal_field.valid[:, :-1] & normal_field.valid[:, 1:]
    normal_row_dots = np.sum(
        normal_field.positive_xyz[:-1, :, :] * normal_field.positive_xyz[1:, :, :],
        axis=-1,
    )[normal_row_valid]
    normal_column_dots = np.sum(
        normal_field.positive_xyz[:, :-1, :] * normal_field.positive_xyz[:, 1:, :],
        axis=-1,
    )[normal_column_valid]
    normal_dots = np.clip(
        np.concatenate((normal_row_dots, normal_column_dots)), -1.0, 1.0
    )
    abrupt_dot_limit = cos(radians(abrupt_normal_angle_degrees))
    abrupt_count = int(np.count_nonzero(normal_dots < abrupt_dot_limit))

    quad_valid = (
        valid[:-1, :-1]
        & valid[:-1, 1:]
        & valid[1:, :-1]
        & valid[1:, 1:]
    )
    p00 = points_physical[:-1, :-1, :]
    p01 = points_physical[:-1, 1:, :]
    p10 = points_physical[1:, :-1, :]
    p11 = points_physical[1:, 1:, :]
    top = np.linalg.norm(p01 - p00, axis=-1)
    bottom = np.linalg.norm(p11 - p10, axis=-1)
    left = np.linalg.norm(p10 - p00, axis=-1)
    right = np.linalg.norm(p11 - p01, axis=-1)
    edge_lengths = np.stack((top, bottom, left, right), axis=-1)
    minimum_edge = np.min(edge_lengths, axis=-1)
    maximum_edge = np.max(edge_lengths, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        distortion = maximum_edge / minimum_edge

    triangle_one = np.cross(p01 - p00, p10 - p00)
    triangle_two = np.cross(p11 - p10, p11 - p01)
    triangle_one_magnitude = np.linalg.norm(triangle_one, axis=-1)
    triangle_two_magnitude = np.linalg.norm(triangle_two, axis=-1)
    quad_area = 0.5 * (triangle_one_magnitude + triangle_two_magnitude)
    valid_quad_areas = quad_area[quad_valid]
    valid_quad_distortion = distortion[quad_valid]
    valid_quad_count = int(np.count_nonzero(quad_valid))
    area_scale = median_distance * median_distance if np.isfinite(median_distance) else 1.0
    degenerate_area_limit = max(
        np.finfo(np.float64).eps, area_scale * degenerate_area_relative_epsilon
    )
    degenerate = quad_valid & (
        ~np.isfinite(quad_area)
        | ~np.isfinite(distortion)
        | (quad_area <= degenerate_area_limit)
        | (minimum_edge <= np.sqrt(degenerate_area_limit))
    )
    degenerate_count = int(np.count_nonzero(degenerate))
    distorted_count = int(
        np.count_nonzero(quad_valid & (distortion > metric_distortion_limit))
    )

    triangle_dot = np.sum(triangle_one * triangle_two, axis=-1)
    triangle_denominator = triangle_one_magnitude * triangle_two_magnitude
    with np.errstate(divide="ignore", invalid="ignore"):
        triangle_cosine = triangle_dot / triangle_denominator
    folded = quad_valid & ~degenerate & (triangle_cosine < 0.0)
    folded_count = int(np.count_nonzero(folded))

    if normal_offsets_um is None:
        offset_in_bounds_fraction = float("nan")
    else:
        offsets = np.asarray(normal_offsets_um, dtype=np.float64)
        if offsets.ndim != 1 or not np.isfinite(offsets).all():
            raise ValueError("normal_offsets_um must be a finite 1-D sequence")
        offset_denominator = int(offsets.size * normal_count)
        offset_in_bounds_count = 0
        signed_normal_voxels = (
            normal_sign * normal_field.positive_xyz / voxel_xyz_um
        )
        for offset in offsets:
            offset_points = points_xyz + offset * signed_normal_voxels
            offset_in_bounds = normal_field.valid & (
                (offset_points[..., 0] >= 0.0)
                & (offset_points[..., 0] <= x_size - 1)
                & (offset_points[..., 1] >= 0.0)
                & (offset_points[..., 1] <= y_size - 1)
                & (offset_points[..., 2] >= 0.0)
                & (offset_points[..., 2] <= z_size - 1)
            )
            offset_in_bounds_count += int(np.count_nonzero(offset_in_bounds))
        offset_in_bounds_fraction = _safe_fraction(
            offset_in_bounds_count, offset_denominator
        )

    warnings: list[str] = []
    if valid_count == 0:
        warnings.append("surface has no valid finite vertices")
    if in_bounds_count < valid_count:
        warnings.append("one or more valid surface vertices are outside the volume")
    if normal_count < valid_count:
        warnings.append("one or more valid vertices lack a non-degenerate normal")
    if discontinuity_count:
        warnings.append("surface contains unusually long neighbor edges")
    if wrap_count:
        warnings.append("surface contains neighbor-wrap-risk coordinate jumps")
    if degenerate_count:
        warnings.append("surface contains degenerate quads")
    if folded_count:
        warnings.append("surface contains folded or self-inverting quads")
    if distorted_count:
        warnings.append("surface contains strongly metric-distorted quads")
    if abrupt_count:
        warnings.append("surface contains abrupt adjacent-normal flips")
    if np.isfinite(offset_in_bounds_fraction) and offset_in_bounds_fraction < 1.0:
        warnings.append("one or more requested normal-offset samples leave the volume")

    return GeometryValidationReport(
        surface_shape=surface.shape,
        volume_shape_zyx=volume_shape,
        coordinate_min_xyz=coordinate_min,
        coordinate_max_xyz=coordinate_max,
        valid_vertex_count=valid_count,
        valid_vertex_fraction=_safe_fraction(valid_count, total_vertices),
        in_bounds_vertex_count=in_bounds_count,
        in_bounds_vertex_fraction=_safe_fraction(in_bounds_count, valid_count),
        normal_valid_vertex_count=normal_count,
        normal_valid_vertex_fraction=_safe_fraction(normal_count, valid_count),
        possible_neighbor_edge_count=possible_edge_count,
        valid_neighbor_edge_count=valid_edge_count,
        valid_neighbor_edge_fraction=_safe_fraction(valid_edge_count, possible_edge_count),
        median_neighbor_distance_um=median_distance,
        p95_neighbor_distance_um=p95_distance,
        max_neighbor_distance_um=max_distance,
        continuity_limit_um=continuity_limit,
        discontinuity_edge_count=discontinuity_count,
        discontinuity_edge_fraction=_safe_fraction(discontinuity_count, valid_edge_count),
        neighbor_wrap_limit_um=wrap_limit,
        neighbor_wrap_risk_edge_count=wrap_count,
        neighbor_wrap_risk_edge_fraction=_safe_fraction(wrap_count, valid_edge_count),
        valid_quad_count=valid_quad_count,
        median_quad_area_um2=_safe_percentile(valid_quad_areas, 50.0),
        degenerate_quad_count=degenerate_count,
        degenerate_quad_fraction=_safe_fraction(degenerate_count, valid_quad_count),
        folded_quad_count=folded_count,
        folded_quad_fraction=_safe_fraction(folded_count, valid_quad_count),
        metric_distortion_limit=float(metric_distortion_limit),
        metric_distortion_p95=_safe_percentile(
            valid_quad_distortion[np.isfinite(valid_quad_distortion)], 95.0
        ),
        distorted_quad_count=distorted_count,
        distorted_quad_fraction=_safe_fraction(distorted_count, valid_quad_count),
        abrupt_normal_angle_degrees=float(abrupt_normal_angle_degrees),
        abrupt_normal_flip_edge_count=abrupt_count,
        abrupt_normal_flip_edge_fraction=_safe_fraction(abrupt_count, normal_dots.size),
        offset_sample_in_bounds_fraction=offset_in_bounds_fraction,
        warnings=tuple(warnings),
    )


__all__ = [
    "GeometryValidationReport",
    "NativeSurfaceSampler",
    "NormalField",
    "NormalSamplingSpec",
    "RenderResult",
    "RenderedTile",
    "SurfaceGrid",
    "TrilinearSample",
    "estimate_surface_normals",
    "sample_trilinear_zyx",
    "validate_surface_geometry",
]
