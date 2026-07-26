# Methodology

`tifxyz-doctor` has two layers with different evidentiary weight:

1. **Contract integrity** inspects the bytes and models pinned reader
   behavior. Its findings are deterministic, machine-checkable statements
   about the package under the pinned implementation assumptions below.
2. **Geometry review** computes thresholded measurements. A cue identifies a
   location or distribution worth inspecting; it is not a correctness label.

The separation is preserved in the CLI, JSON schema, HTML report, benchmark,
and exit codes.

## Reference implementation

The interoperability model is pinned to ScrollPrize/Villa commit
[`1162bcab4bc769b12993fc320c69c14fdb4a4fa5`](https://github.com/ScrollPrize/villa/tree/1162bcab4bc769b12993fc320c69c14fdb4a4fa5).
The relevant sources are:

- [Python `TifxyzReader`](https://github.com/ScrollPrize/villa/blob/1162bcab4bc769b12993fc320c69c14fdb4a4fa5/vesuvius/src/vesuvius/tifxyz/reader.py#L270-L610);
- [C++ `load_quad_from_tifxyz_impl`](https://github.com/ScrollPrize/villa/blob/1162bcab4bc769b12993fc320c69c14fdb4a4fa5/volume-cartographer/core/src/QuadSurface.cpp#L2161-L2510);
- [C++ finite point-sample predicate](https://github.com/ScrollPrize/villa/blob/1162bcab4bc769b12993fc320c69c14fdb4a4fa5/volume-cartographer/core/src/QuadSurface.cpp#L275-L279);
- [Villa's `p10-p01` triangle split](https://github.com/ScrollPrize/villa/blob/1162bcab4bc769b12993fc320c69c14fdb4a4fa5/volume-cartographer/core/src/ABFFlattening.cpp#L356-L379).

The real-data controls use files and assertions from the earlier pinned Villa
fixture revision
[`f6376d7`](https://github.com/ScrollPrize/villa/tree/f6376d715f9568dd8d976c8cfb57fe7619387302):

- [fixture provenance](https://github.com/ScrollPrize/villa/blob/f6376d715f9568dd8d976c8cfb57fe7619387302/volume-cartographer/core/test/test_quadsurface_fixtures.cpp);
- [real TIFXYZ load tests](https://github.com/ScrollPrize/villa/blob/f6376d715f9568dd8d976c8cfb57fe7619387302/volume-cartographer/core/test/test_load_quad_from_tifxyz.cpp).

These links are evidence for the modeled behavior, not a proposal that quirks
in either implementation should become the permanent file-format
specification.

### Executable differential

Source inspection is backed by
[`verification/run_reader_differential.py`](../verification/run_reader_differential.py).
It dynamically loads the exact pinned Python `reader.py` and `types.py`, writes
five synthetic TIFXYZ packages, and sends the same packages through the real
C++ `load_quad_from_tifxyz` path using the official stable
`vc_tifxyz2obj` binary.

As observed on 2026-07-26, the tested pairing is the official stable Apple
Silicon release at commit
`05ff9ea39bf1077f8749fc1fe06f43a548350143` and the exact Python reader files
at audit-date repository HEAD
`1162bcab4bc769b12993fc320c69c14fdb4a4fa5`. The stable commit is 94 commits
older, so this is intentionally a stable-release-versus-repository comparison
rather than a same-commit binary/source comparison. Intervening path history
shows no Python-reader edit and no change to the relevant C++ full-load
`z <= 0` gate, canonical `uint8` mask threshold, or integer-multiple mask
application.

The converter reports that each fixture loads as the full 5×5 grid. At the
tested release, its `loc_valid` predicate uses a half-open `rows-2` by `cols-2`
rectangle, so OBJ emission retains 3×3 quad origins, 18 triangles, and their
4×4 corner union of 16 vertices. This explains the all-valid control without
treating its OBJ vertex count as a raw loader-valid count. Rejecting the center
removes four retained quads and yields the calibrated 12-vertex/10-face
signature.

The official release asset and binary hashes, recorded source blob identities,
controls, loaded-grid dimensions, eight exporter-size probes, observable OBJ
signatures, and all five declared reader-case results are preserved in
[`verification/reader-differential-results-v1.json`](../verification/reader-differential-results-v1.json).
The runner enforces the asset and executable hashes. A separate read-only
repository-history check established that the recorded converter and geometry
blobs are identical at the release and pinned repository commits; the runner
does not fetch GitHub source.

## Contract-integrity layer

### Package and metadata

A package is expected to contain `x.tif`, `y.tif`, `z.tif`, and `meta.json`;
`mask.tif` is optional. The checker reports missing files rather than filling
them in.

`meta.json` must parse as an object. The checker also rejects NaN and infinity,
which Python's default decoder accepts even though they are not strict JSON
numbers. It requires:

- a non-empty string `uuid`;
- `scale` beginning with two positive, finite numbers in on-disk
  `[x_scale, y_scale]` order;
- when present, `bbox` shaped as
  `[[min_x,min_y,min_z],[max_x,max_y,max_z]]`, with finite ordered bounds.

Non-finite JSON numbers are errors. Nonstandard `format`, `type`, coordinate
dtypes, mask dtypes, or an invalid optional area value are warnings because the
pinned readers may still accept or coerce them.

The checker records `area`, `area_vx2`, and `area_cm2` separately. When
`area_vx2` or `area_cm2` is present but `area` is absent, it emits
`metadata-area-schema-divergence`: the pinned Python reader retains the
suffixed fields only in `extra` and sets `Tifxyz.area` from `area`, so the
property remains unset. The warning identifies an observable producer/consumer
schema mismatch; it does not reject the suffixed measurements.

For a recursively scanned collection, UUID reuse is a collection-level
warning. Duplicate UUIDs may be intentional, so the tool reports the paths and
does not relabel each individual package as corrupt.

### TIFF and coordinate invariants

Each component must decode as one two-dimensional numeric image, and the three
coordinate shapes must match. Canonical invalid vertices are exactly
`(-1, -1, -1)`. A partial sentinel, with `-1` in only some channels, is an
error because pinned C++ validity paths do not all treat it identically.
NaN or infinity in any coordinate is also an error.

Canonical Villa writers emit `float32` coordinates and `uint8` masks with
values 0 and 255. Other numeric dtypes are retained for diagnosis and reported
as warnings rather than silently cast by the contract layer.

### Reader validity models

Let `P` denote the Python-valid mask and `C` the conservative finite C++
point-valid mask at stored coordinate resolution.

Without an applicable mask:

```text
P = finite(z) and z > 0
```

The C++ loader first rewrites every point with `z <= 0` to the full sentinel.
Its later sample predicate requires all three channels to be finite and each
channel to differ from `-1`.

For an exact-size mask `M`, Python uses `M != 0`. C++ uses `M >= 255` only to
invalidate and cannot undo the earlier `z <= 0` gate. Consequently, low
nonzero mask values and a mask that selects nonpositive or nonfinite `z` are
reported as interoperability errors.

For an integer-multiple higher-resolution mask, Python ignores the shape
mismatch and derives validity from `z`. C++ maps every high-resolution mask
sample to a stored vertex; all samples in the covered block must be at least
255 for that vertex to survive. This shape relation is reported because the
same package can produce a different surface.

The report defines:

```text
portable_valid = P and C
```

This is a conservative interoperability diagnostic, not a new official
standard. A portable face is a stored grid cell whose four corner vertices are
portable-valid.

### Bounding box

The measured bounding box is computed only from portable-valid vertices. It is
compared with the declared box using a per-axis tolerance:

```text
tolerance = 1e-4 + 1e-6 * max(max(abs(actual_axis)), 1)
```

This comparison is omitted when either side is unavailable.

## Geometry-review layer

All geometry calculations operate in stored grid coordinates. A cell is
renderable only if all four of its corner vertices are valid under the loaded
surface mask.

### Topology

Valid vertices and renderable cells are counted with 4-connectivity. Enclosed
face-lattice gaps pair a 4-connected foreground with an 8-connected
background, avoiding an ambiguous diagonal pinch. Invalid background connected
to the array boundary is exterior; every remaining 8-connected invalid region
is reported as an enclosed-gap cue with size, bounding box, and centroid.

This is a topological observation only. A hole can be deliberate.

### Directional spacing

For every valid horizontal and vertical grid edge, the tool computes its 3-D
Euclidean length. By default, the direction-specific positive finite median is
the reference spacing:

```text
r_x = horizontal_edge_length / median(horizontal_edge_length)
r_y = vertical_edge_length   / median(vertical_edge_length)
```

The default short- and long-edge cues are `r < 0.5` and `r > 2.0`.

Explicit `--expected-spacing-x` or `--expected-spacing-y` values replace the
corresponding median for Jacobian normalization and enable global target
comparisons. A single supplied target enables its directional ratio; both
targets are needed for the isotropic-factor and anisotropy cues. The edge
outlier ratios themselves remain relative to the observed directional medians.

`meta.scale` is **not** automatically interpreted as expected 3-D edge
spacing. It describes the stored parameterization, is written in x/y order,
and is reversed by the pinned Python reader for row/column indexing. Treating
it as an edge-length target would produce false conclusions for existing
surfaces.

### Triangles, area, and folds

For a cell

```text
p00 = coordinates[row,   col]
p01 = coordinates[row,   col+1]
p10 = coordinates[row+1, col]
p11 = coordinates[row+1, col+1]
```

the primary split matches Villa's `p10-p01` diagonal:

```text
triangle A = (p00, p01, p10)
triangle B = (p01, p11, p10)
```

The normalized triangle-area sine is the cross-product magnitude divided by
the product of its two originating edge lengths. Values at or below `1e-6`, or
non-finite values, are degenerate cues; values at or below `1e-3` are also
counted as near-degenerate.

A primary-split fold cue occurs when the two unit triangle normals have a
negative dot product. The alternate `p00-p11` split is computed as a
sensitivity check. A quad whose fold classification changes with the diagonal
is labeled triangulation-sensitive rather than unconditionally folded.

Quad surface area is the sum of the two triangle areas. It is reported in
squared coordinate units, described as voxel² when the input coordinates are
voxel coordinates.

### Bilinear-center Jacobian

The four points define a bilinear patch. At its center, the directional
derivatives are:

```text
twist = p11 - p10 - p01 + p00
du = (p01 - p00) + 0.5 * twist
dv = (p10 - p00) + 0.5 * twist
```

After dividing `du` and `dv` by their respective normalization spacings, they
form a 3-by-2 Jacobian `J`. The eigenvalues of the Gram matrix
`G = JᵀJ` are `lambda_max` and `lambda_min`; the singular values are
`sigma_max = sqrt(lambda_max)` and `sigma_min = sqrt(lambda_min)`.

The recorded metrics and default cue thresholds are:

| Metric | Definition | Default cue |
| --- | --- | --- |
| Condition number | `sigma_max / sigma_min` | `> 4` |
| Symmetric stretch | `max(sigma_max, 1 / sigma_min)` | `> 2` |
| Area ratio | `sigma_max * sigma_min` | `< 0.25` or `> 4` |
| Shear | `abs(dot(du,dv)) / (norm(du) * norm(dv))` | `> cos(30°) ≈ 0.8660` |
| Symmetric Dirichlet | `lambda_max + lambda_min + 1/lambda_max + 1/lambda_min` | `> 20` |

These are standard singular-value-derived local parameterization measures.
The Challenge's July 2026
[flattening documentation](https://scrollprize.org/2026_open_problems#2d-parameterization-and-flattening)
states that VC3D's production `flatboi` tool uses
[SLIM](https://igl.ethz.ch/projects/slim/) to minimize Symmetric Dirichlet
energy. The audit exposes that same energy locally as a diagnostic; it does not
run or replace SLIM.

For additional Vesuvius-context motivation for measuring flattening distortion,
see Parker, Seales, and Shor,
[*Quantitative Distortion Analysis of Flattening Applied to the Scroll from
En-Gedi*](https://arxiv.org/abs/2007.15551). The formulas and thresholds above
are transparent choices made by this tool; they are not values endorsed by
that paper or guaranteed by the Challenge.

### Normal continuity

The cell normal is the normalized sum of the two primary-split unit triangle
normals. Horizontally and vertically adjacent renderable cells are compared by
the angle between their normals. The default cue threshold is greater than
75 degrees; a negative dot product is additionally counted as an orientation
flip.

A sharp normal change may be real papyrus geometry. The report provides the
largest locations for inspection rather than treating them as defects.

### Sampled nonlocal proximity

The proximity pass searches for valid vertices that are close in 3-D but not
near each other in grid coordinates:

- at most 100,000 vertices are selected by a deterministic regular stride;
- the 3-D distance threshold is `0.25 * sqrt(spacing_x * spacing_y)`;
- pairs within Chebyshev grid distance 4 are excluded;
- at most 10,000 pair records are retained, while the total count continues;
- a deterministic spatial hash checks the 27 neighboring buckets.

This is a vertex-sampling heuristic. It can miss contacts between triangles,
and a returned pair is a contact candidate—not an intersection proof.

### Overlay and examples

Each cue stores bounded, deterministic examples in row/column coordinates.
The overlay colors only the exact union of cells touched by a configured local
threshold crossing. Edge cues are projected to every incident valid cell,
normal-jump cues to both adjacent cells, and nonlocal-proximity endpoints to
their incident cells. Direct cell cues include degeneracy, fold and
triangulation sensitivity, Jacobian condition, symmetric stretch, area ratio,
shear, and Symmetric Dirichlet energy. Enclosed face-lattice gaps are overlaid
separately in magenta.

Within that binary cue mask, color intensity uses the maximum normalized local
score as a navigation aid. Dark valid cells have no localized threshold
crossing. The color is not a calibrated probability or aggregate quality
score. Report-level cues without a unique cell support, such as explicit
global-spacing drift, remain in the findings list rather than being painted
across the surface.

## Reproducibility

Public JSON excludes internal NumPy arrays, rejects NaN and infinity during
serialization, sorts contract findings deterministically, and records every
configuration threshold. Each benchmark snapshot also records the SHA-256 of
its source manifest.

The downloader verifies every file against its recorded byte size and SHA-256.
Downloaded datasets are ignored by version control and keep their source
license. See [Recorded results](results.md) for the current snapshot and its
interpretation boundary.
