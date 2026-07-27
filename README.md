# tifxyz-doctor

`tifxyz-doctor` is a deterministic preflight and geometry-diagnostics tool for
the TIFXYZ surface meshes used by Vesuvius Challenge software. In an exhaustive
metadata census of the 450 original/normalized roots in the registry snapshot
observed on 2026-07-22, 277 roots—including all 264 normalized roots—used the
literal UUID `output_tifxyz`. Across 138 inspected PHerc1203 current/version
snapshots, `area_vx2` and `area_cm2` were present while the `area` key consumed
by the pinned Python API was absent, leaving `Tifxyz.area` unset. A separate
recorded regression reproduced three normalized packages that were registered
at the time and contained only a 2×2 sentinel grid; the affected entries were
repaired upstream after the report. The tool also produces exact grid locations
for thresholded geometry review cues.

The tool is read-only and CPU-only: it does not repair, rewrite, or silently
normalize source data.

The distinction between its two result classes is intentional:

| Result class | Meaning | Suitable for automation? |
| --- | --- | --- |
| **Contract finding** | A machine-checkable file, metadata, numeric, or pinned-reader interoperability condition. | Yes. `check` and `scan` can fail with exit status 2. |
| **Geometry review cue** | A thresholded observation that points a human toward an unusual location. It is not a claim that the surface is wrong. | No. Inspect the JSON, HTML, and overlay in context. |

The tool is early-stage research software. Its useful narrow claim is that a
TIFXYZ package can be checked against explicit, pinned reader behavior before
expensive rendering, training, or manual tracing work begins.

## What it checks

`check` audits the raw package without first passing it through a reader that
could normalize the evidence:

- required files, JSON metadata, non-finite numbers, UUID, scale, bounding box,
  and area field compatibility;
- single-image, two-dimensional TIFF decoding, dimensions, and dtypes;
- coordinate shape agreement, finite values, and canonical
  `(-1, -1, -1)` sentinels;
- exact-size and integer-multiple `mask.tif` behavior in pinned Python and C++
  reader models;
- Python/C++ validity disagreements;
- a conservative count of vertices and faces usable under both reader models;
- declared versus measured bounding boxes.

`scan` applies those checks recursively and also reports duplicate UUID groups
within the scanned collection.

`audit` adds geometry and grid-topology measurements:

- valid vertex and face components, face-lattice holes, and isolated vertices;
- direction-specific edge-length deviations;
- degenerate or folded triangles using Villa's `p10-p01` split, plus
  alternate-diagonal sensitivity;
- bilinear-center Jacobian condition, symmetric stretch, area ratio, shear,
  and symmetric Dirichlet energy;
- adjacent-cell normal jumps; and
- sampled nonlocal vertex proximity candidates.

See [Methodology](docs/methodology.md) for definitions and thresholds and
[Recorded results](docs/results.md) for the reproducible benchmarks and
bounded corpus evidence.

## Evidence snapshot

A bounded public-data scan on 2026-07-26 produced three concrete contract and
collection results:

- 277 of 450 registered original/normalized roots reuse the literal UUID
  `output_tifxyz`, which the tool treats as a collection warning rather than a
  per-package error; and
- all 138 inspected PHerc1203 current/version metadata snapshots use
  `area_vx2` and `area_cm2` without the `area` key consumed by the pinned
  Python API; and
- three metadata-registered normalized packages contained no valid vertex or
  face in a 2026-07-26 audit against the registry snapshot last modified on
  2026-07-22; their exact, hash-pinned objects formed a 3/3 recorded regression
  before upstream remediation.

The first two observations do not depend on whether the semantic-empty registry
entries were intentional. One empty entry contains `z_dbg` in its identifier;
the regression establishes only that the exact registry paths resolved to the
recorded sentinel-only bytes, not why they exist or whether a production
workflow uses them.

### Upstream resolution (2026-07-27)

A Villa administrator confirmed that the three empty packages came from
legacy OBJ data sent through the wrong pipeline, then removed or regenerated
the affected artifacts. The primary Discord evidence is linked directly:
[diagnosis](https://discord.com/channels/1079907749569237093/1243576621722767412/1531206054682165309)
and
[remediation confirmation](https://discord.com/channels/1079907749569237093/1243576621722767412/1531220162190245909).
A fresh audit against the registry published at 2026-07-27 08:41:44 UTC found:

- the two PHerc0332 segments now register valid `tifxyz_original` packages
  with 132,462 and 134,880 portable-valid vertices; and
- the PHerc0500P2 normalized path now contains a valid 583×339 package with
  113,448 portable-valid vertices.

All three current packages pass with zero findings. The exact current registry
identity, replacement object hashes, and audit counts are recorded in
[`benchmarks/public-empty-resolution-2026-07-27.json`](benchmarks/public-empty-resolution-2026-07-27.json).
The original 3/3 manifest and result remain a dated, content-addressed record
of the reported bytes; the affected entries were repaired upstream after the
report, and a fresh live fetch is no longer expected to reproduce those
retired/replaced objects.

The complete bounded-scan evidence—including selection limits, exact URLs,
hashes, timestamps, and caveats—is recorded in
[`benchmarks/public-corpus-scan-2026-07-26.json`](benchmarks/public-corpus-scan-2026-07-26.json).
These are package/interoperability observations, not claims about whether a
surface traces the correct papyrus sheet. The concise
[claims review](docs/claims-review.md) states what each number
supports and what must not be inferred.

## Relation to the current unwrapping problem

The Challenge's July 2026
[Open Problems](https://scrollprize.org/2026_open_problems#meshes-adding-connectivity)
page identifies holes, mergers, and sheet switches as recurring mesh failure
modes and explicitly invites geometry-processing, optimization, and C++
contributions. TIFXYZ Doctor is a diagnostic contribution at that boundary:
it detects unusable or nonportable mesh packages before downstream work and
localizes several topology and distortion signals for inspection. It does not
claim to repair a wrong sheet trace or measure sheet-switch recall.

The geometry report includes per-cell Symmetric Dirichlet energy. This is not
just a generic mesh statistic: VC3D's production `flatboi` flattener uses
[SLIM](https://igl.ethz.ch/projects/slim/), which minimizes Symmetric
Dirichlet energy, according to the Challenge's
[flattening documentation](https://scrollprize.org/2026_open_problems#2d-parameterization-and-flattening).
TIFXYZ Doctor exposes the same distortion quantity as a local diagnostic; it
does not reproduce or replace SLIM optimization.

## Quick start

Python 3.10 or newer is required. NumPy and Pillow are the only required
runtime dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The optional TIFF extra installs `tifffile`; the geometry loader still falls
back to Pillow when a TIFF codec is unavailable.

```bash
python -m pip install -e '.[tiff]'
```

Check one package and write strict, deterministic JSON:

```bash
tifxyz-doctor check /path/to/tifxyz \
  --json contract-report.json
```

Treat warnings as a nonzero result in CI:

```bash
tifxyz-doctor check /path/to/tifxyz \
  --json contract-report.json \
  --fail-on-warning
```

Scan a collection:

```bash
tifxyz-doctor scan /path/to/segments \
  --json collection-report.json
```

Generate the full geometry report:

```bash
tifxyz-doctor audit /path/to/tifxyz \
  --json audit.json \
  --html audit.html \
  --overlay audit.png \
  --fail-on-integrity
```

The HTML file is self-contained. The PNG colors only cells touched by an exact
local threshold crossing, in stored grid coordinates; dark valid cells have no
localized cue. Run `check` first on unknown input: `audit` must load the arrays
to compute geometry and therefore cannot always continue after a structural
error.

Exit behavior is deliberately asymmetric:

- `check` exits 2 for contract errors, and also for warnings with
  `--fail-on-warning`;
- `scan` exits 2 for any contained contract error or collection error, and
  also for warnings with `--fail-on-warning`;
- `audit` exits 2 when structural input prevents geometry loading and emits a
  `tifxyz-audit-failure-v1` JSON result when `--json` is requested; otherwise
  it does not fail merely because geometry cues exist, and
  `--fail-on-integrity` additionally makes embedded contract errors fatal.

Run `tifxyz-doctor COMMAND --help` for every configurable audit threshold.

## Pinned synthetic reader-interoperability checks

This is deliberately not presented as a public-corpus finding. None of the 450
registered original/normalized roots in the bounded scan exposed a
`mask.tif`; the matrix below documents pinned implementation behavior and
synthetic regression cases.

At the 2026-07-26 audit-date repository HEAD,
[`1162bcab`](https://github.com/ScrollPrize/villa/tree/1162bcab4bc769b12993fc320c69c14fdb4a4fa5),
the Python and C++ readers do not assign identical meaning to every possible
mask:

| Input condition | Python reader at `1162bcab` | C++ full-load rules at `1162bcab` |
| --- | --- | --- |
| No mask | Uses finite `z > 0`. | Invalidates `z <= 0` before later validity checks. |
| Exact-size mask | Any nonzero value is valid; the mask can select a point whose original `z <= 0`. | Only values at least 255 retain a point, and the earlier `z <= 0` gate cannot be undone. |
| Integer-multiple higher-resolution mask | Ignores the mismatched mask and falls back to `z > 0`. | Applies the mask by invalidating the corresponding stored vertex if a covered high-resolution sample is below 255. |
| Other mismatched mask | Ignores it and falls back to `z > 0`. | Does not apply it. |

As observed on 2026-07-26, Villa's official stable Apple Silicon
`vc_tifxyz2obj` release (`05ff9ea`) and the exact Python `TifxyzReader` files at
the audit-date repository HEAD (`1162bcab`) produce different outcomes on each
of three synthetic mask fixtures. The harness passes all five declared fixture
expectations: two controls and three confirmed disagreements. In every
disagreement, Python retains all 25 input-grid vertices, the stable converter
matches the calibrated 12-vertex/10-face OBJ signature, and TIFXYZ Doctor
models 25 Python-valid points versus 24 finite C++ points.

Why does the all-valid 5×5 control produce 16 OBJ vertices and 18 faces rather
than 25 vertices and 32 faces? The executable reports that it loaded the full
5×5 array. Its
[`loc_valid` call](https://github.com/ScrollPrize/villa/blob/05ff9ea39bf1077f8749fc1fe06f43a548350143/volume-cartographer/apps/src/vc_tifxyz2obj.cpp#L291-L308)
uses a
[half-open `rows-2` by `cols-2` rectangle](https://github.com/ScrollPrize/villa/blob/05ff9ea39bf1077f8749fc1fe06f43a548350143/volume-cartographer/core/src/Geometry.cpp#L126-L146),
so only 3×3 quad origins survive. Two triangles per quad give 18 faces, and the
converter emits only their 4×4 corner union: 16 vertices. Rejecting the center
removes four of those nine quads, leaving the calibrated 12 vertices and 10
faces. This is boundary filtering during OBJ emission, not loader resizing or
decimation. The authenticated harness tests five square sizes from 3×3 through
7×7 plus three rectangular sizes. For those eight all-valid `H×W` grids with
`H,W ≥ 3`, the emitted counts match `(H-1)(W-1)` vertices and
`2(H-2)(W-2)` faces.

The stable release commit is 94 commits older than the audit-date repository
revision. That separation defines the tested stable-release-versus-repository
pairing; it is not a same-commit binary/source comparison. A separate read-only
path-history check found no intervening Python-reader edit, no change to the
relevant C++ full-load mask rules, and identical converter and `Geometry.cpp`
blobs at both commits. The executable harness records those source identities
but does not itself fetch or compare GitHub source. The
[harness and provenance](verification/README.md) and
[machine-readable result](verification/reader-differential-results-v1.json)
record the complete pairing and hashes.

These observations come directly from the pinned
[Python mask reader and load path](https://github.com/ScrollPrize/villa/blob/1162bcab4bc769b12993fc320c69c14fdb4a4fa5/vesuvius/src/vesuvius/tifxyz/reader.py#L437-L446)
and
[C++ TIFXYZ load path](https://github.com/ScrollPrize/villa/blob/1162bcab4bc769b12993fc320c69c14fdb4a4fa5/volume-cartographer/core/src/QuadSurface.cpp#L2161-L2510).
They describe that revision, not a permanent specification. The report records
both modeled rules and a `portable_valid_vertex_count` so a future revision can
be compared rather than guessed.

One related trap: `meta.json` stores scale in `[x_scale, y_scale]` order, while
the Python reader reverses it for row/column indexing. `tifxyz-doctor` does
**not** assume those values are the expected 3-D edge lengths. Geometry is
normalized by observed directional median spacing unless the caller explicitly
provides `--expected-spacing-x` and/or `--expected-spacing-y`.

The checker also preserves `area`, `area_vx2`, and `area_cm2` separately. If
only the suffixed fields exist, it emits a warning: the pinned Python reader
keeps them in `extra` but leaves `Tifxyz.area` unset. This is an observable
schema-consumer divergence, not a claim that the suffixed values are wrong.

## Reproduce the real-data benchmarks

The repository does not redistribute Vesuvius data. The manifest records an
HTTPS source URL, byte count, and SHA-256 for every object. The current ten
cases total 1,818,055 bytes.

```bash
python scripts/fetch_benchmark.py
python scripts/run_benchmark.py
```

The first command verifies already-downloaded files or downloads missing ones
to the ignored `benchmark-data/` directory. The second regenerates
`benchmarks/realdata-results-v0.1.0.json`; it returns status 2 if any benchmark
case has a contract error.

Fetch a single named case with:

```bash
python scripts/fetch_benchmark.py \
  --case pherc0172-20241113070770
```

The three PHerc0172 controls are files used by pinned Villa tests. The other
seven cases are unlabeled auto-grown surfaces selected for format and stress
coverage. Their roles are provenance labels, not geometry ground truth.

The public empty-artifact regression suite records the three normalized
packages identified by the 2026-07-26 corpus scan. The original inputs totaled
81,252 bytes and were size- and SHA-256-pinned. Upstream repaired the entries
on 2026-07-27, so the live fetch command is now expected to fail with a
missing object or hash mismatch. The regression remains runnable only with an
exact cache downloaded before remediation.

```bash
python scripts/run_public_regressions.py
```

A successful regression run means all three inputs reproduced the expected
`no-portable-valid-vertex` and `no-valid-face` detections. The source packages
themselves correctly retain `error` status in the integrity output.

## Development

The test suite uses only the standard-library test runner:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Useful release checks:

```bash
PYTHONPATH=src python -m tifxyz_doctor.cli --help
python scripts/fetch_benchmark.py
python scripts/run_benchmark.py
```

Run `scripts/run_public_regressions.py` separately only when the exact
pre-remediation cache is present.

## Limits

- There is no authoritative labeled corpus of correct and incorrect TIFXYZ
  geometry in this repository. Review-cue precision and recall are therefore
  not claimed.
- Enclosed gaps may be intentional. Sharp normal changes may represent real
  folds. Stretch and edge thresholds depend on the intended parameterization.
- Nonlocal proximity samples vertices; it does not test triangle-triangle
  intersection or prove self-contact.
- The default normalization detects local irregularity but cannot identify a
  globally uniform scale error. Supply explicit expected spacing to test a
  target.
- Results model the pinned Villa reader revision above. Reader changes should
  trigger a model and benchmark review.
- The implementation is CPU-only and keeps coordinate arrays in memory. Very
  large surfaces may need tiling in a future release.

## Licensing

The code, documentation, benchmark manifest, and recorded numeric snapshot are
MIT licensed; see [LICENSE](LICENSE).

Downloaded Vesuvius data is separate and is not covered by the code license.
The benchmark manifest declares
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) and the
downloader prints that notice before fetching. Review the
[Vesuvius Challenge data publication](https://scrollprize.org/pdf/main.pdf)
and the source dataset terms before reuse, especially for commercial work.
