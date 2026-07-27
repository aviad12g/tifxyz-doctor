# July 2026 Progress Prize submission: TIFXYZ Doctor

Update, 2026-07-27: a Villa administrator confirmed that the three
sentinel-only packages came from legacy OBJ data processed through the wrong
pipeline, then removed or regenerated them. Fresh audits of all three current
registrations pass. The original observation is now a dated, content-addressed
finding whose report was followed by an upstream repair; see
[`public-empty-resolution-2026-07-27.json`](../benchmarks/public-empty-resolution-2026-07-27.json).

## One-sentence summary

TIFXYZ Doctor is a deterministic, CPU-only preflight for Vesuvius TIFXYZ
surfaces that identified collection-identity reuse across 277 of 450 registered
roots and an area-schema/API mismatch in all 138 inspected PHerc1203 snapshots,
reproduced three registry-listed sentinel-only normalized packages that were
repaired upstream the next day, and
localizes thresholded topology and flattening-distortion cues for review.

## Concrete public-data findings

The strongest evidence is corpus-first:

| Observation | Bounded evidence | Operational consequence | Interpretation boundary |
| --- | ---: | --- | --- |
| Registered roots using the literal UUID `output_tifxyz` | 277 of 450 roots | UUID cannot uniquely identify those packages for collection consumers that use it as a key. | Reported as a collection warning, not per-surface coordinate corruption. |
| PHerc1203 snapshots with `area_vx2`/`area_cm2` but no `area` | 138 of 138 snapshots | The pinned Python reader retains the suffixed values in `extra` while `Tifxyz.area` remains unset. | The suffixed measurements are preserved and are not called numerically wrong. |
| Registered normalized packages with no valid vertex or face on 2026-07-26 | 3 packages | A consumer selecting one received no renderable surface. All three exact inputs are covered by a hash-pinned recorded 3/3 regression; the live entries were repaired on 2026-07-27. | One identifier contains `z_dbg`. The audit establishes the observed bytes and registry paths, not intent or production use. |

The UUID and area-schema observations do not depend on whether the
semantic-empty registry entries were intentional.

The breadth scan covered 242 TIFXYZ directories, 8,836,120 stored grid points,
and 726 coordinate TIFFs. Selection limits, exact URLs, sizes, SHA-256 hashes,
timestamps, and caveats are recorded in
`benchmarks/public-corpus-scan-2026-07-26.json`.

The three semantic-empty artifacts were independently reproduced from an
81,252-byte pinned manifest before remediation. All 3 of 3 recorded
expectations pass: each exact source produced `status: error`, zero
portable-valid vertices and faces, and both `no-portable-valid-vertex` and
`no-valid-face`. The original objects are no longer all available at those
live paths, so this is a dated, content-addressed result rather than a
currently refetchable corpus. One identifier contains `z_dbg`; no inference is
made about why any of the three entries existed or whether a production
workflow used it.

## Why this fits the current unwrapping problem

The Vesuvius Challenge's July 2026
[Open Problems](https://scrollprize.org/2026_open_problems#meshes-adding-connectivity)
page identifies holes, mergers, and sheet switches as recurring mesh failure
modes and invites geometry-processing, optimization, and C++ contributions.
TIFXYZ Doctor contributes diagnostics at the TIFXYZ mesh boundary:

- deterministic checks catch unusable and nonportable packages before
  rendering, flattening, training, or manual inspection;
- topology and geometry reports retain exact row/column locations instead of
  returning only an aggregate score; and
- the overlay displays threshold-triggered review locations rather than
  coloring ordinary sub-threshold variation.

This is deliberately narrower than automatic topology repair. The tool does
not claim sheet-switch recall, infer the correct papyrus layer, or repair a
surface. Its contribution is making several failure modes observable,
reproducible, and cheap to triage.

The geometry audit also computes Symmetric Dirichlet energy per cell. VC3D's
production `flatboi` flattener uses
[SLIM](https://igl.ethz.ch/projects/slim/), which minimizes Symmetric
Dirichlet energy, according to the Challenge's
[flattening documentation](https://scrollprize.org/2026_open_problems#2d-parameterization-and-flattening).
The report therefore exposes locally the same distortion quantity used by the
production optimization objective; it does not reproduce or replace SLIM.

## What is released

- an MIT-licensed Python package and command-line interface;
- `check` for one raw TIFXYZ package;
- `scan` for recursive collection checks and duplicate UUID discovery;
- `audit` for deterministic JSON, self-contained HTML, and PNG review
  overlays;
- tests for malformed files, metadata, sentinels, masks, topology, geometry,
  CLI behavior, TIFF-reader fallback, and overlay semantics;
- a ten-case externally fetched real-data smoke benchmark with pinned URLs,
  byte counts, and SHA-256 hashes;
- a three-case public empty-artifact regression; and
- machine-readable evidence from the bounded 242-package corpus scan.

Repository: **https://github.com/aviad12g/tifxyz-doctor**

Release: **v0.1.0**

## Real-surface geometry evidence

The ten-case smoke manifest covers 40 source objects totaling 1,818,055 bytes.
With the committed default configuration:

- all 10 packages have zero modeled contract errors;
- all 3 pinned Villa real-fixture controls pass with zero warnings and produce
  zero geometry review cues;
- the other 7 packages each reproduce the area-schema warning described
  above;
- all 3 PHerc0800 format-smoke cases produce zero geometry review cues; and
- the 4 unlabeled PHerc1447 hard candidates produce only enclosed-gap review
  cues: 63 regions and 1,105 face-lattice cells.

On a separately supplied 634×1,217 PHerc1667 surface, the tool evaluated
725,986 portable-valid faces in approximately 9.3 seconds on an Apple M1 Pro
with 16 GB of memory. It localized two enclosed regions, one short-edge cue,
133 adjacent-cell normal jumps, and four symmetric-stretch cues. A
deterministic nonlocal pass sampled 80,897 vertices and returned no close
nonlocal vertex pairs. The exported overlay colors 270 of the 725,986 valid
face-lattice cells (0.037%) because only cells touched by configured cues are
painted; the 13 enclosed-gap cells are shown separately.

These are review candidates, not claimed defects. The PHerc1667 package is a
local spot check rather than part of the downloadable benchmark manifest.

## Pinned synthetic reader-interoperability cases

This is supporting evidence, not the corpus headline. None of the 450
registered roots in the bounded scan exposed a `mask.tif`, so no public
mask-value disagreement is claimed.

At Villa commit
`1162bcab4bc769b12993fc320c69c14fdb4a4fa5`, the pinned source paths treat
several synthetic mask cases differently:

- Python treats any nonzero exact-size `uint8` mask value as valid, while C++
  retains only value 255;
- an exact-size Python mask can select a point whose original `z <= 0`, while
  the C++ load order applies `z <= 0` as an earlier hard gate; and
- Python ignores an integer-multiple higher-resolution mask and falls back to
  `z > 0`, while C++ applies the higher-resolution mask.

TIFXYZ Doctor reports modeled Python-valid, finite C++ point-valid, and
conservative portable-valid counts. The model is revision-pinned rather than
presented as a permanent file-format specification.

As observed on 2026-07-26, an executable differential pairs the exact Python
reader files at audit-date repository HEAD (`1162bcab`) with the real
`load_quad_from_tifxyz` path through Villa's official stable Apple Silicon
`vc_tifxyz2obj` release (`05ff9ea`). The harness satisfies all five declared
fixture expectations: two controls and three confirmed disagreements.

- the all-valid control yields 25 Python-valid vertices; the converter reports
  a loaded 5×5 grid and emits the expected 16-vertex/18-face OBJ signature;
- the no-mask `z <= 0` control is rejected by both paths; and
- for mask value 1, mask-over-`z <= 0`, and an integer-multiple mask, Python
  retains all 25 grid vertices while the C++ output matches the known
  single-invalid-center signature. TIFXYZ Doctor models 25 Python-valid versus
  24 finite C++ vertices in each disagreement case.

The 25-to-16 reduction is an independently explained exporter boundary effect,
not loader resizing. At this release, `loc_valid` uses a half-open `rows-2` by
`cols-2` rectangle. Only 3×3 origins survive for a 5×5 grid; two triangles per
quad produce 18 faces, and the converter writes only their 4×4 corner union.
Rejecting the center removes four of those nine quads, producing the calibrated
12-vertex/10-face signature. The authenticated harness also probes five
all-valid square sizes from 3×3 through 7×7 and three rectangular sizes. For
the eight sampled `H×W` grids with `H,W ≥ 3`, the observed counts match
`(H-1)(W-1)` vertices and `2(H-2)(W-2)` faces.

The stable release commit is 94 commits older than the audit-date repository
revision. That separation defines the tested stable-release-versus-repository
pairing; it is not a same-commit binary/source comparison. Intervening path
history shows no Python-reader edit and no change to the relevant C++ full-load
mask rules. Exact source, release, asset, executable, and result hashes are
recorded.

## Reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/fetch_benchmark.py
python scripts/run_benchmark.py
python verification/run_reader_differential.py \
  --villa-root ../villa \
  --cpp-binary ../villa-release/extracted/VC3D.app/Contents/MacOS/vc_tifxyz2obj \
  --cpp-asset ../villa-release/VC3D-05ff9ea-2026-07-12-macos-arm64.zip \
  --output verification/reader-differential-results-v1.json
```

The historical public-empty regression can be rerun with
`python scripts/run_public_regressions.py` only when the exact
pre-remediation cache is already present. A live refetch is expected to fail
because two paths were retired and one was replaced.

Example use:

```bash
tifxyz-doctor check /path/to/tifxyz --json contract.json
tifxyz-doctor scan /path/to/segments --json collection.json
tifxyz-doctor audit /path/to/tifxyz \
  --json audit.json \
  --html audit.html \
  --overlay audit.png \
  --fail-on-integrity
```

## Technical choices

- Raw contract inspection is independent of the geometry loader.
- JSON rejects NaN/infinity and records every threshold.
- Geometry defaults normalize by observed directional medians; `meta.scale` is
  not assumed to be 3-D edge spacing.
- The face split matches Villa's `p10-p01` diagonal, with an
  alternate-diagonal sensitivity check.
- Topology uses 4-connected faces and 8-connected background for enclosed
  regions.
- The nonlocal pass is a deterministic sampled vertex search and is explicitly
  not presented as triangle-intersection detection.
- Only NumPy and Pillow are required; no GPU or SciPy is needed.

## Scope and limitations

TIFXYZ Doctor does not repair surfaces, rank them with a learned model, or
claim ground-truth geometry quality. There is not yet a labeled benchmark of
known sheet switches against which to estimate review-cue precision or recall.
The current proximity pass can miss triangle contacts, and the in-memory
implementation may need tiling for the largest surfaces.

The interoperability model is pinned to Villa commit
`1162bcab4bc769b12993fc320c69c14fdb4a4fa5` and should be reviewed when the
official readers change.

## Licensing and data

The code and repository-produced documentation/results are MIT licensed.
Downloaded Vesuvius data is not redistributed by the repository and retains
the source license declared in each benchmark manifest, CC BY-NC 4.0.

## Suggested release note

> TIFXYZ Doctor v0.1.0 adds deterministic preflight and review diagnostics for
> Vesuvius TIFXYZ surfaces. A bounded metadata census found 277 of 450
> registered roots reusing `output_tifxyz` and all 138 inspected PHerc1203
> snapshots omitting the `area` key consumed by the pinned Python API. A
> separate hash-pinned regression recorded three then-registered sentinel-only
> normalized packages, including one whose identifier contained `z_dbg`; the
> affected entries were repaired upstream after the report and all three
> current registrations now pass. The geometry audit localizes topology and
> Symmetric Dirichlet distortion cues without relabeling them as proven defects.
