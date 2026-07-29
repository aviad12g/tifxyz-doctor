# Real-segment trace-impact check

## Outcome

A controlled four-way source-build experiment on a public PHerc1447 TIFXYZ
surface shows that the stale `SurfTrackerData` bounds change the final saved
output of the production tracer path. It independently reproduces an earlier
official-release binary ablation.

The public 160 x 136 source contains exactly six valid quads whose origins lie
on the final legal row or column:

```text
[51,134] [52,134] [53,134] [158,88] [158,89] [158,90]
```

All four source variants ran three times. TIF arrays, masks, selected metadata,
and parsed scientific log observations were identical within every variant.

| Variant | Exact source | Resume points | Output shape | Valid vertices | Valid quads |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | `208faea` | 15,696 | 158 x 134 | 15,692 | 15,074 |
| Helper only | `d522691` | 15,702 | 159 x 135 | 15,692 | 15,074 |
| Minimal full bounds | `9e15071` | 15,702 | 159 x 135 | 15,698 | 15,078 |
| Actual PR | `6e2bba9` | 15,702 | 159 x 135 | 15,698 | 15,078 |

Baseline and helper-only have the same 15,692 valid vertices and bit-identical
XYZ values on those vertices; the helper-only output merely has an extra
sentinel-padded row and column. Minimal-full versus helper-only recovers six
valid vertices and four valid quads at exactly the six precomputed boundary
locations. One adjacent common vertex at `[51,133]` moves by `0.001953125`
stored coordinate units; all other common XYZ values are identical.

The minimal full-bounds build and actual PR have identical X, Y, Z,
generation, and valid-mask hashes in all three replicates. The helper commit is
one commit over baseline and changes only `Geometry.cpp`; the minimal-full
commit is one commit over helper and changes only `SurfTrackerData.cpp`.

The narrow defensible finding is:

> On this real public source, widening only the shared helper changes resume
> admission and the saved grid extent, but widening the three stale
> `SurfTrackerData` copies is necessary for the six admitted boundary samples
> to survive into the saved XYZ surface.

## Source-build evidence

The successful
[GitHub Actions run](https://github.com/aviad12g/villa/actions/runs/30445977638)
built and compared exact AppImages for baseline, helper-only, minimal-full, and
[PR #1264](https://github.com/ScrollPrize/villa/pull/1264). The
[compact evidence artifact](https://github.com/aviad12g/villa/actions/runs/30445977638/artifacts/8721453558)
has ZIP SHA-256
`0d734809b454ad2b965f46d9223cbac34cb622e814835dbde09147e974508eb7`.
Its durable checked-in copy, evaluator, and workflow are under
[`source-confirmation/`](source-confirmation/).

The four build logs resolve the same GitHub runner/image, Dockerfile frontend,
Villa builder image, GNU 15.2.0 toolchain, Ninja preset, Qt 6.10.2, and
packaging-tool versions. This is controlled same-toolchain evidence, not a
claim of bit-for-bit hermetic builds: the build workflow still used live APT
and continuously published packaging downloads.

The actual PR also contains non-finite-location checks, lookup sentinel
unification, and a small-grid normal guard. Its exact equality with the
minimal-full build on this fixture means those additional changes do not drive
this observed differential.

## Independent release-binary replication

Before the source builds completed, the same result was obtained through a
byte-validated bounds ablation of the official
[`05ff9ea` macOS-arm64 stable release](https://github.com/ScrollPrize/villa/releases/tag/stable).
The exact archive is
[`VC3D-05ff9ea-2026-07-12-macos-arm64.zip`](https://github.com/ScrollPrize/villa/releases/download/stable/VC3D-05ff9ea-2026-07-12-macos-arm64.zip),
SHA-256
`d5fca566c8bc843f2350ab60e86e0def25ab88240e3f0ffcbd04662ba522c9a8`.
A hash-pinned script copied two dylibs and changed only eight decoded ARM64
instructions:

```text
sub Wbound, Wdimension, #2  ->  sub Wbound, Wdimension, #1
```

The helper-only variant changes two comparisons in
`loc_valid(cv::Mat_<cv::Vec3f>, ...)`. The full-bounds variant changes those
two plus the six comparisons in `SurfTrackerData::lookup_int`, `valid_int`,
and `lookup_int_loc`. Before signing, only the expected eight immediate bytes
differ. After ad-hoc signing, only those bytes differ inside each `__text`
section. Low-side checks, sentinel checks, interpolation, and every other
release instruction remain unchanged. `DYLD_PRINT_LIBRARIES` logs prove which
copied or original dylib each run loaded.

That ablation implements only the high-side-bound portion relevant to the six
all-corners-valid cells. Its three-way result is numerically identical to the
source-built baseline/helper/minimal-full result above.

The source is real published Challenge data but is labeled an unverified
auto-grown hard candidate; it has no official quality label or ground-truth
sheet assignment. The result proves an output difference, not that the six
recovered points improve scientific accuracy. Six points are about 0.038% of
the baseline valid-vertex count.

## What was run

- Entrypoint: `vc_grow_seg_from_segments`
- Source revisions:
  `208faea7d7fec62da03ff75e118e26be0c08a117`,
  `d5226910817be064d2ae1577f1fae0f850acc8c7`,
  `9e15071df41a139ce77ada6aa64367e000387c5d`, and
  `6e2bba940d8f93b53b0d1bcf7f66af19c5b81295`
- Source-build environment: GitHub `ubuntu-24.04`, runner image
  `20260720.247.2`, Villa builder image digest
  `sha256:77be7988b71fca1d8a0657e1915368274729400e03e2e7edaacc9bb84ec707a2`
- Source: public
  `PHerc1447/segments/20250502185519-auto_grown_20250502164303733/mesh/intermediate/tifxyz_original/`
- Source URL:
  `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1447/segments/20250502185519-auto_grown_20250502164303733/mesh/intermediate/tifxyz_original/`
- Source hashes and the raw boundary scan are in
  `input-boundary-scan.json`.
- Mode: `resume_growth`, one generation, `step=1`, `src_step=1`, expansion
  disabled, one deterministic rightward growth direction, CUDA disabled
- Volume: metadata-only PHerc1447 Zarr stub, 7.91 micrometers per voxel. This
  `--src-segment` path reads only shape and voxel-size metadata before
  `grow_surf_from_surfs`; all completed runs opened no volume chunk.
- Controls: `OMP_NUM_THREADS=1`, `OMP_DYNAMIC=FALSE`
- Replicates: three per source-built variant and three per release-ablation
  variant

The three committed PHerc0172 test fixtures were rejected as differential
candidates before the PHerc1447 run: their correct-minus-old boundary-cell
counts are all zero, so they cannot exercise this bug.

## Exact execution path

```text
vc_grow_seg_from_segments
  -> load source QuadSurface(s)
  -> grow_surf_from_surfs
  -> grow_surf_from_surfs_impl
     -> resume initialization
        -> loc_valid(seed_points, [row, col])
        -> SurfTrackerData::lookup_int
     -> candidate probing / Ceres surface losses
        -> SurfTrackerData::lookup_int_loc
        -> SurfaceLossD / ZLocationLoss -> loc_valid
     -> optional re-optimization
        -> loc_valid(points_hr, l)
     -> surftrack_genpoints_hr
        -> SurfTrackerData::valid_int
        -> loc_valid
        -> SurfTrackerData::lookup_int_loc
     -> final QuadSurface save
```

The two highest-value causal sites are resume initialization
(`GrowSurface.cpp:1708-1723`) and final high-resolution reconstruction
(`GrowSurface.cpp:861-915`). The latter requires all four tracked low-resolution
corners to pass `valid_int`, then samples through `lookup_int_loc`.

## Recommendation

Use the source-built result as the primary causal evidence and the independent
release-binary ablation as replication. Keep the scope label intact: this is a
deterministic production-output difference on a real published fixture, not
evidence of improved scientific accuracy.

## Files

- [input-boundary-scan.json](input-boundary-scan.json): pre-run candidate
  selection and exact six boundary origins
- `patch-manifest.json`: original/patched hashes, instruction
  offsets and code-section diff verification
- `run-manifest.json`: commands,
  environments, linkage proof, logs, hashes, repeatability and comparisons
- `comparisons.json`: compact numerical
  release baseline/helper/full output differential
- [source-confirmation/](source-confirmation/): source-built evidence,
  preregistration, evaluator, and workflow
- [experiment-protocol.md](experiment-protocol.md): completed source-build and
  release-ablation protocol
- [hendrik-qa.md](hendrik-qa.md): eight likely maintainer questions and answers
- `resume-one-generation.json`: bounded tracer parameters
- `make_release_bounds_ablation.py`: hash- and byte-validating ablation builder
- `run_release_bounds_ablation.py`: three-variant, three-replicate runner
- `compare_tifxyz.cpp`: raw XYZ/mask differential
- `scan_tifxyz_boundary.cpp`: candidate boundary-cell scanner
- `fake-volume-pherc1447/`: metadata-only Zarr fixture
