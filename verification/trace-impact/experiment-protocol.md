# Paired real-segment tracer protocol

## Question

Does the corrected final-cell contract change tracer output on a real segment,
and is any change specifically caused by eliminating the stale
`SurfTrackerData` copies?

## Completed source-build confirmation

Yes. The
[successful source-build run](https://github.com/aviad12g/villa/actions/runs/30445977638)
used four exact revisions:

1. baseline `208faea7d7fec62da03ff75e118e26be0c08a117`;
2. helper-only `d5226910817be064d2ae1577f1fae0f850acc8c7`, one commit
   over baseline changing only `Geometry.cpp`;
3. minimal-full `9e15071df41a139ce77ada6aa64367e000387c5d`, one commit over
   helper changing only `SurfTrackerData.cpp`; and
4. actual PR `6e2bba940d8f93b53b0d1bcf7f66af19c5b81295`.

Each variant was deterministic across three replicates. Baseline and
helper-only both save 15,692 valid vertices and 15,074 valid quads. Helper-only
expands the saved extent but does not change any common XYZ value. Minimal-full
and the actual PR both save 15,698 valid vertices and 15,078 valid quads, with
identical X, Y, Z, generation, and mask hashes. The six added locations are
exactly the six preregistered boundary cells.

The run therefore isolates the stale `SurfTrackerData` accessors as necessary
for the final saved-surface change on this fixture. The additional PR changes
do not drive this differential because the actual PR exactly matches the
minimal-full build. See
[`source-confirmation/`](source-confirmation/) for the durable evidence,
evaluator, and workflow.

## Completed release-binary ablation

The same result was first obtained independently for the high-side-bound
behavior of the official `05ff9ea` macOS-arm64 release. That experiment used:

1. unmodified release dylibs;
2. a helper-only dylib with the two `loc_valid(Vec3f)` high-side subtract
   immediates changed from `#2` to `#1`; and
3. the helper change plus the six equivalent subtract immediates in
   `SurfTrackerData::lookup_int`, `valid_int`, and `lookup_int_loc`.

The patcher pins source dylib SHA-256s, validates exact original bytes at every
file offset, verifies that no other pre-sign bytes change, ad-hoc signs the
copies, and verifies that no other `__text` bytes change after signing. Run
logs use `DYLD_PRINT_LIBRARIES=1` to prove which variants were loaded.

On the public PHerc1447 source selected below, three replicates per variant are
deterministic. Helper-only expands the resume domain and output extent but
emits no new valid XYZ point. Full-bounds recovers all six precomputed boundary
locations. See `run-manifest.json`.

## Executed variants

All variants were built through the same Villa workflow with matching resolved
compiler, flags, builder image, and packaging-tool versions.

1. **Baseline:** current-main parent `208faea7d7fec62da03ff75e118e26be0c08a117`
2. **Helper-only ablation:** baseline plus only the `Geometry.cpp` `loc_valid`
   correction, leaving the three old `SurfTrackerData` implementations
   untouched
3. **Minimal-full ablation:** helper-only plus only the
   `SurfTrackerData.cpp` synchronization
4. **Full PR:** `6e2bba940d8f93b53b0d1bcf7f66af19c5b81295`

Baseline versus PR answers the user-facing output question. Helper-only versus
minimal-full isolates the three `SurfTrackerData` accessors. Minimal-full
versus PR tests whether any additional PR change drives the observed
differential. Resume initialization first calls the shared `loc_valid` and then
calls `SurfTrackerData::lookup_int`; a helper-only build can admit a final cell
that the stale accessor still rejects.

## Candidate selection

The initial candidates were the three committed real PHerc0172 fixtures:

```text
volume-cartographer/core/test/data/segments/20241113070770
volume-cartographer/core/test/data/segments/20241113080880
volume-cartographer/core/test/data/segments/20241113090990
```

For each `H x W` source, compute two valid-cell masks from the raw `-1`
sentinel contract:

```text
correct origins: row 0..H-2, col 0..W-2
old origins:     row 0..H-3, col 0..W-3
```

A cell is valid only if the first coordinate of all four corners is not `-1`.
Choose the smallest fixture for which:

```text
count(correct_valid_cells) - count(old_valid_cells) > 0
```

Record separately how many added cells originate on row `H-2`, column `W-2`,
and their intersection. All three committed fixtures have zero added cells and
therefore cannot demonstrate an output change.

The completed ablation instead uses this previously hash-pinned TIFXYZ Doctor
benchmark:

```text
PHerc1447/segments/
20250502185519-auto_grown_20250502164303733/
mesh/intermediate/tifxyz_original/
```

It is public Challenge data and its metadata records `vc_grow_seg_from_seed`,
but it is an unverified auto-grown hard candidate with no official quality
label. Its 160 x 136 grid contains 15,702 valid quads. Correct-minus-old is six:
three on the final row and three on the final column, with no corner overlap.
Their origins are:

```text
[51,134] [52,134] [53,134] [158,88] [158,89] [158,90]
```

This selection criterion was evaluated before the tracer differential: the
input must contain a complete all-valid 2x2 cell that the old high-side bound
rejects. Exact hashes and counts are in `input-boundary-scan.json`.

## Bounded parameters

Use the checked-in `resume-one-generation.json` in this directory:

```json
{
  "resume_growth": true,
  "disable_grid_expansion": true,
  "growth_directions": ["right"],
  "steps": 1,
  "step": 1,
  "src_step": 1,
  "max_width": 256,
  "max_height": 256,
  "global_steps_per_window": 1,
  "debug_images": false,
  "use_cuda": false,
  "consensus_default_th": 2
}
```

`step=1` is essential: it makes resume initialization visit every possible
integer cell origin, including row `H-2` and column `W-2`. One generation makes
this a tracer run while keeping runtime bounded. Disabling grid expansion keeps
the comparison on the source footprint.

## Build and provenance

The completed source experiment used Villa's `vc3d-linux.yml` workflow to
produce one AppImage per exact source revision. The evidence workflow verified
the build run, source commit, artifact ID, artifact byte size, archive SHA-256,
exact AppImage filename, and AppImage SHA-256 before extraction.

The four build logs resolve the same GitHub runner 2.336.0, runner image
`ubuntu-24.04` version `20260720.247.2`, Dockerfile frontend digest, Villa
builder image digest, GNU 15.2.0 toolchain, Ninja `ci-release-gcc` preset, Qt
6.10.2, and packaging-tool versions. This controls the practical toolchain but
does not make live APT or continuous packaging downloads bit-for-bit hermetic.

## Run

The evidence workflow extracted each verified AppImage once, dispatched the
tracer through its bundled `AppRun`, and created a distinct fixture copy,
working directory, home, and temporary directory for every replicate. The
native child environment contained only a fixed allowlist and no GitHub token.
The effective command was:

```bash
vc_grow_seg_from_segments \
  --volume /isolated/fixture/fake-volume-pherc1447 \
  --src-dir /isolated/fixture/input-parent \
  --target-dir /isolated/output \
  --params /isolated/fixture/resume-one-generation.json \
  --src-segment /isolated/fixture/input-parent/pherc1447-source
```

The output UUID contains a timestamp, so the evaluator compares array contents,
complete valid masks, selected metadata fields, and parsed scientific log
observations rather than directory names or the entire `meta.json` byte string.
Complete fixture-tree hashes before and after each of the 12 executions show
that no input was modified or added.

The metadata-only Zarr is adequate for this `--src-segment` path: the app reads
dataset shape and voxel size before entering `grow_surf_from_surfs`; the bounded
run completed without a chunk file. The completed PHerc1447 ablation uses
`fake-volume-pherc1447/`, with the correct 7.91 micrometer voxel size and
metadata-only dimensions that contain the source coordinates. If a maintainer
prefers full production fidelity, substitute the matching PHerc1447 OME-Zarr
without changing any other input.

## Required observables

Capture stdout and record:

1. `resume_growth initialized` count
2. `used_area`
3. generation-zero accepted/fringe counts
4. final `area_vx2`
5. final grid width and height
6. effective valid-vertex and valid-quad counts
7. valid-mask symmetric difference between variants
8. on common valid vertices, count of changed XYZ values plus mean and maximum
   Euclidean displacement
9. on newly valid vertices, their source row/column and XYZ values

Also verify that each variant is deterministic across its three replicates.
If not, report distributions and do not attribute a one-off difference to the
patch.

## Causal interpretation

- **Baseline and helper-only have identical valid XYZ, while minimal-full adds
  valid XYZ and minimal-full equals PR:** the `SurfTrackerData` consolidation
  is necessary for the observed final-surface change, while the other PR
  changes do not drive this fixture's differential. This is the completed
  source-build and release-ablation result.
- **Baseline differs from helper-only; helper-only differs from full:** both
  the shared helper and stale accessors affect the trace; report both deltas.
- **Baseline differs from helper-only; helper-only matches full:** the shared
  helper drives the output change; do not market it specifically as a
  `SurfTrackerData` tracing result.
- **No differences:** the chosen segment does not exercise an added final cell,
  or later filtering removes the difference. Use instrumentation before trying
  a larger run.

For a no-difference result, add local counters (not an upstream patch) at
`GrowSurface.cpp:1711`, `SurfTrackerData::lookup_int`,
`SurfTrackerData::valid_int`, and `SurfTrackerData::lookup_int_loc` whenever
`floor(row) == rows-2` or `floor(col) == cols-2`. This distinguishes "path not
exercised" from "path exercised but output converged."

## Resource estimate

- Public PHerc1447 input: about 269 KiB across XYZ and metadata
- Each completed release-ablation run: 0.24-0.41 seconds on Apple Silicon
- Each completed source-built tracer execution: 0.262-0.319 seconds on the
  GitHub-hosted runner after AppImage extraction
- Main cost: four source builds and their artifacts; the compact final evidence
  artifact contains only JSON
