# Likely maintainer questions

## 1. What exactly is the `loc_valid` contract?

`l` is a fractional `[row, col]` location to be bilinearly sampled. Its floored
origin must identify a complete 2x2 cell, so
`0 <= row < rows-1` and `0 <= col < cols-1`. The maximum legal integer origin is
therefore `(rows-2, cols-2)`. All four corners must also pass the existing
`-1` sentinel check.

## 2. Why was the old rectangle wrong?

`cv::Rect::contains` is half-open. Giving it width `rows-2` and height `cols-2`
accepted origins only through `rows-3` and `cols-3`, dropping the final complete
cell. The old code also encoded `[row,col]` through `cv::Point(x,y)`, which made
the axis intent hard to audit. Explicit scalar inequalities state the sampling
contract directly and work for rectangular grids.

## 3. Why change all three `SurfTrackerData` methods instead of only the helper?

They carried independent copies of the same stale bound, so fixing only
`loc_valid` would create a real internal disagreement: resume initialization
could admit a final cell, then `lookup_int` or `valid_int` could reject it.
Delegating `lookup_int`, `valid_int`, and `lookup_int_loc` to the shared helper
makes admission, interpolation, Ceres losses, and final reconstruction use the
same 2x2-cell and sentinel contract.

## 4. Does this change sentinel behavior beyond the off-by-one?

It makes `lookup_int` and `lookup_int_loc` refuse interpolation when any of the
four cell corners has the existing `-1` first-component sentinel. Previously
`valid_int` checked those corners, but the two lookup paths could interpolate
without the same check. That is intentional contract unification and is covered
by final-cell sentinel tests. It does not add a new rule for non-finite stored
XYZ components; only non-finite *locations* are newly rejected before flooring.

## 5. Could widening `loc_valid` make `grid_normal` read out of bounds?

No. `grid_normal` has a different 4x4 stencil contract and does not use
`loc_valid` to establish that stencil. It clamps to the supported interior and
now explicitly returns NaNs for grids smaller than 4x4. The one caller that
combines `loc_valid` with a normal (`ObjAlphaCompRefinement`) obtains the normal
through `QuadSurface::gridNormal`, so the normal's own stencil rules remain in
force.

## 6. What is the tracing blast radius?

The direct path is `vc_grow_seg_from_segments -> grow_surf_from_surfs`.
`GrowSurface` uses the shared helper during resume initialization and
re-optimization, and uses all three `SurfTrackerData` accessors during candidate
sampling and high-resolution reconstruction. The Ceres `SurfaceLossD` and
`ZLocationLoss` templates also gate bilinear interpolation with `loc_valid`.
Outside tracing, callers are bilinear samplers/exporters or normal consumers;
the 4x4 normal stencil remains separately guarded.

## 7. Why use `const auto points = sm->rawPoints()` in the accessors?

`cv::Mat` copies are shallow, reference-counted headers, so this does not copy
the coordinate image. It avoids repeatedly obtaining separate headers inside
one check/interpolation operation and guarantees that the bounds/sentinel check
and `at_int_inv` address the same matrix snapshot. These paths treat the
underlying surface points as immutable.

## 8. Do we already have proof that this changes a real trace?

Yes. A public GitHub Actions run built four exact Linux AppImages from the
baseline, helper-only, minimal-full, and actual PR revisions, then ran three
isolated replicates per variant on the same preregistered PHerc1447 fixture.

The public PHerc1447 input has six all-valid quads on the final legal row or
column. Across the deterministic source-built runs:

- baseline emits 15,692 valid XYZ vertices and 15,074 valid quads;
- helper-only expands the saved grid but emits the same 15,692 valid vertices
  and 15,074 valid quads with numerically identical common XYZ;
- minimal-full emits 15,698 valid vertices and 15,078 valid quads, recovering
  exactly the six precomputed boundary locations; and
- the actual PR has exactly the same X, Y, Z, generation, and mask hashes as
  minimal-full in all three replicates.

One adjacent common vertex moves by 0.001953125 stored coordinate units; all
other common XYZ values are identical. That isolates the stale
`SurfTrackerData` copies as necessary for the final-surface difference on this
fixture, while the actual-PR equality shows its additional changes do not drive
this differential.

The evidence workflow pins source commits, build runs, artifact IDs, archive
sizes and SHA-256s, exact AppImage filenames and SHA-256s, fixture hashes, and
the evaluator/workflow hashes. Complete fixture-tree hashes are unchanged
before and after all 12 executions. The build logs resolve matching toolchain
and packaging versions, although live APT and continuous packaging downloads
mean the builds are not claimed to be bit-for-bit hermetic.

The same numerical pattern was also obtained independently through a
byte-validated bounds ablation of the official `05ff9ea` macOS-arm64 release.
See [`source-confirmation/`](source-confirmation/) for the durable source-build
evidence. The real source is an unverified auto-grown candidate, so this proves
a production-path output difference, not improved sheet accuracy.
