# Reviewed same-wrap benchmark (v0.2)

## Result

TIFXYZ Doctor v0.2 adds a conservative `coherent-normal-step` review cue and
tests it on real, human-reviewed PHercParis4 surface patches. The key result is
narrow:

- on a sealed, overlap-component-isolated holdout, the cue localized
  **124/128 abrupt 8-voxel proxies (96.875%)** and **128/128 abrupt
  16-voxel proxies (100%)**;
- it localized **0/128** 8-voxel transitions spread across four cells and
  only **8/128** 16-voxel transitions spread across twelve cells when all cue
  families were combined; and
- all **128/128 holdout nulls** were exactly unchanged in coordinate bytes,
  validity bytes, the complete public audit report, and a reduced audit
  signature.

This demonstrates sensitivity to abrupt, spatially coherent normal steps. It
does **not** measure recall on naturally occurring sheet switches, and it shows
that gradual transitions remain largely unsolved.

## Data and leakage control

The source is the official
[`PHercParis4/verified_patches`](https://huggingface.co/buckets/scrollprize/datasets/tree/spiral/PHercParis4/verified_patches)
bucket. The benchmark uses all 709 complete `same_wrap*` directories that
contain `meta.json`, `corr_points_results.json`, and x/y/z TIFFs.

The first 64 synthetic cases were used while developing the cue, so they are
reported only as **development** results. Patch IDs alone are not independent:
different patches can overlap the same physical surface. The frozen split
therefore uses the official `patch-overlap-pcls.json` graph:

| Split fact | Count |
| --- | ---: |
| Selected reviewed patches | 709 |
| Overlap edges among them | 478 |
| Overlap-connected components | 358 |
| Development patches | 64 |
| Patches in a development-touched component | 217 |
| Related non-development patches excluded from holdout | 153 |
| Clean holdout pool | 492 patches / 300 components |
| Synthetic holdout opened after detector freeze | 128 patches / 82 components |

The detector and original development protocol were frozen at commit
`d3c8309ca707e2f18e7d64e38fb7be4ff4ca77c0`. The exact component split,
source identities, thresholds, metric definitions, and holdout IDs were then
committed at
`cdec03e72f78fcc5eef0d8381fea77b4871ca9ce` before the holdout was run.
Development and holdout results are never pooled.

## Controlled proxy results

One side of a deterministic seam is moved along locally averaged surface
normals. Results below are the percentage of cases with at least one new cue
inside a one-cell evaluation band after subtracting the unmodified baseline.
Offsets are reported only in voxels; no constant “winding fraction” is
inferred.

| Offset | Transition width | Development | Sealed holdout |
| ---: | ---: | ---: | ---: |
| 4 voxels | 1 cell | 0/64 (0%) | 0/128 (0%) |
| 4 voxels | 4 cells | 0/64 (0%) | 0/128 (0%) |
| 4 voxels | 12 cells | 0/64 (0%) | 1/128 (0.781%) |
| 8 voxels | 1 cell | 62/64 (96.875%) | 124/128 (96.875%) |
| 8 voxels | 4 cells | 0/64 (0%) | 0/128 (0%) |
| 8 voxels | 12 cells | 0/64 (0%) | 2/128 (1.563%) |
| 16 voxels | 1 cell | 64/64 (100%) | 128/128 (100%) |
| 16 voxels | 4 cells | 1/64 (1.563%) | 2/128 (1.563%) |
| 16 voxels | 12 cells | 4/64 (6.25%) | 8/128 (6.25%) |

For the central abrupt cases, the new cue is load-bearing:

- at 8 voxels × 1 cell, v0.1 cues detected **0/128** holdout cases while
  `coherent-normal-step` detected **124/128**; and
- at 16 voxels × 1 cell, v0.1 cues detected **2/128** while the new cue
  detected **128/128**.

Intervals in the machine-readable result are deterministic cluster-bootstrap
intervals over overlap-connected components, not independent-patch Wilson
intervals.

## Reviewed same-wrap alert rates

`corr_points_results.json` supplies mapped annotation samples, not an ordered
polyline. The benchmark therefore scores only dilated **annotation
neighborhoods** and does not claim labels between samples.

Within the entire overlap-isolated 492-patch holdout pool, 491 patches had
nonempty mapped neighborhoods:

| Observation | Holdout result |
| --- | ---: |
| Evaluated annotation-neighborhood cells | 373,904 |
| Cells touched by any v0.2 cue | 305 (0.0816%) |
| Patches touched by any cue | 20/491 (4.07%) |
| Cells touched by `coherent-normal-step` | 26 (0.00695%) |
| Patches touched by `coherent-normal-step` | 2/491 (0.407%) |

All 26 new-cue cells were already touched by at least one v0.1 cue, so adding
the new family increased the same-wrap neighborhood union by **zero cells**.
These are descriptive alert rates, not general false-positive rates: a
same-wrap label negates a sheet switch, but it does not make a real fold,
stretch, or other geometric cue incorrect.

## Cue definition

For every valid horizontal and vertical grid edge, the audit estimates a local
surface normal by averaging adjacent cell normals. It computes the absolute
edge component along that normal and divides by the direction-specific median
grid spacing. An edge becomes a candidate at a ratio of at least `0.25`.
Candidate edges are projected into cell space; only 8-connected bands of at
least eight cells are retained.

The coherence requirement rejects isolated spikes. It also explains the
observed limitation: a large displacement spread smoothly across several
cells can keep each individual normal component below threshold.

## Reproduce

The repository does not redistribute the source data. Download the exact
official subset and overlap graph:

```bash
hf buckets sync \
  hf://buckets/scrollprize/datasets/spiral/PHercParis4/verified_patches \
  DATA \
  --include 'same_wrap*/meta.json' \
  --include 'same_wrap*/corr_points_results.json' \
  --include 'same_wrap*/x.tif' \
  --include 'same_wrap*/y.tif' \
  --include 'same_wrap*/z.tif'

hf buckets cp \
  hf://buckets/scrollprize/datasets/spiral/PHercParis4/patch-overlap-pcls.json \
  DATA/patch-overlap-pcls.json
```

Regenerate the frozen split and result:

```bash
PYTHONPATH=src python scripts/build_reviewed_patch_split.py DATA
PYTHONPATH=src python scripts/run_reviewed_patch_benchmark.py DATA
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

Pinned identities:

| Artifact | SHA-256 |
| --- | --- |
| Official overlap graph | `11fc0ef6112a2b9829f80242b7c67530c841b1159aed1ed5bb45e4362aad5097` |
| Selected five-file data tree | `df51daa45ac762242c044a1fbe95914030664ab96536e157289b4f7077964982` |
| Frozen split manifest | `cbb13f8b18ab38ac32ea010348763fc65d0200fc36193b81ae372dc498731920` |
| Complete v0.2 result | `74f243eb57a84eb2ee9aa8df2f5674ef92d86fc1d54a8333c7b0b9e1fe21173c` |

Two full runs produced the same complete result hash; the second used a fresh
direct download into a non-iCloud temporary directory rather than the original
source directory. The machine-readable artifacts are
[`reviewed-same-wrap-split-v1.json`](../benchmarks/reviewed-same-wrap-split-v1.json)
and
[`reviewed-same-wrap-results-v0.2.0.json`](../benchmarks/reviewed-same-wrap-results-v0.2.0.json).
Downloaded Vesuvius data remains under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) and is not
redistributed here.
