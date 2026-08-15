# Reproducibility guide

## Environment

- Python 3.11 recommended.
- CPU execution is sufficient for inventory, geometry, m7, raw-stack rendering, exact-increment, self-intersection, and model-independent raw audits.
- GPU dependencies are optional and confined to the closed coarse-model experiments.
- Network access is needed only when streaming public Vesuvius Zarr/TIFFXYZ assets that are not already cached.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -m pytest -q toolkit/tests
```

The canonical workspace passed 273 focused and regression tests before release assembly. The compact release subset passed 129 tests after assembly; the exact command and environment are recorded in `RELEASE_TEST_RESULT.json`.

After any intentional release edit, rebuild the byte manifest and verify it:

```bash
python toolkit/finalize_progress_prize_release.py
shasum -a 256 -c SHA256SUMS
```

## Stage 1 — compact public-asset inventory

The inventory tool paginates public S3 prefixes without recursively listing every chunk inside large Zarr containers. It records object keys, metadata, digests, grouped assets, and mesh-only TIFFXYZ candidates.

```bash
python toolkit/inventory_pherc_assets.py \
  --help
```

Expected output: deterministic JSON with raw volumes, prediction containers, normal grids, mesh/TIFFXYZ assets, surface volumes, and extraction-required candidates.

## Stage 2 — native geometry and m7 support

`native_surface_sampler.py` defines the coordinate contract:

- surface points are `(x, y, z)`;
- volume arrays are `(z, y, x)`;
- voxel spacing is stated in physical micrometers;
- both normal orientations are evaluated explicitly;
- invalid vertices remain invalid rather than being clamped.

`preflight_debug_m7.py` and the calibrated grower/salvage tools add public-m7 support, centered selected-run continuity, exact-area subwindow search, and deterministic topology handling.

```bash
python toolkit/preflight_debug_m7.py --help
python toolkit/salvage_calibrated_m7_component.py --help
python toolkit/audit_tifxyz_self_intersection.py --help
```

Fail closed on any hard geometry defect, invalid sampling, mismatched hashes, topology change, or an acceptance criterion below its precommitted threshold.

## Stage 3 — full-resolution preflight and raw rendering

Only a stored-grid candidate that passes geometry, selected-run, topology, and provenance checks may be resampled. The full-resolution object is written, reloaded in its serialized dtype, and revalidated before raw-volume access.

```bash
python toolkit/full_resolution_tifxyz_preflight.py --help
python toolkit/render_gated_tifxyz_raw_stack.py --help
```

The renderer uses one compact source ROI per tile across every offset, writes a positive 93-layer stack at offsets `-46..+46`, and represents the opposite normal sign as exact reversal when the offsets are symmetric. Every layer is masked and hashed.

## Stage 4 — model-independent raw audit

```bash
python toolkit/independent_raw_ink_audit.py \
  STACK_DIR MASK_TIF RAW_STACK_MANIFEST OUTPUT_DIR
```

The audit validates all 93 input hashes and then computes:

- mask-normalized local dark residuals;
- adjacent seven-layer depth persistence;
- distant-offset specificity controls;
- structure-tensor and Frangi fiber evidence;
- bounded row topology;
- native-resolution review crops and depth-control sheets.

Candidate generation is not a letter verdict. A human or papyrological reviewer must inspect the raw-depth evidence; model maps cannot substitute for CT support.

## Stage 5 — model consensus, if independently justified

`build_depth_consensus.py` treats forward and reverse orientations symmetrically. Low/mid/high support is fused only within an orientation; cross-orientation overlap is diagnostic and never an automatic veto. Physical frame sequences are deduplicated before inference.

GPU work should not start unless geometry and raw evidence justify it. The 13-scroll campaign intentionally stopped many candidates before inference.

The compact PHerc1203 canonical-model demonstration is recorded in:

- `evidence/pherc1203-canonical-r152-summary.md`;
- `evidence/pherc1203-canonical-r152-order-stability.json`;
- `evidence/pherc1203-canonical-r152-manual-review.json`;
- `evidence/pherc1203-canonical-r152-contact-sheet.png`.

These files preserve the pinned model identity, forward/reverse stability metrics, fixed manual-review verdict, and a derived review image. Raw CT, probability maps, model weights, provider credentials, and compute-provider artifacts are not redistributed.

## Stage 6 — exact saturation / novelty audit

```bash
python toolkit/audit_tifxyz_exact_increment.py \
  OLD_TIFXYZ NEW_TIFXYZ --voxel-um 9.362 --output increment.json
```

This compares active quads by exact serialized coordinates. It distinguishes genuinely new accepted surface from proposal growth, bbox expansion, tolerance overlap, or duplicated geometry.

## Data and licenses

The release does not redistribute raw CT. Public inputs should be streamed from the official Vesuvius Challenge data endpoints and remain governed by their upstream license. Derived compact evidence included here is for reproducibility and review; consult `ATTRIBUTION.md` before redistribution.

## Submission reproducibility boundary

This release reproduces the validation framework and compact demonstrations. Re-running the complete 13-scroll campaign requires the public datasets and substantial local storage/network time. The report therefore binds every retained real-data conclusion to exact artifact hashes while keeping synthetic tests lightweight.
