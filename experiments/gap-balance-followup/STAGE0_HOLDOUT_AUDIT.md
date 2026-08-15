# GapBalance Stage-0 holdout feasibility audit

Status: **BLOCKED — do not train**

This audit concerns a follow-up to the immutable preregistered Gap8 result. It
does not reopen, overwrite, reinterpret, or tune against the previously opened
Scroll-4/5 or synthetic-test results.

## Decision

Fresh synthetic confirmation is feasible. A genuinely untouched, provenance-
complete real confirmation set is not currently available. The experiment must
therefore stop before training.

The blocking requirement is not merely a new filename or a previously unused
crop. The real confirmation data must be demonstrably outside:

1. the `Dataset059_s1_s4_s5_patches_frangiedt` training lineage of the pinned
   `surface_recto_059_redo` source checkpoint;
2. the original Gap8 Scroll-1 development and Scroll-4/5 confirmation data;
3. every earlier diagnostic, model-selection, and evaluation endpoint; and
4. model-generated or pseudo-labelled surfaces whose errors could make the
   evaluation self-confirming.

## Audited candidates

### Dataset059 and the Gap8 patch release — rejected

- Dataset059 contains 1,754 image/label pairs from Scrolls 1, 4, and 5.
- The original Gap8 release contains 200 seed-fixed Dataset059 pairs: 162 from
  Scroll 1, 36 from Scroll 4, and 2 from Scroll 5.
- Gap8 used 138 Scroll-1 patches for training, 24 Scroll-1 patches for
  validation, and all 38 Scroll-4/5 patches for final confirmation.
- The pinned source checkpoint is itself the Dataset059 model. Unused
  Dataset059 filenames therefore cannot become an independent real holdout.

### 2025 Kaggle Surface Detection public training set — not certifiable

- Current public model documentation describes 786 labelled training volumes;
  an older public mirror contains 806 rows across six anonymized `scroll_id`
  values.
- The official competition file inventory is visible, but the current account
  receives HTTP 403 when requesting the authoritative `train.csv`. No terms or
  competition access were changed during this audit.
- Public mirrors are not authoritative for the corrected 786-volume version.
- Physical-scroll provenance for the anonymized IDs is not published in the
  metadata available to this audit.
- A pre-existing image-overlap audit of the older competition corpus located
  189 crops in Scroll 1 and found 122 competition crops intersecting Dataset059
  regions. This proves that "different competition dataset" is not sufficient
  evidence of independence.

This corpus could become usable only if an authoritative manifest maps a
whole held-out `scroll_id` to a physical scroll outside Scrolls 1/4/5, proves
that the source checkpoint did not train on it, and supplies exact current
image/label identities without requiring new terms to be accepted silently.

### Current open-data scans and segment meshes — rejected

- The current official metadata index lists 45 samples and many raw volumes,
  predicted surface zarrs, and segment meshes.
- It does not expose an independent dense voxel-label corpus suitable for the
  official surface metrics.
- Many segment identifiers are explicitly `auto_grown`, and the metadata also
  lists `surface-prediction-zarr` products. Voxelising those outputs would make
  a pseudo-labelled confirmation set, not an independent real test.

### Official cube registry — rejected

- The current public `cubes.yaml` contains a single Scroll-1 example cube.
- That does not provide a new physical scroll or an adequate confirmation set.

### Hidden Kaggle test and prior reliability patches — rejected

- Hidden Kaggle labels are not locally available, cannot support fixed-panel
  inspection, and would require an external competition action.
- The prior reliability/geometry material was already inspected and is not a
  compatible untouched dense surface-label confirmation set.

## Synthetic feasibility

The deterministic synthetic painter can provide disjoint confirmation cells.
The proposed protocol reserves seeds 400–403 for development and 500–504 for
sealed confirmation. The generator revision, configuration, cell identities,
and file hashes must be frozen before any candidate training. Confirmation
labels and endpoints remain sealed until a single candidate is selected.

## Exact unblock condition

Training may begin only after a real-data manifest passes all of these checks:

- at least 48 fully labelled 3-D cubes from a whole physical scroll absent from
  Dataset059, or from at least two independently annotated new regions;
- at least 24 cubes meeting a GT-only compressed/multi-sheet eligibility rule;
- authoritative physical-scroll and annotation provenance;
- proof of no source-checkpoint, Gap8, diagnostic, or model-selection exposure;
- no pseudo-labels, model-selected crops, or prediction-derived annotations;
- coordinates and fixed qualitative panels selected from labels/geometry only;
- exact image, label, source-manifest, and selection-manifest SHA-256 values;
- compatible official-metric labels with ignore regions defined in advance.

If any item cannot be proved, the confirmation experiment remains blocked.

## Spend decision

Stage 0 authorizes no training and no RunPod spend. Current spend: **USD 0**.
If the holdout is unblocked, Kaggle GPU is the preferred first route. A timed
smoke test and a written job-count/runtime/cost estimate must precede any paid
RunPod request.
