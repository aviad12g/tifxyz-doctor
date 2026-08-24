# GapBalance Stage-0 holdout feasibility audit

> **Superseded for confirmation eligibility on 2026-08-21.** The PHerc1218
> author subsequently withdrew every published version after finding faults in
> crop CT and label stitching plus a unit error in the cross-validation-set
> gap-census comparison. Version 1 and version 2 are both forbidden. See the
> active hashed record `PHERC1218_HOLDOUT_QUARANTINE.json`. The metadata-only
> reconciliation below remains historical audit evidence; it is not a current
> scientific-validity certification.

Status: **`BLOCKED_REAL_HOLDOUT_NOT_CERTIFIED`**

This audit concerns a follow-up to the immutable preregistered Gap8 result. It
does not reopen, overwrite, reinterpret, or tune against the previously opened
Scroll-4/5 or synthetic-test results.

## Decision

Fresh synthetic confirmation is feasible. The PHerc1218 tight-contact release
is physically outside the Dataset059 source-checkpoint lineage and did not
exist when Gap2/Gap4 were proposed, so it can support an external real-CT
confirmation proxy with a deliberately narrow claim. It is not hand-annotated
truth: its repaired-v2 labels are automatic labels, and the experiment may
claim only agreement with that independently produced proxy.

Kaggle dataset version 2 resolves the version-1 membership discrepancy. The
provider explicitly identifies six pilot-extraction leftovers in version 1,
publishes the authoritative 254-contact/60-control set, and reports 209
contacts carrying both split instances. Exact provider-path reconciliation and
the metadata-only Stage-0 validator pass. Version 1 remains forbidden.

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

### PHerc1218 tight-contact validation release — certified external proxy

- Dataset reference: `jhjeong0815/pherc1218-tight-contact-val`, version 2.
- Physical source: PHerc1218, which the official scroll metadata identifies as
  a different physical scroll from Scroll 1 (PHercParis4), Scroll 4
  (PHerc1667), and Scroll 5 (PHerc0172).
- Checkpoint lineage: the frozen source checkpoint is
  `surface_recto_059_redo`, trained from Dataset059 S1/S4/S5. No PHerc1218
  material is named in that lineage.
- Temporal independence: Gap2 and Gap4, matched seeds, development-only
  selection, synthetic confirmation seeds, and all primary safety targets were
  specified publicly before this dataset was released.
- Selection independence: the upstream crop rule was frozen before extraction
  and does not use any GapBalance model or prediction.
- Label limitation: repaired-v2 automatic instance labels are an external
  evaluation proxy, not hand annotation or ground truth. Every real-data result
  must be described as agreement with that proxy.

Only provider metadata, GitHub Issue #191 comments, and `MANIFEST.jsonl` were inspected. No NPZ, CT crop,
surface label, instance label, probability, endpoint, or model prediction was
opened.

#### Version-2 corrected package audit

- provider dataset id: 11704096;
- provider version/status: 2 / `Ready`, created
  `2026-08-19T08:43:33.110000`;
- provider inventory: 324 files and 605,568,763 bytes, comprising 314 NPZs and
  10 public metadata/summary files;
- `MANIFEST.jsonl`: 90,251 bytes, SHA-256
  `b62e5a8f3e8ff4e2de7a5b3626f85fcee91df338c9e250867b2342b903ea9dfe`;
- manifest inventory: 314 unique paths and 314 unique file hashes, comprising
  254 contact and 60 control crops;
- contact bands: 14 / 60 / 60 / 60 / 60 for
  0–2 / 2–4 / 4–6 / 6–10 / 10+ voxels;
- 209 contacts contain both declared split instances and 45 do not;
- the manifest's 314 NPZ paths exactly equal the provider's 314 NPZ paths,
  with no missing or extra path;
- the frozen primary rule retains 171 contacts
  (12 / 39 / 42 / 32 / 46 by band), including 51 below four voxels, and 48
  controls;
- the dataset author publicly confirmed that the six version-1 excess files
  were pilot leftovers, that version 2 is authoritative, and that all v1
  derived numbers are void:
  <https://github.com/ScrollPrize/villa/issues/191#issuecomment-5339946340>.

The exact contract is `GAPBALANCE_STAGE0_CONTRACT.json`; the validator output
`PHERC1218_V2_SELECTION.json` freezes all 219 eligible file identities and the
four fixed panel files from manifest metadata only. Its payload SHA-256 is
`3a099b4aa55d6de16c289c8aafc59dc618c4c6a3ab78c800b229b1e9bdc444d8`;
the eligible-identity payload SHA-256 is
`5eb0690759c0c743f661de1390eb9357d39794419d4620791290a67aebd79882`;
the selected-path ledger SHA-256 is
`57aaa943adfc66959da99a660913e0a1e00ae525a01f18829d6b8f2de6306f41`;
and the fixed-panel payload SHA-256 is
`ce9f57adad7e5044dc44b7f9fc415b51734fcba7a5b50f53df000142ef5d5cd0`.

#### Version-1 package audit

- provider dataset id: 11704096;
- provider version: 1;
- provider bytes: 615,710,185;
- metadata SHA-256:
  `f41ba2257616411df54647b10228b7f98c81382fa8b35681e57004a740815f5a`;
- `MANIFEST.jsonl` SHA-256:
  `407951ebfd1cadb378c093b156ff68f320ebcb7995a4b69198029ac644d170b3`;
- manifest inventory: 320 unique files and 320 unique file hashes, comprising
  260 contact and 60 control crops;
- manifest contact bands: 17 / 60 / 60 / 60 / 63 for
  0–2 / 2–4 / 4–6 / 6–10 / 10+ voxels;
- frozen `crops_summary.json`: 254 accepted contact crops with bands
  14 / 60 / 60 / 60 / 60;
- `labels_summary.json` and `emptiness_summary.json`: both report 260 contact
  crops, consistent with downstream enumeration of the directory and
  inconsistent with the current extraction run's accepted count.

The public extractor creates `OUT/crops` with `exist_ok=True` but does not
require it to be absent or empty. Later label and emptiness passes glob every
NPZ in that directory. The exact six-file excess is three additional 0–2 crops
and three additional 10+ crops. This was sufficient to fail closed before
author confirmation. The author was asked publicly to confirm the authoritative
membership and issue a corrected immutable version:
<https://github.com/ScrollPrize/villa/issues/191#issuecomment-5339174393>.
The author then named the six orphans in the public correction record and
confirmed the authoritative version-2 membership:
<https://github.com/ScrollPrize/villa/issues/191#issuecomment-5339946340>.

#### Frozen eligibility rule

The real primary subset is selected from version-2 manifest metadata only:

- contact crop;
- both split instance ids present;
- `ct_empty_frac <= 0.10`;
- exact file and SHA-256 identity present in the corrected manifest.

The single-sheet control subset requires `ct_empty_frac <= 0.10`. Version 2
yields the frozen 171 eligible contacts and 48 eligible controls above.

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

## Exact certification condition

The version-2 real-data manifest passes all of these Stage-0 checks:

- corrected immutable provider version with reconciled extraction and package
  inventories;
- at least 48 eligible labelled 3-D contact crops and 24 eligible crops below
  a four-voxel measured gap;
- at least 48 eligible single-sheet controls;
- authoritative physical-scroll and annotation provenance;
- proof of no source-checkpoint, Gap8, model-selection, or GapBalance-prediction
  exposure;
- no model-selected crops or GapBalance-prediction-derived annotations;
- automatic-label status recorded in every real-result claim;
- fixed qualitative panels selected mechanically from corrected-manifest
  metadata before predictions;
- exact provider version, source-manifest, selected-file, and panel-list
  SHA-256 values;
- compatible metric labels and any ignore rule defined in advance.

If any identity or count changes, the confirmation experiment fails closed.

## Spend decision

This protocol authorizes no paid compute or RunPod spend. Current spend:
**USD 0**. Free Kaggle GPU is the preferred route. The complete public
protocol, exact six-job plan, original-control reuse proof, and a verify-only
launcher check must pass before free training. Any paid fallback still
requires a written provider/runtime/cost cap and fresh explicit approval.
