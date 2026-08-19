# GapBalance protocol draft

Status: **scientifically specified; execution blocked on corrected PHerc1218 package**

This draft becomes executable only after the package-identity condition in
`STAGE0_HOLDOUT_AUDIT.md` is satisfied, the corrected provider version and
metadata-only selections are inserted, a fail-closed validator passes, and the
complete protocol is public. No result may be used to weaken these rules.

## Question

Can lower exact inter-sheet gap supervision preserve at least a five-point
conditional-fusion benefit while avoiding the detection and false-splitting
regressions observed with immutable Gap8?

## Arms and pairing

- control: gap-background weight 1
- Gap2: gap-background weight 2
- Gap4: gap-background weight 4
- matched initialization seeds: 11, 23, and 47

All other bytes and choices are identical: pinned source checkpoint,
architecture, frozen encoder, decoder initialization, crop sequence, real to
synthetic sampling schedule, augmentations, optimizer, learning-rate schedule,
step count, normalization, runtime, and inference code. The training contract
remains 160-cube crops, batch 1, accumulation 2, deterministic 3:1 real to
synthetic sampling, 1,500 steps, AdamW at 5e-5 with 1e-5 weight decay and cosine
decay to 1e-6, and no test-time augmentation.

The three original control checkpoints are reused by exact file SHA-256:

- seed 11: `7a2e6168f32b3a3389bdc2b43b39a6654568a6da467e71948f523e7cbef6248f`;
- seed 23: `d40c4b856c9a65cc2de127e98d37d08633e03e6b7b5e375bec8991b1bf9487b3`;
- seed 47: `fd86b40b3b25f45f86f2ba43668997d4351fec7d3ebd35fa41856e4c0d505e44`.

They were trained under the exact same frozen 1,500-step source-checkpoint,
data-order, initialization-seed, optimizer, schedule, and augmentation contract
required here. Only six new jobs are permitted: Gap2 and Gap4 for seeds 11,
23, and 47. The new-arm implementation must leave the original control path
unchanged and prove that the sole per-sample numerical difference is the
synthetic inter-sheet-gap background weight.

## Data partitions

### Training

- exact original Scroll-1 training manifest;
- exact original synthetic training seeds 100–115;
- no confirmation data.

### Development and candidate selection

- exact original 24-patch Scroll-1 validation manifest;
- new synthetic development seeds 400–403;
- development endpoints may be opened only after the complete draft is public.

### Sealed confirmation

- real: a corrected immutable version of
  `jhjeong0815/pherc1218-tight-contact-val`, pinned by provider dataset id,
  version, manifest SHA-256, exact selected-file list, and exact fixed-panel
  list. Version 1 is explicitly forbidden because its package membership does
  not reconcile with its extraction summary;
- synthetic: seeds 500–504 over the same frozen 4-by-4 physical
  pitch-by-papyrus factorial plus the 700-um no-kollesis single-sheet control;
- fixed real panels selected mechanically from manifest metadata and
  repaired-v2 labels before any candidate prediction exists.

The confirmation cache is generated, identity-hashed, access-sealed, and held
separately from development. No confirmation prediction is run until one
candidate and every threshold are frozen.

The PHerc1218 labels are repaired-v2 automatic labels. Real confirmation
therefore measures agreement with an external repaired-label proxy on a new
physical scroll; it is never described as ground-truth accuracy or as
independent human validation.

### Real eligibility and fixed panels

The primary contact subset is selected from the corrected manifest without
opening NPZ files:

- `arm == "crops"`;
- `both_instances_present == true`;
- `ct_empty_frac <= 0.10`;
- band in `{0-2, 2-4, 4-6, 6-10, 10+}`;
- unique relative file path and unique 64-character lowercase SHA-256.

The primary single-sheet control subset uses `arm == "control"` and
`ct_empty_frac <= 0.10`. The protocol fails closed unless it retains at least
48 contacts, at least 24 contacts below four voxels, and at least 48 controls.

One panel is selected from each of the four contact bands below ten voxels.
Within a band, choose the eligible file minimizing
`SHA256(corrected_manifest_sha256 + "\\0" + relative_file)`. After the file
list is public, choose the display plane using only its repaired instance
label: among all orthogonal planes containing both declared split ids, maximize
their combined pixel count; ties prefer axes z, y, x and then the lower slice.
Each panel is rendered for all three matched seeds, creating exactly 12
candidate/control comparisons. No panel may be removed, replaced, recropped,
or rerendered because of a model result.

## Threshold rule

For each arm and matched seed, select one threshold from
`{0.30, 0.35, ..., 0.70}` using only the Scroll-1 development patches. Maximize
mean official blend; ties prefer the threshold nearest 0.5 and then the lower
threshold. Freeze validation-cache hashes, every per-patch score, selected
thresholds, and the selection payload before confirmation. Report thresholds
0.4, 0.5, and 0.6 only as labelled sensitivity analyses.

## Candidate-selection rule

Evaluate Gap2 and Gap4 against their matched controls on development data.

An arm is eligible only if all pooled development constraints hold:

- conditional-fusion reduction at least 5.0 percentage points;
- detection decline no worse than 2.0 percentage points;
- false-split increase no worse than 2.0 percentage points;
- mean real-development blend delta at least -0.005;
- mean real-development TopoScore delta at least -0.005.

Among eligible arms, select the arm with the larger conditional-fusion
reduction. If the reductions differ by less than 1.0 percentage point, select
Gap2. If neither arm is eligible, stop without opening confirmation data.

Write and hash a candidate-freeze record containing every development count,
denominator, threshold, arm/seed checkpoint hash, selection decision, and the
statement `confirmation_outputs_inspected: false` before downloading or
running any confirmation NPZ.

## Confirmation metrics

Synthetic primary counts and rates:

- conditional fusion among detected neighbour-sheet sites;
- site-centre detection;
- false splitting on the no-kollesis single-sheet control;
- pooled raw counts and matched-seed results.

Real primary metrics:

- official blend;
- TopoScore;
- Surface Dice and VOI as required decompositions;
- paired per-crop deltas overall and by published gap band;
- one-sided 95% cluster-bootstrap intervals from 10,000 draws with seed 1218,
  resampling source slabs and retaining every crop and all three matched seeds
  within each sampled slab.

## Seven immutable pass/fail gates

Overall success requires all seven gates.

1. Pooled synthetic conditional fusion is at least 5.0 percentage points lower
   than matched control.
2. Conditional-fusion delta is negative in all three matched seeds.
3. Pooled detection delta is at least -2.0 percentage points, and no matched
   seed is below -4.0 points.
4. Pooled false-split delta is at most +2.0 percentage points, and no matched
   seed exceeds +4.0 points.
5. Real blend is non-inferior: the lower one-sided 95% cluster-bootstrap bound
   is at least -0.005, and every matched-seed point delta is at least -0.005.
6. Real TopoScore is non-inferior: the lower one-sided 95% cluster-bootstrap
   bound is at least -0.005, and every matched-seed point delta is at least
   -0.005. A lower bound above zero is separately labelled improvement.
7. All 12 fixed candidate/control comparisons are published with no exclusion,
   replacement, recrop, or result-dependent rendering. Visual separation,
   missed-sheet, break, and split outcomes are reported for every comparison
   as descriptive evidence and cannot alter the quantitative verdict.

No margin, seed rule, panel rule, threshold, metric, or primary/secondary label
may change after confirmation inference starts.

## Reporting

Publish all raw counts, denominators, per-seed values, per-cube deltas,
intervals, fixed panels, sensitivity thresholds, hashes, operational failures,
nulls, adverse outcomes, and limitations. Keep the original Gap8 evidence and
its 5/7 outcome unchanged and separately identified.

## Compute gate

No paid compute is authorized by this draft. The immutable Gap8 metadata gives
a direct timing baseline: its six jobs each took 4,164–4,209 seconds on a
Kaggle Tesla P100, a mean of about 69.7 minutes. Exact control reuse leaves six
new Gap2/Gap4 jobs, or approximately 6.98 observed P100 GPU-hours. Expected
training wall time is about 2.4 hours with three concurrent Kaggle jobs, plus
staging and result-blind development evaluation. Kaggle is therefore the
preferred route at USD 0 provider cost.

Before training:

1. run the complete hash/determinism preflight without a GPU;
2. freeze the corrected PHerc1218 provider version and manifest-derived lists;
3. prove exact original-control reuse and six-job scope;
4. verify Kaggle quota and run one bounded verify-only launcher check;
5. present any paid fallback with provider, GPU type, expected hours, price,
   and hard cap; and
6. use paid compute only after explicit approval.
