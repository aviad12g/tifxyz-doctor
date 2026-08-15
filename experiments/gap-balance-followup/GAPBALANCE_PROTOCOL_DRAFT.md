# GapBalance protocol draft

Status: **scientifically specified but not executable or public-frozen**

This draft becomes eligible for public preregistration only after the real
holdout condition in `STAGE0_HOLDOUT_AUDIT.md` is satisfied. No result may be
used to weaken these rules.

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

All nine arm/seed jobs are rerun unless the public freeze proves that an
existing control checkpoint is byte-for-byte identical in source, data order,
runtime, and initial state. Reuse is forbidden on merely semantic equivalence.

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

- real: a new manifest satisfying every Stage-0 unblock condition;
- synthetic: seeds 500–504 over the same frozen 4-by-4 physical
  pitch-by-papyrus factorial plus the 700-um no-kollesis single-sheet control;
- fixed real panels selected by GT-only spacing/curvature rules before any
  candidate prediction exists.

The confirmation cache is generated, identity-hashed, access-sealed, and held
separately from development. No confirmation prediction is run until one
candidate and every threshold are frozen.

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
- paired per-cube deltas and one-sided 95% cluster-bootstrap intervals, with
  physical region as the resampling cluster.

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
7. Every fixed panel is published; blinded assessment finds at least one clear
   separation improvement and no new nearby break, missed target sheet, or
   qualitatively worse split in any panel.

No margin, seed rule, panel rule, threshold, metric, or primary/secondary label
may change after confirmation inference starts.

## Reporting

Publish all raw counts, denominators, per-seed values, per-cube deltas,
intervals, fixed panels, sensitivity thresholds, hashes, operational failures,
nulls, adverse outcomes, and limitations. Keep the original Gap8 evidence and
its 5/7 outcome unchanged and separately identified.

## Compute gate

No paid compute is authorized by this draft. Before training:

1. run the complete hash/determinism preflight without a GPU;
2. obtain a timed single-job estimate on Kaggle or another free allocation;
3. state whether six jobs (verified control reuse) or nine jobs are required;
4. present expected wall time, paid-hour estimate, provider price, and a hard
   campaign cap to the user;
5. use RunPod only after explicit approval.
