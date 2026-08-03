# Contrast-at-fixed-geometry fusion follow-up (draft preregistration)

Status: **DRAFT — no cells generated and no inference run.** This protocol is
to be frozen only after Jinho Jeong and Diego-dcv confirm the readout and the
long-pitch control geometry in ScrollPrize/villa issue #191.

## Question

For the frozen `surface_recto_059_redo` checkpoint, does increasing material
contrast improve separation between neighbouring finite-thickness sheets, or
does it only increase the probability that the target sheet is detected?

The previous physical pilot found that, conditional on detecting the site's
own sheet, the checkpoint fused it to its neighbour about 75–78% of the time
over nominal gaps from 10 to 150 µm. Detection, not conditional fusion, drove
the humped unconditional curve. The detection-conditioned decomposition was
exploratory there; it is preregistered here.

## Frozen sources

- finite-thickness painter: `Diego-dcv/vesuvius-topological-grid` commit
  `2c483dd16ebfdcd35740864a5e1e30d1fcc03af9` (merged PR #2);
- Villa: `4d5c9e605df86eb5060c0753faadc8b2aba44175`;
- inference harness SHA-256:
  `54cbc7521ee34f559724427f34ca723b6ecec004954c389b57b6a82cbe1be881`;
- checkpoint: `scrollprize/surface_recto_059_redo/Model_epoch499.pth`,
  SHA-256
  `f1990a02ac91889c1f989522ae0e45421a91cb666320448aaf579d42b081636f`.

No checkpoint, inference, painter, threshold, or reader tuning is permitted.

## Paired primary grid

- papyrus levels: **35 / 50 / 65 / 90**;
- additive noise sigma: **6**;
- pitch: **200 µm**;
- sheet thickness: **150 µm** (nominal gap 50 µm);
- voxel: **30 µm**;
- columns: **12**;
- Z window: **8 mm**;
- generation seeds: **0 / 1 / 2**;
- total primary cells: **12**.

Within a seed, all four contrast cells must have byte-identical `gt_surface`
and `turn_id`, identical geometry, and the same random-noise draw. Only the
papyrus/ink intensity convention changes (`ink = papyrus + 110`). Any failure
of those invariants aborts the experiment.

Pitch 200 µm is chosen before running because it is an interior physical-pilot
condition, has a resolved 50 µm nominal gap, and had neither the lowest nor the
highest site-centre detection rate in the prior pilot.

## Long-pitch control arm

To avoid repeating the underpowered single-sheet control:

- pitch: **700 µm**;
- papyrus levels: **35 / 90**;
- all other generation parameters and seeds unchanged;
- total control cells: **6**.

This arm is not used to estimate the contrast effect on neighbour fusion. It
exists only to estimate false-split behaviour at the contrast extremes. Before
freezing, Jinho should confirm that pitch 700 µm with the existing 360 µm ray
span creates an adequately large single-sheet-site population. If it does not,
the control geometry must be revised *before* any inference.

## Frozen inference and output contract

- 192³ patches, 50% step;
- separable Gaussian blending, sigma = patch/8, minimum weight `1e-4`;
- foreground channel 1;
- no test-time augmentation;
- deterministic inference;
- each output NPZ contains exactly:
  - `prob`: float16 ZYX;
  - `gt_surface`: uint8 ZYX;
  - `turn_id`: int16 ZYX.

The manifest will pin every input, array, NPZ, source, and runtime by SHA-256.
Probabilities will be published before any endpoint is computed or inspected.

## Preregistered readout

The primary operating threshold is **0.5**. Thresholds **0.4 and 0.6** are
sensitivity analyses and cannot replace the primary.

Two co-primary measurements are reported at every papyrus level, pooled and
separately for each generation seed:

1. **site-centre detection rate** over true neighbour-sheet sites;
2. **conditional fusion rate** among those sites where the target sheet is
   detected at the site centre.

The preregistered contrast effects are the paired papyrus-90 minus papyrus-35
differences:

- `Δdetection = detection(90) - detection(35)`;
- `Δconditional_fusion = conditional_fusion(90) - conditional_fusion(35)`.

Decision labels:

- **separation benefit:** pooled `Δconditional_fusion <= -5` percentage
  points and the delta is negative in all three seeds;
- **detection only:** pooled `Δdetection >= +5` points and
  `abs(Δconditional_fusion) < 5` points;
- **mixed/inconclusive:** every other result.

The five-point separation margin is fixed before running and is larger than
the full conditional-fusion range observed across the previous gap pilot.

Preregistered secondary measurements at threshold 0.5:

- unconditional fusion rate;
- true-gap-preserved separation rate;
- missing-neighbour separation rate;
- own-sheet-miss rate;
- seed range for every measurement;
- false-split rate and exact control-site count in the 700 µm arm.

All counts and denominators must be published. No p-value is planned: seeds
measure variation across this analytic generator, not a random sample of real
scrolls.

## Interpretation limits

- This is a synthetic, checkpoint-specific mechanism probe, not real-scroll
  accuracy.
- Papyrus and ink intensity follow the painter's existing coupled convention;
  the experiment tests the generator's full material-contrast axis, not a
  papyrus-only photometric intervention.
- Any analysis invented after viewing the primary results must be labelled
  exploratory and kept separate from the preregistered table.
