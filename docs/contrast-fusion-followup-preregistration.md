# Contrast-at-fixed-geometry fusion follow-up (frozen preregistration)

Status: **FROZEN — no cells generated and no inference run.** Jinho Jeong
confirmed the co-primary reader definitions, the five-point extreme contrast
margin, and Diego-dcv's amended long-pitch control in
[ScrollPrize/villa issue #191](https://github.com/ScrollPrize/villa/issues/191#issuecomment-5171858277).

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

- finite-thickness contrast painter: `Diego-dcv/vesuvius-topological-grid`
  `scripts/contrast_phantom.py` at commit
  `f3a8b8afcf794988a077bcca2c9ed210dc927fbe`, SHA-256
  `41f2a097b819bc486819d6702ed3949d0bdfe722c2405b1fe86828062da1d9b8`;
- Villa: `4d5c9e605df86eb5060c0753faadc8b2aba44175`;
- inference harness SHA-256:
  `54cbc7521ee34f559724427f34ca723b6ecec004954c389b57b6a82cbe1be881`;
- checkpoint: `scrollprize/surface_recto_059_redo/Model_epoch499.pth`,
  SHA-256
  `f1990a02ac91889c1f989522ae0e45421a91cb666320448aaf579d42b081636f`.

The primary arm keeps the painter's kollesis joins, matching the completed
physical pilot and retaining real papyrus anatomy. The 700 µm false-split
control alone uses the painter's `--no-kollesis` contract. Fibers are omitted
in both arms.

The frozen reader at diagnostic-repository commit `3a46b63` constructs its
sites and normals from `turn_id`; it does not use `gt_surface` in the readout.
Kollesis can extend inward beyond the owning turn's finite-thickness label, so
the primary contract is `turn_id > 0` as a subset of `gt_surface > 0`, with
every additional surface voxel required to equal the exported kollesis-only
footprint. The no-kollesis control retains exact equality. These invariants are
checked before inference.

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

Within a seed, all four contrast cells must have byte-identical `gt_surface`,
`turn_id`, and kollesis mask, identical geometry, and the same random-noise
draw over all surface material after subtracting the clean papyrus, ink, and
kollesis signals. Only the papyrus/ink/join intensity convention changes. Any
failure of those invariants aborts the experiment.

The published reader deterministically samples its 20,000 sites from ground
truth. Because the four contrast cells within a seed have byte-identical
geometry and labels, their readouts use the same sites. The preregistered
contrast deltas are therefore site-paired comparisons, not population
comparisons.

Pitch 200 µm is chosen before running because it is an interior physical-pilot
condition, has a resolved 50 µm nominal gap, and had neither the lowest nor the
highest site-centre detection rate in the prior pilot.

## Long-pitch control arm

To avoid repeating the underpowered single-sheet control:

- pitch: **700 µm**;
- papyrus levels: **35 / 90**;
- kollesis: **off** (`--no-kollesis` contract);
- all other generation parameters and seeds unchanged;
- total control cells: **6**.

At pitch 700 µm the minimum centre spacing between turns is 450 µm, 90 µm
larger than the reader's 360 µm span, so every perimeter site qualifies as
single-sheet by `turn_id`. With joins enabled, the nearest own-turn material
instead falls to 135 µm at about 9% of sites and would bias the false-split
control low. The control therefore omits joins; the primary keeps them.

This arm is not used to estimate the contrast effect on neighbour fusion. It
exists only to estimate false-split behaviour at the contrast extremes.

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

The five-point separation margin is fixed before running, is evaluated only
between the preregistered axis extremes (papyrus 90 versus 35), and is larger
than the full conditional-fusion range observed across the previous gap
pilot. It is not an adjacent-level decision margin.

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
