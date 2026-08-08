# Fusion-aware surface fine-tuning (preregistration)

Status: **FROZEN BEFORE TRAINING — no model update has been run.** The
supervision primitive, source pins, data splits, evaluation panels, primary
gates, and mechanical design tests were fixed before any model update or
validation/test output was inspected. Any future amendment must be published
before the affected run and reported beside the results.

## Question

Can exact inter-sheet gap supervision reduce merged-sheet predictions in
compressed regions without sacrificing surface detection or the official
topology-aware score on unseen real-scroll patches?

## Intervention

Start both arms from the public `surface_recto_059_redo` checkpoint. Both see
the same ordered stream of real Dataset059 replay crops and finite-thickness
synthetic crops, use the same augmentations, optimizer, schedule, and seeds,
and update the same parameters.

- **Control:** CE + soft Dice with ordinary background weight.
- **Gap arm:** identical loss, except exact air-gap voxels lying within a
  radius-3 overlap of two distinct synthetic `turn_id` instances receive
  background weight 8.

The weight is applied only to synthetic crops with exact instance labels. Real
binary labels never receive inferred gap supervision. Join-only surface
material is explicitly excluded from the gap mask.

## Source and data pins

- source checkpoint: `scrollprize/surface_recto_059_redo/Model_epoch499.pth`,
  SHA-256 `f1990a02ac91889c1f989522ae0e45421a91cb666320448aaf579d42b081636f`;
- finite-thickness painter: `Diego-dcv/vesuvius-topological-grid`
  `scripts/contrast_phantom.py` at commit
  `f3a8b8afcf794988a077bcca2c9ed210dc927fbe`, file SHA-256
  `41f2a097b819bc486819d6702ed3949d0bdfe722c2405b1fe86828062da1d9b8`;
- model loader/evaluation reference:
  `Jinhojeong/vesuvius-surface-geometry-diagnostic` commit
  `3a46b63df4bec66efb9ba614ee58e944fd990bba`, loader SHA-256
  `49a1d4ea3ee611236d53b1291ca2e8cd6e1450f2fc032223ff89a5b53d9ec903`;
- model architecture source: `ScrollPrize/villa` commit
  `4d5c9e605df86eb5060c0753faadc8b2aba44175`, with the committed
  `vesuvius/src/vesuvius/models` subtree bundled byte-for-byte. An empty,
  documented `vesuvius/__init__.py` packaging shim is added outside that
  subtree solely to prevent a site-installed package from shadowing the
  pinned model source;
- physical fusion ray reader from that repository at the same commit,
  `results/fusion_pilot/fusion_readout.py` SHA-256
  `a533ee940712f1a47111705e4d8fff4990ccffb0afb103c7337a68bff244781c`;
- public Dataset059 patch release `patches-v2` assets:
  - `images_s1.tar`: 1,478,256,640 bytes,
    `cce01d96c77cc7966a41a805b2690db3e7705ce92cbc52d35d0a83a5c7ba35a5`;
  - `images_s4_s5.tar`: 444,723,200 bytes,
    `960a152238df8fc60d108a13302af9676c096019c21f204447aa3bf7d9dce4ae`;
  - `labels.tar`: 83,087,360 bytes,
    `6e01e4d5f0591796a060bda0a1357ed8cc8b801912a7d3e569ecb246764741a6`.
- official topology-aware evaluator: Kaggle dataset
  `sohier/vesuvius-metric-resources`, downloaded ZIP SHA-256
  `64d24044b7381dbb660a0e9f602ad0f8fd37d1935c09a8ee3b78088433910e68`;
  `src/topometrics/leaderboard.py` SHA-256
  `f0db94436eea4464a30f252ebc7c35553e539da5e4832e4efaf523ed664cd811`.

## Normalization contract

Both arms use the normalization declared by the pinned checkpoint rather than
the current `vesuvius.predict` CLI default. The run must fail before training
unless the checkpoint declares `CTNormalization` with mean
`129.09779357910156`, standard deviation `42.188316345214844`, and clipping
percentiles `[44, 236]`. Every real and synthetic crop is clipped to that
interval before applying the checkpoint mean and standard deviation.

This is frozen explicitly because ScrollPrize/villa issue #1364 demonstrates
that the current nnU-Net inference entry point can silently substitute
per-volume instance z-scoring for the checkpoint plan. The control, gap arm,
validation, and test paths all call the same checked normalization primitive;
none relies on the CLI default. The saved checkpoints record the resolved
scheme and all four intensity properties. The `[0, 212]` plan values reported
in #1364 belong to the separate `surface_m7` checkpoint; they are not reused
for `surface_recto_059_redo`.

## Frozen data split

- Real replay training: Scroll 1 patches only.
- Real validation: exactly 24 Scroll 1 patches with the lexicographically
  lowest SHA-256 hashes of their filenames, used only for threshold selection;
  all remaining Scroll 1 patches are replay training data.
- Real test: all available Scroll 4 and Scroll 5 patches; never used for
  training, early stopping, weight selection, or threshold selection.
- Synthetic training seeds: 100–115.
- Synthetic validation seeds: 200–203.
- Synthetic test seeds: 300–304.
- Synthetic physical pitches: 170 / 200 / 230 / 260 µm at 150 µm sheet
  thickness and 30 µm voxels.
- Synthetic papyrus levels: 35 / 50 / 65 / 90; noise sigma 6.

The sealed synthetic test is the full five-seed 4×4 pitch-by-papyrus
factorial (80 finite-thickness primary cells, kollesis retained) plus, for
each seed and papyrus level, a 700 µm no-kollesis single-sheet control (20
cells). Exactly 100 cells are assigned round-robin to 10 deterministic shards.
The cache retains only the published reader's sampled ray probabilities and
instance labels; it cannot calculate or print endpoints.

The four real visual panels are selected without model output from the held-out
Scroll 4/5 labels. For every eligible test patch (at least 5,000 labelled
surface voxels), sample up to 30,000 foreground sites with a seed derived from
the SHA-256 of its filename, measure next-sheet distance along both directions
of a sigma-3 signed-distance normal, and rank patches by the descending share
of sites at spacing <=8 voxels. Ties use lower median spacing and then filename.
The panel centre is a sampled compressed site nearest the coordinate-wise
median of that patch's compressed samples. The selected filenames, ZYX centres,
all candidate measurements, input hashes, and selection-payload hash are frozen
before any baseline or intervention prediction is viewed.

The 16 synthetic training seeds map in ascending order to the lexicographic
pitch-major 4×4 `(pitch, papyrus)` factorial, so every combination appears
exactly once in the cached training cells. Even-numbered seeds keep kollesis;
odd-numbered seeds omit it. This anatomy toggle and every intensity/geometry
choice are identical between the control and gap arms.

No test output may be inspected until the checkpoint hashes and selected
thresholds have been frozen from training/validation data.

## Training contract

- 160³ crops, batch 1, gradient accumulation 2;
- deterministic 3:1 real-to-synthetic sample schedule;
- 1,500 optimizer steps;
- AdamW, learning rate 5e-5, weight decay 1e-5, cosine decay to 1e-6;
- decoder-only updates; encoder frozen;
- three training seeds: 11 / 23 / 47;
- Kaggle CUDA runtime with PyTorch 2.5.1 from the CUDA 12.1 wheel index;
- no test-time augmentation.

If memory or runtime makes this contract impossible, the protocol must be
amended and republished before any result is inspected.

## Threshold selection

Every validation and test prediction uses 192-cube sliding windows at 96-voxel
steps (50% overlap), separable Gaussian blending with sigma `patch/8` and a
minimum weight of `1e-4`, the surface softmax channel, reflect padding at short
edges, and no test-time augmentation. The explicit checked CT normalization
above is applied once to the full input volume before windowing. The same
inference function is used for baseline, control, and gap arms.

For the released baseline and for each arm and training seed, select one threshold from
`{0.30, 0.35, ..., 0.70}` using only the real Scroll 1 validation split. The
selection maximizes the mean official leaderboard blend over validation
patches; ties prefer the threshold nearest 0.5 and then the lower threshold.
The validation cache hashes, every per-patch score, selected thresholds, and
selection-payload hash are frozen before test inference. Apply that frozen
threshold unchanged to real Scroll 4/5 and synthetic test cells. Also report
fixed-threshold 0.5 and 0.4/0.6 sensitivity results.

## Primary success gates

All gates must pass; otherwise the intervention is a negative result.

1. On held-out synthetic neighbour-sheet sites, pooled conditional fusion is
   at least 10 percentage points lower than control at the selected threshold,
   with a negative delta in all three training seeds.
2. Pooled site-centre detection decreases by no more than 2 percentage points.
3. On the 700 µm no-kollesis single-sheet control, false-split rate increases
   by no more than 2 percentage points.
4. On unseen Scroll 4/5 patches, the mean official leaderboard blend is not
   lower than control by more than 0.005 and mean TopoScore is higher.
5. At least one preregistered real compressed-region panel shows a visually
   verifiable separation improvement without a new nearby break; panels are
   selected by coordinates before intervention predictions are viewed.

The synthetic primary readout uses the published issue-191 ray geometry
(span 12 voxels, 0.5-voxel steps, 20,000 surface sites sampled with seed 1218).
Fusion is pooled over detected neighbour-sheet sites; detection is the ray
centre clearing threshold. Raw counts are retained before rates are formed.

## Required reporting

- every count, denominator, per-seed value, and paired per-patch delta;
- bootstrap interval over real test patches as descriptive uncertainty;
- checkpoint, source, input, split, and output SHA-256 values;
- failures and regressions alongside improvements;
- baseline, control fine-tune, and gap-arm outputs using identical evaluation.
- the exact normalization scheme, clipping bounds, mean, and standard
  deviation used for every reported arm.

## Interpretation limits

- Synthetic instance labels are exact but do not make the renderer a real
  scroll.
- Scroll 4/5 provide cross-scroll evaluation, not universal generalization.
- Passing the synthetic gates without the real non-inferiority/topology gate
  is not a production improvement and will not be presented as one.
