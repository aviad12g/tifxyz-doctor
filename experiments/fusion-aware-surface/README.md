# Fusion-aware surface model

August 2026 core-pipeline experiment for Vesuvius surface detection.

**Status: completed preregistered experiment; sealed held-out evidence is public.**

The final experiment passed 5 of 7 preregistered gates. Gap8 reduced pooled
synthetic conditional fusion by 10.667 percentage points and improved pooled
held-out real blend by 0.003359 and TopoScore by 0.004928, but it failed both
safety guardrails: detection fell by 7.805 points and false splitting rose by
13.351 points. Because every primary gate was required to pass, Gap8 did not
meet the overall preregistered success criterion and is not a drop-in
production replacement. See the
[complete evidence report](../../docs/fusion-aware-final-evidence/FINAL_EVIDENCE.md)
for every gate, fixed panel, adverse result, hash, and limitation.

## One-command mechanism demo

Run a pinned, checkpoint-free toy demonstration of the exact supervision
primitive:

```bash
./experiments/fusion-aware-surface/run_gap8_demo.sh
```

The command creates an isolated virtual environment and writes three files to
`experiments/fusion-aware-surface/gap8-demo-output/`:

- `gap8_demo.png`: exact sheet instances, their shared air-gap mask, and the
  resulting weight map;
- `demo_summary.json`: fixed geometry counts and the normalized gap-gradient
  multiplier; and
- `demo_manifest.json`: SHA-256 and byte size for both artifacts.

With the pinned demo environment, the expected invariants are 423 exact gap
voxels and a 6.054693274 normalized gap-gradient multiplier. The expected
artifact identities are:

| file | bytes | SHA-256 |
|---|---:|---|
| `gap8_demo.png` | 3545 | `f1ff3c088f2e6e255beb325a3a2477dfb0a2d19c7d51de051f8bd5066033fc2b` |
| `demo_summary.json` | 442 | `85720feb8e371720f3e1f6711f3204c8e83b3992ecf88c7b0cea8f8a53bbbd13` |

The demo uses public toy geometry only. It demonstrates how Gap8 identifies
the compressed-layer gap and changes supervision; it does not download a
checkpoint, rerun training, or claim the published held-out result. To place
the primitive in another training loop, derive the mask with
`inter_sheet_gap_mask(...)`, broadcast it over ZYX if needed, and pass it to
`fusion_aware_surface_loss(..., gap_weight=8)`. Use `gap_weight=1` for the
matched control.

The released binary surface model often turns two tightly packed sheets into
one broad probability peak. Ordinary binary fine-tuning can improve voxel AUC
while making topology worse because it has no explicit representation of the
air gap separating distinct sheet instances.

This project tests one isolated intervention: derive exact inter-sheet air-gap
supervision from the synthetic painter's `turn_id` labels and give those voxels
extra background weight during otherwise identical fine-tuning.

The control and intervention arms use the same checkpoint, samples, crops,
optimizer, schedule, and random seeds. Only the gap weight differs.

All arms use a fail-closed implementation of the checkpoint's declared
`CTNormalization` (clip to `[44, 236]`, then mean `129.09779357910156` and
standard deviation `42.188316345214844`). This avoids the silent nnU-Net
wrapper mismatch reported in
[ScrollPrize/villa#1364](https://github.com/ScrollPrize/villa/issues/1364).
The different `[0, 212]` plan values in that report belong to `surface_m7`,
not to this pinned `surface_recto_059_redo` checkpoint.

The experiment is not considered successful unless it improves held-out
fusion topology without sacrificing target-sheet detection or real-patch
official score. See `PREREGISTRATION.md` for the frozen gates.

## Local checks

```bash
/Users/mazalcohen/miniforge3/bin/python3 -m pytest -q
python validate_design.py
python verify_official_metric.py
```

The official evaluator is the source-pinned Kaggle dataset
`sohier/vesuvius-metric-resources`. By default its extracted
`topological-metrics-kaggle` directory is expected beside this repository at
`../metric-resources/`; set the task-specific `FUSION_TOPOMETRICS_ROOT` to the
project root if it lives elsewhere. The worker verifies the Python evaluator
and Betti-Matching source hashes and refuses to score if they differ.

## Model-blind data freeze

The public 200-patch release is downloaded, size-checked, SHA-256 checked, and
extracted without predictions. The split manifest and four real-panel centres
are then selected from filenames and held-out labels only.

```bash
python prepare_real_data.py --out work/real-data
python freeze_real_split.py \
  --root work/real-data/extracted \
  --out real_split_manifest.json
python select_real_panels.py \
  --root work/real-data/extracted \
  --split-manifest real_split_manifest.json \
  --out real_panel_manifest.json
```

## Matched training arms

Run the control and gap-weighted arms with the same seed and arguments. The
full preregistration requires seeds 11, 23, and 47 for each arm.

The production Kaggle jobs use `kaggle_train_runner.py`. That launcher
fail-closes on all archive, checkpoint, painter, loader, model-source, split,
and project hashes; emits one arm/seed checkpoint; and deliberately performs
no validation or test inference. The private asset bundle can be checked
without installing packages, extracting data, or starting a GPU job:

```bash
python kaggle_train_runner.py \
  --arm control --seed 11 \
  --asset-root /path/to/vesuvius-fusion-aware-training-assets \
  --verify-only
```

Only after the preregistration is public should the six production jobs run:
`control` and `gap8`, each at seeds 11, 23, and 47.

```bash
python train_fusion_aware.py \
  --arm control --seed 11 \
  --real-data work/real-data/extracted \
  --split-manifest real_split_manifest.json \
  --painter-scripts ../contrast-painter-source/scripts \
  --diagnostic-scripts ../surface-geometry-diagnostic/scripts \
  --checkpoint ../models/surface_recto_059_redo/Model_epoch499.pth \
  --work work/control-seed11

python train_fusion_aware.py \
  --arm gap8 --seed 11 \
  --real-data work/real-data/extracted \
  --split-manifest real_split_manifest.json \
  --painter-scripts ../contrast-painter-source/scripts \
  --diagnostic-scripts ../surface-geometry-diagnostic/scripts \
  --checkpoint ../models/surface_recto_059_redo/Model_epoch499.pth \
  --work work/gap8-seed11
```

Validation probabilities are cached for all seven runs before thresholds are
selected. Test caching refuses to start unless given that matching frozen
threshold manifest.

```bash
python cache_real_predictions.py --run baseline --split validation \
  --real-data work/real-data/extracted \
  --split-manifest real_split_manifest.json \
  --diagnostic-scripts ../surface-geometry-diagnostic/scripts \
  --source-checkpoint ../models/surface_recto_059_redo/Model_epoch499.pth \
  --out work/validation-cache

python freeze_thresholds.py \
  --validation-root work/validation-cache \
  --split-manifest real_split_manifest.json \
  --out frozen_thresholds.json
```

After all seven test caches have passed the frozen-threshold gate, score them
once with the official metric and paired patch bootstrap:

```bash
python score_real_test.py \
  --test-root work/test-cache \
  --split-manifest real_split_manifest.json \
  --threshold-manifest frozen_thresholds.json \
  --out sealed_real_test_results.json
```

Synthetic test inference is separately sealed into ten shards per run. Each
shard stores only the preregistered sampled rays and prints no endpoints. Once
all 70 manifests exist, the one-shot scorer verifies every hash and pools raw
counts before testing the three matched-seed gates:

```bash
python cache_synthetic_rays.py --run control_seed11 \
  --shard-index 0 --shard-count 10 \
  --painter-scripts ../contrast-painter-source/scripts \
  --diagnostic-scripts ../surface-geometry-diagnostic/scripts \
  --reference-reader ../surface-geometry-diagnostic/results/fusion_pilot/fusion_readout.py \
  --source-checkpoint ../models/surface_recto_059_redo/Model_epoch499.pth \
  --model-state work/control-seed11/checkpoints/surface059_control_seed11.pth \
  --threshold-manifest frozen_thresholds.json \
  --split-manifest real_split_manifest.json \
  --out work/synthetic-rays

python score_synthetic_test.py \
  --ray-root work/synthetic-rays \
  --split-manifest real_split_manifest.json \
  --threshold-manifest frozen_thresholds.json \
  --out sealed_synthetic_test_results.json
```
