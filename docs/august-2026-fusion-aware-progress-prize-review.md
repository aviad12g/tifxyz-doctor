# August 2026 Progress Prize form payload — review draft

**Status:** Prepared for Aviad Cohen's review. Do not submit without Aviad's explicit approval.

## Email

Use the email address recorded by Aviad's signed-in Google account.

## Your full name

Aviad Cohen

## Team description

Individual submission by Aviad Cohen. The work was developed with OpenAI Codex as an AI coding and research assistant; Aviad is the sole entrant and team leader.

## URL to your open source / publicly available contribution

- https://github.com/aviad12g/tifxyz-doctor/tree/agent/freeze-fusion-aware-checkpoints/experiments/fusion-aware-surface
- https://github.com/aviad12g/tifxyz-doctor/tree/agent/freeze-fusion-aware-checkpoints/docs/fusion-aware-final-evidence

## Short description of how your contributions substantially increase the probability of reading complete scrolls

This contribution addresses the compressed/high-curvature surface-segmentation problem discussed in ScrollPrize/villa issue #191 (https://github.com/ScrollPrize/villa/issues/191): two nearby sheets can collapse into one broad foreground prediction, creating topological bridges between distinct surfaces and blocking downstream flattening and reading.

I implemented a fusion-aware training intervention, `gap8`, that uses the synthetic painter's exact sheet-instance (`turn_id`) geometry to identify the air gap between neighboring sheets and give those voxels extra background supervision. The repository includes the complete modular PyTorch implementation, deterministic data and split builders, matched three-seed control/intervention training, one-time threshold freezing, the official TopoScore evaluator, synthetic fusion diagnostics, fixed real-panel rendering, provenance manifests, and fail-closed result validators.

The evaluation was preregistered and result-blind. Control and gap8 arms used the same architecture, starting checkpoint, samples, optimizer schedule, seeds (11, 23, 47), and evaluation protocol; only the gap supervision differed. Thresholds were selected once on Scroll 1 validation. Real held-out scoring used 38 Scroll 4/5 patches, and the synthetic test pooled 4,795,200 neighboring-sheet sites. Real and synthetic outputs were sealed and collected together before either result or any fixed panel was opened.

The method passed 5 of 7 preregistered gates. Because the preregistration required every primary gate to pass, gap8 did not meet the overall preregistered success criterion. Relative to matched controls on held-out real patches, pooled mean blend improved by 0.003359 (95% descriptive bootstrap interval 0.001333 to 0.005520), surface Dice by 0.003096 (0.002415 to 0.003854), VOI score by 0.002277 (0.001024 to 0.003707), and TopoScore by 0.004928 (-0.001584 to 0.011839). On the synthetic fusion stress test, conditional fusion fell from 60.681% to 50.014%, a 10.667 percentage-point reduction, and it decreased in all three matched seeds. One of the 12 preregistered real panel comparisons showed a clear removal of isolated bridges/blobs between neighboring layers without a new nearby break.

The adverse results are equally important and are published in full. Synthetic neighboring-sheet detection fell by 7.805 percentage points, and false splitting rose by 13.351 points, so those two safety gates failed. Seed 47 also had negative real blend, TopoScore, and surface-Dice deltas. All fine-tuned arms remained below the released baseline in absolute blend and TopoScore. This is therefore not presented as a drop-in production replacement or as evidence of recovered text. It is a reproducible mechanism result: explicit gap supervision can reduce cross-sheet fusion, but the current weight over-penalizes foreground and must be rebalanced before deployment.

This increases the probability of reading complete scrolls in two practical ways. First, it supplies an open, auditable training primitive for a documented compressed-layer failure mode, with measured improvement in the intended topology direction on both real and synthetic held-out data. Second, it supplies a reusable evaluation harness that detects the exact recall/over-segmentation tradeoff that ordinary voxel metrics can hide, preventing an apparently cleaner surface model from silently damaging downstream surface extraction. The complete evidence report publishes all seven gates, all four fixed panels, all 12 visual comparisons, sensitivity tables, artifact hashes, limitations, and the full operational audit.

## Terms and Conditions

Select **Yes, I agree** only after Aviad has reviewed this complete payload and the public evidence links resolve correctly.
