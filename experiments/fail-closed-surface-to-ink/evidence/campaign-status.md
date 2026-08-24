# First Letters campaign status — 2026-08-12

## Decision

The current automated breadth-first route across all 13 eligible scrolls is **exhausted fail-closed**. The search found no defensible text candidate approaching the First Letters requirement of at least **10 visible and legible letters within one 4 cm² area**.

This is a stop decision for the present public assets and methods—not a claim that the scrolls contain no writing. Reopen the physical search when there is materially new evidence: a new scan or public surface, an expert-corrected independent single-sheet ROI, or direct repeatable raw strokes that survive fiber/crack/fold/void controls.

No RunPod work is justified now. New paid compute for this breadth closure: **$0.00**.

## What changed in this run

The seven remaining eligible scrolls—PHerc0191, PHerc0211, PHerc0268, PHerc0358, PHerc0813, PHerc0826, and PHerc1545—were inventoried and searched using the current April-2026 m7 prediction, exact seed checks, raw seed views, and strict calibrated geometry gates. None had an existing public surface or TIFFXYZ asset to screen directly.

| Scroll | Strongest result | Decision |
|---|---|---|
| PHerc0191 | 0.00556 cm² calibrated patch | Far below internal 0.5 cm² raw-render gate. |
| PHerc0211 | Two ~0.00562 cm² strict bootstraps | Neither extended. |
| PHerc0268 | Best candidates failed the first complete ring | No established patch. |
| PHerc0358 | 35/36 exact candidates failed; one boundary pass | Specimen/mask-boundary artifact. |
| PHerc0813 | 0.509167 cm² calibrated patch and complete 93-layer raw stack | Geometry passed; independent raw review found 0/6 defensible and 0/6 borderline ROIs. |
| PHerc0826 | Early candidates were boundaries or failed independent geometry | No established patch. |
| PHerc1545 | Boundary/fold candidates; strict bootstraps failed | No established patch. |

The other six eligible scrolls had already been audited: PHerc0125, PHerc0257, PHerc0800, PHerc1203, PHerc1218, and PHerc1447. PHerc0125, PHerc0800, and PHerc1203 supplied ≥0.5 cm²-class raw surfaces but independent raw/model review was negative; the other three did not yield an eligible clean patch under the current route.

## PHerc0813: the one new raw surface

The official pinned VC3D GrowPatch run from seed `[6051, 3882, 4298]` produced a 2.1333 cm² proposal. Strict subtractive calibration retained one 0.509167 cm² component with 1,692 vertices and 1,444 quads. It passed:

- selected public-m7 run centering at every retained vertex;
- bounds, normals, both orientation signs, and ±46-voxel full-resolution sampling;
- zero folds, flips, degeneracies, discontinuities, wraps, or distortion;
- manifold topology and an exact stored self-intersection audit.

The final stack contains 93 lossless `1640×1640` uint8 layers at offsets `-46..+46`, with exact mask identity and zero outside-mask pixels. A model-independent raw audit generated six row hypotheses; all six were rejected on seven-depth controls as fibers, folds, void/hole margins, or mask-boundary structure. No stable baseline, consistent glyph scale, or legible letter body survived review.

Key evidence:

- [Stored calibrated validation](pherc0813-vc3d-growpatch/rank18/calibrated-mask-salvage-v2/validation.json) — `f09b9469a8b17a48c64ae8c21fefdd271a14f99deb4f9bb92fc54093d88b0e96`
- [Stored self-intersection audit](pherc0813-vc3d-growpatch/rank18/calibrated-mask-salvage-v2/stored-self-intersection-audit.json) — `5d53665231f4e7e2f3533c50bc71149e39daa0fb4a31da80cf9bccd50b26b3bb`
- [Full-resolution preflight](pherc0813-vc3d-growpatch/rank18/full-resolution-preflight-v1/full-resolution-preflight.json) — `787de7b2d9a310d87ca62e2c90e3574e8e15d7baa8380dd815a26e79773700aa`
- [Raw-stack manifest](pherc0813-r18-growpatch-raw-stack-93/raw-stack-manifest.json) — `ba6237958e31972eac34283dc8222279f0d05b654a49069f7a02b312c53eb863`
- [Independent audit](pherc0813-r18-growpatch-independent-ink-audit/audit-report.json) — `cb86d84f5b9bf311eefe104d242ab729c45622c87ddf75b578cecf1251c69e77`
- [Manual review](pherc0813-r18-growpatch-independent-ink-audit/manual-review.json) — `feefc6d4cc96ea7024a6e74174d3a6c8fb2d5d12207987aaef07d5fd75a51ff5`

## Saturation test

To avoid stopping merely because generation 40 was too small, the same official surface was continued to generations 60 and 80.

| Stage | Whole proposal | Strict calibrated component |
|---|---:|---:|
| Generation 40 | 2.1333 cm² | 0.509167 cm² |
| Generation 60 | 5.3284 cm² | 0.536269 cm² |
| Generation 80 | 10.1028 cm² | 0.536269 cm² |

The generation-60 and generation-80 accepted components are byte-exactly identical: all 1,520 quads shared and zero new quads. Relative to generation 40, 92.07% of the accepted generation-80 area is byte-identical overlap and only 0.042508 cm² is new. A second raw render would therefore be almost entirely duplicate evidence.

- [Generation 60→80 exact increment audit](pherc0813-vc3d-growpatch/rank18-resume80/exact-increment-vs-g60.json) — `e8c760e889857c68a62755941692f9fd6bca9de556c5c54666c80e7489132312`
- [Generation 80 calibrated validation](pherc0813-vc3d-growpatch/rank18-resume80/calibrated-mask-salvage-v1/validation.json) — `a775204abc1b82ffae7d1a136dbb02933fb4e0afd2cb32f3fa6b21f351761537`

An independent second PHerc0813 seed produced a 2.2293 cm² proposal, but its largest calibrated component was only 0.1936 cm², so it stopped before raw rendering.

## Closed ML branch

The ~9 µm supervised/target-domain-SSL model-family branch remains closed. In the decisive three-seed same-scroll falsification, the stronger model memorized train-A (AUROC `0.99595 ± 0.00344`) but failed nearby non-overlapping local-B (AUROC `0.51207 ± 0.05094`). Do not spend on decoder retuning, full fine-tuning on the same coarse labels, target-domain SSL intended to rescue the same representation, or generative super-resolution.

## Reopen policy

Only three developments justify renewed work:

1. New official physical data: a better scan, new surface, or materially improved geometry prediction.
2. A genuinely independent expert-corrected surface, not another default GrowPatch seed or duplicate slab.
3. Direct raw stroke evidence that persists across depth and remains after fiber, crack, fold, void, and boundary controls.

Machine-readable campaign record: [FIRST_LETTERS_BREADTH_STATUS_2026-08-12.json](FIRST_LETTERS_BREADTH_STATUS_2026-08-12.json) — `3c2591d2b3d4e7e3ac4111ee1ba7439e15dd38404925c30eebb98fa370260344`.
