# Fail-Closed Surface-to-Ink

## Technical summary

This contribution is a modular quality-control and stopping framework for Vesuvius Challenge surface reconstruction and ink searches. It was built because several common failure modes—sheet switches, smooth but off-surface growth, invalid normal sampling, duplicated geometry, papyrus texture false positives, orientation instability, and training-region memorization—can all produce outputs that look plausible before they are tested against an independent physical control.

The framework was exercised across all 13 scroll volumes currently eligible for the 2027 Grand Prize. Four scrolls reached clean surfaces at or above the campaign’s internal 0.5 cm² raw-render gate. All four were negative after independent raw-depth and/or model review. The search found no defensible candidate approaching the First Letters requirement of at least 10 visible and legible letters within one region up to 4 cm².

The main contribution is therefore not a text claim. It is a reusable pipeline that makes geometry and ink evidence auditable, rejects invalid branches early, and proves when apparent progress is physically duplicated or artifact-driven. That directly matches the official Progress Prize preference for analytic tools that detect actionable failure cases on real scroll data.

## The framework closes silent failure paths before they become claims

The pipeline is organized as a sequence of independent gates:

1. **Asset inventory.** Paginate public prefixes, group volumes/predictions/normal grids/TIFFXYZ surfaces, and identify mesh-only or unseen assets without recursively expanding large Zarr containers.
2. **Stored-grid geometry.** Check finite coordinates, in-volume vertices, valid neighbor edges, discontinuities, wrap risk, quad degeneracy, folds, distortion, abrupt normal flips, topology, and non-adjacent self-intersection.
3. **Public-m7 support.** Sample the current surface prediction along fresh physical normals, identify the intended centered connected run, and measure continuity and competitor clearance.
4. **Full-resolution closure.** Resample the selected component to its final raster, serialize to float32, reload it, and repeat geometry and ±normal-offset gates on the object that will actually be rendered.
5. **Raw normal-offset rendering.** Read compact Zarr regions per tile, render a physically calibrated 93-layer stack, enforce exact mask identity, and bind every layer to its offset and SHA-256.
6. **Model-independent raw audit.** Nominate candidate rows using local residual/depth persistence while treating fiber, fold, crack, void, hole, and mask-boundary evidence as negative controls. Inspect native raw data across depth before model maps.
7. **Optional model consensus.** Keep normal signs, frame orders, models, and depth bands explicit and symmetric. Treat agreement as evidence, not a substitute for raw support.
8. **Terminal decision.** Emit an explicit pass, reject, or reopen condition with thresholds, seeds, source identifiers, and hashes. A failed gate cannot be relaxed after seeing an attractive image.

All major inputs and outputs use community formats: OME-Zarr/Zarr, TIFFXYZ quadmeshes, numbered TIFF stacks, PNG review aids, and JSON manifests.

## Thirteen-scroll breadth test found four renderable surfaces and no defensible text

The campaign covered every volume on the official eligible list current on August 12, 2026.

| Outcome class | Scrolls | Count |
|---|---|---:|
| Clean ≥0.5 cm²-class raw surface, independent review negative | PHerc0125, PHerc0800, PHerc0813, PHerc1203 | 4 |
| Clean patch below the internal 0.5 cm² raw gate | PHerc0191, PHerc0211, PHerc0257, PHerc1218, PHerc1447 | 5 |
| No established clean patch; boundary/fold/first-ring failure | PHerc0268, PHerc0358, PHerc0826, PHerc1545 | 4 |

The 0.5 cm² threshold is a campaign cost-control rule, not a prize condition. It was chosen to avoid raw rendering and inference on tiny fragments that could not plausibly contain a useful line of text. The actual First Letters rule remains at least 10 visible and legible letters in one region up to 4 cm².

The breadth result closes only the tested automated route against the public assets available in this campaign. It does not establish that any scroll lacks text or that future data cannot change the outcome.

## PHerc0813 separated proposal growth from real accepted geometry

PHerc0813 was the strongest final physical test because it produced a clean, renderable surface from a scroll that had no public TIFFXYZ surface asset in the inventory.

The official VC3D GrowPatch seed `[6051, 3882, 4298]` produced a 2.1333 cm² whole proposal at generation 40. Strict subtractive calibration retained a 0.5091672599 cm² component with 1,692 vertices and 1,444 quads. The retained component had centered public-m7 support everywhere, valid ±46-voxel normal sampling, manifold topology, and zero folds, flips, discontinuities, wrap risks, degeneracies, or distortion. An exact stored non-adjacent triangle intersection audit also passed.

The full render was a lossless 93×1640×1640 uint8 stack at offsets -46 through +46, with exact explicit-mask identity and zero outside-mask pixels in every layer. A model-independent audit nominated six row regions. Seven-depth review rejected all six as fibers, folds, void/hole margins, or mask-boundary structures. None had a stable baseline, consistent glyph scale, or legible letter bodies.

The saturation continuation then made the key methodological point:

| Stage | Whole GrowPatch proposal | Accepted calibrated component |
|---|---:|---:|
| Generation 40 | 2.1333 cm² | 0.509167 cm² |
| Generation 60 | 5.3284 cm² | 0.536269 cm² |
| Generation 80 | 10.1028 cm² | 0.536269 cm² |

Generations 60 and 80 contained exactly the same 1,520 accepted quads and zero new accepted quads. Relative to generation 40, only 0.042508 cm² of the generation-80 accepted area was new; 92.07% was byte-exact overlap. Whole-proposal area increased nearly fivefold, but defensible physical evidence stopped changing.

This is actionable beyond PHerc0813: unwrapping systems should report accepted-component novelty, not proposal area alone. Otherwise repeated render/inference cycles can be mistaken for exploration when they revisit the same sheet fragment.

## Calibration changed two naïve rules

### Exact-one-run is not a valid universal sheet gate

The first implementation treated any additional m7 foreground run along a normal transect as evidence of a sheet switch. A published clean control falsified that assumption: 399 of 400 sampled points were multi-run while centered support was 98.75%. Dense rolled papyrus naturally places other predicted sheets within a wide transect.

The corrected rule identifies a selected centered run, measures its continuity and neighbor smoothness, and treats competitor clearance as a calibrated diagnostic. Exact-one-run may still be useful for strict bootstraps, but it is not a universal rejection condition.

### Reverse orientation is evidence, not a veto

With symmetric offsets, reversing the physical normal is exactly equivalent to reversing frame order. The framework records that mapping explicitly and treats both logical orientations as co-equal. Cross-orientation agreement is reported but not used as an automatic veto. Deduplicating the physical sequences reduced a nominal 24 logical-hypothesis matrix to 8 distinct frame sequences evaluated by two models, preserving evidence while reducing compute.

## Raw-depth controls exposed fibers and boundaries that models called ink

The independent audit deliberately avoids model probabilities when nominating raw candidates. It computes mask-normalized dark residuals, adjacent depth persistence, distant-offset specificity, structure-tensor orientation, and Frangi ridge evidence. The output is a ranked review set, not an automatic letter declaration.

Across PHerc0813, PHerc1203, and PHerc0800, the strongest apparent rows repeatedly followed papyrus strands, cross-hatching, cracks, folds, void margins, or mask holes. These structures change or migrate across depth and often align with tile boundaries. Native-resolution depth-control sheets made those explanations visible.

The model audits corroborated the need for independent controls. PHerc0800’s latest six-map screen had same-depth forward/reverse Pearson correlations from -0.066 to 0.043, near-zero top-region overlap, and grid-boundary gradients 15.6× to 25.4× controls. Its TimeSformer candidates were also orientation-dependent and carried 2.22× to 2.36× stride-boundary gradients. Those numerical signals were plausible in isolation but not stable physical ink evidence.

## A canonical ResNet-152 case failed the precommitted order-stability gate

A later PHerc1203 pilot tested the published `scrollprize/ink_canonical_2um` ResNet-152 checkpoint at pinned revision `075855bc69317ef6febf39a0d9d687b27d2b7c29`. The 9.362 µm native stack was interpolated only to align the checkpoint’s physical depth span and tile width; this cannot recreate ink information absent from the coarser scan. Forward and reverse middle-window inference each covered 96.401446% of valid pixels and passed input, checkpoint, and output integrity checks.

The maps nevertheless failed the precommitted order-stability screen decisively:

| Measure | Result |
|---|---:|
| Forward/reverse Pearson correlation | 0.032425 |
| Top-5% Jaccard | 0.049263 |
| Top-1% Jaccard | 0.003361 |
| Top-0.5% Jaccard | 0.000000 |
| Pixels ≥0.9 | 0 in either map |

Fixed review of the eight strongest components in each orientation placed the responses on fiber crossings, cracks or void margins, and mask-hole boundaries. It found zero defensible or borderline text candidates. The branch therefore stopped before shifted-origin inference. This is not evidence that PHerc1203 contains no writing; it is evidence that, under this physically mismatched checkpoint/application pair, the strongest responses were not stable to an exact reversal of depth order.

## The coarse-ML branch was falsified locally, not merely across scrolls

Cross-scroll failure can be explained by domain shift, so the decisive experiment used three spatial partitions on one labeled scroll:

- train region A;
- nearby non-overlapping local holdout B, separated by 1.228 mm;
- distant holdout C, sealed unless at least two of three seeds reached AUROC ≥0.95 on A and ≥0.80 on B.

Across three seeds, the stronger model nearly memorized A (mean AUROC 0.99595 ± 0.00344) but performed at chance on B (AUROC 0.51207 ± 0.05094; AP 0.27209 ± 0.01882 at prevalence 0.26490). Zero seeds passed the precommitted A+B gate, so C remained sealed.

This result does not prove that 9 µm CT contains no ink information. It shows that this supervised/coarse-representation model family learned non-transferable local correlates despite excellent training fit. Decoder retuning, full fine-tuning on the same small labels, target-domain SSL intended to rescue the same representation, and generative super-resolution were therefore closed as low-value branches.

## Reproducibility and provenance are part of the method

Every retained result is bound to its source identity and serialized content. The campaign records:

- public asset identifiers and timestamps;
- TIFFXYZ and Zarr coordinate/axis conventions;
- physical voxel spacing and frame offsets;
- seeds, thresholds, model revisions, and exact logical-to-physical mappings;
- per-layer and per-report SHA-256 values;
- candidate terminal statuses and skipped downstream actions;
- paid-compute observations separately from local/free work;
- explicit reopen conditions.

The canonical workspace passed 273 tests before packaging. Synthetic tests cover trilinear XYZ↔ZYX correctness, anisotropic normals, offset reversal, masks/holes, folds, geometry metrics, Zarr chunk boundaries, deterministic transects, exact increment identity, raw-depth controls, consensus symmetry, and state/provenance tampering.

The compact release does not include raw CT or multi-gigabyte stacks. It includes derived review images, machine-readable manifests, selected decision records, the tool suite, and synthetic tests. Exact hashes let reviewers connect the compact package to the immutable local evidence if a deeper reproduction is requested.

## Limitations and robustness boundaries

- The 13-scroll conclusion is conditional on the public assets and methods tested through August 12, 2026.
- Surface-prediction support is a geometry cue, not ground truth.
- A calibrated accepted component may still omit a real neighboring sheet or text region.
- Exact stored self-intersection checks do not automatically certify every possible interpolation method; full-resolution closure is reported separately.
- Model-independent raw nomination is conservative and can miss weak carbon ink.
- Manual review is not papyrological validation and cannot establish absence of writing.
- The internal 0.5 cm² render gate is an efficiency choice, not an official criterion.
- Some historical detector replays are integration/orientation checks rather than clean held-out generalization tests because model training membership is not independent.
- The PHerc1203 canonical-model pilot is an exploratory physical-resampling sensitivity check, not a held-out accuracy benchmark; scale alignment adds no missing high-frequency information.

These boundaries are intentionally visible. The framework is designed to say “not established” rather than convert missing evidence into a pass.

## Recommended use by the community

1. Run the compact asset inventory before rerunning existing surfaces.
2. Validate stored geometry and selected public-m7 run support before raw access.
3. Materialize, serialize, reload, and revalidate the exact full-resolution surface used for rendering.
4. Inspect raw evidence across depth before launching model sweeps.
5. Keep orientation and depth hypotheses symmetric and preserve contradictions.
6. Measure exact accepted-surface novelty before extending a grown patch.
7. Record a terminal reason and reopen condition whenever a branch stops.

For the First Letters hunt specifically, this campaign should remain dormant until new physical evidence appears: a materially new scan or prediction, an independently expert-corrected single sheet, or raw strokes that survive the artifact controls. For the broader Vesuvius effort, the toolkit can be applied immediately to new surfaces and unwrapping systems as a reproducible quality layer.

## Further questions

- Can accepted-component novelty become a native VC3D metric during growth?
- Can selected-run continuity and competitor clearance be exposed as interactive overlays rather than offline JSON?
- Which clean published surfaces are best suited for scroll- and geometry-specific calibration of m7 thresholds?
- Can the raw-depth artifact controls be used to generate hard negatives for future ink models without accidentally encoding mask boundaries?
- Which new acquisition or expert-corrected surface would most efficiently test whether weak ink is information-limited rather than model-limited?

## Official sources

- [Open prizes and Progress Prize criteria](https://scrollprize.org/prizes)
- [Vesuvius Challenge overview and open problems](https://scrollprize.org/)
- [ScrollPrize/villa](https://github.com/ScrollPrize/villa)

Official rules were verified on August 12, 2026. The next stated Progress Prize deadline was August 31, 2026 at 11:59pm Pacific.
