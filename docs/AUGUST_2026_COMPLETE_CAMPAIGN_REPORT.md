# August 2026 Vesuvius Campaign: Complete Technical and Operational Record

**Reporting period:** 1–22 August 2026

**Status cutoff:** 22 August 2026, 11:30 UTC

**Entrant:** Aviad Cohen (`aviad12g`)

**Repository:** [aviad12g/tifxyz-doctor](https://github.com/aviad12g/tifxyz-doctor), branch `agent/freeze-fusion-aware-checkpoints`

**Evidence state at cutoff:** public through commit `ad1f0e065a5c1dbd0cbd066dd2c6337073bc31ef`

**Scope note:** This report records completed work, methods, results, submissions, failures, corrections, costs, and unresolved blockers. It intentionally excludes prize-outcome forecasts.

## Executive summary

The August campaign produced four distinct public Progress Prize contributions and a fifth research program that remains deliberately unfinished:

1. **Gap8 fusion-aware surface training** tested whether exact synthetic inter-sheet geometry could reduce destructive merging of adjacent papyrus sheets. It reduced conditional fusion by **10.667 percentage points** on a large synthetic held-out stress test and slightly improved pooled real-scroll metrics, but it also reduced detection by **7.805 points** and increased false splitting by **13.351 points**. The intervention therefore passed **5 of 7** preregistered gates and failed the all-gates success rule. The complete mixed result—including adverse outcomes—was published and submitted.
2. **TIFXYZ UUID reliability and interoperability** converted a reproducible metadata warning into a backward-compatible upstream fix. A frozen census found that 277 of 450 registry roots used the literal UUID `output_tifxyz`. Upstream PR [#1299](https://github.com/ScrollPrize/villa/pull/1299) added explicit UUID support to both official converters and was merged.
3. **Fail-Closed Surface-to-Ink** audited 13 scrolls with a staged geometry, raw-evidence, model, and stopping framework. Four scrolls reached raw review; none produced defensible text. The contribution is the reusable fail-closed framework and negative-result evidence, not a text claim.
4. **`vesuvius.surface_preflight`**, upstream PR [#1463](https://github.com/ScrollPrize/villa/pull/1463), added a fail-closed TIFXYZ-to-volume compatibility check. It passed real PHerc1667 validation, matched seven of seven independently supplied expected verdicts, and had four green upstream GitHub Actions checks at the audited head. It was submitted separately on 22 August.
5. **GapBalance** was designed to rebalance Gap8 with lower gap weights, trained all six planned Gap2/Gap4 checkpoints, and prepared result-blind development caches. It did **not** score development results or open confirmation data. The independent PHerc1218 holdout was first corrected after our membership audit and was later withdrawn by its author after additional crop-CT, label-stitching, and unit-conversion faults were found. The experiment is now frozen as `BLOCKED_REAL_HOLDOUT_NOT_CERTIFIED`.

Across the campaign, the central operating principle was **fail closed**: freeze the question and decision rule before seeing results; separate scientific endpoints from operational logs; retain failed attempts rather than rewrite history; bind artifacts by hash; publish adverse findings; and stop when the evidence source became uncertified.

## What was actually submitted

The Google Forms receipt audit found **five receipt messages containing four distinct response bodies**. One response was an identical duplicate of the Gap8 entry; it was not a fifth contribution.

| Contribution | Receipt time (UTC) | State at cutoff |
|---|---:|---|
| Gap8 fusion-aware surface training | 2026-08-15 09:33:13 | Confirmed by Google Forms receipt |
| Gap8 duplicate, byte-identical response | 2026-08-15 10:33:50 | Confirmed duplicate; not a distinct contribution |
| TIFXYZ UUID reliability and interoperability | 2026-08-15 10:53:49 | Confirmed by Google Forms receipt |
| Fail-Closed Surface-to-Ink | 2026-08-15 13:37:08 | Confirmed by Google Forms receipt |
| `vesuvius.surface_preflight` PR #1463 | 2026-08-22 11:29:38 | Confirmed by Google Forms receipt |

No award decision is claimed in this report. Receipt confirmation proves form delivery, not judging or payment.

## Campaign chronology

### 1–8 August: define the topology problem and establish public scope

- The campaign preregistered a contrastive fusion follow-up rather than starting from an unbounded architecture search.
- The Gap8 intervention, matching rules, controls, and intended topology measurements were made public.
- Existing July work was separated from new August claims. This was especially important for TIFXYZ Doctor v0.2 and earlier PRs: they remained useful provenance, but they were not relabelled as new August work.
- The UUID reliability issue was converted into a bounded upstream contribution rather than a broad claim that TIFXYZ coordinates were corrupt.

### 9–12 August: freeze checkpoints, data, thresholds, and held-out execution

- Checkpoint identities, evaluation inputs, validation thresholds, held-out split, fixed-panel rules, and scoring code were frozen by commit and SHA-256.
- The Gap8 protocol used matched seeds 11, 23, and 47 and kept the architecture, source checkpoint, data order, optimizer, schedule, and inference path identical between control and intervention.
- The campaign added fail-closed provider-neutral cache and delivery tooling so that operational provider changes would not silently alter scientific inputs.
- The synthetic primary was moved to a sealed RunPod execution after operational preflights; the real scorer eventually used sealed CPU-only execution because completed cache scoring did not scientifically require a GPU.
- In parallel, the Surface-to-Ink audit expanded to all 13 eligible scrolls and froze its eight-gate stopping framework.

### 12–15 August: complete result-blind execution, open results once, and submit three entries

- Numerous provider/runtime failures occurred before final scoring. Every failure was preserved in the public operational audit, and each correction was limited to transport, dependency, runtime, manifest projection, or collection behavior.
- The final accepted Gap8 evidence used the authoritative RunPod synthetic primary and the accepted sealed real-score source. Scientific results and fixed panels were opened only after a dedicated mixed-provenance validator was frozen.
- Gap8 was reported as a mixed mechanism result, not as a production model.
- The UUID remediation evidence package and the Fail-Closed Surface-to-Ink package were finalized with explicit prior-work boundaries and limitations.
- Three distinct entries—Gap8, UUID reliability, and Fail-Closed Surface-to-Ink—were receipt-confirmed on 15 August.

### 15–21 August: design and execute the bounded GapBalance follow-up

- The original Gap8 verdict remained immutable. GapBalance asked a new question: whether weights 2 or 4 could preserve the fusion benefit while avoiding Gap8’s detection and splitting damage.
- The experiment was constrained to Control, Gap2, and Gap4; seeds 11, 23, and 47; reuse of exact control hashes; and six new training jobs only.
- PHerc1218 version 1 failed a metadata membership reconciliation. The audit identified six leftover pilot crops, and the dataset author publicly confirmed the diagnosis and published version 2.
- Version 2 was initially frozen by exact manifest identity and metadata-only eligibility rules. All six Gap2/Gap4 training jobs completed and were mechanically verified without opening validation or confirmation endpoints.
- A result-blind development pipeline was built for the 24 Scroll-1 validation patches and synthetic seeds 400–403. Real development caches and two synthetic shards were mechanically verified.
- Free Kaggle throughput became a schedule constraint. A bounded RunPod fallback was publicly frozen, approved with exact cost ceilings, and used only for the development-cache stage. Confirmation data was excluded from the private bundle.
- On 21 August, before any development scoring, the PHerc1218 author withdrew every published version after finding further crop-CT faults, label-stitching faults, and a unit error in the cross-validation-set gap comparison. Compute and scoring stopped. The experiment was quarantined rather than retuned around the loss of the holdout.

### 21–22 August: package `surface_preflight`, audit receipts, and freeze current state

- PR #1463’s exact public head, test state, real-scroll evidence, independent validation, and limitations were assembled into a separate Progress Prize payload.
- The campaign audited all Google Forms receipts and distinguished the duplicate Gap8 response from the four genuinely distinct entries.
- The complete local suite passed **472 tests** after the submission receipt record was added.
- GapBalance remained blocked. No paid pod was restarted, no free Kaggle job was added, no development endpoint was scored, and no PHerc1218 or synthetic confirmation payload was opened.

## Operating methodology

### Public freeze before scientific access

Each scientific stage followed the same pattern:

1. define the question and arms;
2. freeze inputs, hashes, seeds, thresholds, gates, and panel rules;
3. publish the plan and focused tests;
4. run result-blind compute;
5. inspect only operational logs, manifests, counts, and hashes while incomplete;
6. open scientific endpoints only after the frozen completion rule is satisfied; and
7. publish every pass, failure, null, and adverse outcome.

This prevented several common failure modes: choosing thresholds on the test set, discarding bad seeds, replacing unattractive panels, mixing partially completed providers, or changing success criteria after seeing results.

### Separation of operational and scientific evidence

Provider failures were treated as operational events unless a scientific endpoint had actually been created and opened. Dependency failures, transfer failures, asset-ledger mismatches, runtime installation errors, stale return-code files, and strict-bundle verification failures did not become scientific results. Conversely, they were not deleted from history: the repository records what failed, when it failed, whether billing was stopped, and whether any endpoint was inspected.

### Hash-bound provenance

The campaign used exact commit IDs, file SHA-256 values, payload SHA-256 values, manifest identities, and provider receipts to bind:

- source checkpoints and control reuse;
- data splits and registry snapshots;
- cache indexes and manifests;
- scoring and rendering code;
- private provider bundles;
- candidate-selection and quarantine records; and
- submission evidence.

The purpose was not decorative reproducibility. The hash checks repeatedly caught real problems: stale files, extra generated bytecode, wrong import closure, incorrect asset bindings, and incompatible provider projections.

### Conservative claim boundaries

The campaign distinguished among:

- a completed experiment and a successful intervention;
- a form receipt and an award decision;
- real-scroll proxy agreement and ground-truth accuracy;
- structural/spatial compatibility and proof of correct geometry;
- model response and legible text; and
- completed training artifacts and a scientifically scored result.

Those distinctions are essential to interpreting the work below.

## Contribution 1: Gap8 fusion-aware surface training

### Problem addressed

In compressed or highly curved papyrus, adjacent sheets may appear close enough that a segmentation model predicts a bridge between them. Such fusion can turn two surfaces into one topological object, damaging extraction and flattening even when ordinary voxel overlap looks acceptable.

Gap8 tested whether synthetic geometry could supervise that failure mode directly. The synthetic painter already knew the exact sheet instance through `turn_id`. The intervention identified the inter-sheet air-gap voxels and increased their background weight from 1 to 8. It did not change the architecture, source checkpoint, training schedule, or inference code.

### Experimental design

| Design element | Frozen choice |
|---|---|
| Arms | matched control and Gap8 |
| Seeds | 11, 23, 47 |
| Sole intended numerical difference | gap-background weight 1 vs 8 on exact synthetic inter-sheet gap voxels |
| Threshold selection | once, using Scroll-1 validation only |
| Real held-out set | 38 Scroll-4/5 patches |
| Synthetic held-out stress test | 4,795,200 neighbouring-sheet sites |
| Visual review | four fixed real panels × three seeds = 12 comparisons |
| Overall success rule | all seven preregistered gates must pass |

The real and synthetic outputs were sealed and collected together before either result or any fixed panel was opened. Sensitivity thresholds were labelled as secondary analyses and could not replace the primary threshold rule.

### Execution and operational corrections

The final result required a substantial amount of provider engineering:

- The first sealed RunPod synthetic executor failed before inference because SciPy was missing.
- A second attempt failed because strict bundle verification detected 13 generated Python bytecode files left by the stopped attempt.
- Early Kaggle scorer versions failed on asset-ledger validation, immutable-plan projection, cache-schema assumptions, or unavailable CUDA-linked dependencies.
- A real scorer completed its long subprocess but stopped during separate panel rendering; its scientific outputs remained sealed and unused.
- The retry moved completed-cache scoring to an exact CPU runtime and parallelized only independent official-metric subprocesses, preserving output order and scientific functions.
- Multiple CPU-pod transfer and runtime problems—missing `rsync`, capacity loss, archive layout, credential transport, aliasing, and import closure—were corrected publicly and result-blind.

The authoritative synthetic primary ultimately completed all seven jobs and was mechanically verified as seven indexes, 70 manifests, and 700 sealed cache identities. The final real and synthetic sources were accepted through a frozen mixed-provenance validator before scientific access.

### Results

#### Held-out real-scroll deltas

| Metric | Pooled matched Gap8 − control delta | Descriptive 95% bootstrap interval |
|---|---:|---:|
| Official blend | +0.003359 | +0.001333 to +0.005520 |
| TopoScore | +0.004928 | −0.001584 to +0.011839 |
| Surface Dice | +0.003096 | +0.002415 to +0.003854 |
| VOI score | +0.002277 | +0.001024 to +0.003707 |

These pooled real-scroll changes were small and generally favorable, but they were not uniform across every seed. Seed 47 had negative real blend, TopoScore, and surface-Dice deltas. All fine-tuned arms also remained below the released baseline in absolute blend and TopoScore.

#### Synthetic topology deltas

| Endpoint | Control | Gap8 | Delta | Gate result |
|---|---:|---:|---:|---|
| Conditional fusion among detected neighbour-sheet sites | 60.681% | 50.014% | **−10.667 pp** | Pass |
| Neighbour-sheet detection | — | — | **−7.805 pp** | Fail |
| False splitting on single-sheet controls | — | — | **+13.351 pp** | Fail |

Conditional fusion decreased in all three matched seeds. However, only seed 23 satisfied the stricter 10-point secondary criterion. The safety regressions were large enough that the intervention could not be considered deployable.

#### Fixed panels

All 12 preregistered candidate/control comparisons were published. One comparison—panel 1, seed 23—showed a clear removal of isolated bridges or blobs between neighboring layers without a new nearby break. The other panels remained part of the evidence; none was removed or replaced because of appearance.

### Verdict and value

Gap8 passed 5 of 7 gates and therefore **failed the preregistered all-gates success criterion**. Its value is a reproducible mechanism finding:

- exact gap supervision can materially reduce cross-sheet fusion;
- weight 8 over-penalizes foreground enough to damage detection and induce splitting; and
- voxel-level improvements alone would have hidden the topology tradeoff.

The repository provides a reusable training primitive, threshold freezer, official-metric evaluator, synthetic topology diagnostic, fixed-panel renderer, sealed collection path, provenance manifests, and fail-closed validator. It does **not** claim recovered text or a drop-in production model.

## Contribution 2: TIFXYZ UUID reliability and interoperability

### Problem addressed

A frozen registry census covered 450 original and normalized TIFXYZ roots: 186 original roots and 264 normalized roots. It found:

- 277 of 450 roots used the literal UUID `output_tifxyz`;
- all 264 normalized roots and 13 original roots used that value; and
- only 174 distinct UUID strings appeared across the 450 roots.

This does not mean TIFXYZ coordinates were corrupted, nor does it establish a formal uniqueness requirement for every consumer. The bounded risk is that software using UUIDs as keys for caches, joins, deduplication, provenance, or result stores can conflate unrelated surfaces.

### Upstream remediation

August PR [ScrollPrize/villa #1299](https://github.com/ScrollPrize/villa/pull/1299), “Allow explicit TIFXYZ UUIDs in obj2tifxyz,” was opened on 2 August and merged on 6 August as commit `8b7c9df4191f01e52e28a8a9fd68add344349841`.

The five-file upstream change:

- added optional `--uuid=<id>` support to both official OBJ-to-TIFXYZ converters;
- centralized identity resolution in shared `TifxyzIdentity.hpp`;
- preserved the historical basename-derived default when `--uuid` is omitted;
- retained input-stem fallback behavior; and
- added four resolver regression cases covering explicit override, backward compatibility, fallback, and empty-input rejection.

The remediation is deliberately conservative: existing commands and existing artifacts remain valid, while producers that need stable staging paths can supply a collection-specific identity.

### Evidence and claim boundary

The evidence package binds the merged diff, exact Git blob identities, machine-readable census, dependencies, license, and material limitations. TIFXYZ Doctor v0.2 and PRs #1264 and #1278 were disclosed as July provenance and not reclaimed as August work.

This contribution improves reliability and interoperability. It does not claim better segmentation, unwrapping, surface geometry, or ink detection.

## Contribution 3: Fail-Closed Surface-to-Ink

### Goal

The Surface-to-Ink campaign asked whether currently public surfaces and volumes justified expensive inference or a claim of visible letters. Instead of optimizing until something looked text-like, it built a stopping framework that requires each route to pass physical and evidentiary gates.

### Eight-gate framework

1. inventory and identity check;
2. stored-geometry validation;
3. public `m7` support review;
4. full-resolution geometry closure;
5. raw normal-offset rendering;
6. model-independent raw review;
7. optional model-consensus checks; and
8. a terminal pass, reject, or explicitly conditioned reopen decision.

The package supports standard OME-Zarr/Zarr, TIFXYZ, TIFF, PNG, and JSON artifacts and records why a branch stopped.

### Thirteen-scroll audit

The audit was current through 12 August and covered 13 eligible scrolls:

| Outcome class | Scrolls | Result |
|---|---|---|
| Clean raw surfaces at or above the internal 0.5 cm²-class gate | PHerc0125, PHerc0800, PHerc0813, PHerc1203 | All negative after independent review |
| Clean but smaller patches | PHerc0191, PHerc0211, PHerc0257, PHerc1218, PHerc1447 | Stopped below the internal render gate |
| No established clean patch | PHerc0268, PHerc0358, PHerc0826, PHerc1545 | Stopped before raw review |

No defensible candidate approached ten visible or legible letters within 4 cm². This is a conditional negative result under the audited public assets and methods, not proof that the scrolls contain no writing.

### PHerc0813 geometry saturation experiment

PHerc0813 was used to test whether expanding a geometry proposal actually added defensible surface area.

- Generation 40 whole proposal: 2.1333 cm².
- Generation 40 accepted region: 0.5091672599 cm², 1,692 vertices, 1,444 quads.
- Raw stack: 93 layers, 1640 × 1640 pixels, `uint8`, offsets −46 through +46.
- Six model-independent proposed rows: all rejected after depth and artifact review.
- Generation 60 whole proposal: 5.3284 cm²; accepted: 0.536269 cm².
- Generation 80 whole proposal: 10.1028 cm²; accepted: 0.536269 cm².

Generations 60 and 80 contained exactly the same 1,520 accepted quads and zero new accepted quads. Relative to generation 40, generation 80 added only 0.042508 cm² of accepted area and overlapped 92.07% byte-for-byte. The experiment showed that proposal area could grow nearly fivefold while defensible physical evidence plateaued.

### PHerc1203 orientation and model checks

The canonical ResNet-152 revision was pinned to `075855…`. Forward/reverse evidence was treated as calibration rather than using one orientation as an automatic veto.

- forward/reverse correlation: 0.032425;
- top-5% Jaccard: 0.049263;
- top-1% Jaccard: 0.003361;
- top-0.5% Jaccard: 0;
- pixels at probability ≥0.9: 0; and
- fixed review of the strongest components: zero defensible or borderline text candidates.

An exact-one-run assumption was therefore replaced with a physically deduplicated orientation/model plan: 24 logical hypotheses reduced to eight distinct physical sequences evaluated with two models.

### Same-scroll transfer test

A coarse ML test separated apparent training performance from nearby spatial transfer:

- training region A: mean AUROC 0.99595 ± 0.00344;
- nearby region B, 1.228 mm away: AUROC 0.51207 ± 0.05094;
- region B average precision: 0.27209 ± 0.01882 at prevalence 0.26490; and
- zero seeds passed the frozen transfer rule.

The distant region C remained sealed because the nearby transfer gate had already failed.

### Verification and verdict

The complete package had 273 tests before packaging. The compact release validation reported **129 passed** in 4.97 seconds on macOS arm64 with Python 3.12.12. No additional RunPod breadth run was justified, so that closure stage spent $0 on new provider compute.

The contribution is the fail-closed diagnostic and stopping layer. It does not claim readable letters, successful ink recovery, or a proof that negative scrolls contain no text.

## Contribution 4: `vesuvius.surface_preflight` upstream PR #1463

### Problem addressed

Surface-rendering and inference pipelines can produce plausible-looking output even when a surface is paired with the wrong CT volume, wrong spatial region, malformed mesh, or nearly empty scan support. PR [#1463](https://github.com/ScrollPrize/villa/pull/1463), “Add fail-closed TIFXYZ volume preflight,” introduces an early compatibility gate so those pairings fail before expensive downstream work.

### What the tool checks

The `vesuvius.surface_preflight` command checks:

- required TIFXYZ files and metadata;
- coordinate and mask raster consistency;
- finite coordinates;
- valid mesh vertices and quads;
- exact volume-array bounds; and
- deterministic sampled CT support.

It writes machine-readable JSON atomically and returns a nonzero exit status when a required gate fails.

### Real and independent validation

The exact audited head was `0ea5aec2c8a2de7c3a080d7d0d3a7524ac8fe928`.

On the real PHerc1667 validation pair, it passed 9 of 9 required gates with:

- 728,218 valid vertices;
- 725,986 valid quads;
- zero out-of-bounds coordinates; and
- signal in all 16 deterministic CT samples.

Independent testing by Jinhojeong supplied seven pairings spanning a full remote OME-Zarr, wrong-region cases, sparse predictions, and a nonexistent path. The tool matched all **7 of 7** expected pass/fail verdicts. Its default 1,024-sample support estimate was within 1.5 percentage points of the independently computed exact per-quad reference in the reported tests.

### Upstream state at the cutoff

- PR state: open and non-draft.
- Mergeability observed: mergeable.
- Merged: no.
- Requested reviewers: `jrudolph` and `bruniss`.
- Green upstream checks: Large PR review gate, CodeQL, Continuous Integration, and Test vesuvius Python.
- Vercel: authorization-required deployment failure, not a project-test failure.

### Bounded limitation

The preflight establishes structural and spatial compatibility under its checks. It does not prove that the surface geometry is scientifically correct, that segmentation is accurate, that ink or text is present, or that PHerc1218 is a certified validation set. This limitation became especially important when PHerc1218 was later withdrawn for faults outside the preflight’s claim scope.

## GapBalance: completed training, preserved development work, and an active holdout quarantine

### Scientific question

GapBalance was an additive follow-up, not a reinterpretation of Gap8:

> Can lower exact inter-sheet gap supervision preserve at least a five-point conditional-fusion benefit while avoiding the detection and false-splitting regressions observed with immutable Gap8?

The arms were:

- Control: gap-background weight 1;
- Gap2: gap-background weight 2; and
- Gap4: gap-background weight 4.

Seeds 11, 23, and 47 were matched. The three original controls were reused only by exact checkpoint SHA-256. All other bytes and scientific choices were frozen: source checkpoint, architecture, encoder state, decoder initialization, crop sequence, real/synthetic schedule, augmentations, optimizer, learning-rate schedule, steps, normalization, runtime, and inference code.

Only six new training jobs were permitted: Gap2 and Gap4 for the three seeds. The frozen training contract used 160-cube crops, batch 1, accumulation 2, deterministic 3:1 real-to-synthetic sampling, 1,500 AdamW steps at 5e-5 with 1e-5 weight decay, and cosine decay to 1e-6.

### Frozen selection rule

Candidate selection could use only:

- the original 24-patch Scroll-1 validation set; and
- synthetic development seeds 400–403.

Thresholds came from `{0.30, 0.35, …, 0.70}` by maximum mean official blend, with fixed tie rules. An arm was eligible only if it met all five development constraints:

1. conditional-fusion reduction at least 5.0 points;
2. detection decline no worse than 2.0 points;
3. false-split increase no worse than 2.0 points;
4. mean real-development blend delta at least −0.005; and
5. mean real-development TopoScore delta at least −0.005.

Among eligible arms, the larger fusion reduction would win; if the difference were under one point, Gap2 would win. If neither arm were eligible, the protocol required stopping without opening confirmation.

### Training and result-blind cache completion

All six free Kaggle training jobs completed and were mechanically verified. Each required training manifest, checkpoint, metadata, and log was present; output hashes, arm/seed identities, and frozen input identities matched. No validation or confirmation endpoint was computed during training verification.

The real development cache was recovered and mechanically verified as exactly:

- 216 NPZ files;
- nine cache manifests; and
- one job index.

Synthetic seed-11 shards q0 and q1 were each verified as:

- 60 NPZ files;
- three manifests; and
- one job index.

Every downloaded file was hashed. These checks established completeness and identity only; NPZ scientific contents were not scored or opened.

The authoritative private development-package manifest payload was
`8ead7fbb99eb74bb3bf81538b9b81fdaff24924a6739f87c4f3a402d7720fa0a`.
The mechanically verified real-development split identity was
`20c600d6061bf8715ada20423a05c2f03a9bc14827b2c79bac7b7e1b3cc0499d`.

### PHerc1218 audit sequence

The real confirmation holdout was intended to be the independently published PHerc1218 tight-contact set.

1. **Version 1 failed membership reconciliation.** Its published totals did not match the package. Our metadata audit identified six leftover pilot crops.
2. **The author confirmed the diagnosis.** The extraction directory had not been required to be empty; later passes globbed the pilot leftovers. The author published version 2 with 254 contact crops plus 60 controls and a recomputed manifest.
3. **Version 2 was frozen only by exact identity.** The plan recorded provider version, manifest SHA-256, metadata-only eligible rows, and fixed-panel paths. It explicitly prohibited opening NPZs, CT, labels, predictions, probabilities, panels, or endpoints before one candidate and all thresholds were publicly frozen.
4. **The author later withdrew all versions.** On 21 August, Jinhojeong reported faults in crop CT, faults in label stitching, and a unit error in the comparison of this set’s gap census with another validation set. The authoritative notice is [ScrollPrize/villa issue #191 comment 5366060707](https://github.com/ScrollPrize/villa/issues/191#issuecomment-5366060707).
5. **The campaign quarantined rather than substituted.** Versions 1 and 2 became forbidden. Historical version-2 metadata remained audit evidence only and no longer certified scientific validity.

The active quarantine payload SHA-256 is `663cf15a2db85c26a6a895dafbacf259de0b2fe3a4c5c76f3d734cdf6ae60c01`.

### RunPod acceleration and cost control

Free Kaggle throughput was slow, so a paid development-only fallback was authorized with these hard boundaries:

- development cutoff: $12;
- total campaign cap: $25;
- confirmation reserve: $13;
- aggregate hourly ceiling: $2.38/hour;
- at most seven GPUs; and
- no confirmation or PHerc1218 data in the uploaded bundle.

The exact private bundle v4 was approximately 4.51 GB with SHA-256 `64cf1fdf4f2f0d160367caad7b134421956d3d6225818c32db01732bb6a7eec4`. It contained development source, launchers, assets, and private model weights, but excluded PHerc1218 and synthetic confirmation seeds 500–504.

Multiple community-cloud allocations and transport paths failed result-blind because of host capacity changes, SSH readiness, Jupyter terminal behavior, relay readiness, transfer resumability, generated `__pycache__`, and a stale verifier return-code marker. Each retry was publicly frozen before action. The final exact pods were:

- `jsoco4r0ijfswk`: four RTX 3090 GPUs; and
- `bij7jheyqbpggf`: three RTX 3090 GPUs.

The extracted bundle was hash-verified on both pods. The scientific executor did not reach a valid complete 12-job development cache: one pod reported the four seed-11 jobs in error and the other remained at runtime installation. Both pods were stopped to halt billing.

The latest public retry ledger recorded $6.078399 of conservative development spend before the final executor attempt. Provider-side stop monitoring conservatively placed total development spend at approximately **$8.55**, still below the $12 cutoff. This report does not treat the difference as a scientific datum: it reflects the timing boundary between the last committed retry ledger and the later provider stop.

### Why the experiment stopped without scoring

The holdout withdrawal occurred before the complete development-cache rule was satisfied. The public quarantine therefore forbids:

- restarting or creating RunPod pods;
- scheduling additional free Kaggle jobs;
- scoring partial development endpoints;
- selecting Gap2 or Gap4;
- opening PHerc1218 versions 1 or 2;
- opening synthetic confirmation seeds 500–504; and
- silently replacing the confirmation source or relaxing thresholds.

No GapBalance scientific result exists yet. The completed checkpoints and verified result-blind caches remain valuable and reproducible, but they are not evidence that Gap2 or Gap4 works.

### Conditions required to reopen

GapBalance can resume only if all of the following occur publicly:

1. the author publishes a complete verified correction and corrected provider release;
2. a new metadata-only identity and inventory audit passes;
3. an additive record freezes the corrected provider identity, manifest, eligibility list, and fixed panels;
4. the original candidate, threshold, gate, metric, and selection rules remain unchanged; and
5. a hash-bound supersession explicitly deactivates the current quarantine.

Until then, the exact status is **`BLOCKED_REAL_HOLDOUT_NOT_CERTIFIED`**.

## Public collaboration and independent evidence

The August work was not limited to private experiments:

- PR #1299 was merged upstream, turning the UUID audit into a backward-compatible producer fix.
- PR #1463 was opened upstream with documentation, implementation, tests, and requested maintainer reviews.
- Jinhojeong independently tested `surface_preflight` on seven pairings and publicly reported seven expected verdicts matched.
- The PHerc1218 exchange demonstrated bidirectional audit value: our package-membership audit found stale pilot files; the author confirmed and corrected that defect; the author’s later deeper audit found additional faults and withdrew the set before it could be used scientifically.
- Issue #191 served as the public context for the compressed-sheet fusion problem and the later holdout discussion.

External validation is reported only within its actual scope. Jinhojeong’s seven-pair test supports preflight pairing behavior; it does not validate Gap8, certify PHerc1218, or establish recovered text.

## Testing, CI, and reproducibility

### Test evidence

- Gap8 and its frozen evaluators had focused fail-closed tests throughout the execution and scoring lifecycle.
- Fail-Closed Surface-to-Ink had 273 tests before packaging and a compact release validation of 129 passing tests.
- PR #1463’s exact head had four successful upstream GitHub Actions workflows.
- The campaign repository’s final local suite at commit `ad1f0e0` reported **472 passed in 10.28 seconds**.

### Reproducibility assets

The repository includes:

- public protocols and machine-readable plans;
- checkpoint and source hashes;
- data split and manifest identities;
- frozen threshold and panel-selection rules;
- cache producers, collectors, and validators;
- focused and full test suites;
- operational attempt audits;
- release evidence packages;
- exact submission payload reviews; and
- Google Forms receipt audit records that omit entrant email and Gmail message IDs.

Private model weights and private provider bundles were not published. Their hashes, permitted destinations, exclusion rules, and scientific role were recorded without exposing credentials or confirmation data.

## Failures and corrections that materially improved the work

### Scientific failures

- Gap8 failed two safety gates despite reducing fusion. Publishing those failures prevented a one-metric success narrative.
- PHerc0813 geometry growth saturated; more proposal area did not produce more defensible surface.
- PHerc1203 model responses were unstable under physical reversal and did not yield defensible text candidates.
- Same-scroll transfer from region A to nearby region B collapsed to chance-like AUROC; region C correctly remained sealed.
- GapBalance never reached scientific scoring because its real holdout lost certification.

### Data and provenance failures

- The UUID census exposed repeated placeholder-like identifiers and led to a merged upstream producer option.
- PHerc1218 version 1 contained six stale pilot crops; the mismatch was caught before use.
- PHerc1218 version 2 was later withdrawn for deeper CT, label-stitching, and unit-comparison faults; the experiment was quarantined before confirmation access.

### Operational failures

- Missing dependencies, generated bytecode, immutable-plan projection errors, unavailable GPU quota, transfer tools, runtime paths, provider host capacity, SSH readiness, relay behavior, archive layout, stale return-code files, and strict import closures all caused fail-closed stops.
- Corrections were public and result-blind. Scientific arms, thresholds, metrics, gates, and panels did not change.
- Billing was stopped when a provider path failed or could not complete within its cap.

These failures consumed time and some compute, but they also hardened the evidence chain. The campaign ended with fewer unqualified claims and better reusable validation tooling than it would have had if operational success had been treated as more important than evidentiary integrity.

## Current state at the report cutoff

| Workstream | Current state | What remains |
|---|---|---|
| Gap8 | Complete, published, submitted; 5/7 gates | Preserve immutable result; do not retune it |
| UUID reliability / PR #1299 | Merged, published, submitted | Downstream adoption and any future registry cleanup are separate work |
| Fail-Closed Surface-to-Ink | Complete, published, submitted; negative text verdict | Reopen only for materially new scans, predictions, or expert-corrected geometry |
| `surface_preflight` / PR #1463 | Public, tested, submitted; open and unmerged at cutoff | Maintainer review and possible upstream merge |
| GapBalance | Six trainings complete; partial result-blind development caches preserved; no scoring | Wait for a complete certified corrected holdout and perform a new metadata-only audit |
| Submission audit | Four distinct entries receipt-confirmed; one duplicate Gap8 response | Retain receipts; do not resubmit existing entries |

## What should happen next

1. **Preserve the four submitted evidence packages unchanged.** Corrections should be additive and should not rewrite result history.
2. **Support PR #1463 review.** Respond to concrete maintainer feedback, keep the bounded compatibility claim, and rerun exact tests for any code change.
3. **Monitor issue #191 metadata only.** Do not download or inspect a future PHerc1218 payload until the author publishes a complete correction with a frozen identity.
4. **Keep GapBalance sealed.** No partial scoring, candidate selection, or substitute confirmation source should occur while the quarantine is active.
5. **Retain compute and receipt records.** They document both cost control and why completed training is not being represented as a completed experiment.
6. **Use the campaign’s reusable tooling in future work.** The most transferable outputs are the topology stress tests, fail-closed surface/volume preflight, provenance remediation, sealed cache lifecycle, and explicit stop rules.

## Evidence map

| Topic | Primary repository evidence |
|---|---|
| Gap8 final result | `docs/fusion-aware-final-evidence/FINAL_EVIDENCE.md` |
| Gap8 operational history | `docs/fusion-aware-final-evidence/OPERATIONAL_ATTEMPT_AUDIT.md` |
| Gap8 implementation and reproduction | `experiments/fusion-aware-surface/README.md` |
| UUID reliability evidence | `docs/reliability-toolkit-final-evidence/FINAL_EVIDENCE.md` |
| August UUID submission review | `docs/august-2026-reliability-toolkit-progress-prize-review.md` |
| Surface-to-Ink report | `experiments/fail-closed-surface-to-ink/TECHNICAL_REPORT.md` |
| Surface-to-Ink form payload | `experiments/fail-closed-surface-to-ink/FORM_RESPONSE.md` |
| PR #1463 evidence | `docs/august-2026-surface-preflight-progress-prize-review.md` and `docs/august-2026-surface-preflight-evidence.json` |
| Receipt audit | `docs/august-2026-progress-prize-submission-audit.md` and `.json` |
| GapBalance protocol | `experiments/gap-balance-followup/GAPBALANCE_PROTOCOL.md` |
| GapBalance provider plan | `experiments/gap-balance-followup/GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json` |
| GapBalance quarantine | `experiments/gap-balance-followup/GAPBALANCE_HOLDOUT_QUARANTINE_AMENDMENT.md` and `PHERC1218_HOLDOUT_QUARANTINE.json` |
| Exact current GapBalance state | `experiments/gap-balance-followup/GAPBALANCE_STATUS` |

## Final assessment

August produced one mixed but scientifically informative segmentation experiment, two upstream reliability contributions, one comprehensive negative-result diagnostics package, and a carefully preserved follow-up that stopped when its confirmation source became invalid. The strongest common result was not a single model score. It was an evidence-governed workflow that repeatedly prevented plausible-looking but invalid conclusions:

- topology benefits were not allowed to hide recall and splitting damage;
- proposal growth was not allowed to masquerade as new physical evidence;
- model heatmaps were not allowed to become text claims without raw and geometric support;
- successful training was not allowed to become a result before complete frozen scoring;
- a corrected dataset version was not allowed to remain trusted after its author withdrew it; and
- form receipts were not allowed to become award claims.

That discipline is the durable output of the campaign alongside the code, checkpoints, tests, reports, and upstream pull requests.
