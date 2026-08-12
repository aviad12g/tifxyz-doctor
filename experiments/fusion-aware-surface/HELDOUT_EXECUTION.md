# Fusion-aware held-out execution preparation

This directory contains local, result-blind preparation for the preregistered
held-out phase.  It does **not** contain Scroll-4/5 predictions, synthetic ray
probabilities, selected thresholds, or scientific endpoints.

`PROTOCOL_CLARIFICATION.md` records a pre-inference implementation correction:
the written preregistration's synthetic magnitude gates pool raw counts across
the three matched training seeds, while requiring a negative fusion delta in
every seed. `score_synthetic_test_v2.py` implements that contract; the original
stricter per-seed numeric checks remain reported as secondary diagnostics. The
clarification and scorer were publicly frozen before held-out inference.

`build_heldout_execution_plan.py` is run only after the private threshold job
has completed and both threshold artifacts have been published at an immutable
GitHub commit.  It validates the threshold artifact identities and emits a
canonical job plan for:

- seven private Scroll-4/5 cache jobs, one per frozen model run;
- seven private synthetic ray jobs, one per frozen run, each invoking the ten
  preregistered shards separately;
- one real and one synthetic one-shot scorer after all sealed caches exist.

Each synthetic job emits the ten original shard manifests, with exactly ten
preregistered cells per shard.  The successful seven-run validation kernel
needed about 22 minutes of inference in total, so this operational grouping
has ample headroom under Kaggle's session limit. Queue orchestration may keep
at most the available GPU concurrency active, but may not change run order,
shard membership, thresholds, checkpoints, or scientific code.

`orchestrate_heldout_queue.py` is the result-blind operational controller for
those 14 packages. On every invocation it revalidates the complete package
index, every package file and checksum ledger, and the prior acceptance
receipt. It queries only Kaggle worker status, rejects a non-prefix launch
order or any failed/unknown worker state, and fills at most two active slots in
the frozen package order. A push counts only when the CLI reports both a kernel
version and successful acceptance; the version is then recorded in a
canonical receipt for the later delivery freeze. It never downloads outputs.

Kaggle uploads only the configured source file, so each generated source
contains its canonical per-job config as hex in the first comment line. The
remaining bytes must exactly equal the public base launcher, and the launcher
recomputes that identity before inference. No sidecar config is assumed.

`collect_heldout_delivery_inputs.py` runs only after all 14 jobs are complete.
For each job it checks that Kaggle's latest version equals the queue receipt and
that the server source equals the accepted package. It downloads only the
sealed job index and cache-manifest JSON files. The CLI's automatic log file is
suppressed with an explicit first-page token and a page size larger than the
complete manifest set; any emitted log is rejected. NPZ probability caches are
also rejected. Its canonical source record is the only input accepted by the
delivery freezer.

The output plan must itself be publicly frozen before any held-out inference.
`validate_final_results.py` is also hash-bound by that plan. After the two
one-shot scorers finish, it independently recomputes per-patch means,
deterministic bootstrap summaries, pooled ray rates, paired deltas, and every
preregistered Boolean gate. It also requires both scoring run manifests and
proves that the two result files came from the same public plan and cache
delivery through the exact frozen launcher and scorer identities; a
self-consistently rehashed but altered gate or provenance record is rejected.

`render_real_panels.py` is likewise hash-bound before held-out inference. It
uses only the four model-blind locations in `real_panel_manifest.json`, derives
the same sigma-3 signed-distance normal at each frozen centre, and renders two
fixed 97×97 normal/tangent planes. Ground-truth agreement is gray, missed
surface is blue, and false-positive material—including an inter-sheet bridge—is
red. The real one-shot scorer invokes it automatically after scoring from the
same publicly frozen cache delivery; the renderer verifies the public plan,
delivery, staging index, thresholds, source-panel manifest, and its own source
identity. Its four PNGs and render manifest stay sealed until both the real and
synthetic scorers complete. The renderer deliberately does not decide the
human visual gate.

`record_visual_assessment.py` is the separately hash-bound post-render gate
recorder. After both scorer jobs complete and the fixed panels are opened, it
requires an observation for every panel × matched-seed comparison (4×3, in
fixed order), including explicit separation-improvement and new-break Booleans
and a note. It recomputes the preregistered visual gate from all 12 records;
panels or comparisons cannot be selectively omitted. The raw observations,
recorder output, all images, and their hashes are required by the final
validator and evidence renderer.

`stage_heldout_for_scoring.py` is the result-blind delivery handoff. It requires
the immutable public plan and later public 14-job delivery manifest, resolves
each mounted job index by exact bytes and SHA-256, rechecks every manifest and
all 966 cache-file hashes, and creates scorer-facing symlinks without loading an
NPZ. Fixture tests cover the full 7+7 delivery and reject a byte-tampered cache
before scoring.

`render_final_evidence.py` is hash-bound by the public held-out plan before any
held-out inference. It first runs the independent result validators, then emits
a fixed Markdown report containing every preregistered gate, every selected-run
summary, all matched-seed deltas, and the complete fixed-threshold sensitivity
readout. It cannot selectively hide failed, null, mixed, or adverse outcomes.

`one_shot_scoring_launcher.py`, `prepare_metric_runtime.py`, and
`generate_scoring_job_packages.py` close the post-cache operational gap. After
the 14 cache-job identities are publicly frozen, the generator emits the real
and synthetic scorer packages together. Each package verifies the public plan,
public delivery, exact seven cache kernels, frozen thresholds, source ledger,
and scorer identity. The real scorer rebuilds and smoke-tests the pinned
official metric. Both scientific stdout streams are captured and hashed rather
than echoed. Kaggle again uploads only one configured source file, so each
scorer source embeds its canonical mode/commit config in its first comment and
fetches the hash-bound stager and metric preparer from the immutable public-plan
commit. It does not depend on missing sidecar files.

`orchestrate_scoring_pair.py` validates both complete packages and records both
successful Kaggle push versions in one canonical acceptance receipt. It does
not download outputs. `collect_scoring_results.py` refuses to download either
result until both receipt entries exist and both workers are complete, then
proves that the latest server sources equal the two accepted package wrappers.
It downloads only each sealed result JSON and scoring run manifest, plus the
four fixed real-panel PNGs and their render manifest, and does not open them.
`validate_final_results.py` is the first process allowed to read the two result
files; it requires the collector record, package index, pair receipt, complete
panel set, raw visual observations, and assessment record, and validates the
complete cryptographic chain before recomputing the scientific results.
Downloader logs are neither downloaded nor opened.

Example (after threshold publication):

```bash
python build_heldout_execution_plan.py \
  --thresholds /path/to/frozen_thresholds.json \
  --threshold-run-manifest /path/to/threshold_run_manifest.json \
  --panel-manifest /path/to/real_panel_manifest.json \
  --threshold-kernel-version 6 \
  --public-threshold-commit FULL_40_HEX_COMMIT \
  --out heldout_execution_plan.json
```

After that plan and all generated packages are publicly bound, one queue tick
is:

```bash
python orchestrate_heldout_queue.py \
  --packages /path/to/generated-heldout-packages \
  --index /path/to/generated-heldout-packages/generated_packages_index.json \
  --receipt /path/to/heldout_queue_receipt.json
```

After all 14 caches are complete, mechanically collect their manifest-only
delivery inputs, publish the resulting delivery freeze, generate both scoring
packages, and launch the fixed pair:

```bash
python collect_heldout_delivery_inputs.py \
  --packages /path/to/generated-heldout-packages \
  --package-index /path/to/generated-heldout-packages/generated_packages_index.json \
  --queue-receipt /path/to/heldout_queue_receipt.json \
  --out /path/to/heldout-delivery-inputs

python orchestrate_scoring_pair.py \
  --packages /path/to/generated-scoring-packages \
  --index /path/to/generated-scoring-packages/generated_scoring_packages_index.json \
  --receipt /path/to/scoring_pair_receipt.json
```

Only after both scorer kernels report complete may the sealed pair be
collected. The independent validator and renderer must receive the resulting
`scoring_result_sources.json`, the exact scoring-package index, and the paired
acceptance receipt in addition to the two result and run-manifest paths.
