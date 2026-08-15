# Fail-Closed Surface-to-Ink

**A reproducible validation and stopping framework for Vesuvius Challenge geometry and ink searches**

This contribution turns a 13-scroll First Letters breadth search into reusable infrastructure. It accepts standard Vesuvius inputs—OME-Zarr/Zarr volumes and TIFFXYZ quadmeshes—and produces hash-bound geometry, sampling, raw-depth, model-consensus, and decision evidence. Its central design rule is simple: a candidate advances only when every precommitted gate passes; a failure is recorded with enough provenance to reproduce it.

The framework was exercised across all 13 scroll volumes currently eligible for the 2027 Grand Prize. It did **not** produce a First Letters claim. That negative result is itself useful: it exposes silent failure modes that otherwise generate plausible surfaces, apparent glyphs, and unnecessary GPU runs.

## Why this is a Progress Prize contribution

The current Progress Prize rules favor open-source work that improves real scroll results, detects actionable failure cases, uses standard formats, integrates modularly, and is thoroughly documented. This toolkit does exactly that:

- validates TIFFXYZ geometry before raw rendering;
- calibrates surface selection against public m7 predictions without treating globally common multi-run transects as an automatic veto;
- renders physically calibrated normal-offset stacks from OME-Zarr/Zarr with explicit axis and sign conventions;
- detects folds, flips, discontinuities, wrap risk, out-of-bounds sampling, topology defects, and non-adjacent self-intersections;
- audits raw candidates across depth with fiber, crack, fold, void, and mask-boundary controls;
- keeps forward/reverse and depth-band model evidence symmetric rather than hiding contradictory orientations;
- proves when additional growth adds no new accepted physical surface;
- records terminal decisions, hashes, seeds, model revisions, and reopen conditions.

Official Progress Prize rules and the August 31, 2026 deadline are at [scrollprize.org/prizes](https://scrollprize.org/prizes#progress-prizes).

## Quantitative demonstration

- **13/13** eligible scroll volumes audited under one fail-closed decision framework.
- **4** scrolls reached clean ≥0.5 cm²-class raw surfaces; all four were rejected after independent raw and/or model review.
- **0** defensible candidates approached the First Letters requirement of 10 visible, legible letters in one area up to 4 cm².
- **PHerc0813 saturation:** whole GrowPatch area rose from 2.1333 to 10.1028 cm², while the accepted calibrated component plateaued at 0.536269 cm²; generations 60 and 80 had exactly the same 1,520 accepted quads.
- **Coarse-ML falsification:** a stronger model memorized train-A (AUROC 0.99595 ± 0.00344) but failed a non-overlapping local-B region only 1.228 mm away (AUROC 0.51207 ± 0.05094) across three seeds.
- **Canonical-model order check:** on PHerc1203, physically scale-matched forward and reverse ResNet-152 maps had Pearson correlation 0.032425, top-1% Jaccard 0.003361, and zero overlap in the top 0.5%; fixed raw review found 0 defensible text candidates, so shifted-origin inference was skipped.
- **273 tests** passed in the canonical campaign workspace before release packaging.

These figures close only the automated route against the public assets and methods tested here. They do not imply that the scrolls contain no writing.

## Package map

- [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) — contribution, experiments, results, limitations, and actionable lessons.
- [`FORM_RESPONSE.md`](FORM_RESPONSE.md) — concise copy for the official August 2026 form.
- [`CONTRIBUTION_MAP.md`](CONTRIBUTION_MAP.md) — direct mapping to the official judging criteria.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — installation, synthetic tests, and real-data CLI workflow.
- [`FAILURE_MODE_CATALOG.md`](FAILURE_MODE_CATALOG.md) — discovered failure modes and the gate that catches each one.
- [`report/progress-prize-report.html`](report/progress-prize-report.html) — self-contained technical report.
- [`toolkit/`](toolkit/) — modular Python tools and focused tests.
- [`evidence/`](evidence/) — compact machine-readable records and representative derived images; no raw CT stack is redistributed.
- [`SHA256SUMS`](SHA256SUMS) — byte-level release manifest generated after assembly.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m pytest -q toolkit/tests
```

Representative commands:

```bash
# Inventory public PHerc assets without recursively entering large Zarr containers.
python toolkit/inventory_pherc_assets.py --help

# Validate and render a TIFFXYZ surface against a Zarr volume.
python toolkit/render_tifxyz_volume.py --help

# Run native geometry and full-resolution preflight gates.
python toolkit/full_resolution_tifxyz_preflight.py --help

# Audit a complete 93-layer raw stack without model predictions.
python toolkit/independent_raw_ink_audit.py --help

# Prove how much exact accepted geometry is genuinely new between two surfaces.
python toolkit/audit_tifxyz_exact_increment.py --help
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for input contracts and a staged workflow.

## What is intentionally absent

- No claim of legible text.
- No raw CT volume or 93-layer stack redistribution.
- No credentials, private endpoints, model weights, or billing data.
- No generative super-resolution or post-hoc letter enhancement.

The intended public location is
<https://github.com/aviad12g/tifxyz-doctor/tree/agent/freeze-fusion-aware-checkpoints/experiments/fail-closed-surface-to-ink>.
Publication of the prepared commit and submission of the official form remain separate, approval-gated actions.

## Submission scope

This is a distinct infrastructure and evidence contribution. It does not reuse the scientific claim of the separate Gap8 surface-fusion experiment or the TIFXYZ UUID/provenance reliability fix. Its claim is narrower: fail-closed geometry, raw-depth, model-stability, and provenance checks can prevent invalid or non-transferable candidates from advancing.

## Reopen policy

The First Letters search should reopen only when one of these changes the physical evidence:

1. a new official scan, surface, or materially improved geometry prediction;
2. a genuinely independent expert-corrected single-sheet ROI;
3. direct repeatable raw strokes that survive depth, fiber, crack, fold, void, and boundary controls.

## License and attribution

Original code and documentation in this package are offered under the [MIT License](LICENSE). Vesuvius Challenge data and site content are not relicensed; the site states CC BY-NC 4.0 unless otherwise specified. Upstream tools and models retain their own licenses. See [`ATTRIBUTION.md`](ATTRIBUTION.md).
