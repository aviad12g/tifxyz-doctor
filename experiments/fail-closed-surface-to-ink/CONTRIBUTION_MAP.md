# Mapping to the official Progress Prize criteria

Rules verified from <https://scrollprize.org/prizes#progress-prizes> on August 12, 2026.

| Official criterion | Evidence in this contribution |
|---|---|
| Address a specific challenge using Vesuvius Challenge scroll data | Prevent silent sheet switches, invalid geometry, raw-texture false positives, and non-transferable coarse ink models from advancing as candidate text. Demonstrated across all 13 eligible scroll volumes. |
| Provide a clear implementation path and a demonstration | Modular command-line tools in `toolkit/`, synthetic tests, real-data manifests, and PHerc0813/PHerc1203/PHerc0800 case studies, including a pinned canonical-model order-stability check. |
| Demonstrate significant advantages over existing solutions | Converts optimistic whole-surface area into defensible accepted area; catches GrowPatch drift, exact plateaus, topology defects, orientation contradictions, tiling seams, and texture-driven model hits before downstream spend or claims. |
| Comprehensive documentation | README, technical report, reproducibility guide, failure-mode catalog, form copy, attribution, release manifest, and self-contained HTML report. |
| Usage examples | Commands and staged workflows in README and `REPRODUCIBILITY.md`; every principal tool has a CLI and focused tests. |
| Accept standard community formats | OME-Zarr/Zarr arrays, TIFFXYZ quadmeshes, numbered TIFF stacks, PNG diagnostics, and JSON manifests. |
| Maintain consistent output formats | All gates emit deterministic JSON with explicit status, thresholds, source hashes, and terminal reasons; rendered stacks are numbered TIFFs with companion masks/manifests. |
| Designed for modular integration | Inventory, geometry, m7 support, topology, rendering, raw audit, model consensus, and incremental-area tools can be run independently or chained. |
| Improve results quantitatively/qualitatively on real data | 13/13 coverage; four ≥0.5 cm²-class surfaces; PHerc0813 saturation proof; exact geometry/self-intersection gates; independent depth-controlled false-positive reviews. |
| Reveal insightful, actionable information | Shows exactly when GrowPatch leaves the intended sheet, when growth saturates, why exact-one-run was an invalid gate, when model orientation/tile seams dominate, and when coarse labels are memorized without local transfer. |
| Well documented and usable by others | Permissive MIT release, compact evidence, no private credentials/data, deterministic tests, explicit limitations, and a reopen policy. |

## Strongest judging argument

This is not a negative-results archive. It is a **quality-control layer for the community pipeline**. It materially increases the chance of reading scrolls by directing attention toward geometry and data regimes that survive independent physical checks, while stopping expensive or misleading branches early. The PHerc0813 saturation audit is a concrete example: naïve area increased nearly fivefold, but exact accepted geometry stopped changing. Without this tool, the same physical patch could have been rendered and screened repeatedly under the illusion of progress.

## Publication and submission boundary

The package is prepared for publication at the stable URL recorded in `FORM_RESPONSE.md` and `CITATION.cff`. Before submission, the prepared commit must be pushed and the public links and `SHA256SUMS` must be verified. Publishing the commit and submitting the official form are separate approval-gated actions.

This local build has not been submitted through the official form.
