# External natural-data probe

This probe asked whether the frozen v0.2 `coherent-normal-step` cue could move
from controlled normal-offset proxies to a defensible natural sheet-switch
claim. It could not. Publishing that negative result immediately before the
submission deadline matters: it makes the benchmark's stated limits testable
rather than decorative.

The detector was run unchanged with `normal_step_ratio = 0.25` and
`normal_step_min_component_cells = 8`. No threshold was selected or adjusted
after inspecting either external source.

| External source | Public description | Valid cells | Normal-step cells | Cell rate | Components |
| --- | --- | ---: | ---: | ---: | ---: |
| PHerc1451, 17 raw tracer surfaces | Vesuvius team member: sheet switches should be visible as black gaps | 2,490,289 | 9,245 | 0.371% | 723 |
| PHerc1667 `02231955` | Segmenter: no large-scale sheet switches apparent; a few outer half-wraps remain | 3,506,175 | 31,774 | 0.906% | 1,723 |

Fourteen of the 17 PHerc1451 surfaces contained at least one coherent-step
component. That is a real frozen-detector result, but it is not natural recall:
the collection-level description supplies no exact per-cell labels, and the
cue can also respond to folds, deformation, seams, and other strong
surface-normal changes.

The PHerc1667 comparator produced a higher normal-step cue-cell rate and a
larger review burden. That surface is not a perfect clean label—the public
description excludes only *large-scale* switches and notes a few outer
half-wraps. The two collections also differ in scroll, pipeline, shape,
surface count, and provenance. The ordering is therefore descriptive rather
than a controlled accuracy comparison. It establishes only that the frozen
normal-step cue does not separate these two populations.

## The missing-data observable

The team's phrase "black gaps" describes absence: the tracer stopped producing
a renderable face lattice. That is not a coherent normal step because there is
no surface across the gap on which to measure a discontinuity. The applicable
existing cue is `enclosed-gaps`.

| External source | Surfaces with enclosed gaps | Enclosed regions | Gap cells | Share of face-lattice cells | Regions / million valid cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| PHerc1451 raw `z_dbg_gen_*` | 16/17 | 1,725 | 294,510 | 4.551% | 692.7 |
| PHerc1447 raw `z_dbg_gen_*` | 23/23 | 514 | 22,230 | 1.372% | 594.0 |
| PHerc1667 curated `02231955` | 0/1 | 0 | 0 | 0% | 0 |

The PHerc1447 row is the matched-pipeline control missing from the initial
comparison. It was found by searching the public S3 catalog for another raw
`z_dbg_gen_*` collection and was evaluated after the PHerc1451/PHerc1667 result
had already been frozen. All 23 public coordinate packages were run with the
same unchanged thresholds.

This control reverses the categorical interpretation. Enclosed gaps are present
in both raw tracer collections, so presence alone is compatible with ordinary
tracer incompleteness. PHerc1451 has 3.32 times the gap-cell share of PHerc1447,
but only 1.17 times as many enclosed regions per million valid cells. The median
PHerc1451 region is 36 cells and the largest is 46,838 cells, so a relatively
small number of large holes contributes materially to the cell-share
difference.

PHerc1447 is a pipeline-matched control, not a labeled clean control. No public
statement establishes that its 23 surfaces are free of sheet switches.
Consequently, the remaining difference cannot be attributed to switches,
scroll geometry, or run quality.

This is evidence for two distinct diagnostic signatures:

- missing face-lattice data is represented by the topology cue
  `enclosed-gaps`; and
- connected surfaces with abrupt normal-direction displacement are represented
  by `coherent-normal-step`.

It is not natural sheet-switch recall. No exact switch coordinates are labeled,
some black gaps may have other causes, and `enclosed-gaps` deliberately excludes
invalid regions connected to the grid boundary. The result shows which
observable maps to the collection description, not that every enclosed gap is
a switch. The matched control further shows that this observable is common in
raw tracer output.

The complete aggregate counts, per-segment PHerc1451 and PHerc1447 results,
source URLs, configuration, caveats, and PHerc1667 file hashes are recorded in
[`benchmarks/external-natural-probe-2026-07-29.json`](../benchmarks/external-natural-probe-2026-07-29.json).

The controlled benchmark remains valid for its narrow claim: abrupt 8- and
16-voxel normal-offset proxies on reviewed real geometry. The external probe
adds a signature-classification result, not natural recall or precision.
