# Progress Prize submission draft: TIFXYZ Doctor v0.2

## TIFXYZ Doctor v0.2 — a sealed reviewed-patch stress benchmark

TIFXYZ Doctor v0.2 adds a conservative `coherent-normal-step` cue for abrupt,
spatially coherent changes in TIFXYZ surface geometry, plus a reproducible
benchmark built from the official PHercParis4 human-reviewed `same_wrap`
patches.

The benchmark was designed to avoid a subtle leakage problem: reviewed patches
can overlap the same physical surface. The 64 patches used while developing the
cue touch overlap-connected components containing 217 of the 709 eligible
patches. I froze the detector, removed all 217 development-connected patches
from holdout, and committed the exact split before opening 128 holdout patches
from 82 untouched overlap components.

On the sealed holdout:

- abrupt 8-voxel normal-offset proxies were localized in **124/128 cases
  (96.875%)**;
- abrupt 16-voxel proxies were localized in **128/128 cases (100%)**;
- the v0.1 cue union localized **0/128** and **2/128** of those cases,
  respectively; and
- **128/128 nulls** were identical to baseline in coordinate bytes, validity
  bytes, the complete public audit report, and a reduced audit signature.

The negative result matters too: gradual four- and twelve-cell transitions
were mostly missed. I report that directly. These are controlled normal-offset
proxies constructed from real reviewed surfaces—not naturally occurring sheet
switches and not a claim of real-world sheet-switch recall.

The full 492-patch overlap-isolated pool also provides a descriptive same-wrap
check. The new cue touched 26 of 373,904 mapped annotation-neighborhood cells
and 2 of 491 patches with nonempty neighborhoods. All 26 cells were already
flagged by a v0.1 cue, so the new family increased the same-wrap neighborhood
union by zero cells. Same-wrap labels negate a sheet switch, not every possible
fold or distortion, so I report alert rates rather than calling them false
positives.

The deliverable includes:

- the v0.2 CLI cue with exact grid locations and overlay integration;
- an official-overlap-graph split builder;
- a 709-patch / 1,920-case deterministic benchmark;
- overlap-component cluster-bootstrap intervals;
- 59 passing tests, including leakage, exact-null, and result-snapshot checks;
  and
- source/data/split/result SHA-256 provenance.

Two full runs—including one against a fresh direct download in a separate
source directory—produced the same complete result SHA-256:
`74f243eb57a84eb2ee9aa8df2f5674ef92d86fc1d54a8333c7b0b9e1fe21173c`.

Repository: https://github.com/aviad12g/tifxyz-doctor

Evidence:
`docs/reviewed-same-wrap-benchmark.md`,
`benchmarks/reviewed-same-wrap-split-v1.json`, and
`benchmarks/reviewed-same-wrap-results-v0.2.0.json`.
