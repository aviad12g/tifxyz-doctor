# GapBalance holdout-quarantine amendment

Status: **public, result-blind, and fail-closed; confirmation is blocked**

This additive amendment does not rewrite the hash-bound
`GAPBALANCE_PROTOCOL.md`, training contract, completed checkpoints,
development inputs, thresholds, gates, metrics, or selection rule.

On 2026-08-21, after the version-2 metadata audit and all six trainings had
completed, the PHerc1218 dataset author instructed users to hold off pinning
any published version. The author reported faults in crop CT and label
stitching and a unit error in the comparison of its gap census with another
validation set. The authoritative notice is GitHub Issue #191 comment
5366060707:

<https://github.com/ScrollPrize/villa/issues/191#issuecomment-5366060707>

`PHERC1218_HOLDOUT_QUARANTINE.json` freezes the exact affected version-2
identity, author notice, reasons, sealed-state assertions, and supersession
requirements. Its payload SHA-256 is
`e25f56f18d189f3e91a22814cc5feb25cb1049773294206ac53a5ca0f46be6a6`.

## Immediate consequences

- PHerc1218 version 1 and version 2 are forbidden for confirmation.
- No future corrected version is pre-authorized.
- The historical version-2 manifest, selection, and panel records remain
  metadata-audit evidence only; they do not certify scientific validity.
- No PHerc1218 NPZ, CT, label, prediction, probability, panel, or endpoint has
  been opened in this experiment.
- Development caching, development scoring, and the frozen Gap2/Gap4
  candidate-selection rule may continue unchanged because those inputs exclude
  PHerc1218 and all confirmation data.
- Even after one candidate and every threshold are publicly frozen, the entire
  one-shot confirmation remains sealed. Synthetic confirmation seeds 500--504
  must not be opened separately while the real confirmation holdout is under
  quarantine.

## Requirements to supersede the quarantine

All of the following must be public before confirmation access:

1. the author publishes the complete verified correction account and a
   corrected provider release;
2. that release passes a new metadata-only identity and inventory audit;
3. an additive public record freezes its provider identity, manifest,
   eligibility list, and fixed panels;
4. the chosen candidate, thresholds, gates, metrics, and selection rule remain
   unchanged; and
5. a hashed public supersession explicitly deactivates
   `PHERC1218_HOLDOUT_QUARANTINE.json`.

If no corrected release satisfies these requirements, the experiment stops
after development selection and reports that sealed confirmation was not run.
