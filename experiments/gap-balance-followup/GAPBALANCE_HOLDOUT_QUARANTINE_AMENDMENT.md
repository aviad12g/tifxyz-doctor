# GapBalance holdout-quarantine amendment

Status: **`BLOCKED_REAL_HOLDOUT_NOT_CERTIFIED`**

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
identity, author notice, reasons, sealed-state assertions, compute stop, and
supersession requirements. Its payload SHA-256 is
`663cf15a2db85c26a6a895dafbacf259de0b2fe3a4c5c76f3d734cdf6ae60c01`.

## Immediate consequences

- PHerc1218 version 1 and version 2 are forbidden for confirmation.
- No future corrected version is pre-authorized.
- The historical version-2 manifest, selection, and panel records remain
  metadata-audit evidence only; they do not certify scientific validity.
- No PHerc1218 NPZ, CT, label, prediction, probability, panel, or endpoint has
  been opened in this experiment.
- Completed training checkpoints and already verified result-blind caches are
  preserved byte-for-byte, but no additional GapBalance job is scheduled, no
  RunPod pod is restarted, and no development endpoint is scored while the
  real holdout is uncertified.
- Gap2/Gap4 configurations, thresholds, gates, metrics, and the
  candidate-selection rule remain frozen and ready for a later reassessment;
  they are not executed or reinterpreted now.
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

If no corrected release satisfies these requirements, the experiment remains
blocked without development selection or sealed confirmation.
