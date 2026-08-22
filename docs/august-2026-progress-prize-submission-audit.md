# August 2026 Progress Prize submission audit

**Audit time:** 2026-08-22T11:06:07Z
**Scope:** provider receipts only; no form was submitted during this audit.

The authenticated mailbox contains four Google Forms response receipts for
the exact August 2026 Progress Prizes form. They prove three distinct public
contributions were submitted on 2026-08-15:

1. Gap8 fusion-aware surface training at 09:33:13Z.
2. TIFXYZ UUID reliability and interoperability at 10:53:49Z.
3. Fail-Closed Surface-to-Ink at 13:37:08Z.

There is also a second Gap8 receipt at 10:33:50Z. Its complete plain-text
response is mechanically identical to the first Gap8 response, so it is
recorded as a duplicate rather than another contribution or result.

Provider message identifiers and the entrant's email address are intentionally
not published. The associated public contribution URLs and exact receipt
timestamps are recorded in
[`august-2026-progress-prize-submission-audit.json`](august-2026-progress-prize-submission-audit.json).

## Separate PR #1463 payload

The `vesuvius.surface_preflight` review payload is a separate contribution. No
Google Forms receipt for it was found in this audit, and no new response was
submitted. Its public source remains available in
[ScrollPrize/villa PR #1463](https://github.com/ScrollPrize/villa/pull/1463)
at exact head
[`0ea5aec2c8a2de7c3a080d7d0d3a7524ac8fe928`](https://github.com/aviad12g/villa/commit/0ea5aec2c8a2de7c3a080d7d0d3a7524ac8fe928).

As of the audit time, the PR is open, mergeable, and unmerged. All four GitHub
Actions workflows on the exact head completed successfully. The separate
Vercel deployment status still requires Scroll-team authorization and is not
treated as a project-test result.

## Scientific boundary

This audit changes no result, threshold, gate, interpretation, or scientific
artifact. Gap8 remains the published mixed 5/7-gate result. PHerc1218 versions
1 and 2 remain quarantined, and GapBalance remains
`BLOCKED_REAL_HOLDOUT_NOT_CERTIFIED` with no development scoring or
confirmation access permitted.
