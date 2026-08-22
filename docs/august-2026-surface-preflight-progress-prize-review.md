# August 2026 Progress Prize — surface-pairing preflight review payload

**Status:** **Submitted once and confirmed by Google Forms receipt** at
2026-08-22T11:29:38Z after Aviad Cohen's explicit authorization. This is a
distinct contribution from the already-submitted Gap8 and UUID-reliability
entries. Do not create another response for this contribution or either
earlier entry.

Provider and public state were rechecked on 2026-08-22. ScrollPrize/villa PR
[#1463](https://github.com/ScrollPrize/villa/pull/1463) is open, mergeable, and
not merged. Its exact public head is
[`0ea5aec2c8a2de7c3a080d7d0d3a7524ac8fe928`](https://github.com/aviad12g/villa/commit/0ea5aec2c8a2de7c3a080d7d0d3a7524ac8fe928).
All four GitHub Actions workflows currently associated with that exact head
completed successfully: Large PR review gate, CodeQL, Continuous Integration
(CI), and Test vesuvius Python. The separate Vercel deployment status remains
failed because a Scroll team member must authorize the deployment; it is not
recorded as a project-test failure.

## Email

Use the email address recorded by Aviad's signed-in Google account.

## Your full name

Aviad Cohen

## Team description

Individual submission by Aviad Cohen (`aviad12g`). OpenAI Codex was used as
an AI coding and research assistant for implementation, evidence auditing, and
submission preparation; Aviad is the sole entrant and team leader.

## URL to your open source / publicly available contribution

- Primary contribution: https://github.com/ScrollPrize/villa/pull/1463
- Exact public head: https://github.com/aviad12g/villa/commit/0ea5aec2c8a2de7c3a080d7d0d3a7524ac8fe928
- Independent seven-pair validation: https://github.com/ScrollPrize/villa/pull/1463#issuecomment-5307369457
- Scope clarification responding to that validation: https://github.com/ScrollPrize/villa/pull/1463#issuecomment-5307416420

## Short description of how the contribution substantially increases the probability of reading complete scrolls

I contributed `vesuvius.surface_preflight`, a deterministic, fail-closed
command that checks a TIFXYZ surface together with its source Zarr/OME-Zarr CT
volume before rendering, label transfer, or model inference. A surface folder
can be structurally readable yet paired with the wrong scroll, crop,
multiscale array, or partially populated volume. Those errors can otherwise
waste expensive downstream runs or, worse, produce apparently valid output
from coordinates with little or no CT support.

The five-file Villa contribution adds the installed command, documentation,
tests, and package entry point. It verifies required TIFXYZ files and metadata,
compatible coordinate and mask rasters, finite selected coordinates, valid
vertices and connected quads, exact coordinate bounds against the selected CT
array, and deterministic sampled CT-signal support. It writes an atomic,
machine-readable JSON report and exits nonzero when any required gate fails,
so a downstream pipeline can require an explicit complete `PASS` record rather
than infer safety from partial output.

The author validation reported eight focused tests passing and a clean-install
end-to-end run against the real PHerc1667 `w013_7.91um` surface and official
Scroll 4 level-0 CT volume: 9/9 required gates passed, covering 728,218 valid
vertices, 725,986 valid quads, zero out-of-bounds coordinates, and signal at
all 16 deterministic CT samples.

The sampled support gate also received a public independent referee test.
Jinhojeong predicted the correct result from exact per-quad support
classifications before running seven surface/volume pairings. The command
matched all seven expected PASS/FAIL outcomes. With its default 1,024-vertex
sample, the reported support fraction remained within 1.5 percentage points
of the per-quad reference in every measured pairing; the cases included a
remote full OME-Zarr, a wrong-region store, a sparse prediction store, and a
nonexistent volume path. That comparison provides real failure-case evidence
for the deterministic sampled gate rather than only synthetic unit fixtures.

This contribution does not claim that a surface is geometrically correct and
does not replace self-intersection or local-orientation diagnostics. A support
failure means the selected volume does not support enough sampled coordinates;
it can indicate a wrong pairing, cropped/sparse/partially populated CT, or
genuinely unsupported coverage. The documentation was tightened after the
independent comparison so users do not misdiagnose every support failure as
bad geometry.

The practical benefit is a small, modular stop gate at the point where surface
and CT identities first meet. It makes invalid pairings reproducibly visible
before they consume inference resources or contaminate later surface and ink
evidence, increasing the reliability of the pipeline used to reconstruct and
ultimately read complete scrolls.

## Evidence boundary

- PR #1463 is **open and unmerged** as of the stated review date. The source is
  public in Aviad's fork at the exact head commit above.
- The independent seven-pair comparison tests input-pairing gate behavior. It
  does not certify surface geometry, segmentation quality, ink, or recovered
  text.
- It also does not certify any version of the withdrawn PHerc1218 tight-contact
  validation dataset, its labels, or its gap census. PHerc1218 versions 1 and 2
  remain forbidden for GapBalance.
- The previously reported Python-compatibility workflow interruption was an
  external dependency-fetch `503`; it was not a project-test failure. All four
  GitHub Actions workflows now pass on the exact head. The separate Vercel
  deployment authorization status remains failed, so this payload does not
  describe every provider status as green.

## Terms and Conditions

Select **Yes, I agree** only after Aviad has reviewed this exact payload and
explicitly decides to file this distinct contribution. Do not use the form to
resubmit Gap8 or the UUID-reliability entry.
