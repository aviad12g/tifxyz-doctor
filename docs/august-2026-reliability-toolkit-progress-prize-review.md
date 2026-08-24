# August 2026 Progress Prize — exact review payload

This is a separate submission from the Gap8 experiment. It claims only the
new August upstream identity remediation. July work is linked as provenance
and explicitly disclosed as prior work.

## Full name

Aviad Cohen

## Team description

Individual submission by Aviad Cohen (`aviad12g`), with OpenAI Codex used as
an AI coding and research assistant for evidence auditing, packaging, and
submission preparation.

## Public contribution URL(s)

- Primary immutable evidence commit: https://github.com/aviad12g/tifxyz-doctor/commit/723e4f04ea297ac2d51457d571b0aa036ad06472
- Merged August contribution: https://github.com/ScrollPrize/villa/pull/1299
- TIFXYZ Doctor repository: https://github.com/aviad12g/tifxyz-doctor
- Historical v0.2 release: https://github.com/aviad12g/tifxyz-doctor/releases/tag/v0.2.0

## Short description

I turned a reproducible TIFXYZ provenance warning into a conservative upstream
fix. An exhaustive metadata census of all 450 original/normalized roots in a
frozen public registry snapshot found that 277 roots—including all 264
normalized roots—used the literal UUID `output_tifxyz`; only 174 distinct UUID
strings appeared. This is not coordinate corruption and does not affect
path-keyed consumers, but UUID-keyed caches, joins, deduplication, provenance,
or result stores can conflate unrelated surfaces.

In August I contributed ScrollPrize/villa PR #1299, merged as commit
`8b7c9df4191f01e52e28a8a9fd68add344349841`. It adds optional `--uuid=<id>`
support to both official OBJ-to-TIFXYZ converters, centralizes identity
resolution, preserves the historical basename-derived default when the option
is omitted, and adds regression coverage for explicit override, backward
compatibility, input-stem fallback, and empty-input rejection. Existing
artifacts and existing commands are unchanged; producers can now retain stable
staging paths while supplying a collection-specific identity.

The evidence package binds the merged five-file upstream diff, exact Git blob
identities, machine-readable census, package dependencies, license, and all
material limitations. TIFXYZ Doctor v0.2, the final-quad PR #1264, and the
community-project PR #1278 were already submitted in July and are included
only as historical provenance—not claimed again as new August work. This is a
reliability and interoperability contribution, not a claim of improved
segmentation, unwrapping, or ink quality.

## Final review gates

- [x] Bind the payload to evidence commit
  `723e4f04ea297ac2d51457d571b0aa036ad06472`.
- [ ] Verify every public link without authentication.
- [ ] Confirm the official form fields match this document exactly.
- [ ] Submit only after Aviad reviews the final URL-complete payload.
- [ ] Retain the emailed Google Forms response copy.
