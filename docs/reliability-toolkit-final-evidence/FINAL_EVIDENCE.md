# TIFXYZ identity reliability: August 2026 final evidence

## Result in one sentence

A public, exhaustive metadata census found that 277 of 450 registered
original/normalized TIFXYZ roots reused the literal UUID `output_tifxyz`; in
August, a backward-compatible upstream change added explicit UUID control to
both official OBJ-to-TIFXYZ converters so stable staging paths no longer have
to determine collection identity.

## What is new in this August submission

The claimed August contribution is
[ScrollPrize/villa PR #1299](https://github.com/ScrollPrize/villa/pull/1299),
**Allow explicit TIFXYZ UUIDs in obj2tifxyz**, opened on 2026-08-02 and merged
on 2026-08-06 as commit
[`8b7c9df4191f01e52e28a8a9fd68add344349841`](https://github.com/ScrollPrize/villa/commit/8b7c9df4191f01e52e28a8a9fd68add344349841).

The upstream change:

- adds optional `--uuid=<id>` support to `vc_obj2tifxyz` and
  `vc_obj2tifxyz_legacy`;
- centralizes identity resolution in `TifxyzIdentity.hpp`;
- preserves the historical output-directory-basename behavior when the
  option is omitted; and
- adds four resolver regression cases covering explicit override, backward
  compatibility, input-stem fallback, and empty-input rejection.

The merged diff changed five public files: 36 additions in the shared helper,
31 additions in its regression test, two CMake registration lines, and the
two converter integrations. Exact paths and Git blob identities are frozen in
[`evidence_manifest.json`](evidence_manifest.json).

## Evidence that motivated the change

The registry snapshot observed on 2026-07-22 contained 186 registered
`tifxyz_original` roots and 264 registered `tifxyz_normalized` roots. The
metadata census covered all 450:

- 277 of 450 roots used the literal UUID `output_tifxyz`;
- all 264 normalized roots and 13 original roots used that value; and
- only 174 distinct UUID strings appeared across the 450 roots.

This does **not** imply coordinate corruption or a formal global-uniqueness
requirement. Paths still distinguish every object, so path-keyed consumers are
unaffected. The operational risk is limited to UUID-keyed caches, joins,
deduplication, provenance, or result stores, where unrelated packages may be
overwritten or conflated.

The exact census, selection scope, timestamps, URLs, and caveats are preserved
in
[`benchmarks/public-corpus-scan-2026-07-26.json`](../../benchmarks/public-corpus-scan-2026-07-26.json)
and summarized in
[`docs/claims-review.md`](../claims-review.md). The audit exhaustively covered
metadata and object presence for the 450 original/normalized roots; it did not
download the coordinate payload of every root and did not cover every
transformed or flattened registry root.

## Why the remediation is conservative

The change does not rewrite existing artifacts, mandate UUID policy, infer the
private production command, or silently change established converters. A
caller that does nothing gets the previous behavior. A producer that needs a
stable, collection-specific identity can now provide one explicitly while
retaining a fixed staging directory.

This is deliberately a reliability contribution rather than a segmentation
quality claim. It prevents an avoidable provenance ambiguity before packages
enter rendering, training, evaluation, or collaborative result stores.

## Prior work and non-duplication boundary

The following work is useful context but is **not claimed as new August work**
in this submission:

- TIFXYZ Doctor v0.2 and its 709-patch/1,920-case reviewed-surface benchmark,
  released and submitted in July;
- Villa PR #1264, the final-quad geometry correction, created and submitted
  in July and merged on 2026-08-01; and
- Villa PR #1278, the TIFXYZ Doctor community-project listing, created and
  submitted in July and merged on 2026-08-01.

Those artifacts establish the diagnostic lineage and upstream adoption. They
are linked for reproducibility, not presented again as a new prize result.

## Reproduction and validation

The August contribution can be reviewed without private data:

1. inspect the merged PR and its five-file diff;
2. run the upstream resolver test in Villa's C++ test suite;
3. inspect the hash-pinned corpus census and claims review; and
4. run `python scripts/verify_reliability_evidence.py` in this repository to
   verify every local evidence identity and the explicit prior-work boundary.

The original PR also records a standalone strict C++ build and 4/4 resolver
regression tests. The evidence verifier does not refetch mutable live data and
does not reinterpret the July scientific benchmark.

## Limitations

- UUID reuse is a collection-level interoperability warning, not proof that
  any surface geometry is wrong.
- The format itself is not claimed to require global UUID uniqueness.
- Existing public objects remain unchanged; the new option affects only
  producer invocations that use it.
- Producer attribution is not claimed because the private production command
  is not observable.
- The upstream resolver tests validate identity selection and compatibility;
  they do not measure downstream segmentation, unwrapping, or ink quality.
- The July benchmark and earlier PRs are historical context, not new August
  contributions.

## Public references

- [Merged August UUID remediation](https://github.com/ScrollPrize/villa/pull/1299)
- [TIFXYZ Doctor repository](https://github.com/aviad12g/tifxyz-doctor)
- [Machine-readable census](../../benchmarks/public-corpus-scan-2026-07-26.json)
- [Claims review](../claims-review.md)
- [Historical v0.2 release](https://github.com/aviad12g/tifxyz-doctor/releases/tag/v0.2.0)
- [Historical final-quad PR](https://github.com/ScrollPrize/villa/pull/1264)
- [Historical community-project PR](https://github.com/ScrollPrize/villa/pull/1278)
