# Claims review

This is the short human-readable companion to
[`public-corpus-scan-2026-07-26.json`](../benchmarks/public-corpus-scan-2026-07-26.json).
Read the machine-readable record before publishing under your name; this page
states exactly what the release claims and what it does not.

## Registry scope

- Registry snapshot observed: 2026-07-22 14:17:40 UTC.
- Metadata/object-presence census: all 186 registered original roots and all
  264 registered normalized roots, 450 total.
- Coordinate-content pass: 242 directories, 726 coordinate TIFFs, 8,836,120
  stored grid points.
- Only 103 registered coordinate payloads were downloaded. The remaining
  coordinate-content coverage came from 138 PHerc1203 snapshots and one local
  PHerc1667 spot check. This is not a full coordinate download of all 450
  roots.

## Claim 1 — UUID reuse

Defensible statement:

> In the observed registry snapshot, 277 of 450 registered
> original/normalized TIFXYZ roots used the literal UUID `output_tifxyz`.
> This included 264 of 264 normalized roots and 13 of 186 original roots.

Operational meaning: `uuid` cannot uniquely identify those packages for a
consumer that keys caches, joins, provenance, deduplication, or result stores
on that field.

Do not claim:

- that the coordinates are corrupt;
- that the format formally requires global uniqueness; or
- that path-keyed consumers are affected.

## Claim 2 — PHerc1203 area schema/API mismatch

Defensible statement:

> All 138 inspected PHerc1203 current/version snapshots—22 current and 116
> version directories—contained `area_vx2` and `area_cm2`, while none
> contained the `area` key consumed by the pinned Python API. The suffixed
> fields remain in `extra`, but `Tifxyz.area` is unset.

Do not claim:

- that the area values are numerically wrong;
- that the values are destroyed; or
- that the scan covered every PHerc1203 artifact outside those 138
  current/version snapshots.

## Claim 3 — registry-listed semantic-empty packages (dated)

Defensible statement:

> In a 2026-07-26 audit against the registry snapshot last modified on
> 2026-07-22, three registered normalized paths resolved to hash-pinned 2×2
> coordinate
> packages containing only the full `(-1,-1,-1)` sentinel. They therefore had
> zero portable-valid vertices and faces, and the recorded 3/3 regression
> reproduced both expected contract errors.

One identifier contained `z_dbg`; that string alone did not establish the
entry's cause or intended use.

The original scan alone did not establish:

- why any entry exists;
- that the entries are consumed by production;
- that `z_dbg` proves intent; or
- that the files are corrupt TIFF/JSON. They are syntactically valid and
  semantically empty under the pinned reader model.

Later evidence resolved the cause: on 2026-07-27 a Villa administrator
attributed the entries to legacy OBJ data processed through the wrong pipeline
and removed or regenerated them. Fresh audits of the three current
registrations pass. Treat Claim 3 as a dated finding whose report was followed
by an upstream repair, not as a statement about current live objects. See
the exact
[diagnosis](https://discord.com/channels/1079907749569237093/1243576621722767412/1531206054682165309),
[remediation confirmation](https://discord.com/channels/1079907749569237093/1243576621722767412/1531220162190245909),
and
[`public-empty-resolution-2026-07-27.json`](../benchmarks/public-empty-resolution-2026-07-27.json).

The UUID and area findings are independent of this remediation.

## Synthetic Python/C++ differential

- None of the 450 probed original/normalized roots exposed `mask.tif`; the
  reader disagreements are synthetic interoperability fixtures, not public
  mask failures.
- The dated pairing is the official stable Apple Silicon VC3D release
  (`05ff9ea`) and exact Python reader files at the 2026-07-26 repository HEAD
  (`1162bcab`).
- Five declared fixture expectations are satisfied: two controls and three
  confirmed disagreements.
- The executable reports loading the full 5×5 fixture. Its 16-vertex/18-face
  all-valid OBJ is explained by boundary filtering during emission, not by a
  4×4 loader result. The three disagreement cases match the calibrated
  12-vertex/10-face known-invalid-center signature.
- TIFXYZ Doctor's 24 finite C++ points are a source-derived model. The direct
  executable observables are the loaded 5×5 dimensions and OBJ counts.

## Geometry claims

- Geometry events are review cues, not labeled defects or repair targets.
- The proximity pass samples vertices; it is not triangle-intersection proof.
- The PHerc1667 example is a local exploratory spot check, not a downloadable
  benchmark case.
- Its corrected overlay colors 270 of 725,986 valid cells (0.037%) and shows
  13 enclosed-gap cells separately.

## Personal approval

Before publishing, confirm that you are comfortable defending:

- the 450-root census denominator and its original/normalized scope;
- the distinction between UUID-keyed and path-keyed consumers;
- the 138-snapshot PHerc1203 scope and preservation of suffixed area values;
- the registry-hygiene-only interpretation of the three empty entries; and
- the synthetic, architecture-specific, dated scope of the reader
  differential.
