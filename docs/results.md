# Recorded results

These results demonstrate reproducibility and show the tool's output shape.
They do **not** establish a geometry classifier, estimate false-positive
rates, or label any reviewed surface as defective.

## Headline public-data results

The strongest demonstrated corpus observations are:

1. 277 of 450 registered original/normalized roots reuse the literal UUID
   `output_tifxyz`, which is reported as a collection identity warning rather
   than per-file corruption;
2. all 138 inspected PHerc1203 current/version snapshots supply
   `area_vx2`/`area_cm2` without `area`, leaving the pinned Python API's
   `Tifxyz.area` property unset; and
3. three metadata-registered normalized packages contained no valid vertex or
   face in a 2026-07-26 audit against the registry snapshot last modified on
   2026-07-22, and all three exact inputs reproduced the expected errors in a
   hash-pinned 3/3 regression before upstream remediation.

The first two observations do not depend on whether the semantic-empty registry
entries were intentional. One empty entry contains `z_dbg` in its identifier;
the regression establishes the recorded bytes and paths, not intent or
production use.

The broader scan also found no public `mask.tif` object among the 450 probed
registered roots. Mask-reader differences are therefore synthetic,
revision-pinned interoperability cases, not a claimed real-corpus finding.
The exact scan scope and caveats appear under
[Broader exploratory corpus scan](#broader-exploratory-corpus-scan).

### Resolution update (2026-07-27)

A Villa administrator identified the three empty packages as legacy OBJ data
processed through the wrong pipeline, then removed or regenerated them. The
[diagnosis](https://discord.com/channels/1079907749569237093/1243576621722767412/1531206054682165309)
and
[remediation confirmation](https://discord.com/channels/1079907749569237093/1243576621722767412/1531220162190245909)
are linked to their exact Discord messages. The current registry no longer
registers the two PHerc0332 normalized roots; each segment now points to a
valid `tifxyz_original` package. The PHerc0500P2 normalized path was replaced
with a valid 583×339 package.

Fresh TIFXYZ Doctor audits of the three currently registered packages all pass
with zero findings:

| Segment | Current registration | Shape | Portable-valid vertices | Portable-valid faces |
| --- | --- | ---: | ---: | ---: |
| PHerc0332 `20240711124827-20240618142020` | `tifxyz_original` | 126×1,286 | 132,462 | 130,952 |
| PHerc0332 `20240828190516-20240716140050` | `tifxyz_original` | 138×1,266 | 134,880 | 133,373 |
| PHerc0500P2 `20250716055236-z_dbg_gen_00356_inp_hr` | `tifxyz_normalized` | 583×339 | 113,448 | 111,973 |

The machine-readable
[`public-empty-resolution-2026-07-27.json`](../benchmarks/public-empty-resolution-2026-07-27.json)
pins the current registry and replacement object hashes. This confirms the
affected entries were repaired after the diagnostic was reported; it also
means the original sentinel-only condition is no longer a claim about live
packages.

## Executable reader differential v1

The recorded
[`verification/reader-differential-results-v1.json`](../verification/reader-differential-results-v1.json)
satisfies all five declared synthetic fixture expectations: two controls and
three confirmed reader disagreements. The
[`verification harness`](../verification/README.md) executes the exact Python
reader files from the 2026-07-26 audit-date Villa repository HEAD
(`1162bcab4bc769b12993fc320c69c14fdb4a4fa5`) and Villa's real C++
`load_quad_from_tifxyz` path through the official stable Apple Silicon
`vc_tifxyz2obj` release (`05ff9ea39bf1077f8749fc1fe06f43a548350143`).

The executable itself reports that it loaded the complete 5×5 grid. The
16-vertex result is produced later during OBJ emission:

- the converter tests candidate quads through
  [`loc_valid`](https://github.com/ScrollPrize/villa/blob/05ff9ea39bf1077f8749fc1fe06f43a548350143/volume-cartographer/apps/src/vc_tifxyz2obj.cpp#L291-L308);
- that predicate uses a
  [half-open `rows-2` by `cols-2` rectangle](https://github.com/ScrollPrize/villa/blob/05ff9ea39bf1077f8749fc1fe06f43a548350143/volume-cartographer/core/src/Geometry.cpp#L126-L146),
  so a 5×5 grid retains 3×3 quad origins;
- two triangles per retained quad produce 18 faces; and
- only the 4×4 union of corners used by those faces is emitted, producing 16
  OBJ vertices. The final source row and column are not emitted.

This is an exporter boundary effect, not loader resizing or decimation.
Rejecting the center removes four of the nine retained quads, leaving the
known-invalid 12-vertex/10-face signature. The controls therefore calibrate
these observable signatures rather than relabeling OBJ counts as raw C++ valid
grid counts. Eight additional authenticated all-valid probes—five square sizes
from 3×3 through 7×7 and the rectangular sizes 3×4, 4×7, and 7×4—confirm, for
the sampled `H×W` grids with `H,W ≥ 3`, the emitted-count formulas
`(H-1)(W-1)` vertices and `2(H-2)(W-2)` faces.

For exact mask value 1, an exact mask over center `z <= 0`, and a 2×
higher-resolution mask with one bad sample, the pinned Python reader retains
all 25 grid vertices. The C++ output is 12 vertices/10 faces in every case,
matching the known-invalid control, while TIFXYZ Doctor models 25 Python-valid
and 24 finite C++ vertices.

The stable release commit is 94 commits older than the audit-date repository
revision. That separation defines the intentional
stable-release-versus-repository comparison; it is not a same-commit
binary/source comparison. A separate read-only repository-history check found
no Python-reader edit and no change to the relevant C++ full-load mask rules,
and found that the converter and `Geometry.cpp` source blobs are identical at
both commits. The executable harness records those blob identities but does not
itself fetch or compare GitHub source.

## Real-data smoke benchmark v0.1.0

The committed snapshot is
[`benchmarks/realdata-results-v0.1.0.json`](../benchmarks/realdata-results-v0.1.0.json).
It was generated from
[`benchmarks/realdata-smoke.json`](../benchmarks/realdata-smoke.json), whose
SHA-256 in the result file is:

```text
4d8945d37ec01ad9d03a39e71d394147d04bbd8b8e759f224c3379357701dc59
```

Reproduce it with:

```bash
python scripts/fetch_benchmark.py
python scripts/run_benchmark.py
```

The fetch step retrieves 40 objects totaling 1,818,055 bytes and verifies every
byte count and SHA-256. The benchmark then runs the default contract and
geometry configuration recorded in the snapshot.

### Summary

- All 10 packages had zero contract errors.
- The 3 Villa real-fixture controls passed with zero warnings.
- The other 7 cases each produced one
  `metadata-area-schema-divergence` warning: they supply `area_vx2` and/or
  `area_cm2`, but no `area`, so the pinned Python reader leaves
  `Tifxyz.area` unset.
- The three Villa controls and three PHerc0800 format-smoke cases produced no
  geometry review cues.
- The four PHerc1447 hard candidates produced only `enclosed-gaps` cues:
  63 enclosed face-lattice regions totaling 1,105 cells.
- Across all ten cases, the tool evaluated 103,516 portable-valid faces.

### Per-case observations

| Case | Provenance role | Shape | Portable faces | Contract | Review observations |
| --- | --- | ---: | ---: | --- | --- |
| `pherc0172-20241113070770` | Villa real-fixture control | 129×129 | 13,632 | pass, 0 warnings | none |
| `pherc0172-20241113080880` | Villa real-fixture control | 129×129 | 13,458 | pass, 0 warnings | none |
| `pherc0172-20241113090990` | Villa real-fixture control | 129×129 | 13,403 | pass, 0 warnings | none |
| `pherc0800-20251029010146-original` | unverified auto-grown format smoke | 42×42 | 1,369 | warning: area schema | none |
| `pherc0800-20251028220955-original` | unverified auto-grown format smoke | 90×84 | 5,451 | warning: area schema | none |
| `pherc0800-20251028220042-original` | unverified auto-grown format smoke | 93×95 | 5,621 | warning: area schema | none |
| `pherc1447-20250502183138-original` | unverified auto-grown hard candidate | 94×160 | 8,527 | warning: area schema | 16 enclosed regions / 371 cells |
| `pherc1447-20250502180708-original` | unverified auto-grown hard candidate | 109×177 | 11,682 | warning: area schema | 8 enclosed regions / 106 cells |
| `pherc1447-20250502184201-original` | unverified auto-grown hard candidate | 113×180 | 14,671 | warning: area schema | 18 enclosed regions / 192 cells |
| `pherc1447-20250502185519-original` | unverified auto-grown hard candidate | 160×136 | 15,702 | warning: area schema | 21 enclosed regions / 436 cells |

The control label means Villa's pinned tests load the fixture and assert basic
properties; it does not mean every geometry metric has been independently
certified. The auto-grown cases have no official correct/incorrect geometry
label. In particular, an enclosed region can be a deliberate boundary in a
surface rather than a repair target.

## Public empty-artifact regression v0.1.0 (historical)

The three zero-valid normalized packages discovered in the broader scan have a
standalone recorded regression:

- manifest:
  [`benchmarks/public-empty-regressions.json`](../benchmarks/public-empty-regressions.json);
- recorded result:
  [`benchmarks/public-empty-results-v0.1.0.json`](../benchmarks/public-empty-results-v0.1.0.json);
- manifest SHA-256:
  `4d46b20b4e82c061a48c04fc19ac08df4b365c3e092c86183da3146c1bdc9443`.

The committed result was produced from the 81,252 original bytes before the
2026-07-27 remediation. A live refetch is now expected to encounter missing
objects for the retired PHerc0332 normalized roots or a hash mismatch for the
replaced PHerc0500P2 path. If the exact pre-remediation inputs are already
cached, reproduce the checks with:

```bash
python scripts/run_public_regressions.py
```

All 3 of 3 expectations pass in the committed result. Here, “pass” means the
test successfully reproduced the expected semantic-empty-artifact detection:
each source package has `status: error`, shape 2×2, four canonical sentinel
vertices, zero portable-valid vertices, zero portable-valid faces, and both
`no-portable-valid-vertex` and `no-valid-face` findings. One package identifier
contains `z_dbg`; this evidence does not establish whether any entry is an
intentional debug artifact or whether a production workflow uses it.
See the [resolution snapshot](../benchmarks/public-empty-resolution-2026-07-27.json)
for the current passing packages.

## Larger-surface exploratory spot check

A separately supplied local PHerc1667 `w013_7.91um` package was used to check
runtime and report usefulness beyond the small benchmark. This case is not in
the downloadable manifest, so the numbers below are an exploratory observation
rather than part of the reproducible benchmark claim. The full audit completed
in approximately 9.3 seconds on an Apple M1 Pro with 16 GB of memory.

| Observation | Value |
| --- | ---: |
| Stored shape | 634×1,217 |
| Portable-valid faces | 725,986 |
| Contract result | pass, 0 errors, 0 warnings |
| Valid face components | 1 |
| Enclosed face-lattice regions | 2 (9 and 4 cells) |
| Short directional edges below 0.5× observed median | 1 |
| Adjacent-cell normal jumps above 75° | 133 |
| Cells above symmetric stretch 2 | 4 |
| Valid cells colored by at least one review cue | 270 (0.037%) |
| Enclosed-gap cells shown separately | 13 |
| Sampled vertices in proximity pass | 80,897 |
| Sampled nonlocal close-vertex pairs | 0 |

No primary-split folded quads, degenerate triangles, long-edge cues, high-area
distortion cells, or high-shear cells were recorded. Again, the four nonzero
cue families are locations to inspect, not confirmed geometry defects. The
overlay count is a union: cells touched by more than one cue are counted once.

## Broader exploratory corpus scan

A separate discovery pass on 2026-07-26 tested breadth beyond the ten-case
manifest against the
[official `metadata.min.json` registry snapshot](https://vesuvius-challenge-open-data.s3.amazonaws.com/metadata.min.json)
observed with a 2026-07-22 14:17:40 UTC Last-Modified value. Because the
downloaded corpus bytes are not committed to this repository, this section is
an exploratory audit record rather than a fully rerunnable corpus benchmark.
The selection, exact public URLs, hashes, timestamps, aggregates, and caveats
are preserved in
[`benchmarks/public-corpus-scan-2026-07-26.json`](../benchmarks/public-corpus-scan-2026-07-26.json).

The geometry/content pass covered 242 TIFXYZ directories:

- 103 packages downloaded from registered public roots;
- 138 PHerc1203 current/version snapshots; and
- the local PHerc1667 `w013_7.91um` spot-check surface.

Together they contained 8,836,120 stored grid points and 726 coordinate TIFFs.
The downloaded payload was 82,150,850 bytes. All 726 coordinate TIFFs were
single-image `float32`. The pass found no coordinate shape mismatches, partial
sentinels, non-finite coordinates, or bounding-box mismatches among packages
with at least one valid vertex.

It did find three registered normalized roots with no valid vertex or face.
Each is a 2×2 package whose x, y, and z channels are entirely `-1`, with both
declared bounding-box corners equal to `[-1,-1,-1]`:

- [PHerc0332 `20240711124827-20240618142020`](https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0332/segments/20240711124827-20240618142020/mesh/intermediate/tifxyz_normalized/meta.json);
- [PHerc0332 `20240828190516-20240716140050`](https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0332/segments/20240828190516-20240716140050/mesh/intermediate/tifxyz_normalized/meta.json); and
- [PHerc0500P2 `20250716055236-z_dbg_gen_00356_inp_hr`](https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0500P2/segments/20250716055236-z_dbg_gen_00356_inp_hr/mesh/intermediate/tifxyz_normalized/meta.json).

For the third root, the
[registered original counterpart](https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0500P2/segments/20250716055236-z_dbg_gen_00356_inp_hr/mesh/intermediate/tifxyz_original/meta.json)
was nonempty (576×336 with 105,111 valid vertices). This establishes an empty
normalized artifact at the observed URL; it does not by itself establish why
it was published or whether a downstream workflow relies on it. The three
detections are independently covered by the reproducible regression above.
The third identifier contains `z_dbg`, which may indicate a debug artifact;
the audit does not treat that naming hint as proof of intent.

A metadata-only census over 450 registered roots—186 original and 264
normalized—found:

- 277 roots use the literal UUID `output_tifxyz` (all 264 normalized roots and
  13 PHerc0500P2 originals);
- only 174 distinct UUID strings occur across the 450 roots; and
- none of the 450 roots exposed a `mask.tif` object at the probed prefix.

UUID reuse is reported as a collection warning because it defeats uniqueness
for callers that use UUID as an identity key, but it does not make the
coordinate content of each package invalid. The absent masks are an
observation, not an error; `mask.tif` is optional.

All 138 scanned PHerc1203 current/version metadata files supplied suffixed area
fields without `area`. Under the pinned Python reader, those fields remain in
`extra` while `Tifxyz.area` is unset. The tool reports this consistently as an
area-schema warning rather than discarding either representation.

## Interpretation boundary

The strongest demonstrated claims are:

1. source bytes that remain live can be fetched and verified from small pinned
   manifests, while retired inputs remain content-addressed historical
   records;
2. contract results and geometry observations serialize deterministically;
3. official loader fixtures pass without spurious default cues in this small
   control set;
4. the reports retain actionable row/column locations instead of returning
   only aggregate scores; and
5. the exact pre-remediation cache for the three then-registered
   semantic-empty packages reproducibly triggers the two expected contract
   errors, without relying on the repaired live entries.

Claims not supported by this evidence include:

- that every contract rule is an official TIFXYZ specification requirement;
- that every geometry cue should be repaired;
- that a surface with no default cues is globally correct;
- that the proximity pass proves absence of self-intersection; or
- that current thresholds generalize to every scroll, resolution, or pipeline.

Downloaded Vesuvius objects are not committed. They retain the source data
license declared in the manifest,
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The manifest,
code, documentation, and recorded numeric snapshots are MIT licensed.
