# Source-built PHerc1447 confirmation

## Result

The source-built four-way experiment confirms the earlier release-binary
ablation on the same public PHerc1447 fixture.

| Variant | Exact source | Resume points | Output shape | Valid vertices | Valid quads |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | `208faea7d7fec62da03ff75e118e26be0c08a117` | 15,696 | 158 x 134 | 15,692 | 15,074 |
| Helper only | `d5226910817be064d2ae1577f1fae0f850acc8c7` | 15,702 | 159 x 135 | 15,692 | 15,074 |
| Minimal full bounds | `9e15071df41a139ce77ada6aa64367e000387c5d` | 15,702 | 159 x 135 | 15,698 | 15,078 |
| Actual PR | `6e2bba940d8f93b53b0d1bcf7f66af19c5b81295` | 15,702 | 159 x 135 | 15,698 | 15,078 |

Every variant produced identical scientifically relevant output across three
replicates. The minimal full-bounds build and actual PR have identical X, Y,
Z, generation, and valid-mask hashes.

That equality applies to every recorded scientifically relevant output listed
above for this fixture and bounded configuration. Generated UUID/path metadata
and complete stdout bytes are intentionally excluded. This does not imply that
the PR's additional finite-location, sentinel, and normal-guard changes are
unnecessary on other inputs.

Helper-only changes the resume count and saved extent but not the valid output.
Adding only the `SurfTrackerData.cpp` synchronization recovers six valid
vertices and four valid quads:

```text
[51,134] [52,134] [53,134] [158,88] [158,89] [158,90]
```

One adjacent common point at `[51,133]` moves by `0.001953125` stored
coordinate units. Every other common XYZ value is unchanged.

This proves a deterministic production-tracer output change on a real public
fixture. The fixture is an unverified auto-grown candidate, so the experiment
does not establish that the recovered points improve scientific accuracy.

## Public execution

- Run:
  <https://github.com/aviad12g/villa/actions/runs/30445977638>
- Job:
  <https://github.com/aviad12g/villa/actions/runs/30445977638/job/90556330511>
- Original compact artifact:
  <https://github.com/aviad12g/villa/actions/runs/30445977638/artifacts/8721453558>
- Workflow/run commit:
  `3a96f08b305b1446aaefb13dbb936acf2ebfcfb1`
- Uploaded artifact ZIP SHA-256:
  `0d734809b454ad2b965f46d9223cbac34cb622e814835dbde09147e974508eb7`
- Evidence runner SHA-256:
  `38d94ef9de01782c44b427f11df6ab4bf8c242d510ed8c24c8fb37090da18133`
- Workflow SHA-256:
  `dd7c28d479cd866541a91b1f73d38a242048a5e31d815eb87cc4b30e955b7262`

`status_during_evaluation` is `in_progress` because the evaluator queried its
own run before that job could finish. The linked job subsequently completed
successfully with every step passing.

The four build logs resolve the same GitHub runner/image, Dockerfile frontend,
Villa builder image, GNU 15.2.0 toolchain, Ninja preset, Qt 6.10.2, and
packaging-tool versions. This is controlled same-toolchain evidence, not a
claim of bit-for-bit hermetic builds: the build workflow still used live APT
and continuously published packaging downloads.

## Files

- `preregistration.json`: fixture hashes and the six boundary cells recorded
  before the trace comparison
- `summary.json`: compact aggregate results and causal comparisons
- `manifest.json`: build/run provenance, all 12 replicate records, integrity
  checks, source comparisons, and output hashes
- `index.json`: SHA-256 and size index for the three evidence JSON files
- `vc3d_trace_evidence.py`: exact evaluator used by the successful run
- `vc3d-trace-confirm.yml`: exact GitHub Actions workflow used by the run

This directory preserves the compact evidence and exact evaluator. The 123 MB
AppImages are not redistributed and remain subject to GitHub artifact
retention. Rebuilding after expiry may not be bit-for-bit identical because the
build used live APT and continuously published packaging tools.
