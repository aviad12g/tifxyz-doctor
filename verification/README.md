# Reader differential verification

This harness converts the source-derived mask model into an executable,
dated stable-release-versus-repository differential.

It executes:

- the exact Villa Python `TifxyzReader` files from commit
  `1162bcab4bc769b12993fc320c69c14fdb4a4fa5`; and
- the official Apple Silicon `vc_tifxyz2obj` binary from Villa's `stable`
  release at commit `05ff9ea39bf1077f8749fc1fe06f43a548350143`.

The official release asset is
`VC3D-05ff9ea-2026-07-12-macos-arm64.zip`, published with SHA-256
`d5fca566c8bc843f2350ab60e86e0def25ab88240e3f0ffcbd04662ba522c9a8`.
The packaged `vc_tifxyz2obj` executable is independently pinned to SHA-256
`71a32235924499dee93bd24944794c930dffdee8e95450220e2665bd2020e118`.
The harness requires both files and rejects either one before execution if its
hash differs.
The executable calls Villa's real `load_quad_from_tifxyz` path before writing
the OBJ; the harness also requires that loader symbol to be present in the
packaged executable's undefined-symbol table.

## Why a 5×5 input emits 16 OBJ vertices

The all-valid fixture contains 25 grid samples and 16 possible quads. The
stable executable confirms that the loader retains the full array:

```text
Point dims: [5 x 5] cols: 5 rows: 5
```

The reduction happens during OBJ emission, not loading or decimation. The
release converter
[iterates candidate quad origins and calls `loc_valid`](https://github.com/ScrollPrize/villa/blob/05ff9ea39bf1077f8749fc1fe06f43a548350143/volume-cartographer/apps/src/vc_tifxyz2obj.cpp#L291-L308).
At this release,
[`loc_valid` constructs a half-open rectangle with width `rows-2` and height
`cols-2`](https://github.com/ScrollPrize/villa/blob/05ff9ea39bf1077f8749fc1fe06f43a548350143/volume-cartographer/core/src/Geometry.cpp#L126-L146).
For a square 5×5 input, only origins 0–2 pass in each direction:

- 3×3 retained quads × 2 triangles = 18 OBJ faces;
- the retained quads touch a 4×4 corner union = 16 emitted OBJ vertices; and
- source row and column 4 never enter an emitted face.

When the center sample is rejected, four of those nine retained quads fail.
Five quads remain, producing 10 faces and 12 referenced vertices. The harness
therefore compares each disputed case with this known-invalid control rather
than treating an OBJ vertex count as a raw C++ valid-grid count. The
machine-readable result records the 5×5 loaded dimensions for every fixture.

The same authenticated harness probes five all-valid square sizes from 3×3
through 7×7 plus the rectangular sizes 3×4, 4×7, and 7×4. For all-valid `H×W`
grids with `H,W ≥ 3`, those eight observations confirm the mapping
`(H-1)(W-1)` referenced vertices and `2(H-2)(W-2)` faces. They are stored
alongside the reader fixtures in the machine-readable result.

## Version pairing

As observed on 2026-07-26, the authenticated official stable Apple Silicon
binary and the exact Python reader implementation at audit-date repository HEAD
were simultaneously available version points. The stable release commit
predates the pinned repository revision by 94 commits. That separation defines
the tested stable-release-versus-repository comparison; it is not a same-commit
binary/source comparison.

A separate read-only GitHub path-history check found no Python-reader edit and
no change to the relevant C++ full-load `z <= 0` gate, canonical `uint8` mask
threshold, or integer-multiple mask application. It also found that the exact
converter blob
`eff32a4d0984bd6614308a4484f8dd70cb699d1a` and `Geometry.cpp` blob
`edfec27becdda604281ad3d2feae26630d4ea62a` are identical at the stable and
pinned repository commits. The harness records these identities but does not
itself fetch or compare GitHub source.

Run from the repository root with a Python environment containing NumPy,
Pillow, and tifffile:

```bash
python verification/run_reader_differential.py \
  --villa-root ../villa \
  --cpp-binary ../villa-release/extracted/VC3D.app/Contents/MacOS/vc_tifxyz2obj \
  --cpp-asset ../villa-release/VC3D-05ff9ea-2026-07-12-macos-arm64.zip \
  --output verification/reader-differential-results-v1.json
```

The five cases include two controls plus exact-mask value `1`, an exact mask
over a nonpositive `z`, and an integer-multiple higher-resolution mask. The
result is successful only when the actual Python reader, actual C++ loader
path, TIFXYZ Doctor model, loaded 5×5 dimensions, and declared expectations all
agree. Passing all five expectations means the two controls behave as declared
and the three disagreements are confirmed; it does not mean the readers agree.
