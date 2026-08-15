# PHerc1203 canonical ResNet-152 origin-0 pilot

## Decision

**Stop this PHerc1203 detector branch.** The two canonical middle-window maps pass integrity and coverage gates but fail depth-order stability decisively. No shifted-origin runs were performed, and no output is a defensible text candidate approaching the First Letters requirement of at least 10 visible and legible letters within one 4 cm² area.

## Verified execution

- Approved input archive: `pherc1203-canonical-r152-middle-v1.tar.gz`, 27,087,137 bytes, SHA-256 `4e97b157e663758cff9c835d94f92baf73f6c4488de324e5d0be9d749df0cb7c`.
- Model: `scrollprize/ink_canonical_2um` at revision `075855bc69317ef6febf39a0d9d687b27d2b7c29`; checkpoint SHA-256 `36dd0de84b7b7aa6590184192c7415466cd8a1ba7c1e59f42c6373846373c3e0`.
- Safe checkpoint load: `weights_only=True`, zero missing keys, zero unexpected keys.
- Forward: 2,466 tiles, 96.401446% valid-mask coverage, 169.814 seconds.
- Reverse: 2,466 tiles, 96.401446% valid-mask coverage, 156.142 seconds.
- Downloaded evidence archive: `pherc1203-canonical-r152-origin0-results-v1.tar.gz`, 20,053,489 bytes, SHA-256 `2fe5c8a9f7a4ffa7f831f66e0ac1b6321b44af4324b4ba0fb0981da03c737b89`; all internal hashes pass.

## Order-stability result

| Measure | Result |
|---|---:|
| Forward/reverse Pearson correlation | 0.032425 |
| Top 5% Jaccard | 0.049263 |
| Top 1% Jaccard | 0.003361 |
| Top 0.5% Jaccard | 0.000000 |
| Top 0.1% Jaccard | 0.000000 |
| Forward pixels ≥0.5 | 729 |
| Reverse pixels ≥0.5 | 926 |
| Pixels ≥0.9 | 0 in either map |

The strongest locations move under depth reversal. The top-component contact sheet places the surviving bright blobs on fiber crossings, cracks/void margins, or mask-hole boundaries; it shows no stable baseline, repeated glyph scale, or legible sequence.

## Cost and safety

- The first community RTX 3090 host was terminated before model download because PyTorch could not initialize CUDA despite `nvidia-smi` visibility.
- The successful secure A40 run was stopped and terminated after the evidence download.
- Both pod IDs now resolve absent; no persistent volume was used.
- Aggregate estimated compute cost: **$0.16337** from pod wall time and advertised rates (not an API billing export), below the authorized $0.60.
- Aggregate pod wall time: **1,508.99 seconds** (25.15 minutes), below the authorized 90 minutes.

## Scientific limitation

This checkpoint is documented for approximately 2 µm CT, while PHerc1203 is 9.362 µm. Physical interpolation aligns the sampled span and tile width but cannot reconstruct ink information absent from the coarser scan. The negative order-stability result is therefore a reason to stop this branch, not proof that the scroll contains no text.
