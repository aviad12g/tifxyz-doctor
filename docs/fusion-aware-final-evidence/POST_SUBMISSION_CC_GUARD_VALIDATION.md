# Post-submission exact-truth validation of the facing-pairs CC guard

Status: **ADDITIVE EXTERNAL VALIDATION RECORDED**

Recorded: 24 August 2026

Scope: interpretation of the synthetic facing-pairs instrument only

This note records evidence published after the August Gap8 submission. It is
strictly additive: it does not modify the frozen Gap8 protocol, thresholds,
predictions, panels, seven-gate verdict, or submission. Gap8 remains a mixed
result that passed five of seven gates. No form was resubmitted.

## Result in one sentence

Jinhojeong independently ran flummoxjr's prospectively specified exact-truth
check on the eight cells of Aviad's physical-fusion pilot and reported that
**97.6% of 94,134 step-3 CC-guard rejections were genuinely different sheets,
while 0 of 2,400 accepted sites were same-sheet false accepts**.

This supports the interpretation that, in the tight-pitch synthetic cells, the
guard mostly rejects real two-sheet contacts that the reconstruction-style
label locally merges. It validates that bounded instrument interpretation. It
does **not** turn Gap8 into a seven-gate success, establish real-scroll efficacy,
certify PHerc1218, unblock GapBalance, or justify retuning any experiment.

## Exact public chain of custody

### Public validation statement

- Author: `Jinhojeong`
- GitHub Issue comment: [ScrollPrize/villa #191 comment 5394005579](https://github.com/ScrollPrize/villa/issues/191#issuecomment-5394005579)
- Created and last updated: `2026-08-24T10:33:40Z`
- Author association reported by the GitHub API: `CONTRIBUTOR`

The comment states that the harness self-test passed, the port verifier was
rerun and found byte-identical output to the archived original, and the eight
pilot cells passed Aviad's shipped validator `156/156`. It also identifies the
run settings as `--gt-instance turn_id`, `--log-rejected`, and
`--seed-mode caseid`.

### Eight-cell physical-fusion pilot identity

The input manifest identity was published earlier in the same issue by Aviad:

- Manifest SHA-256:
  `506f3353ba834bcdb839cf4000d2ec6b1a18e074ce1fb9c10dce24696e7299dd`
- Original public binding:
  [ScrollPrize/villa #191 comment 5165010848](https://github.com/ScrollPrize/villa/issues/191#issuecomment-5165010848)
- Independent earlier manifest/hash check:
  [ScrollPrize/villa #191 comment 5168228180](https://github.com/ScrollPrize/villa/issues/191#issuecomment-5168228180)

Each pilot cell carries `prob`, `gt_surface`, and per-voxel `turn_id`; the new
check uses `turn_id` as exact sheet identity by construction.

### Published result package

Repository: [Jinhojeong/vesuvius-surface-geometry-diagnostic](https://github.com/Jinhojeong/vesuvius-surface-geometry-diagnostic)

Exact commit: [`9cfcaa8eb126264b85612f467ea20154a65dd7c3`](https://github.com/Jinhojeong/vesuvius-surface-geometry-diagnostic/commit/9cfcaa8eb126264b85612f467ea20154a65dd7c3)

| file | bytes | Git blob SHA-1 | SHA-256 |
|---|---:|---|---|
| `results/ccguard_pilot/RESULTS.md` | 3,030 | `b563121f697f0a4bfda03a06f561558f2e27468d` | `8a0d1cdc13c972df13ec7ab39f92453501c0b21efd4b4f2cde00fc3db0a6806a` |
| `results/ccguard_pilot/facing_pairs_case_diag.csv` | 711 | `847286d0dd28bb0dddefa117e1389f5050027f00` | `66e19d10f04d61509826c5e402f8a4fc348a8ffb807773c61d5b954ce819bd73` |
| `results/ccguard_pilot/fp_score.py` | 2,240 | `22f8a5facdfcbbd12d4b05e8c7cf14caf1f4b16b` | `276129cdf2367986375ba998a65872fecde08e3029c992dce759ef4e194cdd6c` |

### Published instrument package

Repository: [flummoxjr/facing-pairs-harness](https://github.com/flummoxjr/facing-pairs-harness)

Exact commit: [`0c5ed007cf6c875c9712cec8677528952a64c14f`](https://github.com/flummoxjr/facing-pairs-harness/commit/0c5ed007cf6c875c9712cec8677528952a64c14f)

| file | bytes | Git blob SHA-1 | SHA-256 |
|---|---:|---|---|
| `README.md` | 11,194 | `78d572708baf9ca26d05ddd494f37570e6ec1345` | `9792e6c5efee0573cc769a9fbe0c603235bc30f547eb3de573f1c564b8bceb6b` |
| `facing_pairs.py` | 24,059 | `e209dd2eee529239f97740420908c6f264fecfc9` | `86336bf7067c1384914505ad2f53c4d9e1ed5b258ef749c1507bf91238463d19` |
| `self_test.py` | 6,978 | `bc0a588a8691b059203e0322b3acc418dd7fb166` | `383705571780ab7e11fe01dae5d9a6636f2d43bb528af3c1abe239fee669013b` |
| `verify_against_original.py` | 8,929 | `9b6e19c0fbbc93285c416c2211b25829e2567b3f` | `383888d289128b499f1893ae281e160f4874bd7831c7175d3e3d61bddaa7cba8` |
| `verify_report.json` | 1,098 | `b7d6aaa3e5ad847ee82cd2bba4a0d62508fd98c1` | `d6f76e5d70103c1bb9c4c72a21bd65fb5b051e7bdadab35ad6f06d9b4914a911` |
| `original/extract_fusion_sites.py` | 10,897 | `c03bfb1b52d2f003d34887194ae60a5fe0108e34` | `209b0141e10768642d537fb88d4324c7f7c6fa248c5896c9608d6c54e4027ab9` |
| `requirements.txt` | 27 | `b2c8174712815cd13bff215d4cd4238b180df145` | `cf4d1dc9a101f290fca565d8a6fb4f3ca3f1ada0912c93a81bf37f3fcdf20973` |

The harness package's own published verification report records `1,200/1,200`
synthetic site rows and all diagnostic rows as identical to the archived
original. Its README additionally reports `342/342` rows identical on a real
Dataset059 prefix. Those are package claims, not reruns performed for this
supplement.

## Reported exact-truth results

| d12 bin | rejected pairs | reported genuinely different sheets | accepted sites | false accepts |
|---|---:|---:|---:|---:|
| 2 to 3 voxels | 93,205 | 97.6% | 790 | 0 |
| 3 to 4 voxels | 913 | 99.2% | 954 | 0 |
| 4 to 5 voxels | 16 | 0.0% | 308 | 0 |
| 5 to 6.01 voxels | 0 | not applicable | 348 | 0 |
| **All** | **94,134** | **97.6%** | **2,400** | **0 (0.00%)** |

The public table arithmetic is internally consistent:

- rejected pairs: `93,205 + 913 + 16 = 94,134`;
- accepted sites: `790 + 954 + 308 + 348 = 2,400`; and
- reported false accepts: `0 / 2,400 = 0.00%`.

## The pitch-300 caveat is part of the result

Jinho reports that at pitch 300, both seeds have a `0.0%` genuinely-different
fraction among the `941` rejected pairs (`470 + 471`). At that loose pitch the
two sheets are too far apart to form tight facing pairs; the tight candidates
are instead one sheet curving toward itself. The CC guard rejects them
correctly under `turn_id` truth because both endpoints belong to the same turn.

Therefore, a rejection does not have one invariant meaning across all
geometries:

- at tight pitch, it predominantly indicates a true two-sheet contact merged
  by the local reconstruction label; and
- at loose pitch, it can indicate self-facing curvature on one sheet.

The `4–5` voxel row has only 16 rejected pairs, all from pitch-300 cells, so its
`0.0%` value must not be generalized as guard failure.

## Verification performed for this supplement

This repository's audit did the following without reopening scientific
endpoints:

1. resolved the exact GitHub comment identity and timestamp through the GitHub
   API;
2. cloned both public repositories and checked out the exact commits above;
3. computed the listed byte sizes, Git blob SHA-1 values, and SHA-256 values;
4. checked the arithmetic of the published bin totals; and
5. bound this note to the original full eight-cell manifest SHA-256; and
6. passed the repository's full local suite: `480 passed`.

It did **not** rerun Jinho's scientific scoring. The published
`fp_score.py` refers to two generated per-pair files,
`facing_pairs_cc_rejected.csv` and `facing_pairs_sites.csv`, but those full
files are not present in the bound `results/ccguard_pilot` directory. The
public `facing_pairs_case_diag.csv` contains per-case funnel counts, not the
per-pair `gt_diff_sheet` values required to recompute 97.6%. Accordingly:

- the hashes, commits, comment identity, and displayed arithmetic are locally
  verified;
- the scientific percentages are accurately transcribed third-party results;
  and
- independent public recomputation of those percentages requires regenerating
  the omitted per-pair CSVs from the hash-bound harness and eight pilot cells.

## What this changes—and what it does not

### Supported additive conclusion

The new exact-truth run strengthens the mechanism-level interpretation of the
facing-pairs CC guard used to diagnose synthetic fusion. In the tested tight
synthetic geometries, the guard's step-3 rejections overwhelmingly correspond
to genuinely distinct sheets, while the accepted population showed no
same-sheet contamination.

### Unchanged conclusions and gates

- Gap8 remains **5/7 gates passed**.
- Gap8's adverse detection and false-split findings remain unchanged.
- The fixed-panel assessment remains unchanged.
- No threshold, candidate, prediction, panel, or metric was retuned.
- No PHerc1218 version was accessed or certified by this supplement.
- GapBalance remains blocked pending its separately frozen prerequisites.
- No Progress Prize or other form was resubmitted.

The machine-readable companion is
`post_submission_cc_guard_validation.json`.
