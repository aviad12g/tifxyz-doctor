# GapBalance blocked-resume runbook

Status: **prepared but blocked**. This runbook does not supersede
`PHERC1218_HOLDOUT_QUARANTINE.json`, authorize compute, or authorize access to
scientific endpoints or confirmation inputs.

## Current mechanical state

- All six Gap2/Gap4 trainings and the three matched controls are frozen.
- The 24-patch real development cache is complete and mechanically verified.
- Synthetic seed 11 shards 0--3 are complete and mechanically verified.
- Eight uniform-Kaggle synthetic development shards remain: seed 23 shards
  0--3, followed by seed 47 shards 0--3.
- No development endpoint has been scored and no confirmation input has been
  opened.

`GAPBALANCE_KAGGLE_RESUME_DRAFT.json` binds the exact completed and pending
Kaggle job identities. It remains non-executable while the quarantine is
active.

## Before any compute

1. Obtain a corrected contact release and a supplemental control release with
   exact provider version, dataset id, manifest byte count, and manifest
   SHA-256.
2. Run `audit_gapbalance_supplemental_controls.py` on the control manifest.
   The manifest must expose `source_slab` and `center_z_level1`; at least 48
   controls must satisfy `ct_empty_frac <= 0.10`, span at least four source
   slabs, avoid more than 35% from one slab, and cover at least half the
   corrected contacts' z range.
3. Perform the complete metadata-only corrected-contact audit, explicitly
   describing native level-1 17.28-um CT as cross-resolution confirmation.
   Do not create a level-0 derivative or reinterpret the old voxel comparison.
4. Publish an additive, hash-bound supersession freezing the corrected contact
   and control identities, eligibility lists, fixed panels, native resolution,
   and unchanged GapBalance gates. Until that commit exists, stop.
5. Publish the finalized uniform-Kaggle resume record. Its only allowed change
   from the draft is filling release identities, activation commit identity,
   and provider-confirmed versions; job science and ordering cannot change.

## Result-blind development completion

Launch at most two free Kaggle jobs at once in the frozen order:

1. seed23 q0 + q1;
2. seed23 q2 + q3;
3. seed47 q0 + q1;
4. seed47 q2 + q3.

After each completion, download to a new absent directory and mechanically
require exactly 60 NPZ files, three cache manifests, and one job index. Hash
every file, verify job/run/input identities, and do not open or score NPZ
scientific contents. Any failed job is handled from operational logs only.

## One-shot selection and confirmation

Only after all twelve synthetic shards and the real cache are mechanically
complete:

1. run the already-frozen scorer exactly once;
2. apply the unchanged threshold grid and five eligibility gates;
3. if neither arm is eligible, publish the stop and keep confirmation sealed;
4. otherwise publicly freeze exactly one arm and every threshold;
5. only then stage PHerc and synthetic seeds 500--504 for one-shot sealed
   confirmation;
6. run fixed panels and every gate without replacement or relaxation; and
7. publish every positive, adverse, null, and operational-failure outcome.

`verify_gapbalance_resume_readiness.py` checks this package without launching,
scoring, or opening confirmation data. Its expected status while waiting is
`BLOCKED_PACKAGE_READY_FOR_LATER_HOLDOUT_REASSESSMENT`.
