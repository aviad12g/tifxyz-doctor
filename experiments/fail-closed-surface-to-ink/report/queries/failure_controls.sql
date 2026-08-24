-- Frozen release snapshot derived from FAILURE_MODE_CATALOG.md and its
-- hash-bound evidence references.
SELECT 1 AS "order", 'GrowPatch seed drift' AS failure_mode,
       'Fresh public-m7 samples along recomputed physical normals' AS control,
       'Smooth patches left the seeded sheet immediately or during growth' AS observed_result
UNION ALL SELECT 2, 'Whole-area inflation',
       'Strict accepted-component area and exact quad increment',
       'PHerc0813 proposal 2.13→10.10 cm²; accepted 0.509→0.536 cm²'
UNION ALL SELECT 3, 'Duplicate continuation',
       'Exact serialized coordinate and quad identity',
       'Generation 60 and 80 accepted components byte-identical'
UNION ALL SELECT 4, 'Exact-one-run veto',
       'Published clean-surface calibration',
       '399/400 clean control transects were multi-run'
UNION ALL SELECT 5, 'Fold / normal flip / self-overlap',
       'Native quad geometry, topology, and exact non-adjacent intersection',
       'Rejected invalid surfaces before raw rendering'
UNION ALL SELECT 6, 'XYZ↔ZYX or sign/order confusion',
       'Synthetic axis tests and published-stack calibration',
       'Median Pearson ≈0.9972 and mask IoU 1.0 on calibrated replay'
UNION ALL SELECT 7, 'Fiber/crack/fold/void false positives',
       'Seven-depth raw review and fiber/specificity maps',
       'PHerc0813 nominated six regions; 0 defensible or borderline'
UNION ALL SELECT 8, 'Tiling seams',
       'Coverage and grid-boundary gradient controls',
       'Observed 15.6×–25.4× boundary gradients in one screen'
UNION ALL SELECT 9, 'Orientation instability',
       'Co-equal forward/reverse maps and linked-region overlap',
       'Putative regions relocated under depth reversal'
UNION ALL SELECT 10, 'Training-set leakage',
       'Model-card/revision audit',
       'A supposed held-out replay segment appeared in model finetuning data'
UNION ALL SELECT 11, 'Local label memorization',
       'Three seeds and sealed train/local/distant spatial partitions',
       'Train AUROC 0.996; local AUROC 0.512 only 1.228 mm away';
