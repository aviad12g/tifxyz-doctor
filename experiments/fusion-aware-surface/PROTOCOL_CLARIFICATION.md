# Pre-inference clarification: synthetic primary-gate pooling

Status: **FROZEN BEFORE HELD-OUT INFERENCE OR ENDPOINT INSPECTION.**

The original preregistration remains immutable at commit
`5ca0444fb31863c8e02466316bf9e560cf567876`. Its primary synthetic criteria
say that conditional fusion, site-centre detection, and false-split rates are
pooled, with the additional requirement that the fusion delta be negative in
all three matched training seeds. It also says that raw counts are retained
before rates are formed.

The original `score_synthetic_test.py` did not implement that wording exactly:
it applied each numerical tolerance separately to every seed. This was found
during the pre-inference execution audit, before any Scroll-4/5 prediction,
synthetic-test inference, or held-out endpoint was run or inspected.

For the primary preregistered synthetic gates, `score_synthetic_test_v2.py`
therefore applies the written rule mechanically:

1. Apply each run's independently frozen Scroll-1 validation threshold.
2. Within each arm, sum the raw numerators and denominators across matched
   seeds 11, 23, and 47 before forming rates.
3. Require pooled gap8-minus-control conditional fusion to be at most `-0.10`.
4. Separately require the conditional-fusion delta to be strictly negative in
   each of the three matched seeds.
5. Require pooled gap8-minus-control site-centre detection to be at least
   `-0.02`.
6. Require pooled gap8-minus-control false-split rate to be at most `+0.02`.

The original per-seed 10-point fusion, 2-point detection, and 2-point
false-split checks are retained and reported as stricter secondary
diagnostics. They are not substituted for the prose-defined primary gates.

No training configuration, checkpoint, data split, synthetic cell, threshold
selection rule, test cache, real-data gate, panel, or interpretation limit is
changed by this clarification. The clarified scorer and its SHA-256 identity
must be published and bound into the held-out execution plan before test
inference begins.
