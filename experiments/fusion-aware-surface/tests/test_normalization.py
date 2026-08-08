import numpy as np
import pytest

from normalization import EXPECTED_CT_PROPERTIES, ct_normalize, validate_ct_contract


def test_ct_contract_accepts_pinned_checkpoint_values():
    observed = validate_ct_contract("CTNormalization", EXPECTED_CT_PROPERTIES)
    assert observed == EXPECTED_CT_PROPERTIES


def test_ct_contract_rejects_wrapper_default_and_changed_properties():
    with pytest.raises(ValueError, match="CTNormalization"):
        validate_ct_contract("instance_zscore", EXPECTED_CT_PROPERTIES)
    changed = dict(EXPECTED_CT_PROPERTIES, mean=90.0)
    with pytest.raises(ValueError, match="changed"):
        validate_ct_contract("ct", changed)


def test_ct_normalization_clips_before_scaling():
    mean = EXPECTED_CT_PROPERTIES["mean"]
    image = np.array([0.0, 44.0, mean, 236.0, 255.0], dtype=np.float32)
    result = ct_normalize(image, EXPECTED_CT_PROPERTIES)
    assert result.dtype == np.float32
    assert result[0] == pytest.approx(result[1])
    assert result[2] == pytest.approx(0.0, abs=1e-6)
    assert result[3] == pytest.approx(result[4])
