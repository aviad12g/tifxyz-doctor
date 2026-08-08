import numpy as np

from fusion_ray_readout import CENTER, K, score_rays


def test_neighbour_detection_and_conditional_fusion_counts() -> None:
    turn = np.zeros((3, K), dtype=np.int16)
    prob = np.zeros((3, K), dtype=np.float32)
    turn[:, CENTER] = 1
    turn[:, CENTER + 3] = 2
    prob[0, CENTER : CENTER + 4] = 0.9
    prob[1, CENTER] = 0.9
    prob[2, CENTER : CENTER + 4] = 0.4
    result = score_rays(prob, turn, 0.5)
    assert result["neighbour_sites"] == 3
    assert result["detected_neighbour_sites"] == 2
    assert result["fused_detected_sites"] == 1
    assert result["site_center_detection_rate"] == 2 / 3
    assert result["conditional_fusion_rate"] == 1 / 2


def test_single_sheet_false_split_matches_contiguous_own_run() -> None:
    turn = np.zeros((2, K), dtype=np.int16)
    prob = np.zeros((2, K), dtype=np.float32)
    turn[:, CENTER - 2 : CENTER + 3] = 1
    prob[0, CENTER - 2 : CENTER + 3] = 0.9
    prob[1, CENTER - 2 : CENTER + 3] = 0.9
    prob[1, CENTER + 1] = 0.1
    result = score_rays(prob, turn, 0.5)
    assert result["control_sites"] == 2
    assert result["false_split_sites"] == 1
    assert result["false_split_rate"] == 0.5
