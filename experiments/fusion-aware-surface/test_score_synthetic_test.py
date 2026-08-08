from score_synthetic_test import pool_counts


def test_pool_counts_forms_rates_from_raw_denominators() -> None:
    rows = [
        {
            "neighbour_sites": 10,
            "detected_neighbour_sites": 8,
            "fused_detected_sites": 4,
            "control_sites": 20,
            "false_split_sites": 2,
        },
        {
            "neighbour_sites": 30,
            "detected_neighbour_sites": 12,
            "fused_detected_sites": 3,
            "control_sites": 10,
            "false_split_sites": 1,
        },
    ]
    pooled = pool_counts(rows)
    assert pooled["site_center_detection_rate"] == 20 / 40
    assert pooled["conditional_fusion_rate"] == 7 / 20
    assert pooled["false_split_rate"] == 3 / 30
