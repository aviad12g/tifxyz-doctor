from cache_synthetic_rays import synthetic_cells


def test_synthetic_test_grid_is_complete_and_unique() -> None:
    cells = synthetic_cells()
    assert len(cells) == 100
    assert len({cell["name"] for cell in cells}) == 100
    primary = [cell for cell in cells if cell["kind"] == "primary"]
    controls = [cell for cell in cells if cell["kind"] == "single_sheet_control"]
    assert len(primary) == 80
    assert len(controls) == 20
    assert all(cell["kollesis"] for cell in primary)
    assert all(not cell["kollesis"] and cell["pitch_um"] == 700 for cell in controls)


def test_ten_frozen_shards_each_have_ten_cells() -> None:
    cells = synthetic_cells()
    assert [sum(index % 10 == shard for index in range(len(cells))) for shard in range(10)] == [10] * 10
