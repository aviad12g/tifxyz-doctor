from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_scorer_exception_freezer_is_result_blind() -> None:
    source = (HERE / "freeze_scorer_exception_projection.py").read_text(encoding="utf-8")
    assert '"stream": "stderr"' in source
    assert '"maximum_exception_lines": 3' in source
    assert 'plan["one_shot_scoring_stager"] = identity(args.stager)' in source
    assert '"scientific_stdout_projected": False' in source
    assert '"scorer_source_changed": False' in source
    assert '"held_out_result_opened_or_used": False' in source
