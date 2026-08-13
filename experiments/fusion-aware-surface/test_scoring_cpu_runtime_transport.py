from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_scoring_runtime_uses_exact_resolved_python312_cpu_wheels() -> None:
    source = (HERE / "one_shot_scoring_launcher.py").read_text(encoding="utf-8")
    assert '"torch==2.5.1+cpu"' in source
    assert '"torchvision==0.20.1+cpu"' in source
    assert '"--only-binary=:all:"' in source
    assert '"https://download.pytorch.org/whl/cpu"' in source
    assert '"--extra-index-url"' in source
    assert '"https://pypi.org/simple"' in source
    assert '"https://download.pytorch.org/whl/cu121"' not in source
