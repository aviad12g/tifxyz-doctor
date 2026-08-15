import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_launcher():
    spec = importlib.util.spec_from_file_location(
        "launcher_exception_projection", HERE / "one_shot_scoring_launcher.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_projects_only_bounded_stderr_exception_lines() -> None:
    launcher = load_launcher()
    stderr = (
        b"private scientific text must not appear\n"
        b"ValueError: exact runtime identity mismatch\n"
        + b"X" * 2_000
        + b"Error: oversized\n"
        b"RuntimeError: scorer failed closed\n"
    )
    assert launcher.operational_exception_lines(stderr) == [
        "ValueError: exact runtime identity mismatch",
        "RuntimeError: scorer failed closed",
    ]


def test_launcher_keeps_scientific_stdout_sealed_on_failure() -> None:
    source = (HERE / "one_shot_scoring_launcher.py").read_text(encoding="utf-8")
    assert "operational_exception_lines(completed.stderr)" in source
    assert "completed.stdout" not in source[source.index("if completed.returncode != 0:") : source.index("return result_path")]
