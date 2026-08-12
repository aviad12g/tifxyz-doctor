from __future__ import annotations

import json
import sys
from pathlib import Path

import collect_threshold_freeze as collector
import orchestrate_heldout_queue as queue


def test_collector_downloads_only_two_jsons_without_log(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "threshold_launcher.py"
    source.write_bytes(b"frozen threshold launcher\n")
    monkeypatch.setattr(
        collector, "THRESHOLD_LAUNCHER_SHA256", queue.sha256_file(source)
    )
    monkeypatch.setattr(queue, "kernel_status", lambda _kaggle, _kernel: "COMPLETE")
    monkeypatch.setattr(
        collector.heldout_collector,
        "current_kernel_version_and_source",
        lambda _kernel: (collector.THRESHOLD_KERNEL_VERSION, source.read_bytes()),
    )

    def fake_download(_kaggle: str, destination: Path) -> None:
        destination.mkdir()
        (destination / "frozen_thresholds.json").write_text("{}\n", encoding="utf-8")
        (destination / "threshold_run_manifest.json").write_text(
            "{}\n", encoding="utf-8"
        )

    monkeypatch.setattr(collector, "download_selected", fake_download)
    output = tmp_path / "collected"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_threshold_freeze.py",
            "--expected-source",
            str(source),
            "--out",
            str(output),
        ],
    )
    assert collector.main() == 0
    record = json.loads(
        (output / "threshold_artifact_sources.json").read_text(encoding="utf-8")
    )
    assert [item["file"] for item in record["artifacts"]] == [
        "frozen_thresholds.json",
        "threshold_run_manifest.json",
    ]
    assert [item["relative_path"] for item in record["artifacts"]] == [
        "artifacts/frozen_thresholds.json",
        "artifacts/threshold_run_manifest.json",
    ]
    assert all("path" not in item for item in record["artifacts"])
    assert record["scientific_gate"]["kernel_log_downloaded_or_opened"] is False


def test_download_command_suppresses_automatic_kernel_log(
    tmp_path: Path, monkeypatch
) -> None:
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(collector.subprocess, "run", fake_run)
    collector.download_selected("kaggle", tmp_path / "selected")
    command = observed["command"]
    assert command[command.index("--page-size") + 1] == "100"
    assert command[command.index("--page-token") + 1] == ""
    assert "--force" not in command
