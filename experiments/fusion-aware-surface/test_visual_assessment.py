from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import record_visual_assessment as recorder
import validate_final_results as validator
from PIL import Image


def _write_hashed(path: Path, payload: dict) -> dict:
    result = dict(payload)
    result["payload_sha256"] = recorder.canonical_sha256(result)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _identity(path: Path, payload: dict | None = None) -> dict:
    record = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": recorder.sha256_file(path),
    }
    if payload is not None:
        record["payload_sha256"] = payload["payload_sha256"]
    return record


def test_all_fixed_panel_comparisons_are_recorded_and_recomputed(
    tmp_path: Path, monkeypatch
) -> None:
    thresholds_path = tmp_path / "frozen_thresholds.json"
    thresholds = _write_hashed(thresholds_path, {"schema_version": "1.0"})
    source_path = tmp_path / "real_panel_manifest.json"
    selected = [
        {"image": f"scroll_4_{index:03d}.tif", "center_zyx": [50, 50, 50]} for index in range(1, 5)
    ]
    source = _write_hashed(
        source_path,
        {
            "schema_version": "1.0",
            "status": "model-blind; selected from held-out labels before prediction",
            "selected": selected,
        },
    )
    images = []
    for index in range(1, 5):
        path = tmp_path / f"real_panel_{index:02d}.png"
        Image.new("RGB", (700, 2349), (index, 0, 0)).save(path, format="PNG")
        images.append(path)
    renderer = Path(__file__).resolve().with_name("render_real_panels.py")
    panel_records = []
    for index, (selected_record, image) in enumerate(zip(selected, images, strict=True), start=1):
        cache_name = Path(selected_record["image"]).with_suffix(".npz").name
        panel_records.append(
            {
                "panel_index": index,
                "image": selected_record["image"],
                "cache_file": cache_name,
                "center_zyx": selected_record["center_zyx"],
                "frame_zyx": {"normal": [1, 0, 0]},
                "source_caches": {
                    run: {"file": cache_name, "bytes": 1, "sha256": "a" * 64}
                    for run in validator.RUN_ORDER
                },
                "output": _identity(image) | {"width": 700, "height": 2349},
            }
        )
    render_path = tmp_path / "real_panel_render_manifest.json"
    _write_hashed(
        render_path,
        {
            "schema_version": "1.0",
            "status": "four model-blind real panels rendered after sealed test delivery",
            "source_thresholds": _identity(thresholds_path, thresholds),
            "source_panel_manifest": _identity(source_path, source),
            "rendering": {},
            "panels": panel_records,
            "scientific_gate": {
                "panel_locations_selected_before_predictions": True,
                "selected_thresholds_used_unchanged": True,
                "panel_locations_or_plane_orientation_tuned_after_predictions": False,
                "visual_gate_assessed_by_renderer": False,
            },
            "renderer": _identity(renderer),
        },
    )
    observations_path = tmp_path / "visual_observations.json"
    comparisons = [
        {
            "panel_index": panel_index,
            "seed": seed,
            "separation_improvement": panel_index == 1 and seed == 11,
            "new_nearby_break": False,
            "notes": "fixed comparison reviewed",
        }
        for panel_index in recorder.PANEL_INDICES
        for seed in recorder.SEEDS
    ]
    observations_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "all fixed real-panel comparisons assessed",
                "assessor": "fixture assessor",
                "assessment_method": "complete fixed-panel visual comparison",
                "criterion": recorder.CRITERION,
                "comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assessment_path = tmp_path / "visual_assessment.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_visual_assessment.py",
            "--panel-render-manifest",
            str(render_path),
            *[item for path in images for item in ("--panel-image", str(path))],
            "--observations",
            str(observations_path),
            "--out",
            str(assessment_path),
        ],
    )
    assert recorder.main() == 0
    plan = {
        "threshold_binding": {"frozen_thresholds": _identity(thresholds_path, thresholds)},
        "real_panel_manifest": _identity(source_path, source),
        "real_panel_renderer": _identity(renderer),
        "visual_assessment_recorder": _identity(Path(recorder.__file__).resolve()),
    }
    validator.validate_real_panel_artifacts(
        plan=plan,
        thresholds=thresholds,
        source_panel_path=source_path,
        render_manifest_path=render_path,
        image_paths=images,
    )
    assessment = validator.validate_visual_assessment(
        plan=plan,
        assessment_path=assessment_path,
        observations_path=observations_path,
        render_manifest_path=render_path,
        image_paths=images,
    )
    assert assessment["visual_gate_pass"] is True

    tampered = dict(assessment)
    tampered["comparisons"] = [dict(record) for record in assessment["comparisons"]]
    tampered["comparisons"][0]["comparison_pass"] = False
    tampered.pop("payload_sha256")
    tampered["payload_sha256"] = recorder.canonical_sha256(tampered)
    tampered_path = tmp_path / "tampered_visual_assessment.json"
    tampered_path.write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="comparison gate mismatch"):
        validator.validate_visual_assessment(
            plan=plan,
            assessment_path=tampered_path,
            observations_path=observations_path,
            render_manifest_path=render_path,
            image_paths=images,
        )
