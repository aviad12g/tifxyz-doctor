#!/usr/bin/env python3
"""Assemble the compact, credential-free Progress Prize release.

This builder deliberately copies code and compact derived evidence only.  It
does not copy raw CT volumes, rendered 93-layer stacks, model weights, RunPod
artifacts, credentials, or machine-local paths.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable


WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE_DIR = WORKSPACE / "work" / "first-letters"
OUTPUT_DIR = (
    WORKSPACE
    / "outputs"
    / "first-letters-progress-prize"
    / "2026-08"
    / "fail-closed-surface-to-ink-v1"
)

ROOT_MODULES = (
    "finalize_progress_prize_release",
    "inventory_pherc_assets",
    "native_surface_sampler",
    "tifxyz_render_pipeline",
    "render_tifxyz_volume",
    "preflight_debug_m7",
    "m7_calibrated_surface_grower",
    "m7_calibrated_directional_grower",
    "m7_calibrated_ragged_grower",
    "salvage_calibrated_m7_component",
    "full_resolution_tifxyz_preflight",
    "render_gated_tifxyz_raw_stack",
    "independent_raw_ink_audit",
    "audit_tifxyz_self_intersection",
    "audit_tifxyz_exact_increment",
    "build_depth_consensus",
    "same_scroll_spatial_holdout",
    "three_way_same_scroll_falsification",
)

SUPPORT_FILES = (
    "coarse_ink_sources.json",
    "same_scroll_three_way_protocol.json",
)

# These tests intentionally bind multi-megabyte/multi-gigabyte canonical
# workspace fixtures.  The compact release keeps their result in provenance
# but ships only synthetic/self-contained tests.
FIXTURE_BOUND_TESTS = {
    "test_m7_calibrated_directional_grower.py",
    "test_m7_calibrated_ragged_grower.py",
    "test_render_pherc1203_auto_grown_raw_stack.py",
    "test_same_scroll_spatial_holdout.py",
    "test_three_way_same_scroll_falsification.py",
}

EVIDENCE = {
    "campaign-status.md": "outputs/first-letters-geometry/FIRST_LETTERS_CAMPAIGN_STATUS.md",
    "breadth-status.json": "outputs/first-letters-geometry/FIRST_LETTERS_BREADTH_STATUS_2026-08-12.json",
    "pherc0813-stored-validation.json": "outputs/first-letters-geometry/pherc0813-vc3d-growpatch/rank18/calibrated-mask-salvage-v2/validation.json",
    "pherc0813-self-intersection.json": "outputs/first-letters-geometry/pherc0813-vc3d-growpatch/rank18/calibrated-mask-salvage-v2/stored-self-intersection-audit.json",
    "pherc0813-full-resolution-preflight.json": "outputs/first-letters-geometry/pherc0813-vc3d-growpatch/rank18/full-resolution-preflight-v1/full-resolution-preflight.json",
    "pherc0813-raw-stack-manifest.json": "outputs/first-letters-geometry/pherc0813-r18-growpatch-raw-stack-93/raw-stack-manifest.json",
    "pherc0813-independent-audit.json": "outputs/first-letters-geometry/pherc0813-r18-growpatch-independent-ink-audit/audit-report.json",
    "pherc0813-manual-review.json": "outputs/first-letters-geometry/pherc0813-r18-growpatch-independent-ink-audit/manual-review.json",
    "pherc0813-generation80-validation.json": "outputs/first-letters-geometry/pherc0813-vc3d-growpatch/rank18-resume80/calibrated-mask-salvage-v1/validation.json",
    "pherc0813-generation60-to-80-increment.json": "outputs/first-letters-geometry/pherc0813-vc3d-growpatch/rank18-resume80/exact-increment-vs-g60.json",
    "coarse-ml-three-way-summary.json": "outputs/first-letters-geometry/coarse-ink-domain-v1/same-scroll-0139-three-way-v1/three-way-summary.json",
    "pherc1203-canonical-r152-summary.md": "outputs/first-letters-geometry/pherc1203-canonical-r152-origin0-pilot-v1/SUMMARY.md",
    "pherc1203-canonical-r152-order-stability.json": "outputs/first-letters-geometry/pherc1203-canonical-r152-origin0-pilot-v1/extracted/results/origin0-pair-audit.json",
    "pherc1203-canonical-r152-manual-review.json": "outputs/first-letters-geometry/pherc1203-canonical-r152-origin0-pilot-v1/manual-audit/manual-review.json",
    "pherc1203-canonical-r152-contact-sheet.png": "outputs/first-letters-geometry/pherc1203-canonical-r152-origin0-pilot-v1/manual-audit/top-components-contact-sheet.png",
    "pherc0813-seven-offset-contact-sheet.png": "outputs/first-letters-geometry/pherc0813-r18-growpatch-raw-stack-93/diagnostics/selected-offset-contact-sheet.png",
    "pherc0813-ranked-raw-candidates.png": "outputs/first-letters-geometry/pherc0813-r18-growpatch-independent-ink-audit/ranked_rows_contact_sheet.png",
    "pherc0813-top-candidate-depth-controls.png": "outputs/first-letters-geometry/pherc0813-r18-growpatch-independent-ink-audit/row_01_depth_controls.png",
}

FORBIDDEN_TEXT = (
    "/Users/" + "mazalcohen",
    "BEGIN OPENSSH" + " PRIVATE KEY",
    "RUNPOD" + "_API_KEY=",
    "HF" + "_TOKEN=",
    "HUGGING_FACE_HUB" + "_TOKEN=",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module.split(".", 1)[0])
    return {
        name
        for name in imports
        if (SOURCE_DIR / f"{name}.py").is_file()
    }


def dependency_closure(roots: Iterable[str]) -> list[str]:
    pending = list(roots)
    resolved: set[str] = set()
    while pending:
        module = pending.pop()
        if module in resolved:
            continue
        path = SOURCE_DIR / f"{module}.py"
        if not path.is_file():
            raise FileNotFoundError(f"Missing release module: {path}")
        resolved.add(module)
        pending.extend(sorted(local_imports(path) - resolved))
    return sorted(resolved)


def safe_copy(
    source: Path,
    destination: Path,
    *,
    sanitize_machine_paths: bool = False,
) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text_suffix = source.suffix.lower() in {".py", ".md", ".json", ".txt", ".cff"}
    if text_suffix:
        text = source.read_text(encoding="utf-8")
        if sanitize_machine_paths:
            text = text.replace(str(WORKSPACE) + "/", "<WORKSPACE>/")
            text = text.replace("/Users/" + "mazalcohen", "<HOME>")
            text = text.replace("/private/tmp", "<TMP>")
            destination.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(source, destination)
    else:
        shutil.copy2(source, destination)
    if source.resolve() != Path(__file__).resolve() and text_suffix:
        for forbidden in FORBIDDEN_TEXT:
            if forbidden in text:
                raise RuntimeError(f"Forbidden release text {forbidden!r} in {source}")
    return {
        "source": str(source.relative_to(WORKSPACE)),
        "source_sha256": sha256_file(source),
        "path": str(destination.relative_to(OUTPUT_DIR)),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "machine_paths_sanitized": sanitize_machine_paths,
    }


def assemble() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    toolkit = OUTPUT_DIR / "toolkit"
    tests_dir = toolkit / "tests"
    evidence_dir = OUTPUT_DIR / "evidence"
    toolkit.mkdir(exist_ok=True)
    tests_dir.mkdir(exist_ok=True)
    evidence_dir.mkdir(exist_ok=True)

    # The output directory is dedicated to this builder.  Remove only prior
    # generated code/test copies so a removed module cannot survive a rebuild.
    for old in toolkit.glob("*.py"):
        old.unlink()
    for old in tests_dir.glob("*.py"):
        old.unlink()

    modules = dependency_closure((*ROOT_MODULES, "build_progress_prize_release"))
    copied_modules = [
        safe_copy(SOURCE_DIR / f"{module}.py", toolkit / f"{module}.py")
        for module in modules
    ]

    copied_tests: list[dict[str, object]] = []
    for module in modules:
        source = SOURCE_DIR / "tests" / f"test_{module}.py"
        if source.is_file() and source.name not in FIXTURE_BOUND_TESTS:
            copied_tests.append(safe_copy(source, tests_dir / source.name))

    copied_support = [
        safe_copy(SOURCE_DIR / name, toolkit / name) for name in SUPPORT_FILES
    ]

    copied_evidence = [
        safe_copy(
            WORKSPACE / relative,
            evidence_dir / name,
            sanitize_machine_paths=True,
        )
        for name, relative in sorted(EVIDENCE.items())
    ]

    provenance = {
        "schema_version": 1,
        "release": "fail-closed-surface-to-ink-v1",
        "generated_date": "2026-08-12",
        "scope": "compact code, tests, documentation, and derived evidence only",
        "excluded": [
            "raw CT volumes",
            "rendered 93-layer TIFF stacks",
            "model weights",
            "credentials",
            "private endpoints",
            "RunPod artifacts",
        ],
        "root_modules": list(ROOT_MODULES),
        "dependency_module_count": len(modules),
        "test_file_count": len(copied_tests),
        "evidence_file_count": len(copied_evidence),
        "modules": copied_modules,
        "tests": copied_tests,
        "support_files": copied_support,
        "fixture_bound_tests_not_shipped": sorted(FIXTURE_BOUND_TESTS),
        "evidence": copied_evidence,
    }
    provenance_path = OUTPUT_DIR / "RELEASE_PROVENANCE.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the assembled provenance record.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = assemble()
    if args.print_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"assembled {result['dependency_module_count']} modules, "
            f"{result['test_file_count']} tests, and "
            f"{result['evidence_file_count']} evidence files at {OUTPUT_DIR}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
