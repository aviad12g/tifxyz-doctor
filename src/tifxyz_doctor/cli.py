"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ._version import __version__
from .audit import AuditConfig, audit_mesh, public_report
from .integrity import audit_tifxyz_integrity, dumps_integrity_report
from .io import load_tifxyz
from .report import write_html, write_json, write_overlay


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tifxyz-doctor",
        description="Deterministic integrity checks and geometry review cues for TIFXYZ.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check",
        help="Check raw format integrity and Python/C++ interoperability",
    )
    check.add_argument("path", type=Path)
    check.add_argument("--json", dest="json_path", type=Path)
    check.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Also exit 2 when the integrity report contains warnings.",
    )

    scan = subparsers.add_parser(
        "scan",
        help="Run integrity checks over TIFXYZ directories below a root",
    )
    scan.add_argument("root", type=Path)
    scan.add_argument("--json", dest="json_path", type=Path)
    scan.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Also exit 2 when any report contains warnings.",
    )

    audit = subparsers.add_parser("audit", help="Audit one TIFXYZ directory")
    audit.add_argument("path", type=Path)
    audit.add_argument("--json", dest="json_path", type=Path)
    audit.add_argument("--html", dest="html_path", type=Path)
    audit.add_argument("--overlay", dest="overlay_path", type=Path)
    audit.add_argument("--expected-spacing-x", type=float)
    audit.add_argument("--expected-spacing-y", type=float)
    audit.add_argument("--long-edge-ratio", type=float, default=2.0)
    audit.add_argument("--short-edge-ratio", type=float, default=0.5)
    audit.add_argument("--normal-jump-degrees", type=float, default=75.0)
    audit.add_argument("--condition-number", type=float, default=4.0)
    audit.add_argument("--symmetric-stretch", type=float, default=2.0)
    audit.add_argument("--symmetric-dirichlet", type=float, default=20.0)
    audit.add_argument("--area-ratio-low", type=float, default=0.25)
    audit.add_argument("--area-ratio-high", type=float, default=4.0)
    audit.add_argument("--shear", type=float, default=0.8660254037844387)
    audit.add_argument("--normal-step-ratio", type=float, default=0.25)
    audit.add_argument("--normal-step-min-component-cells", type=int, default=8)
    audit.add_argument("--nonlocal-distance-ratio", type=float, default=0.25)
    audit.add_argument("--nonlocal-uv-exclusion", type=int, default=4)
    audit.add_argument("--max-proximity-points", type=int, default=100_000)
    audit.add_argument("--max-proximity-pairs", type=int, default=10_000)
    audit.add_argument(
        "--fail-on-integrity",
        action="store_true",
        help="Exit 2 when contract-level integrity errors are found.",
    )
    return parser


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + ("" if content.endswith("\n") else "\n"), encoding="utf-8")


def _integrity_exit_code(report: dict, *, fail_on_warning: bool) -> int:
    if report["summary"]["errors"] > 0:
        return 2
    if fail_on_warning and report["summary"]["warnings"] > 0:
        return 2
    return 0


def _run_check(args: argparse.Namespace) -> int:
    report = audit_tifxyz_integrity(args.path)
    encoded = dumps_integrity_report(report)
    if args.json_path:
        _write_text(args.json_path, encoded)
        print(
            f"{args.path}: {report['status']} "
            f"({report['summary']['errors']} errors, "
            f"{report['summary']['warnings']} warnings)"
        )
    else:
        sys.stdout.write(encoded)
        sys.stdout.write("\n")
    return _integrity_exit_code(report, fail_on_warning=args.fail_on_warning)


def _run_scan(args: argparse.Namespace) -> int:
    root = args.root
    candidates: set[Path] = set()
    if root.is_dir():
        # Discover incomplete surfaces too: scanning only for meta.json would
        # silently miss a TIFXYZ directory whose metadata was deleted.
        for filename in ("meta.json", "x.tif", "y.tif", "z.tif"):
            candidates.update(path.parent for path in root.rglob(filename))

    reports = [audit_tifxyz_integrity(path) for path in sorted(candidates)]
    paths_by_uuid: dict[str, list[str]] = {}
    for report in reports:
        uuid = report["metadata"].get("uuid")
        if isinstance(uuid, str) and uuid:
            paths_by_uuid.setdefault(uuid, []).append(report["path"])
    duplicate_uuids = [
        {"uuid": uuid, "count": len(paths), "paths": sorted(paths)}
        for uuid, paths in sorted(paths_by_uuid.items())
        if len(paths) > 1
    ]
    collection_findings: list[dict] = []
    if not root.is_dir():
        collection_findings.append(
            {
                "severity": "error",
                "code": "scan-root-not-directory",
                "message": f"Scan root is not a directory: {root}",
            }
        )
    elif not reports:
        collection_findings.append(
            {
                "severity": "warning",
                "code": "no-tifxyz-directories",
                "message": f"No directories containing meta.json were found below {root}.",
            }
        )
    collection_findings.extend([
        {
            "severity": "warning",
            "code": "duplicate-uuid",
            "message": (
                f"UUID {item['uuid']!r} is reused by {item['count']} TIFXYZ directories."
            ),
            "uuid": item["uuid"],
            "count": item["count"],
            "paths": item["paths"],
        }
        for item in duplicate_uuids
    ])
    aggregate = {
        "schema_version": "tifxyz-integrity-scan-v1",
        "root": str(root),
        "surface_count": len(reports),
        "status_counts": {
            status: sum(report["status"] == status for report in reports)
            for status in ("pass", "warning", "error")
        },
        "collection_findings": collection_findings,
        "reports": reports,
    }
    encoded = json.dumps(aggregate, allow_nan=False, indent=2, sort_keys=True)
    if args.json_path:
        _write_text(args.json_path, encoded)
        counts = aggregate["status_counts"]
        print(
            f"{root}: {len(reports)} surfaces "
            f"({counts['pass']} pass, {counts['warning']} warning, {counts['error']} error; "
            f"{len(duplicate_uuids)} duplicate UUID group(s))"
        )
    else:
        sys.stdout.write(encoded)
        sys.stdout.write("\n")

    if any(
        _integrity_exit_code(report, fail_on_warning=args.fail_on_warning) != 0
        for report in reports
    ):
        return 2
    if args.fail_on_warning and collection_findings:
        return 2
    if any(item["severity"] == "error" for item in collection_findings):
        return 2
    return 0


def _run_audit(args: argparse.Namespace) -> int:
    config = AuditConfig(
        expected_spacing_x=args.expected_spacing_x,
        expected_spacing_y=args.expected_spacing_y,
        long_edge_ratio=args.long_edge_ratio,
        short_edge_ratio=args.short_edge_ratio,
        normal_jump_degrees=args.normal_jump_degrees,
        condition_number=args.condition_number,
        symmetric_stretch=args.symmetric_stretch,
        symmetric_dirichlet=args.symmetric_dirichlet,
        area_ratio_low=args.area_ratio_low,
        area_ratio_high=args.area_ratio_high,
        shear=args.shear,
        normal_step_ratio=args.normal_step_ratio,
        normal_step_min_component_cells=args.normal_step_min_component_cells,
        nonlocal_distance_ratio=args.nonlocal_distance_ratio,
        nonlocal_uv_exclusion=args.nonlocal_uv_exclusion,
        max_proximity_points=args.max_proximity_points,
        max_proximity_pairs=args.max_proximity_pairs,
    )
    contract = audit_tifxyz_integrity(args.path)
    try:
        data = load_tifxyz(args.path)
    except Exception as exc:
        failure = {
            "schema_version": "tifxyz-audit-failure-v1",
            "tool": {"name": "tifxyz-doctor", "version": __version__},
            "source": {"path": str(args.path)},
            "status": "error",
            "contract": contract,
            "geometry_audit": None,
            "failure": {
                "code": "geometry-load-failed",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        }
        encoded = json.dumps(failure, allow_nan=False, indent=2, sort_keys=True)
        if args.json_path:
            _write_text(args.json_path, encoded)
        if not any((args.json_path, args.html_path, args.overlay_path)):
            sys.stdout.write(encoded)
            sys.stdout.write("\n")
        else:
            print(
                f"{args.path}: geometry audit could not load the package "
                f"({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
            if args.html_path or args.overlay_path:
                print(
                    "HTML and overlay outputs require loadable coordinate arrays "
                    "and were not written.",
                    file=sys.stderr,
                )
        return 2

    report = audit_mesh(data, config)
    report["contract"] = contract

    if args.json_path:
        write_json(report, args.json_path)
    if args.html_path:
        write_html(report, args.html_path)
    if args.overlay_path:
        write_overlay(report, args.overlay_path)

    summary = public_report(report)
    if not any((args.json_path, args.html_path, args.overlay_path)):
        json.dump(summary, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
        sys.stdout.write("\n")
    else:
        levels: dict[str, int] = {}
        for finding in summary["findings"]:
            levels[finding["level"]] = levels.get(finding["level"], 0) + 1
        print(
            f"{summary['source']['uuid']}: "
            f"{summary['topology']['valid_quad_count']:,} valid quads, "
            f"{contract['summary']['errors']} contract errors, "
            f"{levels.get('review', 0)} review cues"
        )

    if args.fail_on_integrity and contract["summary"]["errors"] > 0:
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "check":
        return _run_check(args)
    if args.command == "scan":
        return _run_scan(args)
    if args.command == "audit":
        return _run_audit(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
