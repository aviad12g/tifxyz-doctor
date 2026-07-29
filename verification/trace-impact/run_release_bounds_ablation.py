#!/usr/bin/env python3
"""Run a three-way real-segment trace bounds ablation three times."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path


EXPECTED = {
    "executable": "aaee7adfda92304f5c22464a21aaff54e937131deea615a7c4e523238d7c8e58",
    "input_x": "179da10e706424a8479681555a03a68dee80e91c8ad241877ea62b4044039651",
    "input_y": "dae0c8d118cd8bfa40b8e38c04682801191402a6cc66b14544d649eb7fba8153",
    "input_z": "f7a028ea6121c492bf6c15c7174bcd379b3b00a780ef8361054f1c3266728b4f",
    "input_meta": "158dc7ab72251dffb95df471f1789eb4b020e0bef70bd275714399f72b142634",
    "params": "5606943e4f278ad8e2ec9f5522d2d0cad6914ba908f829464594e564ddd31093",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(label: str, path: Path) -> str:
    observed = sha256(path)
    if observed != EXPECTED[label]:
        raise RuntimeError(
            f"{path}: SHA-256 {observed} != expected {EXPECTED[label]}"
        )
    return observed


def parse_log(stdout: str) -> dict:
    initialized = re.search(
        r"resume_growth initialized (\d+) low-res points, fringe (\d+).*?"
        r"used_area \[(\d+) x (\d+) from \((\d+), (\d+)\)\]",
        stdout,
    )
    generation = re.search(
        r"gen 0 processing (\d+) fringe cands .*? area (\d+) vx\^2",
        stdout,
    )
    area_estimate = re.search(r"area est: (\d+) vx\^2 \(([^)]+) cm\^2\)", stdout)
    if not initialized or not generation or not area_estimate:
        raise RuntimeError("could not parse tracer stdout")
    return {
        "resume_initialized_points": int(initialized.group(1)),
        "resume_fringe_points": int(initialized.group(2)),
        "used_area": {
            "width": int(initialized.group(3)),
            "height": int(initialized.group(4)),
            "x": int(initialized.group(5)),
            "y": int(initialized.group(6)),
        },
        "generation_zero_processed_fringe": int(generation.group(1)),
        "generation_zero_area_vx2": int(generation.group(2)),
        "final_area_estimate_vx2": int(area_estimate.group(1)),
        "final_area_estimate_cm2": float(area_estimate.group(2)),
    }


def loaded_vc_libraries(stderr: str) -> list[str]:
    libraries = []
    for line in stderr.splitlines():
        if "dyld[" not in line or "libvc_" not in line:
            continue
        match = re.search(r"(/[^\n]+/libvc_(?:core|tracer)\.dylib)", line)
        if match and match.group(1) not in libraries:
            libraries.append(match.group(1))
    return libraries


def verify_variant_libraries(
    frameworks: Path, ablation: Path
) -> tuple[dict, dict[str, dict[str, str]]]:
    manifest_path = ablation / "patch-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    helper_core = manifest["helper_only"]["libvc_core.dylib"]
    full_core = manifest["full_bounds"]["libvc_core.dylib"]
    full_tracer = manifest["full_bounds"]["libvc_tracer.dylib"]
    expected = {
        "baseline": {
            "libvc_core.dylib": helper_core["source_sha256"],
            "libvc_tracer.dylib": full_tracer["source_sha256"],
        },
        "helper_only": {
            "libvc_core.dylib": helper_core["patched_signed_sha256"],
            "libvc_tracer.dylib": full_tracer["source_sha256"],
        },
        "full_bounds": {
            "libvc_core.dylib": full_core["patched_signed_sha256"],
            "libvc_tracer.dylib": full_tracer["patched_signed_sha256"],
        },
    }
    paths = {
        "baseline": {
            "libvc_core.dylib": frameworks / "libvc_core.dylib",
            "libvc_tracer.dylib": frameworks / "libvc_tracer.dylib",
        },
        "helper_only": {
            "libvc_core.dylib": ablation / "helper-only" / "libvc_core.dylib",
            "libvc_tracer.dylib": frameworks / "libvc_tracer.dylib",
        },
        "full_bounds": {
            "libvc_core.dylib": ablation / "full-bounds" / "libvc_core.dylib",
            "libvc_tracer.dylib": (
                ablation / "full-bounds" / "libvc_tracer.dylib"
            ),
        },
    }
    verified: dict[str, dict[str, str]] = {}
    for variant, libraries in paths.items():
        verified[variant] = {}
        for name, path in libraries.items():
            observed = sha256(path)
            if observed != expected[variant][name]:
                raise RuntimeError(
                    f"{variant} {path}: SHA-256 {observed} != expected "
                    f"{expected[variant][name]}"
                )
            verified[variant][str(path.resolve())] = observed
    return manifest, verified


def trace_directory(target: Path) -> Path:
    matches = sorted(target.glob("auto_trace_*"))
    if len(matches) != 1:
        raise RuntimeError(f"{target}: expected one auto_trace output, got {matches}")
    return matches[0]


def run_comparison(comparator: Path, baseline: Path, candidate: Path) -> dict:
    result = subprocess.run(
        [str(comparator), str(baseline), str(candidate)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--frameworks", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--volume", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--ablation", required=True, type=Path)
    parser.add_argument("--comparator", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for attribute in (
        "executable",
        "frameworks",
        "source",
        "volume",
        "params",
        "ablation",
        "comparator",
        "output",
    ):
        setattr(args, attribute, getattr(args, attribute).resolve())

    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite existing {args.output}")
    args.output.mkdir(parents=True)

    patch_manifest, variant_library_hashes = verify_variant_libraries(
        args.frameworks, args.ablation
    )
    provenance = {
        "executable": require_hash("executable", args.executable),
        "input": {
            "x.tif": require_hash("input_x", args.source / "x.tif"),
            "y.tif": require_hash("input_y", args.source / "y.tif"),
            "z.tif": require_hash("input_z", args.source / "z.tif"),
            "meta.json": require_hash("input_meta", args.source / "meta.json"),
        },
        "params": require_hash("params", args.params),
        "volume": {
            "meta.json": sha256(args.volume / "meta.json"),
            "0/.zarray": sha256(args.volume / "0/.zarray"),
        },
        "patch_manifest": sha256(args.ablation / "patch-manifest.json"),
        "patch_manifest_release_revision": patch_manifest["release_revision"],
        "variant_library_hashes": variant_library_hashes,
        "comparator_binary": sha256(args.comparator),
        "comparator_source": sha256(
            Path(__file__).resolve().with_name("compare_tifxyz.cpp")
        ),
    }

    variants = {
        "baseline": args.frameworks,
        "helper_only": args.ablation / "helper-only",
        "full_bounds": args.ablation / "full-bounds",
    }
    records: dict[str, list[dict]] = {}
    source_parent = args.source.parent
    for variant, first_library_path in variants.items():
        records[variant] = []
        for replicate in range(1, 4):
            target = args.output / variant / f"replicate-{replicate}"
            target.mkdir(parents=True)
            command = [
                str(args.executable),
                "--volume",
                str(args.volume),
                "--src-dir",
                str(source_parent),
                "--target-dir",
                str(target),
                "--params",
                str(args.params),
                "--src-segment",
                str(args.source),
            ]
            env = os.environ.copy()
            env.update(
                {
                    "OMP_NUM_THREADS": "1",
                    "OMP_DYNAMIC": "FALSE",
                    "DYLD_PRINT_LIBRARIES": "1",
                    "DYLD_LIBRARY_PATH": os.pathsep.join(
                        [str(first_library_path), str(args.frameworks)]
                    ),
                }
            )
            started = dt.datetime.now(dt.timezone.utc)
            result = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True,
            )
            ended = dt.datetime.now(dt.timezone.utc)
            (target / "stdout.log").write_text(result.stdout)
            (target / "stderr.log").write_text(result.stderr)
            if result.returncode != 0:
                raise RuntimeError(
                    f"{variant} replicate {replicate} failed: {result.stderr}"
                )

            trace = trace_directory(target)
            output_hashes = {
                name: sha256(trace / name)
                for name in ("x.tif", "y.tif", "z.tif", "generations.tif")
            }
            meta = json.loads((trace / "meta.json").read_text())
            loaded_libraries = loaded_vc_libraries(result.stderr)
            loaded_library_hashes = {
                library: sha256(Path(library))
                for library in loaded_libraries
                if Path(library).is_file()
            }
            records[variant].append(
                {
                    "replicate": replicate,
                    "started_utc": started.isoformat(),
                    "ended_utc": ended.isoformat(),
                    "duration_seconds": (ended - started).total_seconds(),
                    "command": shlex.join(command),
                    "environment": {
                        "OMP_NUM_THREADS": env["OMP_NUM_THREADS"],
                        "OMP_DYNAMIC": env["OMP_DYNAMIC"],
                        "DYLD_PRINT_LIBRARIES": env["DYLD_PRINT_LIBRARIES"],
                        "DYLD_LIBRARY_PATH": env["DYLD_LIBRARY_PATH"],
                    },
                    "loaded_vc_libraries": loaded_libraries,
                    "loaded_vc_library_hashes": loaded_library_hashes,
                    "trace_directory": str(trace.resolve()),
                    "stdout_observations": parse_log(result.stdout),
                    "meta_observations": {
                        "area_vx2": meta["area_vx2"],
                        "area_cm2": meta["area_cm2"],
                        "grid_offset": meta["grid_offset"],
                    },
                    "output_hashes": output_hashes,
                }
            )

    deterministic = {}
    for variant, variant_records in records.items():
        hash_sets = [
            tuple(sorted(record["output_hashes"].items()))
            for record in variant_records
        ]
        observations = [
            (
                record["stdout_observations"],
                record["meta_observations"],
            )
            for record in variant_records
        ]
        deterministic[variant] = {
            "tiff_hashes_identical": len(set(hash_sets)) == 1,
            "observations_identical": all(
                observation == observations[0] for observation in observations
            ),
        }

    first_traces = {
        variant: Path(variant_records[0]["trace_directory"])
        for variant, variant_records in records.items()
    }
    comparisons = {
        "baseline_to_helper_only": run_comparison(
            args.comparator,
            first_traces["baseline"],
            first_traces["helper_only"],
        ),
        "helper_only_to_full_bounds": run_comparison(
            args.comparator,
            first_traces["helper_only"],
            first_traces["full_bounds"],
        ),
        "baseline_to_full_bounds": run_comparison(
            args.comparator,
            first_traces["baseline"],
            first_traces["full_bounds"],
        ),
    }
    (args.output / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )

    manifest = {
        "schema_version": 1,
        "status": "release_binary_bounds_ablation_complete",
        "warning": (
            "This is a byte-validated bounds ablation of the official 05ff9ea "
            "macOS-arm64 release, not a build of PR #1264."
        ),
        "release_revision": "05ff9ea",
        "input_public_url": (
            "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
            "PHerc1447/segments/"
            "20250502185519-auto_grown_20250502164303733/mesh/intermediate/"
            "tifxyz_original/"
        ),
        "provenance": provenance,
        "variants": records,
        "deterministic": deterministic,
        "comparisons": comparisons,
    }
    (args.output / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(args.output / "run-manifest.json")


if __name__ == "__main__":
    main()
