#!/usr/bin/env python3
"""Mechanically freeze the six preregistered training checkpoint bundles.

This program deliberately does not load checkpoint tensors or read training-log
contents.  It binds each run using only fixed identity fields from the two JSON
files plus file names, byte sizes, and streaming SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PREREGISTRATION_COMMIT = "5ca0444fb31863c8e02466316bf9e560cf567876"
SOURCE_CHECKPOINT_SHA256 = (
    "f1990a02ac91889c1f989522ae0e45421a91cb666320448aaf579d42b081636f"
)
LAUNCHER_LEDGER_SHA256 = (
    "594d23dfe645c6a6ea6d737e81cfdeef015f5e6251049b0ae13eee1413e9d8a4"
)
TRAINING_COMPLETE_STATUS = (
    "training complete; validation and test endpoints not computed"
)
FREEZE_STATUS = (
    "all six training checkpoints frozen before validation or test evaluation"
)

EXPECTED_INTENSITY_PROPERTIES = {
    "mean": 129.09779357910156,
    "std": 42.188316345214844,
    "percentile_00_5": 44.0,
    "percentile_99_5": 236.0,
}


@dataclass(frozen=True)
class Run:
    key: str
    arm: str
    seed: int
    kernel_id: str
    kernel_version: int

    @property
    def stem(self) -> str:
        return f"surface059_{self.arm}_seed{self.seed}"

    @property
    def gap_weight(self) -> float:
        return 1.0 if self.arm == "control" else 8.0

    @property
    def kernel_url(self) -> str:
        return f"https://www.kaggle.com/code/{self.kernel_id}"


RUNS = (
    Run(
        "control-seed11",
        "control",
        11,
        "aviadcohen1/vesuvius-fusion-aware-control-seed-11",
        4,
    ),
    Run(
        "control-seed23",
        "control",
        23,
        "aviadcohen1/vesuvius-fusion-aware-control-seed-23",
        4,
    ),
    Run(
        "control-seed47",
        "control",
        47,
        "aviadcohen1/vesuvius-fusion-aware-control-seed-47",
        1,
    ),
    Run(
        "gap8-seed11",
        "gap8",
        11,
        "aviadcohen1/vesuvius-fusion-aware-gap8-seed-11",
        1,
    ),
    Run(
        "gap8-seed23",
        "gap8",
        23,
        "aviadcohen1/vesuvius-fusion-aware-gap8-seed-23",
        1,
    ),
    Run(
        "gap8-seed47",
        "gap8",
        47,
        "aviadcohen1/vesuvius-fusion-aware-gap8-seed-47",
        1,
    ),
)


class ManifestError(RuntimeError):
    """Raised when a checkpoint bundle violates the frozen contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ManifestError(f"{label}: expected {expected!r}, found {actual!r}")


def _require_regular_file(path: Path) -> None:
    if not path.is_file():
        raise ManifestError(f"missing regular file: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    # This function is intentionally called only for checkpoint metadata and the
    # training-run manifest, never for PTH or log files.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot parse JSON identity file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"JSON identity file is not an object: {path}")
    return value


def _file_record(path: Path, relative_path: str) -> dict[str, Any]:
    _require_regular_file(path)
    return {
        "bytes": path.stat().st_size,
        "path": relative_path,
        "sha256": _sha256(path),
    }


def _run_paths(staging_root: Path, run: Run) -> tuple[Path, dict[str, Path]]:
    payload_root = staging_root / run.key / "fusion-aware" / run.key
    paths = {
        "checkpoint": payload_root / "checkpoints" / f"{run.stem}.pth",
        "metadata": payload_root / "checkpoints" / f"{run.stem}.json",
        "training_log": payload_root / "training.log",
        "training_run_manifest": payload_root / "training_run_manifest.json",
    }
    for path in paths.values():
        _require_regular_file(path)
    return payload_root, paths


def _validate_metadata(
    run: Run,
    metadata: dict[str, Any],
    checkpoint_record: dict[str, Any],
) -> None:
    expected = {
        "checkpoint": f"{run.stem}.pth",
        "checkpoint_sha256": checkpoint_record["sha256"],
        "arm": run.arm,
        "seed": run.seed,
        "steps": 1500,
        "gap_weight": run.gap_weight,
        "real_train_count": 138,
        "real_validation_count": 24,
        "synthetic_count": 16,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "normalization_scheme": "CTNormalization",
        "intensity_properties": EXPECTED_INTENSITY_PROPERTIES,
    }
    for field, expected_value in expected.items():
        _require_equal(
            f"{run.key} checkpoint metadata field {field}",
            metadata.get(field),
            expected_value,
        )


def _validate_run_manifest(
    run: Run,
    run_manifest: dict[str, Any],
    records: dict[str, dict[str, Any]],
) -> None:
    _require_equal(f"{run.key} run schema", run_manifest.get("schema_version"), "1.0")
    _require_equal(f"{run.key} run arm", run_manifest.get("arm"), run.arm)
    _require_equal(f"{run.key} run seed", run_manifest.get("seed"), run.seed)
    _require_equal(
        f"{run.key} run status",
        run_manifest.get("status"),
        TRAINING_COMPLETE_STATUS,
    )

    inputs = run_manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ManifestError(f"{run.key} run inputs is not an object")
    fixed_assets = inputs.get("fixed_assets")
    if not isinstance(fixed_assets, dict):
        raise ManifestError(f"{run.key} fixed_assets is not an object")
    _require_equal(
        f"{run.key} source checkpoint",
        fixed_assets.get("model/Model_epoch499.pth"),
        SOURCE_CHECKPOINT_SHA256,
    )

    outputs = run_manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ManifestError(f"{run.key} run outputs is not an object")
    expected_outputs = {
        "checkpoint": f"{run.key}/checkpoints/{run.stem}.pth",
        "checkpoint_sha256": records["checkpoint"]["sha256"],
        "metadata": f"{run.key}/checkpoints/{run.stem}.json",
        "metadata_sha256": records["metadata"]["sha256"],
        "training_log": f"{run.key}/training.log",
        "training_log_sha256": records["training_log"]["sha256"],
    }
    for field, expected_value in expected_outputs.items():
        _require_equal(
            f"{run.key} run output field {field}",
            outputs.get(field),
            expected_value,
        )


def _build_run_record(staging_root: Path, run: Run) -> dict[str, Any]:
    payload_root, paths = _run_paths(staging_root, run)
    relative_paths = {
        "checkpoint": f"{run.key}/checkpoints/{run.stem}.pth",
        "metadata": f"{run.key}/checkpoints/{run.stem}.json",
        "training_log": f"{run.key}/training.log",
        "training_run_manifest": f"{run.key}/training_run_manifest.json",
    }
    records = {
        name: _file_record(path, relative_paths[name])
        for name, path in paths.items()
    }

    metadata = _load_json(paths["metadata"])
    run_manifest = _load_json(paths["training_run_manifest"])
    _validate_metadata(run, metadata, records["checkpoint"])
    _validate_run_manifest(run, run_manifest, records)

    # Reject an accidentally selected payload directory with extra scientific
    # output.  Kaggle's wrapper log lives above payload_root and is out of scope.
    actual_payload_files = {
        path.relative_to(payload_root).as_posix()
        for path in payload_root.rglob("*")
        if path.is_file()
    }
    expected_payload_files = {
        f"checkpoints/{run.stem}.pth",
        f"checkpoints/{run.stem}.json",
        "training.log",
        "training_run_manifest.json",
    }
    _require_equal(
        f"{run.key} payload file set",
        actual_payload_files,
        expected_payload_files,
    )

    return {
        "arm": run.arm,
        "files": records,
        "kernel": {
            "id": run.kernel_id,
            "url": run.kernel_url,
            "version": run.kernel_version,
        },
        "seed": run.seed,
        "training_status": TRAINING_COMPLETE_STATUS,
    }


def build_manifest(staging_root: Path) -> dict[str, Any]:
    staging_root = staging_root.resolve()
    if not staging_root.is_dir():
        raise ManifestError(f"staging root is not a directory: {staging_root}")

    runs = [_build_run_record(staging_root, run) for run in RUNS]
    runs_payload_sha256 = hashlib.sha256(
        json.dumps(runs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "evaluation_access": {
            "validation_or_test_outputs_inspected": False,
            "statement": (
                "Only checkpoint identity JSON, file names, byte sizes, and "
                "SHA-256 digests were inspected while creating this freeze."
            ),
        },
        "launcher_ledger_sha256": LAUNCHER_LEDGER_SHA256,
        "preregistration": {
            "commit": PREREGISTRATION_COMMIT,
            "repository": "https://github.com/aviad12g/tifxyz-doctor",
        },
        "runs": runs,
        "runs_payload_sha256": runs_payload_sha256,
        "schema_version": "1.0",
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "status": FREEZE_STATUS,
    }


def _write_atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-root",
        type=Path,
        required=True,
        help="root containing the six downloaded Kaggle output directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("checkpoint_manifest.json"),
        help="canonical aggregate manifest path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that --output already equals the canonical generated manifest",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        canonical = _canonical_json(build_manifest(args.staging_root))
        if args.check:
            if not args.output.is_file():
                raise ManifestError(f"manifest does not exist: {args.output}")
            existing = args.output.read_text(encoding="utf-8")
            if existing != canonical:
                raise ManifestError(
                    f"manifest differs from canonical downloaded bundles: {args.output}"
                )
            print(f"checkpoint manifest verified: {args.output}")
        else:
            _write_atomic(args.output, canonical)
            print(f"checkpoint manifest written: {args.output}")
        print(f"checkpoint manifest sha256: {_sha256(args.output)}")
    except (ManifestError, OSError) as exc:
        print(f"checkpoint manifest error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
