"""Score only frozen GapBalance development caches and select one candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import gapbalance_development as contract
import numpy as np

FUSION_ROOT = Path(__file__).resolve().parents[1] / "fusion-aware-surface"
sys.path.insert(0, str(FUSION_ROOT))
from fusion_ray_readout import score_rays

REAL_CACHE_STATUS = "GapBalance Scroll-1 development probabilities cached; endpoints not scored"
SYNTHETIC_CACHE_STATUS = "GapBalance synthetic development rays cached; endpoints not scored"
FREEZE_STATUS = "all GapBalance and matched-control checkpoints frozen before development scoring"
SYNTHETIC_SHARDS = 4
SPLIT_MANIFEST_SHA256 = "dedc881134d9de2ed2605162f82dfb219b52c6e05629c68223100b148b15d4fe"
SPLIT_RECORDS_SHA256 = "20c600d6061bf8715ada20423a05c2f03a9bc14827b2c79bac7b7e1b3cc0499d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_embedded_hash(payload: dict[str, Any]) -> None:
    body = dict(payload)
    observed = body.pop("payload_sha256", None)
    if observed != canonical_sha256(body):
        raise RuntimeError("embedded payload SHA-256 mismatch")


def load_checkpoint_freeze(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_embedded_hash(payload)
    if payload.get("status") != FREEZE_STATUS:
        raise RuntimeError("checkpoint freeze has the wrong status")
    if payload.get("confirmation_outputs_inspected") is not False:
        raise RuntimeError("checkpoint freeze does not preserve the confirmation blind")
    runs = payload.get("runs", {})
    if set(runs) != set(contract.RUNS):
        raise RuntimeError("checkpoint freeze does not contain the exact nine matched runs")
    hashes = {}
    for run in contract.RUNS:
        contract.run_identity(run)
        digest = runs[run].get("checkpoint_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"{run}: invalid checkpoint SHA-256")
        hashes[run] = digest
    return hashes


def validate_real_cache(path: Path) -> None:
    with np.load(path) as data:
        if set(data.files) != {"prob", "gt"}:
            raise RuntimeError(f"{path}: expected exactly prob and gt")
        probability = np.asarray(data["prob"])
        gt = np.asarray(data["gt"])
        if probability.shape != gt.shape or probability.ndim != 3:
            raise RuntimeError(f"{path}: invalid paired ZYX shapes")
        if probability.dtype != np.float16:
            raise RuntimeError(f"{path}: probability must be float16")
        if gt.dtype != np.uint8 or np.any((gt != 0) & (gt != 1)):
            raise RuntimeError(f"{path}: GT must be binary uint8")
        if (
            not np.isfinite(probability).all()
            or probability.min() < 0
            or probability.max() > 1
        ):
            raise RuntimeError(f"{path}: invalid probability values")


def validate_ray_cache(path: Path) -> None:
    with np.load(path) as data:
        if set(data.files) != {"ray_probability", "ray_turn"}:
            raise RuntimeError(f"{path}: wrong ray-cache keys")
        probability = np.asarray(data["ray_probability"])
        turn = np.asarray(data["ray_turn"])
        if probability.shape != turn.shape or probability.ndim != 2:
            raise RuntimeError(f"{path}: ray-cache shape mismatch")
        if probability.dtype != np.float16 or turn.dtype != np.int16:
            raise RuntimeError(f"{path}: ray-cache dtype mismatch")
        if (
            not np.isfinite(probability).all()
            or probability.min() < 0
            or probability.max() > 1
        ):
            raise RuntimeError(f"{path}: invalid ray probabilities")
        if np.any(turn < 0):
            raise RuntimeError(f"{path}: invalid ray instance labels")


def official_score(worker: Path, cache: Path, threshold: float) -> dict[str, float]:
    result = subprocess.run(
        [sys.executable, str(worker), str(cache), str(threshold)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"official metric failed for {cache.name}@{threshold}: "
            f"{result.stderr.strip()}"
        )
    values = json.loads(result.stdout)
    if set(values) != {"blend", "toposcore", "surface_dice", "voi_score"}:
        raise RuntimeError("official metric returned an unexpected schema")
    converted = {key: float(value) for key, value in values.items()}
    if not all(np.isfinite(value) for value in converted.values()):
        raise RuntimeError("official metric returned a non-finite value")
    return converted


def load_real_run(
    root: Path,
    run: str,
    expected_files: set[str],
    split_sha256: str,
    checkpoint_sha256: str,
) -> tuple[list[Path], dict[str, Any]]:
    manifest_path = root / f"cache_manifest_{run}_development.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_embedded_hash(manifest)
    expected = {
        "status": REAL_CACHE_STATUS,
        "run": run,
        "split": "development",
        "source_split_records_sha256": split_sha256,
        "model_state_sha256": checkpoint_sha256,
        "selected_threshold": None,
        "confirmation_outputs_inspected": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"{run}: real-cache manifest mismatch for {key}")
    records = manifest.get("files", [])
    if len(records) != 24 or {record.get("file") for record in records} != expected_files:
        raise RuntimeError(f"{run}: Scroll-1 development cache set changed")
    caches = []
    identities = []
    for record in sorted(records, key=lambda item: item["file"]):
        cache = root / run / record["file"]
        validate_real_cache(cache)
        actual = {
            "file": record["file"],
            "bytes": cache.stat().st_size,
            "sha256": sha256_file(cache),
        }
        if actual["bytes"] != int(record["bytes"]) or actual["sha256"] != record["sha256"]:
            raise RuntimeError(f"{run}: real-cache identity mismatch for {cache.name}")
        caches.append(cache)
        identities.append(actual)
    return caches, {
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_payload_sha256": manifest["payload_sha256"],
        "files": identities,
    }


def score_real_run(worker: Path, caches: list[Path]) -> dict[str, Any]:
    threshold_rows = {}
    mean_blend = {}
    for threshold in contract.THRESHOLDS:
        rows = [
            {"file": cache.name, **official_score(worker, cache, threshold)}
            for cache in caches
        ]
        threshold_rows[str(threshold)] = rows
        mean_blend[threshold] = float(np.mean([row["blend"] for row in rows]))
    selected = contract.choose_threshold(mean_blend)
    selected_rows = threshold_rows[str(selected)]
    return {
        "selected_threshold": selected,
        "mean_blend": {str(key): value for key, value in mean_blend.items()},
        "per_patch_official": threshold_rows,
        "selected_means": {
            "blend": float(np.mean([row["blend"] for row in selected_rows])),
            "toposcore": float(np.mean([row["toposcore"] for row in selected_rows])),
        },
    }


def load_synthetic_run(
    root: Path,
    run: str,
    split_sha256: str,
    checkpoint_sha256: str,
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    expected = {cell["name"]: cell for cell in contract.synthetic_cells()}
    observed: dict[str, Path] = {}
    manifest_identities = []
    for shard in range(SYNTHETIC_SHARDS):
        manifest_path = root / f"synthetic_development_manifest_{run}_shard{shard:02d}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_embedded_hash(manifest)
        checks = {
            "status": SYNTHETIC_CACHE_STATUS,
            "run": run,
            "shard_index": shard,
            "shard_count": SYNTHETIC_SHARDS,
            "source_split_records_sha256": split_sha256,
            "model_state_sha256": checkpoint_sha256,
            "selected_threshold": None,
            "confirmation_outputs_inspected": False,
        }
        for key, value in checks.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"{run} shard {shard}: manifest mismatch for {key}")
        cells = manifest.get("cells", [])
        if len(cells) != 20:
            raise RuntimeError(f"{run} shard {shard}: expected exactly 20 cells")
        for record in cells:
            name = record.get("name")
            if name not in expected or name in observed:
                raise RuntimeError(f"{run}: duplicate or unexpected synthetic cell {name}")
            if any(record.get(key) != value for key, value in expected[name].items()):
                raise RuntimeError(f"{run}: changed synthetic design for {name}")
            cache = root / run / record["file"]
            validate_ray_cache(cache)
            if cache.stat().st_size != int(record["bytes"]) or sha256_file(cache) != record["sha256"]:
                raise RuntimeError(f"{run}: synthetic-cache identity mismatch for {name}")
            observed[name] = cache
        manifest_identities.append(
            {
                "file": manifest_path.name,
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
                "payload_sha256": manifest["payload_sha256"],
            }
        )
    if set(observed) != set(expected):
        raise RuntimeError(f"{run}: incomplete synthetic development grid")
    return observed, manifest_identities


def _sum_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        field: sum(int(row[field]) for row in rows)
        for field in contract.COUNT_FIELDS
    }


def score_synthetic_run(
    caches: dict[str, Path], threshold: float
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    expected = {cell["name"]: cell for cell in contract.synthetic_cells()}
    per_cell = {}
    for name, cache in sorted(caches.items()):
        with np.load(cache) as data:
            scored = score_rays(data["ray_probability"], data["ray_turn"], threshold)
        per_cell[name] = {field: int(scored[field]) for field in contract.COUNT_FIELDS}
    pooled = {
        "primary": _sum_counts(
            [per_cell[name] for name, cell in expected.items() if cell["kind"] == "primary"]
        ),
        "single_sheet_control": _sum_counts(
            [
                per_cell[name]
                for name, cell in expected.items()
                if cell["kind"] == "single_sheet_control"
            ]
        ),
    }
    return pooled, per_cell


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--worker", type=Path, default=FUSION_ROOT / "official_metric.py"
    )
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"candidate-freeze output must start absent: {args.out}")

    checkpoint_hashes = load_checkpoint_freeze(args.checkpoint_freeze)
    if sha256_file(args.split_manifest) != SPLIT_MANIFEST_SHA256:
        raise RuntimeError("Scroll-1 split manifest SHA-256 mismatch")
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split_sha256 = split.get("records_sha256")
    if split_sha256 != SPLIT_RECORDS_SHA256:
        raise RuntimeError("Scroll-1 split records SHA-256 mismatch")
    expected_files = {
        Path(record["image"]).with_suffix(".npz").name
        for record in split.get("records", [])
        if record.get("split") == "validation"
    }
    if len(expected_files) != 24:
        raise RuntimeError("frozen Scroll-1 validation set is not exactly 24 patches")

    validation_results = {}
    validation_cache_identities = {}
    for run in contract.RUNS:
        caches, identities = load_real_run(
            args.validation_root,
            run,
            expected_files,
            split_sha256,
            checkpoint_hashes[run],
        )
        validation_cache_identities[run] = identities
        validation_results[run] = score_real_run(args.worker, caches)

    synthetic_results = {}
    synthetic_cache_identities = {}
    selected_synthetic = {}
    for run in contract.RUNS:
        caches, identities = load_synthetic_run(
            args.synthetic_root,
            run,
            split_sha256,
            checkpoint_hashes[run],
        )
        synthetic_cache_identities[run] = identities
        threshold = float(validation_results[run]["selected_threshold"])
        pooled, per_cell = score_synthetic_run(caches, threshold)
        selected_synthetic[run] = pooled
        synthetic_results[run] = {
            "selected_threshold": threshold,
            "pooled_counts": pooled,
            "per_cell_counts": per_cell,
        }

    real_means = {
        run: validation_results[run]["selected_means"] for run in contract.RUNS
    }
    decision = contract.select_candidate(selected_synthetic, real_means)
    selected_arm = decision["selected_arm"]
    status = (
        "GapBalance development selection frozen before confirmation access"
        if selected_arm is not None
        else "GapBalance development ineligible; confirmation remains sealed"
    )
    payload = {
        "schema_version": "1.0",
        "status": status,
        "development_only": {
            "real": "exact original 24-patch Scroll-1 validation manifest",
            "synthetic_seeds": list(contract.SYNTHETIC_SEEDS),
        },
        "source_split_manifest_sha256": sha256_file(args.split_manifest),
        "source_split_records_sha256": split_sha256,
        "source_checkpoint_freeze_sha256": sha256_file(args.checkpoint_freeze),
        "checkpoint_hashes": checkpoint_hashes,
        "threshold_grid": list(contract.THRESHOLDS),
        "threshold_tie_break": "maximum mean official blend; nearest 0.5; lower threshold",
        "validation_cache_identities": validation_cache_identities,
        "synthetic_cache_identities": synthetic_cache_identities,
        "validation_results": validation_results,
        "synthetic_results": synthetic_results,
        "candidate_decision": decision,
        "selected_candidate": selected_arm,
        "selected_thresholds": {
            run: record["selected_threshold"]
            for run, record in validation_results.items()
        },
        "confirmation_outputs_inspected": False,
        "confirmation_may_open_after_public_freeze": selected_arm is not None,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("candidate-freeze payload SHA-256:", payload["payload_sha256"])
    print("selected candidate:", selected_arm)
    print("GAPBALANCE_DEVELOPMENT_SELECTION_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
