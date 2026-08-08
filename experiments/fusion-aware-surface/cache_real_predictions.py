#!/usr/bin/env python3
"""Cache validation/test probabilities with a hard test-leakage gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch

from inference import predict_volume
from normalization import validate_ct_contract


SOURCE_CHECKPOINT_SHA256 = "f1990a02ac91889c1f989522ae0e45421a91cb666320448aaf579d42b081636f"
RUNS = (
    "baseline",
    "control_seed11",
    "control_seed23",
    "control_seed47",
    "gap8_seed11",
    "gap8_seed23",
    "gap8_seed47",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_identity(run: str) -> tuple[str | None, int | None]:
    if run == "baseline":
        return None, None
    arm, seed_text = run.rsplit("_seed", 1)
    if arm not in {"control", "gap8"} or int(seed_text) not in {11, 23, 47}:
        raise ValueError(f"invalid run identity: {run}")
    return arm, int(seed_text)


def require_test_gate(path: Path | None, run: str, split_records_sha256: str) -> float:
    if path is None:
        raise RuntimeError("test inference requires the frozen threshold manifest")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "thresholds frozen from Scroll-1 validation before test inference":
        raise RuntimeError("threshold manifest has the wrong status")
    if payload.get("source_split_records_sha256") != split_records_sha256:
        raise RuntimeError("threshold manifest belongs to a different real-data split")
    try:
        threshold = float(payload["runs"][run]["selected_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"threshold manifest is missing {run}") from error
    if threshold not in {round(0.30 + 0.05 * index, 2) for index in range(9)}:
        raise RuntimeError(f"{run}: selected threshold is outside the frozen grid")
    return threshold


def validate_existing_cache(path: Path, shape: tuple[int, ...]) -> None:
    with np.load(path) as data:
        if set(data.files) != {"prob", "gt"}:
            raise RuntimeError(f"{path}: expected exactly prob and gt")
        probability = np.asarray(data["prob"])
        gt = np.asarray(data["gt"])
        if probability.shape != gt.shape or probability.shape != shape:
            raise RuntimeError(f"{path}: cached shape mismatch")
        if probability.dtype != np.float16 or gt.dtype != np.uint8:
            raise RuntimeError(f"{path}: cached dtype mismatch")
        if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
            raise RuntimeError(f"{path}: invalid cached probability")
        if np.any((gt != 0) & (gt != 1)):
            raise RuntimeError(f"{path}: non-binary cached GT")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=RUNS, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--real-data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-scripts", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--model-state", type=Path)
    parser.add_argument("--threshold-manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    arm, seed = run_identity(args.run)
    if (args.model_state is None) != (args.run == "baseline"):
        raise RuntimeError("baseline takes no model state; fine-tuned runs require one")
    if sha256_file(args.source_checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("source checkpoint SHA-256 mismatch")

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    records = [record for record in split["records"] if record["split"] == args.split]
    if not records:
        raise RuntimeError(f"real split contains no {args.split} records")
    selected_threshold = None
    if args.split == "test":
        selected_threshold = require_test_gate(
            args.threshold_manifest, args.run, split["records_sha256"]
        )

    sys.path.insert(0, str(args.diagnostic_scripts))
    import loader059

    loader059.CKPT = str(args.source_checkpoint)
    model, normalization, properties = loader059.load_059()
    ct_properties = validate_ct_contract(normalization, properties)
    model_state_sha256 = None
    if args.model_state is not None:
        state = torch.load(args.model_state, map_location="cpu", weights_only=False)
        if state.get("arm") != arm or int(state.get("seed", -1)) != seed:
            raise RuntimeError(f"model state identity does not match {args.run}")
        if state.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256:
            raise RuntimeError("fine-tuned state points to another source checkpoint")
        validate_ct_contract(
            state.get("normalization_scheme"), state.get("intensity_properties")
        )
        model.load_state_dict(state["model"], strict=True)
        model_state_sha256 = sha256_file(args.model_state)
    model.cuda().eval()

    images = {path.name: path for path in args.real_data.rglob("*_0000.tif")}
    labels = {
        path.name: path
        for path in args.real_data.rglob("*.tif")
        if not path.name.endswith("_0000.tif")
    }
    output_directory = args.out / args.run
    output_directory.mkdir(parents=True, exist_ok=True)
    emitted = []
    for index, record in enumerate(records, start=1):
        image_path = images.get(record["image"])
        label_path = labels.get(record["label"])
        if image_path is None or label_path is None:
            raise RuntimeError(f"missing pair for {record['image']}")
        if sha256_file(image_path) != record["image_sha256"]:
            raise RuntimeError(f"image hash mismatch for {record['image']}")
        if sha256_file(label_path) != record["label_sha256"]:
            raise RuntimeError(f"label hash mismatch for {record['label']}")
        image = tifffile.imread(image_path)
        gt = (tifffile.imread(label_path) > 0).astype(np.uint8)
        if image.shape != gt.shape or image.ndim != 3:
            raise RuntimeError(f"paired ZYX shape mismatch for {record['image']}")
        output_path = output_directory / Path(record["image"]).with_suffix(".npz").name
        if output_path.exists():
            validate_existing_cache(output_path, image.shape)
        else:
            probability = predict_volume(
                model, image, ct_properties, torch.device("cuda")
            )
            np.savez_compressed(
                output_path,
                prob=probability.astype(np.float16),
                gt=gt,
            )
            validate_existing_cache(output_path, image.shape)
        emitted.append(
            {
                "file": output_path.name,
                "sha256": sha256_file(output_path),
                "bytes": output_path.stat().st_size,
                "source_image": record["image"],
            }
        )
        print(
            f"CACHE {index}/{len(records)} run={args.run} split={args.split} "
            f"file={output_path.name}",
            flush=True,
        )

    payload = {
        "schema_version": "1.0",
        "status": f"{args.split} probabilities cached",
        "run": args.run,
        "split": args.split,
        "source_split_records_sha256": split["records_sha256"],
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "model_state_sha256": model_state_sha256,
        "selected_threshold": selected_threshold,
        "normalization_scheme": "CTNormalization",
        "intensity_properties": ct_properties,
        "files": emitted,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    manifest_path = args.out / f"cache_manifest_{args.run}_{args.split}.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("payload SHA-256:", payload["payload_sha256"])
    print("REAL_PROBABILITIES_CACHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
