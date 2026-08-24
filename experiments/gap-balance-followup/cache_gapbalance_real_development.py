"""Cache only the frozen Scroll-1 GapBalance development probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import gapbalance_development as contract
import numpy as np
import tifffile
import torch

FUSION_ROOT = Path(__file__).resolve().parents[1] / "fusion-aware-surface"
sys.path.insert(0, str(FUSION_ROOT))
from inference import predict_volume
from normalization import validate_ct_contract

SOURCE_CHECKPOINT_SHA256 = "f1990a02ac91889c1f989522ae0e45421a91cb666320448aaf579d42b081636f"
SPLIT_MANIFEST_SHA256 = "dedc881134d9de2ed2605162f82dfb219b52c6e05629c68223100b148b15d4fe"
SPLIT_RECORDS_SHA256 = "20c600d6061bf8715ada20423a05c2f03a9bc14827b2c79bac7b7e1b3cc0499d"
STATUS = "GapBalance Scroll-1 development probabilities cached; endpoints not scored"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cache(path: Path, shape: tuple[int, ...]) -> None:
    with np.load(path) as data:
        if set(data.files) != {"prob", "gt"}:
            raise RuntimeError(f"{path}: expected exactly prob and gt")
        probability = np.asarray(data["prob"])
        gt = np.asarray(data["gt"])
        if probability.shape != gt.shape or probability.shape != shape:
            raise RuntimeError(f"{path}: cached shape mismatch")
        if probability.dtype != np.float16 or gt.dtype != np.uint8:
            raise RuntimeError(f"{path}: cached dtype mismatch")
        if (
            not np.isfinite(probability).all()
            or probability.min() < 0
            or probability.max() > 1
        ):
            raise RuntimeError(f"{path}: invalid cached probability")
        if np.any((gt != 0) & (gt != 1)):
            raise RuntimeError(f"{path}: non-binary cached GT")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=contract.RUNS, required=True)
    parser.add_argument("--real-data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-scripts", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--model-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    arm, seed = contract.run_identity(args.run)
    if sha256_file(args.source_checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("source checkpoint SHA-256 mismatch")
    if sha256_file(args.split_manifest) != SPLIT_MANIFEST_SHA256:
        raise RuntimeError("Scroll-1 split manifest SHA-256 mismatch")
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if split.get("records_sha256") != SPLIT_RECORDS_SHA256:
        raise RuntimeError("Scroll-1 split records SHA-256 mismatch")
    records = [record for record in split["records"] if record["split"] == "validation"]
    if len(records) != 24:
        raise RuntimeError("frozen Scroll-1 development set is not exactly 24 patches")

    sys.path.insert(0, str(args.diagnostic_scripts))
    import loader059

    loader059.CKPT = str(args.source_checkpoint)
    model, normalization, properties = loader059.load_059()
    ct_properties = validate_ct_contract(normalization, properties)
    state = torch.load(args.model_state, map_location="cpu", weights_only=False)
    if state.get("arm") != arm or int(state.get("seed", -1)) != seed:
        raise RuntimeError(f"model state identity does not match {args.run}")
    if state.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("fine-tuned state points to another source checkpoint")
    validate_ct_contract(
        state.get("normalization_scheme"), state.get("intensity_properties")
    )
    model.load_state_dict(state["model"], strict=True)
    model.cuda().eval()
    model_state_sha256 = sha256_file(args.model_state)

    images = {path.name: path for path in args.real_data.rglob("*_0000.tif")}
    labels = {
        path.name: path
        for path in args.real_data.rglob("*.tif")
        if not path.name.endswith("_0000.tif")
    }
    output = args.out / args.run
    output.mkdir(parents=True, exist_ok=True)
    emitted = []
    for index, record in enumerate(records, start=1):
        image_path = images.get(record["image"])
        label_path = labels.get(record["label"])
        if image_path is None or label_path is None:
            raise RuntimeError(f"missing Scroll-1 pair for {record['image']}")
        if sha256_file(image_path) != record["image_sha256"]:
            raise RuntimeError(f"image hash mismatch for {record['image']}")
        if sha256_file(label_path) != record["label_sha256"]:
            raise RuntimeError(f"label hash mismatch for {record['label']}")
        image = tifffile.imread(image_path)
        gt = (tifffile.imread(label_path) > 0).astype(np.uint8)
        if image.shape != gt.shape or image.ndim != 3:
            raise RuntimeError(f"paired ZYX shape mismatch for {record['image']}")
        cache = output / Path(record["image"]).with_suffix(".npz").name
        if cache.exists():
            validate_cache(cache, image.shape)
        else:
            probability = predict_volume(
                model, image, ct_properties, torch.device("cuda")
            )
            np.savez_compressed(cache, prob=probability.astype(np.float16), gt=gt)
            validate_cache(cache, image.shape)
        emitted.append(
            {
                "file": cache.name,
                "sha256": sha256_file(cache),
                "bytes": cache.stat().st_size,
                "source_image": record["image"],
            }
        )
        print(
            f"GAPBALANCE_DEVELOPMENT_CACHE {index}/24 run={args.run} file={cache.name}",
            flush=True,
        )

    payload = {
        "schema_version": "1.0",
        "status": STATUS,
        "run": args.run,
        "split": "development",
        "source_split_records_sha256": SPLIT_RECORDS_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "model_state_sha256": model_state_sha256,
        "selected_threshold": None,
        "normalization_scheme": "CTNormalization",
        "intensity_properties": ct_properties,
        "files": emitted,
        "confirmation_outputs_inspected": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    manifest = args.out / f"cache_manifest_{args.run}_development.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("payload SHA-256:", payload["payload_sha256"])
    print("GAPBALANCE_REAL_DEVELOPMENT_CACHED_NOT_SCORED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
