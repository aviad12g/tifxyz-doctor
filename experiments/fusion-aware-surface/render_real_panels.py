#!/usr/bin/env python3
"""Render the four frozen real panels with a fixed, result-blind procedure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pillow_version
from scipy import __version__ as scipy_version
from scipy import ndimage as ndi

RUN_ORDER = (
    "baseline",
    "control_seed11",
    "control_seed23",
    "control_seed47",
    "gap8_seed11",
    "gap8_seed23",
    "gap8_seed47",
)
DISPLAY_ORDER = (
    "ground_truth",
    "baseline",
    "control_seed11",
    "gap8_seed11",
    "control_seed23",
    "gap8_seed23",
    "control_seed47",
    "gap8_seed47",
)
PLANE_RADIUS = 48
PLANE_SIZE = 2 * PLANE_RADIUS + 1
PIXEL_SCALE = 3
LABEL_WIDTH = 112
ROW_GAP = 3
COLUMN_GAP = 6
BACKGROUND = (0, 0, 0)
TRUE_POSITIVE = (238, 238, 238)
FALSE_NEGATIVE = (40, 110, 255)
FALSE_POSITIVE = (255, 70, 50)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_identity(path: Path) -> dict:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    observed = payload.get("payload_sha256")
    content = dict(payload)
    content.pop("payload_sha256", None)
    if observed != canonical_sha256(content):
        raise RuntimeError(f"embedded payload SHA-256 mismatch: {path}")
    return payload


def require_hex(value: object, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"invalid {label}")
    return value


def validate_public_context(
    *,
    plan_path: Path,
    delivery_path: Path,
    score_input_index_path: Path,
    thresholds_path: Path,
    panel_manifest_path: Path,
    cache_root: Path,
    public_plan_commit: str,
    public_delivery_commit: str,
) -> tuple[dict, dict, dict, dict]:
    require_hex(public_plan_commit, 40, "public plan commit")
    require_hex(public_delivery_commit, 40, "public delivery commit")
    plan = load_hashed(plan_path)
    delivery = load_hashed(delivery_path)
    score_input = load_hashed(score_input_index_path)
    thresholds = load_hashed(thresholds_path)
    panels = load_hashed(panel_manifest_path)
    if plan.get("status") != "held-out execution plan frozen before held-out inference":
        raise RuntimeError("wrong public held-out plan status")
    if delivery.get("status") != (
        "all 14 publicly planned held-out caches sealed before one-shot scoring"
    ):
        raise RuntimeError("wrong public held-out delivery status")
    if plan.get("real_panel_renderer") != local_identity(Path(__file__).resolve()):
        raise RuntimeError("running panel renderer differs from public plan")
    if plan.get("real_panel_manifest") != local_identity(panel_manifest_path) | {
        "payload_sha256": panels["payload_sha256"]
    }:
        raise RuntimeError("panel manifest differs from public plan")
    threshold_record = plan.get("threshold_binding", {}).get("frozen_thresholds")
    if threshold_record != local_identity(thresholds_path) | {
        "payload_sha256": thresholds["payload_sha256"]
    }:
        raise RuntimeError("threshold artifact differs from public plan")
    if delivery.get("public_execution_plan") != {
        "commit": public_plan_commit,
        "file": plan_path.name,
        "bytes": plan_path.stat().st_size,
        "sha256": sha256_file(plan_path),
        "payload_sha256": plan["payload_sha256"],
    }:
        raise RuntimeError("cache delivery points to another public plan")
    if delivery.get("threshold_binding") != plan.get("threshold_binding"):
        raise RuntimeError("cache delivery threshold binding mismatch")
    real_jobs = plan.get("real_test_jobs", [])
    delivered_by_job = {
        record.get("job_id"): record for record in delivery.get("jobs", [])
    }
    staged_jobs = score_input.get("jobs")
    if not isinstance(staged_jobs, list) or len(staged_jobs) != 7:
        raise RuntimeError("real score-input job set mismatch")
    for job, staged in zip(real_jobs, staged_jobs, strict=True):
        delivered = delivered_by_job.get(job["job_id"])
        if not isinstance(delivered, dict) or staged != {
            "job_id": job["job_id"],
            "run": job["run"],
            "kernel_id": delivered.get("kernel_id"),
            "kernel_version": delivered.get("kernel_version"),
            "job_index": delivered.get("job_index"),
            "cache_manifest_count": 1,
            "sealed_cache_file_count": 38,
        }:
            raise RuntimeError(f"{job['job_id']}: staged real job provenance mismatch")
    if score_input != {
        "schema_version": "1.0",
        "status": "real held-out caches staged from public delivery before one-shot scoring",
        "mode": "real",
        "public_execution_plan": {
            "commit": public_plan_commit,
            "file": plan_path.name,
            "bytes": plan_path.stat().st_size,
            "sha256": sha256_file(plan_path),
            "payload_sha256": plan["payload_sha256"],
        },
        "public_cache_delivery": {
            "commit": public_delivery_commit,
            "payload_sha256": delivery["payload_sha256"],
        },
        "threshold_binding": plan["threshold_binding"],
        "job_order": [job["job_id"] for job in real_jobs],
        "jobs": staged_jobs,
        "counts": {
            "jobs": 7,
            "cache_manifests": 7,
            "sealed_cache_files": 266,
        },
        "scientific_gate": {
            "all_required_cache_jobs_verified": True,
            "cache_delivery_publicly_frozen_before_staging": True,
            "scientific_endpoints_scored": False,
            "scientific_endpoints_printed": False,
            "cache_npz_payloads_opened_or_inspected_by_stager": False,
            "one_shot_scoring_permitted": True,
        },
        "stager": plan["one_shot_scoring_stager"],
        "payload_sha256": score_input["payload_sha256"],
    }:
        raise RuntimeError("real score-input index provenance mismatch")
    if score_input_index_path.resolve().parent != cache_root.resolve():
        raise RuntimeError("panel cache root differs from staged score-input root")
    return plan, delivery, thresholds, panels


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if set(data.files) != {"prob", "gt"}:
            raise RuntimeError(f"unexpected cache schema: {path}")
        probability = np.asarray(data["prob"])
        gt = np.asarray(data["gt"])
    if (
        probability.ndim != 3
        or probability.shape != gt.shape
        or probability.dtype != np.float16
        or gt.dtype != np.uint8
    ):
        raise RuntimeError(f"invalid probability/GT cache: {path}")
    if (
        not np.isfinite(probability).all()
        or probability.min() < 0
        or probability.max() > 1
        or np.any((gt != 0) & (gt != 1))
    ):
        raise RuntimeError(f"invalid cache values: {path}")
    return probability.astype(np.float32), gt.astype(bool)


def local_frame(gt: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, ...]:
    if center.shape != (3,) or np.any(center < 0) or np.any(center >= gt.shape):
        raise ValueError("panel center is outside the cache volume")
    signed = ndi.distance_transform_edt(~gt).astype(np.float32)
    signed -= ndi.distance_transform_edt(gt).astype(np.float32)
    ndi.gaussian_filter(signed, 3.0, output=signed)
    normal = np.asarray(
        [np.gradient(signed, axis=axis)[tuple(center)] for axis in range(3)],
        dtype=np.float64,
    )
    magnitude = float(np.linalg.norm(normal))
    if not math.isfinite(magnitude) or magnitude <= 1e-6:
        raise RuntimeError("frozen panel center has an undefined SDF normal")
    normal /= magnitude
    axes = np.eye(3, dtype=np.float64)
    reference = axes[int(np.argmin(np.abs(axes @ normal)))]
    tangent_one = np.cross(normal, reference)
    tangent_one /= np.linalg.norm(tangent_one)
    tangent_two = np.cross(normal, tangent_one)
    tangent_two /= np.linalg.norm(tangent_two)
    return normal, tangent_one, tangent_two


def coordinates(
    center: np.ndarray, normal: np.ndarray, tangent: np.ndarray
) -> np.ndarray:
    offset = np.arange(-PLANE_RADIUS, PLANE_RADIUS + 1, dtype=np.float64)
    horizontal, vertical = np.meshgrid(offset, offset, indexing="xy")
    points = (
        center[None, None, :]
        + horizontal[..., None] * tangent[None, None, :]
        + vertical[..., None] * normal[None, None, :]
    )
    return np.moveaxis(points, -1, 0)


def sample(volume: np.ndarray, points: np.ndarray, order: int) -> np.ndarray:
    return ndi.map_coordinates(
        volume.astype(np.float32, copy=False),
        points.reshape(3, -1),
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).reshape(PLANE_SIZE, PLANE_SIZE)


def overlay(gt: np.ndarray, prediction: np.ndarray | None) -> Image.Image:
    rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
    if prediction is None:
        rgb[gt] = TRUE_POSITIVE
    else:
        rgb[gt & prediction] = TRUE_POSITIVE
        rgb[gt & ~prediction] = FALSE_NEGATIVE
        rgb[~gt & prediction] = FALSE_POSITIVE
    return Image.fromarray(rgb, mode="RGB").resize(
        (PLANE_SIZE * PIXEL_SCALE, PLANE_SIZE * PIXEL_SCALE),
        resample=Image.Resampling.NEAREST,
    )


def render_panel(
    panel_index: int,
    panel: dict,
    cache_root: Path,
    thresholds: dict,
    output_root: Path,
) -> dict:
    cache_name = Path(panel["image"]).with_suffix(".npz").name
    probabilities = {}
    cache_identities = {}
    baseline_gt = None
    for run in RUN_ORDER:
        path = cache_root / run / cache_name
        probability, gt = load_cache(path)
        if baseline_gt is None:
            baseline_gt = gt
        elif not np.array_equal(gt, baseline_gt):
            raise RuntimeError(f"{cache_name}: GT differs across run caches")
        probabilities[run] = probability
        cache_identities[run] = {
            "file": cache_name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    assert baseline_gt is not None
    center = np.asarray(panel["center_zyx"], dtype=np.int64)
    if not baseline_gt[tuple(center)]:
        raise RuntimeError(f"{cache_name}: frozen panel center is not foreground")
    normal, tangent_one, tangent_two = local_frame(baseline_gt, center)
    planes = {
        "normal_x_tangent1": coordinates(center, normal, tangent_one),
        "normal_x_tangent2": coordinates(center, normal, tangent_two),
    }
    sampled_gt = {
        name: sample(baseline_gt, points, order=0) >= 0.5
        for name, points in planes.items()
    }
    sampled_probabilities = {
        run: {
            name: sample(probability, points, order=1)
            for name, points in planes.items()
        }
        for run, probability in probabilities.items()
    }

    tile = PLANE_SIZE * PIXEL_SCALE
    width = LABEL_WIDTH + 2 * tile + COLUMN_GAP
    height = len(DISPLAY_ORDER) * tile + (len(DISPLAY_ORDER) - 1) * ROW_GAP
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row, run in enumerate(DISPLAY_ORDER):
        top = row * (tile + ROW_GAP)
        draw.text((4, top + 4), run, fill=(255, 255, 255), font=font)
        for column, plane_name in enumerate(planes):
            gt_plane = sampled_gt[plane_name]
            if run == "ground_truth":
                image = overlay(gt_plane, None)
            else:
                threshold = float(thresholds["runs"][run]["selected_threshold"])
                prediction = sampled_probabilities[run][plane_name] >= threshold
                image = overlay(gt_plane, prediction)
            left = LABEL_WIDTH + column * (tile + COLUMN_GAP)
            canvas.paste(image, (left, top))
    output_path = output_root / f"real_panel_{panel_index:02d}.png"
    canvas.save(output_path, format="PNG", compress_level=9, optimize=False)
    return {
        "panel_index": panel_index,
        "image": panel["image"],
        "cache_file": cache_name,
        "center_zyx": panel["center_zyx"],
        "frame_zyx": {
            "normal": normal.tolist(),
            "tangent1": tangent_one.tolist(),
            "tangent2": tangent_two.tolist(),
        },
        "source_caches": cache_identities,
        "output": {
            "file": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "width": width,
            "height": height,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--score-input-index", type=Path, required=True)
    parser.add_argument("--public-plan-commit", required=True)
    parser.add_argument("--public-delivery-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"panel output root must start absent: {args.out}")
    _, _, thresholds, panels = validate_public_context(
        plan_path=args.plan,
        delivery_path=args.delivery,
        score_input_index_path=args.score_input_index,
        thresholds_path=args.thresholds,
        panel_manifest_path=args.panel_manifest,
        cache_root=args.test_root,
        public_plan_commit=args.public_plan_commit,
        public_delivery_commit=args.public_delivery_commit,
    )
    if thresholds.get("status") != (
        "thresholds frozen from Scroll-1 validation before test inference"
    ):
        raise RuntimeError("wrong threshold-freeze status")
    if tuple(thresholds.get("runs", {})) != RUN_ORDER:
        raise RuntimeError("threshold run order mismatch")
    if (
        panels.get("status")
        != "model-blind; selected from held-out labels before prediction"
    ):
        raise RuntimeError("real panels were not frozen model-blind")
    selected = panels.get("selected")
    if not isinstance(selected, list) or len(selected) != 4:
        raise RuntimeError("expected exactly four frozen real panels")
    args.out.mkdir(parents=True)
    rendered = [
        render_panel(index, panel, args.test_root, thresholds, args.out)
        for index, panel in enumerate(selected, start=1)
    ]
    payload = {
        "schema_version": "1.0",
        "status": "four model-blind real panels rendered after sealed test delivery",
        "source_thresholds": {
            "file": args.thresholds.name,
            "bytes": args.thresholds.stat().st_size,
            "sha256": sha256_file(args.thresholds),
            "payload_sha256": thresholds["payload_sha256"],
        },
        "source_panel_manifest": {
            "file": args.panel_manifest.name,
            "bytes": args.panel_manifest.stat().st_size,
            "sha256": sha256_file(args.panel_manifest),
            "payload_sha256": panels["payload_sha256"],
        },
        "rendering": {
            "display_order": list(DISPLAY_ORDER),
            "plane_radius_voxels": PLANE_RADIUS,
            "plane_size": PLANE_SIZE,
            "pixel_scale": PIXEL_SCALE,
            "normal": "sigma-3 signed-distance gradient at frozen center",
            "tangent_reference": "coordinate axis least aligned with normal",
            "sampling": {"ground_truth": "nearest", "probability": "linear"},
            "colors": {
                "background": list(BACKGROUND),
                "true_positive_or_ground_truth": list(TRUE_POSITIVE),
                "false_negative": list(FALSE_NEGATIVE),
                "false_positive": list(FALSE_POSITIVE),
            },
            "pillow_version": pillow_version,
            "scipy_version": scipy_version,
        },
        "panels": rendered,
        "scientific_gate": {
            "panel_locations_selected_before_predictions": True,
            "selected_thresholds_used_unchanged": True,
            "panel_locations_or_plane_orientation_tuned_after_predictions": False,
            "visual_gate_assessed_by_renderer": False,
        },
        "renderer": {
            "file": Path(__file__).resolve().name,
            "bytes": Path(__file__).resolve().stat().st_size,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    manifest_path = args.out / "real_panel_render_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    observed = {path.name for path in args.out.iterdir() if path.is_file()}
    expected = {"real_panel_render_manifest.json"} | {
        f"real_panel_{index:02d}.png" for index in range(1, 5)
    }
    if observed != expected:
        raise RuntimeError(f"panel output set mismatch: {sorted(observed)}")
    print("real panel count:", len(rendered))
    print("real panel manifest payload SHA-256:", payload["payload_sha256"])
    print("FOUR_MODEL_BLIND_REAL_PANELS_RENDERED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
