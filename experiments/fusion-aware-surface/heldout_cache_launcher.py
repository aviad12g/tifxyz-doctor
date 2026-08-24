"""Fail-closed launcher for one publicly planned held-out cache job.

The launcher accepts only a job identity present in an immutable public
``heldout_execution_plan.json``.  It verifies the public threshold freeze and
the exact checkpoint, then runs either one Scroll-4/5 cache or one sealed
synthetic-ray job covering all ten preregistered shards separately.  It never
scores or prints scientific endpoints.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path, PurePosixPath

PUBLIC_REPOSITORY = "aviad12g/tifxyz-doctor"
PUBLIC_EXPERIMENT_PATH = "experiments/fusion-aware-surface"
ASSET_DATASET_ID = "aviadcohen1/vesuvius-fusion-aware-training-assets"
ASSET_LEDGER_SHA256 = "1b3d78b2f85808a4a2953b7b8ed3a5969f07714f341cea269fb11746f7892fba"
ASSET_LEDGER_RECORDS = 278
SOURCE_CHECKPOINT_SHA256 = (
    "f1990a02ac91889c1f989522ae0e45421a91cb666320448aaf579d42b081636f"
)
SOURCE_CHECKPOINT_BYTES = 819_171_665
REAL_SPLIT_MANIFEST_SHA256 = (
    "dedc881134d9de2ed2605162f82dfb219b52c6e05629c68223100b148b15d4fe"
)
REAL_SPLIT_RECORDS_SHA256 = (
    "20c600d6061bf8715ada20423a05c2f03a9bc14827b2c79bac7b7e1b3cc0499d"
)
PROTOCOL_CLARIFICATION = {
    "file": "PROTOCOL_CLARIFICATION.md",
    "bytes": 2036,
    "sha256": "c849d7a268465dd17b8c5d0c6774e7caab5caa343eb2dacb0cf4d862db3166dd",
    "first_public_commit": "1e0ced2c928fa0842aa8c601f39dbcd49272247e",
    "scope": "synthetic primary-gate pooling only",
}
CHECKPOINT_FREEZE_COMMIT = "5d98eb683d692c7d4fe7b8f9e4dd0a285e8bf4ff"
CHECKPOINT_MANIFEST_SHA256 = (
    "b8dec8f2c21d2e3686e3435a136de747b3e95f47820b2eb8a41b7044808f8564"
)
INPUT_FREEZE_COMMIT = "62521d98a1771f736cf48dc1e9609a04a5779bb2"
INPUT_MANIFEST_SHA256 = (
    "9e0e70b1e98d9d879ac0b27ba902cbfdc960f4577580146488f076efebda9fad"
)
THRESHOLD_KERNEL_ID = "aviadcohen1/vesuvius-fusion-aware-freeze-thresholds"
THRESHOLD_LAUNCHER_SHA256 = (
    "115c8f27ca5f0a8dee9fe9e56883f9ecbfa4e76dd22742a0f8ba39d5b95a3af9"
)

SOURCE_HASHES = {
    "project/cache_real_predictions.py": "8f89910c6667a135bcc832a3348cbd264b883f04013eb779d704f3cb3fa99035",
    "project/cache_synthetic_rays.py": "a2d4baac41720d0b83d21a5f30ad3947ba7ca9162791995630eb178561e507bc",
    "project/fusion_ray_readout.py": "ddc02747b9aad432a384aba8aeecd00f40c8ad67888cc699ee95e1c26d09f515",
    "project/inference.py": "1ba249a04809de64ecfbb4856e0f81737e7f278a7dad3e8f17ba2c1b5490dce1",
    "project/normalization.py": "42f51a56eeed337dbce08b9af15fa63096993f6353ae72cc879fc9e104794261",
    "project/real_split_manifest.json": REAL_SPLIT_MANIFEST_SHA256,
    "diagnostic/loader059.py": "49a1d4ea3ee611236d53b1291ca2e8cd6e1450f2fc032223ff89a5b53d9ec903",
    "diagnostic/fusion_readout.py": "a533ee940712f1a47111705e4d8fff4990ccffb0afb103c7337a68bff244781c",
    "painter/contrast_phantom.py": "41f2a097b819bc486819d6702ed3949d0bdfe722c2405b1fe86828062da1d9b8",
}
DIRECT_SOURCE_IDENTITIES = {
    "model/Model_epoch499.pth": (
        SOURCE_CHECKPOINT_BYTES,
        SOURCE_CHECKPOINT_SHA256,
    ),
}
EXPECTED_PROPERTIES = {
    "mean": 129.09779357910156,
    "std": 42.188316345214844,
    "percentile_00_5": 44.0,
    "percentile_99_5": 236.0,
}
RUN_ORDER = (
    "baseline",
    "control_seed11",
    "control_seed23",
    "control_seed47",
    "gap8_seed11",
    "gap8_seed23",
    "gap8_seed47",
)
EMBEDDED_CONFIG_PREFIX = b"# HELDOUT_JOB_CONFIG_HEX="


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload_sha256(payload: dict) -> str:
    content = dict(payload)
    observed = content.pop("payload_sha256", None)
    expected = sha256_bytes(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    )
    if observed != expected:
        raise RuntimeError("embedded payload SHA-256 mismatch")
    return expected


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def embedded_job_config_bytes() -> bytes:
    first_line = Path(__file__).resolve().read_bytes().split(b"\n", 1)[0]
    if not first_line.startswith(EMBEDDED_CONFIG_PREFIX):
        raise RuntimeError("held-out job config is not embedded in the kernel source")
    encoded = first_line[len(EMBEDDED_CONFIG_PREFIX) :]
    try:
        return bytes.fromhex(encoded.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("embedded held-out job config is not valid hex") from error


def launcher_source_identity() -> dict:
    raw = Path(__file__).resolve().read_bytes()
    first_line, separator, remainder = raw.partition(b"\n")
    if first_line.startswith(EMBEDDED_CONFIG_PREFIX):
        if not separator or not remainder:
            raise RuntimeError("embedded held-out launcher has no base source")
        raw = remainder
    return {
        "file": "heldout_cache_launcher.py",
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def fetch_public(commit: str, name: str, expected_sha256: str) -> bytes:
    url = (
        f"https://raw.githubusercontent.com/{PUBLIC_REPOSITORY}/{commit}/"
        f"{PUBLIC_EXPERIMENT_PATH}/{name}"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "vesuvius-heldout-cache/1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    if sha256_bytes(raw) != expected_sha256:
        raise RuntimeError(f"public file SHA-256 mismatch: {name}")
    return raw


def safe_relative(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise RuntimeError(f"unsafe relative path: {relative!r}")
    resolved = (root / Path(*pure.parts)).resolve()
    if root.resolve() not in resolved.parents:
        raise RuntimeError(f"path escapes root: {relative!r}")
    return resolved


def find_asset_root(input_root: Path) -> Path:
    matches = [
        path.parent.resolve()
        for path in input_root.glob("**/SOURCE_SHA256SUMS")
        if path.is_file() and sha256_file(path) == ASSET_LEDGER_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen asset mount; found {matches}")
    return matches[0]


def verify_asset_root(root: Path) -> None:
    ledger = root / "SOURCE_SHA256SUMS"
    if sha256_file(ledger) != ASSET_LEDGER_SHA256:
        raise RuntimeError("asset ledger mismatch")
    lines = [
        line for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(lines) != ASSET_LEDGER_RECORDS:
        raise RuntimeError(f"asset ledger record-count mismatch: {len(lines)}")
    records = {}
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = safe_relative(root, relative)
        if relative in records or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"asset source mismatch: {relative}")
        records[relative] = expected
    for relative, expected in SOURCE_HASHES.items():
        if records.get(relative) != expected:
            raise RuntimeError(f"fixed source is not ledger-bound: {relative}")
    for relative, (expected_bytes, expected_sha256) in DIRECT_SOURCE_IDENTITIES.items():
        path = safe_relative(root, relative)
        if (
            not path.is_file()
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != expected_sha256
        ):
            raise RuntimeError(f"fixed source direct identity mismatch: {relative}")


def load_job_config() -> dict:
    raw = embedded_job_config_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("embedded held-out job config must be a JSON object")
    canonical_payload_sha256(payload)
    if payload.get("schema_version") != "1.0":
        raise RuntimeError("held-out job config schema mismatch")
    commit = payload.get("public_plan_commit")
    digest = payload.get("public_plan_file_sha256")
    job_id = payload.get("job_id")
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("invalid public-plan commit")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("invalid public-plan file SHA-256")
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError("invalid job identity")
    return payload


def load_public_plan(config: dict) -> tuple[dict, bytes]:
    raw = fetch_public(
        config["public_plan_commit"],
        "heldout_execution_plan.json",
        config["public_plan_file_sha256"],
    )
    plan = json.loads(raw)
    canonical_payload_sha256(plan)
    checks = {
        "status": "held-out execution plan frozen before held-out inference",
        "public_checkpoint_freeze_commit": CHECKPOINT_FREEZE_COMMIT,
        "public_checkpoint_freeze_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "public_evaluation_input_freeze_commit": INPUT_FREEZE_COMMIT,
        "public_evaluation_input_freeze_manifest_sha256": INPUT_MANIFEST_SHA256,
        "source_split_manifest_sha256": REAL_SPLIT_MANIFEST_SHA256,
        "source_split_records_sha256": REAL_SPLIT_RECORDS_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "run_order": list(RUN_ORDER),
    }
    for key, expected in checks.items():
        if plan.get(key) != expected:
            raise RuntimeError(f"public held-out plan mismatch for {key}")
    launcher = plan.get("heldout_cache_launcher", {})
    if launcher != launcher_source_identity():
        raise RuntimeError("running launcher differs from the public held-out plan")
    if plan.get("pre_inference_protocol_clarification") != PROTOCOL_CLARIFICATION:
        raise RuntimeError("public held-out plan lacks the frozen clarification")
    clarification_raw = fetch_public(
        config["public_plan_commit"],
        PROTOCOL_CLARIFICATION["file"],
        PROTOCOL_CLARIFICATION["sha256"],
    )
    if len(clarification_raw) != PROTOCOL_CLARIFICATION["bytes"]:
        raise RuntimeError("public protocol clarification byte-size mismatch")
    gate = plan.get("scientific_gate", {})
    if gate.get("thresholds_publicly_frozen_before_held_out_inference") is not True:
        raise RuntimeError("public held-out plan lacks threshold-freeze gate")
    if (
        gate.get("protocol_clarification_publicly_frozen_before_held_out_inference")
        is not True
    ):
        raise RuntimeError("public held-out plan lacks clarification-freeze gate")
    if gate.get("held_out_outputs_inspected_when_plan_frozen") is not False:
        raise RuntimeError("public held-out plan does not preserve blind gate")
    jobs = plan.get("real_test_jobs", []) + plan.get("synthetic_ray_jobs", [])
    selected = [record for record in jobs if record.get("job_id") == config["job_id"]]
    if len(selected) != 1:
        raise RuntimeError(
            f"job is absent or duplicated in public plan: {config['job_id']}"
        )
    return {"plan": plan, "job": selected[0]}, raw


def find_unique_identity(input_root: Path, name: str, digest: str, size: int) -> Path:
    matches = []
    for path in input_root.rglob(name):
        if (
            path.is_file()
            and path.stat().st_size == size
            and sha256_file(path) == digest
        ):
            matches.append(path.resolve())
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one mounted {name} with frozen identity; found {matches}"
        )
    return matches[0]


def verify_threshold_binding(
    input_root: Path, plan: dict, job: dict
) -> tuple[Path, dict]:
    binding = plan.get("threshold_binding", {})
    if binding.get("kernel_id") != THRESHOLD_KERNEL_ID:
        raise RuntimeError("held-out plan points to another threshold kernel")
    if binding.get("launcher_sha256") != THRESHOLD_LAUNCHER_SHA256:
        raise RuntimeError("held-out plan points to another threshold launcher")
    commit = binding.get("public_freeze_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("invalid public threshold-freeze commit")
    record = binding.get("frozen_thresholds", {})
    threshold_raw = fetch_public(commit, record.get("file"), record.get("sha256"))
    if len(threshold_raw) != int(record.get("bytes", -1)):
        raise RuntimeError("public threshold file byte-size mismatch")
    mounted = find_unique_identity(
        input_root, record["file"], record["sha256"], int(record["bytes"])
    )
    if mounted.read_bytes() != threshold_raw:
        raise RuntimeError("mounted and public threshold files differ")
    thresholds = json.loads(threshold_raw)
    if canonical_payload_sha256(thresholds) != record.get("payload_sha256"):
        raise RuntimeError("threshold payload identity mismatch")
    if (
        thresholds.get("status")
        != "thresholds frozen from Scroll-1 validation before test inference"
    ):
        raise RuntimeError("threshold status mismatch")
    if thresholds.get("source_split_records_sha256") != REAL_SPLIT_RECORDS_SHA256:
        raise RuntimeError("threshold split identity mismatch")
    selected = thresholds.get("runs", {}).get(job["run"], {}).get("selected_threshold")
    if selected != job.get("selected_threshold"):
        raise RuntimeError("job threshold differs from public frozen threshold")
    run_record = binding.get("threshold_run_manifest", {})
    run_raw = fetch_public(commit, run_record.get("file"), run_record.get("sha256"))
    if len(run_raw) != int(run_record.get("bytes", -1)):
        raise RuntimeError("public threshold-run manifest byte-size mismatch")
    run_mounted = find_unique_identity(
        input_root, run_record["file"], run_record["sha256"], int(run_record["bytes"])
    )
    if run_mounted.read_bytes() != run_raw:
        raise RuntimeError("mounted and public threshold-run manifests differ")
    threshold_run = json.loads(run_raw)
    if canonical_payload_sha256(threshold_run) != run_record.get("payload_sha256"):
        raise RuntimeError("threshold-run payload identity mismatch")
    return mounted, thresholds


def verify_split(asset_root: Path) -> dict:
    split_path = asset_root / "project" / "real_split_manifest.json"
    split = load_json(split_path)
    records = split.get("records")
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    if split.get("records_sha256") != REAL_SPLIT_RECORDS_SHA256:
        raise RuntimeError("split records identity mismatch")
    if sha256_bytes(canonical) != REAL_SPLIT_RECORDS_SHA256:
        raise RuntimeError("split records payload mismatch")
    counts = {}
    for record in records:
        key = f"{record['scroll']}:{record['split']}"
        counts[key] = counts.get(key, 0) + 1
    if counts != {"s1:train": 138, "s1:validation": 24, "s4:test": 36, "s5:test": 2}:
        raise RuntimeError(f"split counts mismatch: {counts}")
    return split


def verify_model_state(input_root: Path, job: dict) -> Path | None:
    if job["run"] == "baseline":
        if any(
            job.get(key) is not None
            for key in ("kernel_id", "kernel_version", "model_state_sha256")
        ):
            raise RuntimeError("baseline job unexpectedly carries a trained model")
        return None
    digest = job.get("model_state_sha256")
    candidates = []
    for path in input_root.rglob("*.pth"):
        if path.is_file() and sha256_file(path) == digest:
            candidates.append(path.resolve())
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one exact model state for {job['run']}; found {candidates}"
        )
    checkpoint = candidates[0]
    relative = checkpoint.relative_to(input_root)
    mount = input_root / relative.parts[0]
    manifests = list(mount.rglob("training_run_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(
            f"expected one training manifest beside {job['run']} checkpoint"
        )
    if manifests[0].stat().st_size != int(job.get("training_run_manifest_bytes", -1)):
        raise RuntimeError("training manifest byte-size mismatch")
    if sha256_file(manifests[0]) != job.get("training_run_manifest_sha256"):
        raise RuntimeError("training manifest SHA-256 mismatch")
    training = load_json(manifests[0])
    arm, seed = job["run"].rsplit("_seed", 1)
    if (
        training.get("status")
        != "training complete; validation and test endpoints not computed"
    ):
        raise RuntimeError("training manifest status mismatch")
    if (training.get("arm"), int(training.get("seed", -1))) != (arm, int(seed)):
        raise RuntimeError("training manifest identity mismatch")
    if training.get("outputs", {}).get("checkpoint_sha256") != digest:
        raise RuntimeError("training manifest points to another checkpoint")
    return checkpoint


def prepare_test_only(asset_root: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError("held-out scratch must start absent")
    split = verify_split(asset_root)
    records = [record for record in split["records"] if record["split"] == "test"]
    if len(records) != 38:
        raise RuntimeError(f"unexpected held-out split count: {len(records)}")
    source_images = asset_root / "archives" / "images_s4_s5" / "imagesTr"
    source_labels = asset_root / "archives" / "labels" / "labelsTr"
    expected_images = {record["image"] for record in records}
    expected_labels = {record["label"] for record in records}
    observed_images = {path.name for path in source_images.glob("*.tif")}
    observed_labels = {
        path.name
        for path in source_labels.glob("*.tif")
        if path.name.startswith(("s4_", "s5_"))
    }
    if observed_images != expected_images or observed_labels != expected_labels:
        raise RuntimeError("expanded held-out source file set mismatch")
    output_images = destination / "imagesTr"
    output_labels = destination / "labelsTr"
    output_images.mkdir(parents=True)
    output_labels.mkdir(parents=True)
    for record in records:
        for source, target, digest in (
            (
                source_images / record["image"],
                output_images / record["image"],
                record["image_sha256"],
            ),
            (
                source_labels / record["label"],
                output_labels / record["label"],
                record["label_sha256"],
            ),
        ):
            if not source.is_file() or sha256_file(source) != digest:
                raise RuntimeError(f"expanded held-out source hash mismatch: {source.name}")
            shutil.copyfile(source, target)
            if sha256_file(target) != digest:
                raise RuntimeError(f"held-out scratch copy mismatch: {target.name}")
    if list(destination.rglob("s1_*.tif")):
        raise RuntimeError("Scroll-1 leaked into held-out scratch")


def install_runtime() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "torch==2.5.1",
            "torchvision==0.20.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "numpy==1.26.4",
            "tifffile==2025.2.18",
            "imagecodecs==2024.12.30",
            "huggingface-hub==1.11.0",
            "timm==1.0.27",
            "einops==0.8.1",
        ],
        check=True,
    )
    import torch

    if torch.__version__.split("+", 1)[0] != "2.5.1" or not torch.cuda.is_available():
        raise RuntimeError("required CUDA runtime is unavailable")
    expected = {
        "torchvision": "0.20.1",
        "numpy": "1.26.4",
        "tifffile": "2025.2.18",
        "imagecodecs": "2024.12.30",
        "timm": "1.0.27",
        "einops": "0.8.1",
    }
    observed = {
        name: importlib.metadata.version(name).split("+", 1)[0] for name in expected
    }
    if observed != expected:
        raise RuntimeError(f"held-out runtime package mismatch: {observed}")


def runtime_identity() -> dict:
    import torch

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "timm": importlib.metadata.version("timm"),
        "numpy": importlib.metadata.version("numpy"),
        "cuda_runtime": str(torch.version.cuda),
        "gpu_name": torch.cuda.get_device_name(0),
    }


def configure_compat(scratch: Path) -> Path:
    import torch

    if "reason" not in inspect.signature(torch.compiler.disable).parameters:
        original = torch.compiler.disable

        def disable_compat(fn=None, recursive=True, *, reason=None):
            return original(fn=fn, recursive=recursive)

        torch.compiler.disable = disable_compat
    root = scratch / "runtime-compat"
    root.mkdir()
    (root / "sitecustomize.py").write_text(
        "import inspect\n"
        "import torch\n"
        "if 'reason' not in inspect.signature(torch.compiler.disable).parameters:\n"
        "    _original = torch.compiler.disable\n"
        "    def _compat(fn=None, recursive=True, *, reason=None):\n"
        "        return _original(fn=fn, recursive=recursive)\n"
        "    torch.compiler.disable = _compat\n",
        encoding="utf-8",
    )
    return root


def execute_job(
    asset_root: Path,
    job: dict,
    threshold_path: Path,
    model_state: Path | None,
    output: Path,
    scratch: Path,
    compat: Path,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(compat),
            str(asset_root / "network-source" / "vesuvius" / "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    commands = []
    if job["mode"] == "real_test_cache":
        real_data = scratch / "real-test-data"
        prepare_test_only(asset_root, real_data)
        commands.append(
            [
                sys.executable,
                str(asset_root / "project" / "cache_real_predictions.py"),
                "--run",
                job["run"],
                "--split",
                "test",
                "--real-data",
                str(real_data),
                "--split-manifest",
                str(asset_root / "project" / "real_split_manifest.json"),
                "--diagnostic-scripts",
                str(asset_root / "diagnostic"),
                "--source-checkpoint",
                str(asset_root / "model" / "Model_epoch499.pth"),
                "--threshold-manifest",
                str(threshold_path),
                "--out",
                str(output),
            ]
        )
    elif job["mode"] == "synthetic_ray_cache":
        if job.get("shard_indices") != list(range(10)) or job.get("shard_count") != 10:
            raise RuntimeError(
                "public plan does not carry the frozen ten synthetic shards"
            )
        for shard in job["shard_indices"]:
            commands.append(
                [
                    sys.executable,
                    str(asset_root / "project" / "cache_synthetic_rays.py"),
                    "--run",
                    job["run"],
                    "--shard-index",
                    str(shard),
                    "--shard-count",
                    "10",
                    "--painter-scripts",
                    str(asset_root / "painter"),
                    "--diagnostic-scripts",
                    str(asset_root / "diagnostic"),
                    "--reference-reader",
                    str(asset_root / "diagnostic" / "fusion_readout.py"),
                    "--source-checkpoint",
                    str(asset_root / "model" / "Model_epoch499.pth"),
                    "--threshold-manifest",
                    str(threshold_path),
                    "--split-manifest",
                    str(asset_root / "project" / "real_split_manifest.json"),
                    "--out",
                    str(output),
                ]
            )
    else:
        raise RuntimeError(f"unsupported planned mode: {job['mode']}")
    print(f"HELDOUT_CACHE_START job={job['job_id']}", flush=True)
    for index, command in enumerate(commands):
        if model_state is not None:
            command.extend(["--model-state", str(model_state)])
        print(
            f"HELDOUT_CACHE_INVOCATION_START job={job['job_id']} "
            f"invocation={index + 1}/{len(commands)}",
            flush=True,
        )
        subprocess.run(
            command,
            cwd=asset_root / "project",
            env=env,
            check=True,
            timeout=12 * 60 * 60,
        )
        print(
            f"HELDOUT_CACHE_INVOCATION_COMPLETE job={job['job_id']} "
            f"invocation={index + 1}/{len(commands)}",
            flush=True,
        )
    print(f"HELDOUT_CACHE_COMPLETE job={job['job_id']}", flush=True)


def expected_synthetic_names(shard: int) -> set[str]:
    names = []
    for seed in (300, 301, 302, 303, 304):
        for pitch in (170, 200, 230, 260):
            for papyrus in (35, 50, 65, 90):
                names.append(f"primary_seed{seed}_pitch{pitch}_pap{papyrus}")
        for papyrus in (35, 50, 65, 90):
            names.append(f"control_seed{seed}_pitch700_pap{papyrus}")
    return {name for index, name in enumerate(names) if index % 10 == shard}


def freeze_job_index(
    *,
    output: Path,
    config: dict,
    plan: dict,
    public_plan_raw: bytes,
    job: dict,
    thresholds: dict,
    runtime: dict,
    split: dict,
) -> None:
    if job["mode"] == "real_test_cache":
        expected_names = {
            Path(record["image"]).with_suffix(".npz").name
            for record in split["records"]
            if record["split"] == "test"
        }
        manifest_specs = [
            {
                "name": f"cache_manifest_{job['run']}_test.json",
                "expected_names": expected_names,
                "status": "test probabilities cached",
                "file_key": "files",
                "count": 38,
                "extra_checks": {"split": "test"},
            }
        ]
    else:
        if job.get("shard_indices") != list(range(10)):
            raise RuntimeError(
                "synthetic job index does not cover all ten frozen shards"
            )
        manifest_specs = [
            {
                "name": f"synthetic_manifest_{job['run']}_shard{shard:02d}.json",
                "expected_names": {
                    name + ".npz" for name in expected_synthetic_names(shard)
                },
                "status": "sealed synthetic rays cached; endpoints not scored",
                "file_key": "cells",
                "count": 10,
                "extra_checks": {"shard_index": shard, "shard_count": 10},
            }
            for shard in range(10)
        ]
    expected_manifest_names = {spec["name"] for spec in manifest_specs}
    root_files = {path.name for path in output.iterdir() if path.is_file()}
    root_directories = {path.name for path in output.iterdir() if path.is_dir()}
    if root_files != expected_manifest_names or root_directories != {job["run"]}:
        raise RuntimeError(
            f"sealed output-root mismatch: files={sorted(root_files)} "
            f"directories={sorted(root_directories)}"
        )
    manifest_records = []
    sealed_files = []
    expected_all_cache_names = set()
    for spec in manifest_specs:
        manifest_path = output / spec["name"]
        manifest = load_json(manifest_path)
        canonical_payload_sha256(manifest)
        checks = {
            "status": spec["status"],
            "run": job["run"],
            "source_split_records_sha256": REAL_SPLIT_RECORDS_SHA256,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "model_state_sha256": job.get("model_state_sha256"),
            "selected_threshold": job["selected_threshold"],
            "normalization_scheme": "CTNormalization",
            "intensity_properties": EXPECTED_PROPERTIES,
            **spec["extra_checks"],
        }
        for key, expected in checks.items():
            if manifest.get(key) != expected:
                raise RuntimeError(f"sealed cache manifest mismatch for {key}")
        records = manifest.get(spec["file_key"])
        if not isinstance(records, list) or len(records) != spec["count"]:
            raise RuntimeError("sealed cache manifest file count mismatch")
        listed = {record.get("file") for record in records}
        if listed != spec["expected_names"]:
            raise RuntimeError("sealed cache manifest file set mismatch")
        if expected_all_cache_names & listed:
            raise RuntimeError("sealed cache file appears in multiple shard manifests")
        expected_all_cache_names |= listed
        for record in records:
            path = output / job["run"] / record["file"]
            if not path.is_file() or path.stat().st_size != int(record["bytes"]):
                raise RuntimeError(
                    f"sealed cache missing or byte-size mismatch: {record['file']}"
                )
            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"sealed cache SHA-256 mismatch: {record['file']}")
        sealed_files.extend(records)
        manifest_records.append(
            {
                "file": spec["name"],
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
                "payload_sha256": manifest["payload_sha256"],
            }
        )
    observed_cache_names = {
        path.name for path in (output / job["run"]).iterdir() if path.is_file()
    }
    if observed_cache_names != expected_all_cache_names:
        raise RuntimeError("sealed run-directory file set mismatch")
    binding = plan["threshold_binding"]
    payload = {
        "schema_version": "1.0",
        "status": "one publicly planned held-out cache job sealed without scoring",
        "job": job,
        "public_execution_plan": {
            "commit": config["public_plan_commit"],
            "file_sha256": sha256_bytes(public_plan_raw),
            "payload_sha256": plan["payload_sha256"],
        },
        "threshold_binding": binding,
        "threshold_payload_sha256": thresholds["payload_sha256"],
        "cache_manifests": manifest_records,
        "sealed_cache_files": sealed_files,
        "runtime": runtime,
        "launcher": launcher_source_identity(),
        "job_config": {
            "encoding": "hex in first source comment",
            "bytes": len(embedded_job_config_bytes()),
            "sha256": sha256_bytes(embedded_job_config_bytes()),
            "payload_sha256": config["payload_sha256"],
        },
        "scientific_gate": {
            "thresholds_were_publicly_frozen_before_this_job": True,
            "held_out_inference_executed": True,
            "scientific_endpoints_scored": False,
            "scientific_endpoints_printed": False,
            "manual_threshold_override_used": False,
            "cache_payload_opened_or_inspected_by_launcher": False,
        },
    }
    payload["payload_sha256"] = sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )
    destination = output / "heldout_job_index.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("held-out job-index payload SHA-256:", payload["payload_sha256"])
    print("PUBLICLY_PLANNED_HELDOUT_CACHE_SEALED")


def main() -> int:
    input_root = Path(os.environ.get("KAGGLE_INPUT_PATH", "/kaggle/input"))
    working = Path(os.environ.get("KAGGLE_WORKING_PATH", "/kaggle/working"))
    scratch = (
        Path(os.environ.get("KAGGLE_TEMP_PATH", "/kaggle/temp"))
        / "fusion-aware-heldout"
    )
    if scratch.exists():
        raise RuntimeError(f"held-out scratch must start absent: {scratch}")
    scratch.mkdir(parents=True)
    config = load_job_config()
    public, public_plan_raw = load_public_plan(config)
    plan, job = public["plan"], public["job"]
    asset_root = find_asset_root(input_root)
    verify_asset_root(asset_root)
    split = verify_split(asset_root)
    threshold_path, thresholds = verify_threshold_binding(input_root, plan, job)
    model_state = verify_model_state(input_root, job)
    install_runtime()
    compat = configure_compat(scratch)
    runtime = runtime_identity()
    output = working / f"fusion-aware-heldout-{job['job_id']}"
    output.mkdir(parents=True, exist_ok=False)
    execute_job(asset_root, job, threshold_path, model_state, output, scratch, compat)
    freeze_job_index(
        output=output,
        config=config,
        plan=plan,
        public_plan_raw=public_plan_raw,
        job=job,
        thresholds=thresholds,
        runtime=runtime,
        split=split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
