#!/usr/bin/env python3
"""Fail-closed launcher for one publicly frozen held-out scorer.

The real and synthetic scorers are launched as separate private kernels after
all 14 cache jobs and their delivery manifest are publicly frozen. Scientific
stdout is captured and hashed, not echoed. The launcher emits only the sealed
result JSON and an identity-only run manifest.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path, PurePosixPath

PUBLIC_REPOSITORY = "aviad12g/tifxyz-doctor"
PUBLIC_EXPERIMENT_PATH = "experiments/fusion-aware-surface"
ASSET_LEDGER_SHA256 = "1b3d78b2f85808a4a2953b7b8ed3a5969f07714f341cea269fb11746f7892fba"
ASSET_LEDGER_RECORDS = 278
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
RUN_ORDER = (
    "baseline",
    "control_seed11",
    "control_seed23",
    "control_seed47",
    "gap8_seed11",
    "gap8_seed23",
    "gap8_seed47",
)
PROJECT_HASHES = {
    "cache_real_predictions.py": "8f89910c6667a135bcc832a3348cbd264b883f04013eb779d704f3cb3fa99035",
    "cache_synthetic_rays.py": "a2d4baac41720d0b83d21a5f30ad3947ba7ca9162791995630eb178561e507bc",
    "freeze_thresholds.py": "e04e4c4540609388d5d6f1d18e766dd5da87c8b327102b304a3d024555910c40",
    "fusion_loss.py": "a5611df0b1535f033bfc9a7a68600e8303ff5948623f5c983eb95f0bd3977ef4",
    "fusion_ray_readout.py": "ddc02747b9aad432a384aba8aeecd00f40c8ad67888cc699ee95e1c26d09f515",
    "gap_supervision.py": "f7bdd74af92b1de7b507515aa879baa4c2622b3051f6a5b1c053e5329778e08c",
    "inference.py": "1ba249a04809de64ecfbb4856e0f81737e7f278a7dad3e8f17ba2c1b5490dce1",
    "normalization.py": "42f51a56eeed337dbce08b9af15fa63096993f6353ae72cc879fc9e104794261",
    "official_metric.py": "da5236e67117c1ca6a634c38656dad5cad7579d97017e27bd1c3854f6bcec0fb",
    "real_split_manifest.json": REAL_SPLIT_MANIFEST_SHA256,
    "score_real_test.py": "3529b8213237a60d392ffec04efca602988b3242f6af8cadc87423e8e224bb79",
    "score_synthetic_test.py": "d64049b1048dee8274f8416b979384b3342f671811db356ba79052726401063f",
    "score_synthetic_test_v2.py": "d594cea7d58b08bbeccab5ec65f0a3d64191a70d07e9423314cd607d9fe53d05",
    "train_fusion_aware.py": "c793f5d76103a63e4c3d11f460d1603f12d7d55a31e2495eab08d0d070940d07",
    "verify_official_metric.py": "09ba89028aa48405a3fc96390b76b455bd0b26d07c908b976f8d6fdfd1aa4e00",
}
PUBLIC_ONLY_PROJECT_FILES = frozenset({"score_synthetic_test_v2.py"})
RUNTIME_PACKAGES = {
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "numpy": "1.26.4",
    "scipy": "1.15.3",
    "tifffile": "2025.2.18",
    "timm": "1.0.27",
    "einops": "0.8.1",
}
EMBEDDED_CONFIG_PREFIX = b"# SCORING_JOB_CONFIG_HEX="


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


def embedded_scoring_config_bytes() -> bytes:
    first_line = Path(__file__).resolve().read_bytes().split(b"\n", 1)[0]
    if not first_line.startswith(EMBEDDED_CONFIG_PREFIX):
        raise RuntimeError("scoring job config is not embedded in the kernel source")
    encoded = first_line[len(EMBEDDED_CONFIG_PREFIX) :]
    try:
        return bytes.fromhex(encoded.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("embedded scoring config is not valid hex") from error


def launcher_source_identity() -> dict:
    raw = Path(__file__).resolve().read_bytes()
    first_line, separator, remainder = raw.partition(b"\n")
    if first_line.startswith(EMBEDDED_CONFIG_PREFIX):
        if not separator or not remainder:
            raise RuntimeError("embedded scoring launcher has no base source")
        raw = remainder
    return {
        "file": "one_shot_scoring_launcher.py",
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def fetch_public(commit: str, name: str, expected_sha256: str) -> bytes:
    url = (
        f"https://raw.githubusercontent.com/{PUBLIC_REPOSITORY}/{commit}/"
        f"{PUBLIC_EXPERIMENT_PATH}/{name}"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "vesuvius-one-shot-scoring/1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    if sha256_bytes(raw) != expected_sha256:
        raise RuntimeError(f"public file SHA-256 mismatch: {name}")
    return raw


def require_hex(value: object, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"invalid {label}")
    return value


def load_config() -> dict:
    raw = embedded_scoring_config_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("embedded scoring config must be a JSON object")
    canonical_payload_sha256(payload)
    if set(payload) != {
        "schema_version",
        "mode",
        "public_plan_commit",
        "public_plan_file_sha256",
        "public_delivery_commit",
        "public_delivery_file_sha256",
        "payload_sha256",
    }:
        raise RuntimeError("scoring-job config schema mismatch")
    if payload["schema_version"] != "1.0" or payload["mode"] not in {
        "real",
        "synthetic",
    }:
        raise RuntimeError("scoring-job config identity mismatch")
    require_hex(payload["public_plan_commit"], 40, "public plan commit")
    require_hex(payload["public_plan_file_sha256"], 64, "public plan SHA-256")
    require_hex(payload["public_delivery_commit"], 40, "public delivery commit")
    require_hex(payload["public_delivery_file_sha256"], 64, "public delivery SHA-256")
    return payload


def write_public_files(config: dict, scratch: Path) -> tuple[Path, Path, dict, dict]:
    plan_raw = fetch_public(
        config["public_plan_commit"],
        "heldout_execution_plan.json",
        config["public_plan_file_sha256"],
    )
    delivery_raw = fetch_public(
        config["public_delivery_commit"],
        "heldout_cache_delivery_manifest.json",
        config["public_delivery_file_sha256"],
    )
    plan = json.loads(plan_raw)
    delivery = json.loads(delivery_raw)
    canonical_payload_sha256(plan)
    canonical_payload_sha256(delivery)
    if plan.get("pre_inference_protocol_clarification") != PROTOCOL_CLARIFICATION:
        raise RuntimeError("public plan lacks the frozen protocol clarification")
    clarification_raw = fetch_public(
        config["public_plan_commit"],
        PROTOCOL_CLARIFICATION["file"],
        PROTOCOL_CLARIFICATION["sha256"],
    )
    if len(clarification_raw) != PROTOCOL_CLARIFICATION["bytes"]:
        raise RuntimeError("public protocol clarification byte-size mismatch")
    plan_path = scratch / "heldout_execution_plan.json"
    delivery_path = scratch / "heldout_cache_delivery_manifest.json"
    plan_path.write_bytes(plan_raw)
    delivery_path.write_bytes(delivery_raw)
    return plan_path, delivery_path, plan, delivery


def materialize_cache_job_plan(plan: dict, scratch: Path) -> tuple[Path, dict]:
    correction = plan.get("result_blind_scoring_asset_correction", {})
    identity = correction.get("predecessor_public_execution_plan")
    if not isinstance(identity, dict) or set(identity) != {
        "commit",
        "file",
        "bytes",
        "sha256",
        "payload_sha256",
    }:
        raise RuntimeError("cache-job execution-plan identity is absent")
    raw = fetch_public(identity["commit"], identity["file"], identity["sha256"])
    if len(raw) != identity["bytes"]:
        raise RuntimeError("cache-job execution-plan byte-size mismatch")
    payload = json.loads(raw)
    if canonical_payload_sha256(payload) != identity["payload_sha256"]:
        raise RuntimeError("cache-job execution-plan payload mismatch")
    destination = scratch / "cache_job_execution_plan.json"
    destination.write_bytes(raw)
    return destination, payload


def local_identity(path: Path) -> dict:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def materialize_public_helper(commit: str, identity: dict, scratch: Path) -> Path:
    if set(identity) != {"file", "bytes", "sha256"}:
        raise RuntimeError("public helper identity schema mismatch")
    raw = fetch_public(commit, identity["file"], identity["sha256"])
    if len(raw) != identity["bytes"]:
        raise RuntimeError(f"public helper byte-size mismatch: {identity['file']}")
    destination = scratch / identity["file"]
    destination.write_bytes(raw)
    return destination


def verify_public_contract(
    plan: dict,
    delivery: dict,
    mode: str,
    stager: Path,
    metric: Path,
    panel_renderer: Path,
) -> None:
    if plan.get("one_shot_scoring_launcher") != launcher_source_identity():
        raise RuntimeError("scoring launcher differs from public plan")
    if plan.get("one_shot_scoring_stager") != local_identity(stager):
        raise RuntimeError("scoring stager differs from public plan")
    if plan.get("metric_runtime_preparer") != local_identity(metric):
        raise RuntimeError("metric preparer differs from public plan")
    if plan.get("real_panel_renderer") != local_identity(panel_renderer):
        raise RuntimeError("real-panel renderer differs from public plan")
    if plan.get("pre_inference_protocol_clarification") != PROTOCOL_CLARIFICATION:
        raise RuntimeError("public plan lacks the frozen protocol clarification")
    scorer = plan.get("one_shot_scorers", {}).get(mode, {})
    expected_name = (
        "score_real_test.py" if mode == "real" else "score_synthetic_test_v2.py"
    )
    if scorer.get("script") != {
        "file": expected_name,
        "sha256": PROJECT_HASHES[expected_name],
    }:
        raise RuntimeError("public plan scorer identity mismatch")
    if delivery.get("threshold_binding") != plan.get("threshold_binding"):
        raise RuntimeError("delivery threshold binding mismatch")
    if (
        delivery.get("scientific_gate", {}).get(
            "one_shot_scoring_permitted_after_this_public_freeze"
        )
        is not True
    ):
        raise RuntimeError("public delivery does not authorize one-shot scoring")


def safe_relative(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise RuntimeError(f"unsafe asset path: {relative!r}")
    path = (root / Path(*pure.parts)).resolve()
    if root.resolve() not in path.parents:
        raise RuntimeError(f"asset path escapes root: {relative!r}")
    return path


def find_asset_root(input_root: Path) -> Path:
    matches = [
        path.parent.resolve()
        for path in input_root.rglob("SOURCE_SHA256SUMS")
        if path.is_file() and sha256_file(path) == ASSET_LEDGER_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen asset mount; found {matches}")
    root = matches[0]
    records = {}
    lines = (root / "SOURCE_SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if len(lines) != ASSET_LEDGER_RECORDS:
        raise RuntimeError("asset ledger count mismatch")
    for line in lines:
        digest, relative = line.split("  ", 1)
        path = safe_relative(root, relative)
        if relative in records or not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"asset ledger file mismatch: {relative}")
        records[relative] = digest
    for name, digest in PROJECT_HASHES.items():
        if name in PUBLIC_ONLY_PROJECT_FILES:
            continue
        if records.get(f"project/{name}") != digest:
            raise RuntimeError(f"project source is not asset-ledger bound: {name}")
    return root


def materialize_public_synthetic_scorer(commit: str, scratch: Path) -> Path:
    name = "score_synthetic_test_v2.py"
    raw = fetch_public(commit, name, PROJECT_HASHES[name])
    destination = scratch / name
    destination.write_bytes(raw)
    return destination


def verify_split(project: Path) -> dict:
    path = project / "real_split_manifest.json"
    if sha256_file(path) != REAL_SPLIT_MANIFEST_SHA256:
        raise RuntimeError("split manifest identity mismatch")
    split = load_json(path)
    records = split.get("records")
    digest = sha256_bytes(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    )
    if split.get("records_sha256") != REAL_SPLIT_RECORDS_SHA256 or digest != (
        REAL_SPLIT_RECORDS_SHA256
    ):
        raise RuntimeError("split records identity mismatch")
    return split


def find_threshold(input_root: Path, plan: dict) -> tuple[Path, dict]:
    binding = plan["threshold_binding"]
    record = binding["frozen_thresholds"]
    public_raw = fetch_public(
        binding["public_freeze_commit"], record["file"], record["sha256"]
    )
    matches = [
        path.resolve()
        for path in input_root.rglob(record["file"])
        if path.is_file()
        and path.stat().st_size == record["bytes"]
        and sha256_file(path) == record["sha256"]
    ]
    if len(matches) != 1 or matches[0].read_bytes() != public_raw:
        raise RuntimeError("expected one exact mounted threshold freeze")
    thresholds = json.loads(public_raw)
    if canonical_payload_sha256(thresholds) != record["payload_sha256"]:
        raise RuntimeError("threshold payload identity mismatch")
    if thresholds.get("source_split_records_sha256") != REAL_SPLIT_RECORDS_SHA256:
        raise RuntimeError("threshold split identity mismatch")
    if tuple(thresholds.get("runs", {})) != RUN_ORDER:
        raise RuntimeError("threshold run order mismatch")
    return matches[0], thresholds


def install_scoring_import_runtime() -> dict[str, str]:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "--only-binary=:all:",
            "torch==2.5.1+cpu",
            "torchvision==0.20.1+cpu",
            "numpy==1.26.4",
            "scipy==1.15.3",
            "tifffile==2025.2.18",
            "timm==1.0.27",
            "einops==0.8.1",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "--extra-index-url",
            "https://pypi.org/simple",
        ],
        check=True,
    )
    observed = {
        name: importlib.metadata.version(name).split("+", 1)[0]
        for name in RUNTIME_PACKAGES
    }
    if observed != RUNTIME_PACKAGES:
        raise RuntimeError(f"scoring import runtime mismatch: {observed}")
    return observed


def configure_compat(scratch: Path) -> Path:
    import torch

    if "reason" not in inspect.signature(torch.compiler.disable).parameters:
        original = torch.compiler.disable

        def disable_compat(fn=None, recursive=True, *, reason=None):
            del reason
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


def stage_inputs(
    *,
    mode: str,
    input_root: Path,
    scratch: Path,
    plan_path: Path,
    cache_job_plan_path: Path,
    delivery_path: Path,
    config: dict,
    stager: Path,
) -> tuple[Path, dict]:
    staged = scratch / f"staged-{mode}"
    command = [
        sys.executable,
        str(stager),
        "--mode",
        mode,
        "--input-root",
        str(input_root),
        "--plan",
        str(plan_path),
        "--cache-job-plan",
        str(cache_job_plan_path),
        "--delivery",
        str(delivery_path),
        "--public-plan-commit",
        config["public_plan_commit"],
        "--public-delivery-commit",
        config["public_delivery_commit"],
        "--out",
        str(staged),
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("held-out scoring input staging failed")
    index = load_json(staged / "score_input_index.json")
    canonical_payload_sha256(index)
    if (
        index.get("mode") != mode
        or index.get("scientific_gate", {}).get("one_shot_scoring_permitted")
        is not True
    ):
        raise RuntimeError("staged scoring input index gate mismatch")
    return staged, {
        "command": command,
        "stdout_bytes": len(result.stdout),
        "stdout_sha256": sha256_bytes(result.stdout),
        "stderr_bytes": len(result.stderr),
        "stderr_sha256": sha256_bytes(result.stderr),
        "index": local_identity(staged / "score_input_index.json")
        | {"payload_sha256": index["payload_sha256"]},
    }


def load_metric_preparer(path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("frozen_metric_preparer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen metric preparer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepare = getattr(module, "prepare", None)
    if not callable(prepare):
        raise TypeError("frozen metric preparer has no callable prepare")
    return prepare


def execute_scorer(
    *,
    mode: str,
    staged: Path,
    project: Path,
    split_path: Path,
    threshold_path: Path,
    synthetic_scorer_path: Path | None,
    output: Path,
    environment: dict[str, str],
) -> tuple[Path, dict]:
    script_name = (
        "score_real_test.py" if mode == "real" else "score_synthetic_test_v2.py"
    )
    result_name = (
        "sealed_real_test_results.json"
        if mode == "real"
        else "sealed_synthetic_test_results.json"
    )
    result_path = output / result_name
    root_flag = "--test-root" if mode == "real" else "--ray-root"
    script_path = project / script_name
    if mode == "synthetic":
        if synthetic_scorer_path is None:
            raise RuntimeError("public synthetic scorer was not materialized")
        script_path = synthetic_scorer_path
    if script_path.name != script_name or sha256_file(script_path) != PROJECT_HASHES[
        script_name
    ]:
        raise RuntimeError("one-shot scorer source identity mismatch")
    command = [
        sys.executable,
        str(script_path),
        root_flag,
        str(staged),
        "--split-manifest",
        str(split_path),
        "--threshold-manifest",
        str(threshold_path),
        "--out",
        str(result_path),
    ]
    if mode == "real":
        command.extend(["--worker", str(project / "official_metric.py")])
    completed = subprocess.run(
        command,
        cwd=project,
        env=environment,
        capture_output=True,
        check=False,
        timeout=42_000,
    )
    if completed.returncode != 0:
        raise RuntimeError("one-shot scorer failed; scientific stdout remains sealed")
    return result_path, {
        "script": {"file": script_name, "sha256": PROJECT_HASHES[script_name]},
        "command": command,
        "returncode": completed.returncode,
        "stdout_bytes": len(completed.stdout),
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stderr_bytes": len(completed.stderr),
        "stderr_sha256": sha256_bytes(completed.stderr),
        "result": local_identity(result_path),
    }


def execute_real_panels(
    *,
    renderer: Path,
    staged: Path,
    threshold_path: Path,
    panel_manifest_path: Path,
    plan_path: Path,
    delivery_path: Path,
    config: dict,
    output: Path,
    environment: dict[str, str],
) -> dict:
    panel_root = output / "real-panels"
    command = [
        sys.executable,
        str(renderer),
        "--test-root",
        str(staged),
        "--thresholds",
        str(threshold_path),
        "--panel-manifest",
        str(panel_manifest_path),
        "--plan",
        str(plan_path),
        "--delivery",
        str(delivery_path),
        "--score-input-index",
        str(staged / "score_input_index.json"),
        "--public-plan-commit",
        config["public_plan_commit"],
        "--public-delivery-commit",
        config["public_delivery_commit"],
        "--out",
        str(panel_root),
    ]
    completed = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        check=False,
        timeout=3_600,
    )
    if completed.returncode != 0:
        raise RuntimeError("fixed real-panel rendering failed; output remains sealed")
    manifest_path = panel_root / "real_panel_render_manifest.json"
    image_paths = [panel_root / f"real_panel_{index:02d}.png" for index in range(1, 5)]
    expected = {manifest_path.resolve(), *(path.resolve() for path in image_paths)}
    observed = {path.resolve() for path in panel_root.iterdir() if path.is_file()}
    if observed != expected:
        raise RuntimeError("fixed real-panel output set mismatch")
    return {
        "renderer": local_identity(renderer),
        "command": command,
        "returncode": completed.returncode,
        "stdout_bytes": len(completed.stdout),
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stderr_bytes": len(completed.stderr),
        "stderr_sha256": sha256_bytes(completed.stderr),
        "manifest": local_identity(manifest_path),
        "images": [local_identity(path) for path in image_paths],
    }


def main() -> int:
    input_root = Path(os.environ.get("KAGGLE_INPUT_PATH", "/kaggle/input"))
    working = Path(os.environ.get("KAGGLE_WORKING_PATH", "/kaggle/working"))
    scratch = Path(os.environ.get("KAGGLE_TEMP_PATH", "/kaggle/temp")) / (
        "fusion-one-shot-scoring"
    )
    if scratch.exists():
        raise RuntimeError("one-shot scoring scratch must start absent")
    scratch.mkdir(parents=True)
    config = load_config()
    mode = config["mode"]
    plan_path, delivery_path, plan, delivery = write_public_files(config, scratch)
    cache_job_plan_path, cache_job_plan = materialize_cache_job_plan(plan, scratch)
    stager_path = materialize_public_helper(
        config["public_plan_commit"], plan["one_shot_scoring_stager"], scratch
    )
    metric_preparer_path = materialize_public_helper(
        config["public_plan_commit"], plan["metric_runtime_preparer"], scratch
    )
    panel_renderer_path = materialize_public_helper(
        config["public_plan_commit"], plan["real_panel_renderer"], scratch
    )
    verify_public_contract(
        plan,
        delivery,
        mode,
        stager_path,
        metric_preparer_path,
        panel_renderer_path,
    )
    asset_root = find_asset_root(input_root)
    project = asset_root / "project"
    synthetic_scorer_path = None
    if mode == "synthetic":
        synthetic_scorer_path = materialize_public_synthetic_scorer(
            config["public_plan_commit"], scratch
        )
    split = verify_split(project)
    threshold_path, thresholds = find_threshold(input_root, plan)
    staged, staging = stage_inputs(
        mode=mode,
        input_root=input_root,
        scratch=scratch,
        plan_path=plan_path,
        cache_job_plan_path=cache_job_plan_path,
        delivery_path=delivery_path,
        config=config,
        stager=stager_path,
    )
    import_runtime = install_scoring_import_runtime()
    compat = configure_compat(scratch)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(compat),
            str(asset_root / "network-source" / "vesuvius" / "src"),
            str(project),
            environment.get("PYTHONPATH", ""),
        ]
    )
    metric = None
    if mode == "real":
        prepare_metric = load_metric_preparer(metric_preparer_path)
        metric_root, metric = prepare_metric(input_root, scratch, project)
        environment["FUSION_TOPOMETRICS_ROOT"] = str(metric_root)
    output = working / f"fusion-one-shot-{mode}"
    output.mkdir(parents=True, exist_ok=False)
    result_path, invocation = execute_scorer(
        mode=mode,
        staged=staged,
        project=project,
        split_path=project / "real_split_manifest.json",
        threshold_path=threshold_path,
        synthetic_scorer_path=synthetic_scorer_path,
        output=output,
        environment=environment,
    )
    if split.get("records_sha256") != REAL_SPLIT_RECORDS_SHA256:
        raise RuntimeError("verified split identity changed before scoring")
    if cache_job_plan.get("threshold_binding") != plan.get("threshold_binding"):
        raise RuntimeError("cache-job execution-plan threshold binding changed")
    if (
        thresholds.get("payload_sha256")
        != plan["threshold_binding"]["frozen_thresholds"]["payload_sha256"]
    ):
        raise RuntimeError("verified threshold identity changed before scoring")
    panel_invocation = None
    if mode == "real":
        panel_invocation = execute_real_panels(
            renderer=panel_renderer_path,
            staged=staged,
            threshold_path=threshold_path,
            panel_manifest_path=project / "real_panel_manifest.json",
            plan_path=plan_path,
            delivery_path=delivery_path,
            config=config,
            output=output,
            environment=environment,
        )
    run_manifest = {
        "schema_version": "1.0",
        "status": f"{mode} held-out result scored exactly once after public cache-delivery freeze",
        "mode": mode,
        "public_execution_plan": {
            "commit": config["public_plan_commit"],
            "file_sha256": config["public_plan_file_sha256"],
            "payload_sha256": plan["payload_sha256"],
        },
        "public_cache_delivery": {
            "commit": config["public_delivery_commit"],
            "file_sha256": config["public_delivery_file_sha256"],
            "payload_sha256": delivery["payload_sha256"],
        },
        "threshold_binding": plan["threshold_binding"],
        "launcher": launcher_source_identity(),
        "job_config": {
            "encoding": "hex in first source comment",
            "bytes": len(embedded_scoring_config_bytes()),
            "sha256": sha256_bytes(embedded_scoring_config_bytes()),
            "payload_sha256": config["payload_sha256"],
        },
        "staging": staging,
        "project_source_hashes": PROJECT_HASHES,
        "import_runtime": import_runtime,
        "metric_runtime": metric,
        "scorer_invocation": invocation,
        "panel_invocation": panel_invocation,
        "scientific_gate": {
            "all_14_cache_jobs_publicly_frozen_before_scoring": True,
            "scorer_invocations": 1,
            "manual_threshold_override_used": False,
            "test_time_tuning_permitted": False,
            "scientific_stdout_echoed_by_launcher": False,
            "sealed_result_opened_or_parsed_by_launcher": False,
            "fixed_real_panels_rendered": mode == "real",
            "panel_render_manifest_opened_or_parsed_by_launcher": False,
            "panels_opened_or_visually_assessed_by_launcher": False,
        },
    }
    run_manifest["payload_sha256"] = sha256_bytes(
        json.dumps(run_manifest, sort_keys=True, separators=(",", ":")).encode()
    )
    manifest_path = output / "scoring_run_manifest.json"
    manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    expected_entries = {result_path.name, manifest_path.name}
    if mode == "real":
        expected_entries.add("real-panels")
    if {path.name for path in output.iterdir()} != expected_entries:
        raise RuntimeError("one-shot scoring output file set mismatch")
    print("sealed result file SHA-256:", sha256_file(result_path))
    print("scoring run-manifest payload SHA-256:", run_manifest["payload_sha256"])
    print(f"ONE_SHOT_{mode.upper()}_SCORING_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
