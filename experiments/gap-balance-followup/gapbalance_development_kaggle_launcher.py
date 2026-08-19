"""Fail-closed launcher for one publicly frozen GapBalance development-cache job.

The generated Kaggle kernel prepends one hex-encoded job configuration to this
source.  Real jobs cache all nine matched Scroll-1 validation predictions.
Synthetic jobs cache one of four frozen shards for the three models sharing an
initialization seed.  This launcher never selects a threshold or scores an
endpoint, and it has no confirmation-data path.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

EMBEDDED_CONFIG_PREFIX = b"# GAPBALANCE_DEVELOPMENT_JOB_CONFIG_HEX="
PUBLIC_REPOSITORY = "aviad12g/tifxyz-doctor"
ASSET_DATASET_ID = "aviadcohen1/vesuvius-fusion-aware-training-assets"
ASSET_LEDGER_SHA256 = "1b3d78b2f85808a4a2953b7b8ed3a5969f07714f341cea269fb11746f7892fba"
ASSET_LEDGER_RECORDS = 278
SOURCE_CHECKPOINT_SHA256 = "f1990a02ac91889c1f989522ae0e45421a91cb666320448aaf579d42b081636f"
SOURCE_CHECKPOINT_BYTES = 819_171_665
SPLIT_MANIFEST_SHA256 = "dedc881134d9de2ed2605162f82dfb219b52c6e05629c68223100b148b15d4fe"
SPLIT_RECORDS_SHA256 = "20c600d6061bf8715ada20423a05c2f03a9bc14827b2c79bac7b7e1b3cc0499d"
LOADER_SHA256 = "49a1d4ea3ee611236d53b1291ca2e8cd6e1450f2fc032223ff89a5b53d9ec903"
REFERENCE_READER_SHA256 = "a533ee940712f1a47111705e4d8fff4990ccffb0afb103c7337a68bff244781c"
PAINTER_SHA256 = "41f2a097b819bc486819d6702ed3949d0bdfe722c2405b1fe86828062da1d9b8"
REAL_STATUS = "GapBalance Scroll-1 development probabilities cached; endpoints not scored"
SYNTHETIC_STATUS = "GapBalance synthetic development rays cached; endpoints not scored"
RUNS = tuple(
    f"{arm}_seed{seed}"
    for arm in ("control", "gap2", "gap4")
    for seed in (11, 23, 47)
)
ARCHIVES = {
    "images_s1.tar": (
        1_478_256_640,
        "cce01d96c77cc7966a41a805b2690db3e7705ce92cbc52d35d0a83a5c7ba35a5",
    ),
    "labels.tar": (
        83_087_360,
        "6e01e4d5f0591796a060bda0a1357ed8cc8b801912a7d3e569ecb246764741a6",
    ),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def validate_embedded_hash(payload: dict[str, Any]) -> None:
    body = dict(payload)
    observed = body.pop("payload_sha256", None)
    if observed != canonical_sha256(body):
        raise RuntimeError("embedded payload SHA-256 mismatch")


def launcher_source_identity() -> dict[str, Any]:
    raw = Path(__file__).resolve().read_bytes()
    first, separator, remainder = raw.partition(b"\n")
    if first.startswith(EMBEDDED_CONFIG_PREFIX):
        if not separator or not remainder:
            raise RuntimeError("generated launcher has no base source")
        raw = remainder
    return {
        "file": "gapbalance_development_kaggle_launcher.py",
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def load_job_config() -> dict[str, Any]:
    first = Path(__file__).resolve().read_bytes().split(b"\n", 1)[0]
    if not first.startswith(EMBEDDED_CONFIG_PREFIX):
        raise RuntimeError("GapBalance development job config is not embedded")
    try:
        raw = bytes.fromhex(first[len(EMBEDDED_CONFIG_PREFIX) :].decode("ascii"))
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("embedded job config is not valid hex") from error
    payload = json.loads(raw)
    validate_embedded_hash(payload)
    if payload.get("schema_version") != "1.0":
        raise RuntimeError("development job config schema mismatch")
    if payload.get("mode") not in {"real", "synthetic"}:
        raise RuntimeError("development job config mode mismatch")
    if payload.get("confirmation_outputs_inspected") is not False:
        raise RuntimeError("development job config breaks the confirmation blind")
    if payload.get("launcher") != launcher_source_identity():
        raise RuntimeError("running launcher differs from its frozen job config")
    return payload


def safe_relative(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise RuntimeError(f"unsafe relative path: {relative!r}")
    resolved = (root / Path(*pure.parts)).resolve()
    if resolved != root.resolve() and root.resolve() not in resolved.parents:
        raise RuntimeError(f"path escapes root: {relative!r}")
    return resolved


def find_asset_root(input_root: Path) -> Path:
    matches = [
        ledger.parent.resolve()
        for ledger in input_root.glob("**/SOURCE_SHA256SUMS")
        if ledger.is_file() and sha256_file(ledger) == ASSET_LEDGER_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one mounted {ASSET_DATASET_ID}; found {matches}"
        )
    return matches[0]


def verify_asset_root(root: Path) -> None:
    ledger = root / "SOURCE_SHA256SUMS"
    lines = [
        line for line in ledger.read_text(encoding="utf-8").splitlines() if line
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
    fixed = {
        "project/real_split_manifest.json": SPLIT_MANIFEST_SHA256,
        "diagnostic/loader059.py": LOADER_SHA256,
        "diagnostic/fusion_readout.py": REFERENCE_READER_SHA256,
        "painter/contrast_phantom.py": PAINTER_SHA256,
    }
    for relative, digest in fixed.items():
        if records.get(relative) != digest:
            raise RuntimeError(f"required asset is not ledger-bound: {relative}")
    source = root / "model" / "Model_epoch499.pth"
    if (
        not source.is_file()
        or source.stat().st_size != SOURCE_CHECKPOINT_BYTES
        or sha256_file(source) != SOURCE_CHECKPOINT_SHA256
    ):
        raise RuntimeError("source checkpoint identity mismatch")
    split = json.loads((root / "project" / "real_split_manifest.json").read_text())
    if split.get("records_sha256") != SPLIT_RECORDS_SHA256:
        raise RuntimeError("Scroll-1 split records identity mismatch")


def fetch_public_sources(config: dict[str, Any], destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"public source destination must start absent: {destination}")
    commit = config.get("public_source_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("invalid public source commit")
    files = config.get("public_source_files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("public source file mapping is absent")
    for relative, expected in sorted(files.items()):
        url = f"https://raw.githubusercontent.com/{PUBLIC_REPOSITORY}/{commit}/{relative}"
        request = urllib.request.Request(
            url, headers={"User-Agent": "gapbalance-development/1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
        if sha256_bytes(raw) != expected:
            raise RuntimeError(f"public source SHA-256 mismatch: {relative}")
        target = safe_relative(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)


def find_model_states(input_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    records = config.get("runs")
    if not isinstance(records, dict) or not records:
        raise RuntimeError("job config contains no runs")
    found = {}
    all_states = [path for path in input_root.rglob("*.pth") if path.is_file()]
    states_by_hash: dict[str, list[Path]] = {}
    for path in all_states:
        states_by_hash.setdefault(sha256_file(path), []).append(path.resolve())
    for run, record in records.items():
        if run not in RUNS:
            raise RuntimeError(f"unexpected run in job config: {run}")
        expected = record.get("checkpoint_sha256")
        matches = states_by_hash.get(expected, [])
        if len(matches) != 1:
            raise RuntimeError(f"{run}: expected one exact mounted checkpoint; found {matches}")
        checkpoint = matches[0]
        if checkpoint.stat().st_size != int(record.get("checkpoint_bytes", -1)):
            raise RuntimeError(f"{run}: checkpoint byte-size mismatch")
        mount = input_root / checkpoint.relative_to(input_root).parts[0]
        manifests = [
            path
            for path in mount.rglob("training_run_manifest.json")
            if path.is_file()
            and path.stat().st_size == int(record.get("training_manifest_bytes", -1))
            and sha256_file(path) == record.get("training_manifest_sha256")
        ]
        if len(manifests) != 1:
            raise RuntimeError(f"{run}: exact training manifest not found beside checkpoint")
        training = json.loads(manifests[0].read_text(encoding="utf-8"))
        arm, seed_text = run.rsplit("_seed", 1)
        if (
            training.get("status")
            != "training complete; validation and test endpoints not computed"
            or training.get("arm") != arm
            or int(training.get("seed", -1)) != int(seed_text)
            or training.get("outputs", {}).get("checkpoint_sha256") != expected
        ):
            raise RuntimeError(f"{run}: training manifest identity mismatch")
        found[run] = checkpoint
    return found


def _safe_selected_extract(archive: Path, destination: Path, *, kind: str) -> None:
    with tarfile.open(archive, "r") as bundle:
        destination_root = destination.resolve()
        selected = []
        for member in bundle.getmembers():
            relative = PurePosixPath(member.name)
            wanted = (
                kind == "image"
                and len(relative.parts) >= 2
                and relative.parts[-2] == "imagesTr"
                and relative.name.startswith("s1_")
                and relative.name.endswith("_0000.tif")
            ) or (
                kind == "label"
                and len(relative.parts) >= 2
                and relative.parts[-2] == "labelsTr"
                and relative.name.startswith("s1_")
                and relative.name.endswith(".tif")
            )
            if not wanted:
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise RuntimeError(f"invalid selected archive member: {member.name}")
            target = (destination / member.name).resolve()
            if destination_root not in target.parents:
                raise RuntimeError(f"archive path traversal: {member.name}")
            selected.append(member)
        if len(selected) != 162:
            raise RuntimeError(f"unexpected selected {kind} count: {len(selected)}")
        for member in selected:
            bundle.extract(member, destination, filter="data")


def _link_expanded(root: Path, destination: Path, *, kind: str) -> None:
    archive_name = "images_s1" if kind == "image" else "labels"
    directory = "imagesTr" if kind == "image" else "labelsTr"
    pattern = "s1_*_0000.tif" if kind == "image" else "s1_*.tif"
    matches = sorted((root / "archives" / archive_name).glob(f"**/{directory}/{pattern}"))
    if len(matches) != 162:
        raise RuntimeError(f"unexpected expanded {kind} count: {len(matches)}")
    target_root = destination / directory
    target_root.mkdir(parents=True, exist_ok=True)
    for source in matches:
        target = target_root / source.name
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"duplicate selected input: {target}")
        target.symlink_to(source.resolve())


def prepare_scroll1_only(root: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"Scroll-1 scratch must start absent: {destination}")
    destination.mkdir(parents=True)
    tar_paths = {name: root / "archives" / name for name in ARCHIVES}
    if all(path.is_file() for path in tar_paths.values()):
        for name, path in tar_paths.items():
            size, digest = ARCHIVES[name]
            if path.stat().st_size != size or sha256_file(path) != digest:
                raise RuntimeError(f"frozen archive mismatch: {name}")
        _safe_selected_extract(tar_paths["images_s1.tar"], destination, kind="image")
        _safe_selected_extract(tar_paths["labels.tar"], destination, kind="label")
    else:
        _link_expanded(root, destination, kind="image")
        _link_expanded(root, destination, kind="label")
    if (
        len(list((destination / "imagesTr").glob("s1_*_0000.tif"))) != 162
        or len(list((destination / "labelsTr").glob("s1_*.tif"))) != 162
        or list(destination.rglob("s4_*.tif"))
        or list(destination.rglob("s5_*.tif"))
    ):
        raise RuntimeError("Scroll-1-only materialization mismatch")


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

    expected = {
        "torchvision": "0.20.1",
        "numpy": "1.26.4",
        "tifffile": "2025.2.18",
        "imagecodecs": "2024.12.30",
        "timm": "1.0.27",
        "einops": "0.8.1",
    }
    observed = {name: importlib.metadata.version(name).split("+", 1)[0] for name in expected}
    if (
        torch.__version__.split("+", 1)[0] != "2.5.1"
        or not torch.cuda.is_available()
        or observed != expected
    ):
        raise RuntimeError(f"frozen CUDA runtime unavailable: torch={torch.__version__} packages={observed}")


def configure_runtime_compat(scratch: Path) -> Path:
    import torch

    if "reason" not in inspect.signature(torch.compiler.disable).parameters:
        original_disable = torch.compiler.disable

        def disable_compat(fn=None, recursive=True, *, reason=None):
            return original_disable(fn=fn, recursive=recursive)

        torch.compiler.disable = disable_compat
    compat = scratch / "runtime-compat"
    compat.mkdir(parents=True, exist_ok=False)
    (compat / "sitecustomize.py").write_text(
        "import inspect\n"
        "import torch\n"
        "if 'reason' not in inspect.signature(torch.compiler.disable).parameters:\n"
        "    _original_disable = torch.compiler.disable\n"
        "    def _disable_compat(fn=None, recursive=True, *, reason=None):\n"
        "        return _original_disable(fn=fn, recursive=recursive)\n"
        "    torch.compiler.disable = _disable_compat\n",
        encoding="utf-8",
    )
    return compat


def invoke_cache(
    config: dict[str, Any],
    asset_root: Path,
    source_root: Path,
    states: dict[str, Path],
    scratch: Path,
    output: Path,
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
    mode = config["mode"]
    real_data = scratch / "scroll1-development"
    if mode == "real":
        if tuple(states) != RUNS:
            raise RuntimeError("real development job must contain the exact nine runs")
        prepare_scroll1_only(asset_root, real_data)
    else:
        seed = int(config.get("seed", -1))
        expected = {f"{arm}_seed{seed}" for arm in ("control", "gap2", "gap4")}
        if set(states) != expected or int(config.get("shard_index", -1)) not in range(4):
            raise RuntimeError("synthetic development job identity mismatch")

    for index, (run, state) in enumerate(states.items(), start=1):
        if mode == "real":
            script = source_root / "experiments/gap-balance-followup/cache_gapbalance_real_development.py"
            command = [
                sys.executable,
                str(script),
                "--run",
                run,
                "--real-data",
                str(real_data),
                "--split-manifest",
                str(asset_root / "project" / "real_split_manifest.json"),
                "--diagnostic-scripts",
                str(asset_root / "diagnostic"),
                "--source-checkpoint",
                str(asset_root / "model" / "Model_epoch499.pth"),
                "--model-state",
                str(state),
                "--out",
                str(output),
            ]
        else:
            script = source_root / "experiments/gap-balance-followup/cache_gapbalance_synthetic_development.py"
            command = [
                sys.executable,
                str(script),
                "--run",
                run,
                "--shard-index",
                str(config["shard_index"]),
                "--painter-scripts",
                str(asset_root / "painter"),
                "--diagnostic-scripts",
                str(asset_root / "diagnostic"),
                "--reference-reader",
                str(asset_root / "diagnostic" / "fusion_readout.py"),
                "--source-checkpoint",
                str(asset_root / "model" / "Model_epoch499.pth"),
                "--model-state",
                str(state),
                "--split-manifest",
                str(asset_root / "project" / "real_split_manifest.json"),
                "--out",
                str(output),
            ]
        print(
            f"GAPBALANCE_DEVELOPMENT_INVOCATION_START {index}/{len(states)} run={run}",
            flush=True,
        )
        subprocess.run(
            command,
            cwd=source_root / "experiments/gap-balance-followup",
            env=env,
            check=True,
            timeout=12 * 60 * 60,
        )
        print(f"GAPBALANCE_DEVELOPMENT_INVOCATION_COMPLETE run={run}", flush=True)


def freeze_output_index(
    config: dict[str, Any], output: Path, states: dict[str, Path]
) -> None:
    records = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "file": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    expected_npz = 9 * 24 if config["mode"] == "real" else 3 * 20
    expected_manifests = len(states)
    if sum(record["file"].endswith(".npz") for record in records) != expected_npz:
        raise RuntimeError("development cache NPZ count mismatch")
    manifests = [
        record for record in records if Path(record["file"]).name.endswith(".json")
    ]
    if len(manifests) != expected_manifests:
        raise RuntimeError("development cache manifest count mismatch")
    status = REAL_STATUS if config["mode"] == "real" else SYNTHETIC_STATUS
    for record in manifests:
        payload = json.loads((output / record["file"]).read_text(encoding="utf-8"))
        validate_embedded_hash(payload)
        run = payload.get("run")
        if (
            payload.get("status") != status
            or payload.get("model_state_sha256")
            != config["runs"][run]["checkpoint_sha256"]
            or payload.get("source_split_records_sha256") != SPLIT_RECORDS_SHA256
            or payload.get("selected_threshold") is not None
            or payload.get("confirmation_outputs_inspected") is not False
        ):
            raise RuntimeError(f"development cache manifest mismatch: {record['file']}")
    payload = {
        "schema_version": "1.0",
        "status": "GapBalance development cache job complete without endpoint scoring",
        "job_id": config["job_id"],
        "mode": config["mode"],
        "seed": config.get("seed"),
        "shard_index": config.get("shard_index"),
        "runs": {
            run: config["runs"][run]["checkpoint_sha256"] for run in states
        },
        "files": records,
        "selected_thresholds": False,
        "scientific_endpoints_scored": False,
        "confirmation_outputs_inspected": False,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    destination = output / "gapbalance_development_job_index.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("job-index payload SHA-256:", payload["payload_sha256"])


def main() -> int:
    config = load_job_config()
    input_root = Path(os.environ.get("KAGGLE_INPUT_PATH", "/kaggle/input"))
    working = Path(os.environ.get("KAGGLE_WORKING_PATH", "/kaggle/working"))
    scratch = Path(os.environ.get("KAGGLE_TEMP_PATH", "/kaggle/temp")) / config["job_id"]
    output = working / "gapbalance-development"
    if scratch.exists() or output.exists():
        raise RuntimeError("development scratch and output must start absent")
    scratch.mkdir(parents=True)
    output.mkdir(parents=True)

    asset_root = find_asset_root(input_root)
    verify_asset_root(asset_root)
    source_root = scratch / "public-source"
    fetch_public_sources(config, source_root)
    states = find_model_states(input_root, config)
    install_runtime()
    compat = configure_runtime_compat(scratch)
    invoke_cache(config, asset_root, source_root, states, scratch, output, compat)
    freeze_output_index(config, output, states)
    print(
        json.dumps(
            {
                "status": "GAPBALANCE_DEVELOPMENT_CACHE_COMPLETE_NOT_SCORED",
                "job_id": config["job_id"],
                "confirmation_outputs_inspected": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
