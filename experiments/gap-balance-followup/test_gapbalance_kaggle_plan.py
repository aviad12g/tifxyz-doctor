import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


LAUNCHER = load_module("gapbalance_kaggle_launcher", "gapbalance_kaggle_launcher.py")
PREPARE = load_module("prepare_gapbalance_kaggle_jobs", "prepare_gapbalance_kaggle_jobs.py")
PLAN_PATH = HERE / "GAPBALANCE_KAGGLE_TRAINING_PLAN.json"


def test_exact_six_job_plan_and_public_artifact_hashes():
    plan = PREPARE.load_and_validate_plan(PLAN_PATH, REPO_ROOT)
    assert {(job["arm"], job["seed"]) for job in plan["jobs"]} == PREPARE.EXPECTED_JOBS
    assert plan["execution"] == {
        "free_kaggle_execution_authorized": True,
        "form_submission_authorized": False,
        "paid_compute_authorized": False,
        "provider_cost_usd": 0,
        "verify_only_required_before_gpu": True,
    }


def test_prepare_writes_only_six_frozen_private_kernels(tmp_path):
    output = tmp_path / "jobs"
    receipt = PREPARE.prepare(PLAN_PATH, REPO_ROOT, output)
    assert receipt["status"] == "CPU_PREFLIGHT_AND_SIX_FREE_KAGGLE_JOB_PACKAGES_PREPARED_NOT_PUSHED"
    assert receipt["preflight"]["verify_only"] is True
    assert receipt["preflight"]["enable_gpu"] is False
    preflight_root = output / "preflight"
    preflight_metadata = json.loads((preflight_root / "kernel-metadata.json").read_text())
    assert preflight_metadata["enable_gpu"] is False
    assert preflight_metadata["is_private"] is True
    assert 'FROZEN_VERIFY_ONLY = True' in (preflight_root / "kernel_launcher.py").read_text()
    assert len(receipt["jobs"]) == 6
    for record in receipt["jobs"]:
        root = output / f"{record['arm']}-seed{record['seed']}"
        metadata = json.loads((root / "kernel-metadata.json").read_text())
        assert metadata["is_private"] is True
        assert metadata["enable_gpu"] is True
        assert metadata["dataset_sources"] == [LAUNCHER.ASSET_DATASET_ID]
        source = (root / "kernel_launcher.py").read_text()
        assert 'FROZEN_ARM = "__FROZEN_ARM__"' not in source
        assert 'FROZEN_SEED = "__FROZEN_SEED__"' not in source
        assert f'FROZEN_ARM = "{record["arm"]}"' in source
        assert f'FROZEN_SEED = "{record["seed"]}"' in source
        assert "FROZEN_VERIFY_ONLY = False" in source


def test_plan_rejects_any_seventh_or_paid_job(tmp_path):
    plan = json.loads(PLAN_PATH.read_text())
    altered = copy.deepcopy(plan)
    altered["jobs"].append({
        "arm": "gap2",
        "seed": 11,
        "kernel_id": "aviadcohen1/vesuvius-gapbalance-extra",
        "title": "extra",
    })
    path = tmp_path / "altered.json"
    path.write_text(json.dumps(altered))
    with pytest.raises(PREPARE.PlanError, match="exactly six"):
        PREPARE.load_and_validate_plan(path, REPO_ROOT)
    altered = copy.deepcopy(plan)
    altered["execution"]["paid_compute_authorized"] = True
    path.write_text(json.dumps(altered))
    with pytest.raises(PREPARE.PlanError, match="paid compute"):
        PREPARE.load_and_validate_plan(path, REPO_ROOT)


def test_result_blind_public_projection_replaces_only_two_files(tmp_path, monkeypatch):
    asset_root = tmp_path / "assets"
    public_root = REPO_ROOT / "experiments" / "fusion-aware-surface"
    records = []
    for index in range(198):
        relative = f"dummy/file_{index:03d}.txt"
        path = asset_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"dummy-{index}\n")
        records.append((hashlib.sha256(path.read_bytes()).hexdigest(), relative))
    for relative in LAUNCHER.PUBLIC_REPLACEMENTS:
        path = asset_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old frozen source\n")
        records.append((hashlib.sha256(path.read_bytes()).hexdigest(), relative))
    ledger = asset_root / "SOURCE_SHA256SUMS"
    ledger.write_text("".join(f"{digest}  {relative}\n" for digest, relative in records))
    monkeypatch.setattr(LAUNCHER, "SOURCE_LEDGER_SHA256", LAUNCHER.sha256_file(ledger))
    expanded = asset_root / "archives" / "images_s1"
    expanded.mkdir(parents=True)
    (expanded / "fixture.tif").write_bytes(b"fixture")
    monkeypatch.setattr(LAUNCHER, "FIXED_ASSET_PATHS", ("archives/images_s1.tar",))

    projection = tmp_path / "projection"
    identities = LAUNCHER.build_projection(asset_root, projection, public_root)
    assert identities["original_source_ledger_sha256"] == LAUNCHER.sha256_file(ledger)
    for relative, digest in LAUNCHER.PUBLIC_REPLACEMENTS.items():
        projected = projection / relative
        assert projected.is_file() and not projected.is_symlink()
        assert LAUNCHER.sha256_file(projected) == digest
    assert (projection / "dummy/file_000.txt").is_symlink()
    assert (projection / "archives" / "images_s1").is_symlink()
    assert not any(path.suffix == ".npz" for path in projection.rglob("*"))


def test_generic_launcher_accepts_only_gap2_gap4_matched_seeds():
    assert LAUNCHER.resolve_job("gap2", 11) == ("gap2", 11)
    assert LAUNCHER.resolve_job("gap4", 47) == ("gap4", 47)
    with pytest.raises(LAUNCHER.LaunchError, match="outside"):
        LAUNCHER.resolve_job("gap8", 11)
    assert LAUNCHER.resolve_verify_only(True) is True
    assert LAUNCHER.resolve_verify_only(False) is False


def test_asset_root_is_bound_by_unique_frozen_ledger(tmp_path, monkeypatch):
    expected = tmp_path / "mounted" / "bundle"
    expected.mkdir(parents=True)
    ledger = expected / "SOURCE_SHA256SUMS"
    ledger.write_text("frozen ledger\n")
    monkeypatch.setattr(LAUNCHER, "SOURCE_LEDGER_SHA256", LAUNCHER.sha256_file(ledger))
    assert LAUNCHER.find_asset_root(tmp_path) == expected.resolve()

    duplicate = tmp_path / "other"
    duplicate.mkdir()
    (duplicate / "SOURCE_SHA256SUMS").write_text("frozen ledger\n")
    with pytest.raises(LAUNCHER.LaunchError, match="expected one mounted"):
        LAUNCHER.find_asset_root(tmp_path)


def test_expanded_archive_identity_and_training_projection(tmp_path, monkeypatch):
    asset_root = tmp_path / "assets"
    rows = {"images_s4_s5/imagesTr/s4_a_0000.tif": b"image-c"}
    for index in range(162):
        rows[f"images_s1/imagesTr/s1_{index:03d}_0000.tif"] = f"image-{index}".encode()
        rows[f"labels/labelsTr/s1_{index:03d}.tif"] = f"label-{index}".encode()
    for relative, raw in rows.items():
        path = asset_root / "archives" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    identity = LAUNCHER.expanded_archive_identity(asset_root)
    monkeypatch.setattr(LAUNCHER, "EXPANDED_ARCHIVE_COUNT", identity[0])
    monkeypatch.setattr(LAUNCHER, "EXPANDED_ARCHIVE_BYTES", identity[1])
    monkeypatch.setattr(LAUNCHER, "EXPANDED_ARCHIVE_SHA256", identity[2])
    assert identity[0] == 325

    destination = tmp_path / "training"
    LAUNCHER.extract_training_data_compat(asset_root, destination, lambda *_: None)
    assert (destination / "imagesTr" / "s1_000_0000.tif").is_symlink()
    assert (destination / "labelsTr" / "s1_161.tif").is_symlink()
