import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "validate_gapbalance_stage0", ROOT / "validate_gapbalance_stage0.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def load_draft():
    return json.loads((ROOT / "GAPBALANCE_STAGE0_CONTRACT_DRAFT.json").read_text())


def load_public_frozen():
    return json.loads((ROOT / "GAPBALANCE_STAGE0_CONTRACT.json").read_text())


def frozen_contract(manifest_path):
    contract = load_draft()
    contract["status"] = "PUBLIC_FROZEN_READY_FOR_EXPLICIT_TRAINING_APPROVAL"
    real = contract["confirmation"]["real"]
    real["version"] = 2
    real["manifest_sha256"] = MODULE.sha256_file(manifest_path)
    real["manifest_bytes"] = manifest_path.stat().st_size
    real["expected_inventory"] = {
        "contact_bands": {"0-2": 14, "2-4": 60, "4-6": 60, "6-10": 60, "10+": 60},
        "control": 60,
        "both_instances_present": 254,
        "eligible_contact": 254,
        "eligible_contact_under_4_voxels": 74,
        "eligible_control": 60,
    }
    real["provider_inventory"] = {
        "all_files": 324,
        "npz_files": 314,
        "total_bytes": 1,
    }
    return contract


def write_manifest(path):
    rows = []
    bands = {"0-2": 14, "2-4": 60, "4-6": 60, "6-10": 60, "10+": 60}
    index = 0
    for band, count in bands.items():
        for _ in range(count):
            index += 1
            rows.append(
                {
                    "arm": "crops",
                    "file": f"crops/contact_{index:03d}.npz",
                    "sha256": hashlib.sha256(f"contact-{index}".encode()).hexdigest(),
                    "band": band,
                    "ct_empty_frac": 0.0,
                    "both_instances_present": True,
                }
            )
    for control_index in range(60):
        rows.append(
            {
                "arm": "control",
                "file": f"control/control_{control_index:03d}.npz",
                "sha256": hashlib.sha256(f"control-{control_index}".encode()).hexdigest(),
                "ct_empty_frac": 0.0,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def test_public_draft_fails_closed_on_pending_identity():
    with pytest.raises(MODULE.ContractError, match="PENDING"):
        MODULE.validate_contract(load_draft())


def test_public_frozen_contract_validates():
    MODULE.validate_contract(load_public_frozen())


def test_active_author_withdrawal_quarantine_fails_closed():
    quarantine = ROOT / "PHERC1218_HOLDOUT_QUARANTINE.json"
    with pytest.raises(MODULE.ContractError, match="quarantined by the dataset author"):
        MODULE.require_no_active_quarantine(load_public_frozen(), quarantine)


def test_tampered_holdout_quarantine_fails_closed(tmp_path):
    quarantine = json.loads((ROOT / "PHERC1218_HOLDOUT_QUARANTINE.json").read_text())
    quarantine["confirmation_gate"]["real_confirmation_may_open"] = True
    path = tmp_path / "PHERC1218_HOLDOUT_QUARANTINE.json"
    path.write_text(json.dumps(quarantine))
    with pytest.raises(MODULE.ContractError, match="payload mismatch"):
        MODULE.require_no_active_quarantine(load_public_frozen(), path)


def test_metadata_only_manifest_validation_and_panel_selection(tmp_path):
    manifest = tmp_path / "MANIFEST.jsonl"
    write_manifest(manifest)
    contract = frozen_contract(manifest)
    result = MODULE.validate_manifest(contract, manifest)
    assert result["status"] == "STAGE0_MANIFEST_VALIDATED_RESULT_BLIND"
    assert result["inventory"]["eligible_contact"] == 254
    assert result["inventory"]["eligible_contact_under_4_voxels"] == 74
    assert result["inventory"]["eligible_control"] == 60
    assert set(result["fixed_panels"]) == set(MODULE.PANEL_BANDS)
    assert len(result["eligible_file_identities"]["contact"]) == 254
    assert len(result["eligible_file_identities"]["control"]) == 60
    assert result["npz_opened_or_parsed"] is False


def test_inventory_mismatch_fails_closed(tmp_path):
    manifest = tmp_path / "MANIFEST.jsonl"
    rows = write_manifest(manifest)
    contract = frozen_contract(manifest)
    rows.append(
        {
            "arm": "crops",
            "file": "crops/stale.npz",
            "sha256": "a" * 64,
            "band": "0-2",
            "ct_empty_frac": 0.0,
            "both_instances_present": True,
        }
    )
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    contract["confirmation"]["real"]["manifest_sha256"] = MODULE.sha256_file(manifest)
    contract["confirmation"]["real"]["manifest_bytes"] = manifest.stat().st_size
    with pytest.raises(MODULE.ContractError, match="inventory mismatch"):
        MODULE.validate_manifest(contract, manifest)


def test_duplicate_file_fails_closed(tmp_path):
    manifest = tmp_path / "MANIFEST.jsonl"
    rows = write_manifest(manifest)
    rows[1]["file"] = rows[0]["file"]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    contract = frozen_contract(manifest)
    with pytest.raises(MODULE.ContractError, match="duplicate file path"):
        MODULE.validate_manifest(contract, manifest)


def test_changed_gate_or_control_identity_fails_closed(tmp_path):
    manifest = tmp_path / "MANIFEST.jsonl"
    write_manifest(manifest)
    contract = frozen_contract(manifest)
    altered = copy.deepcopy(contract)
    altered["gates"]["detection_delta_pp_min_pooled"] = -3.0
    with pytest.raises(MODULE.ContractError, match="detection gate changed"):
        MODULE.validate_contract(altered)
    altered = copy.deepcopy(contract)
    altered["experiment"]["reused_control_checkpoint_sha256"]["11"] = "b" * 64
    with pytest.raises(MODULE.ContractError, match="control checkpoint identity"):
        MODULE.validate_contract(altered)
