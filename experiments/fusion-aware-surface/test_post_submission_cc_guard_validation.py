import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "docs" / "fusion-aware-final-evidence"
PAYLOAD_PATH = EVIDENCE_DIR / "post_submission_cc_guard_validation.json"
NOTE_PATH = EVIDENCE_DIR / "POST_SUBMISSION_CC_GUARD_VALIDATION.md"


def _load_payload() -> dict:
    return json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))


def _canonical_payload_sha256(payload: dict) -> str:
    unhashed = dict(payload)
    unhashed.pop("payload_sha256")
    encoded = json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_post_submission_payload_is_self_bound_and_additive() -> None:
    payload = _load_payload()

    assert payload["payload_sha256"] == _canonical_payload_sha256(payload)
    assert payload["scope"] == "POST_SUBMISSION_SYNTHETIC_INSTRUMENT_INTERPRETATION_ONLY"
    assert payload["gap8_immutable_verdict"] == {
        "gates_failed": 2,
        "gates_passed": 5,
        "retuned": False,
        "status": "MIXED_RESULT_UNCHANGED",
    }
    assert payload["gapbalance"] == {
        "holdout_certified_by_this_evidence": False,
        "unblocked": False,
    }
    assert payload["submission"] == {
        "form_resubmitted": False,
        "post_submission_addendum_only": True,
    }


def test_reported_totals_and_pitch_caveat_are_preserved() -> None:
    results = _load_payload()["reported_results"]
    bins = results["bins"]

    assert sum(row["rejected_pairs"] for row in bins) == results["rejected_pairs"] == 94134
    assert sum(row["accepted_sites"] for row in bins) == results["accepted_sites"] == 2400
    assert sum(row["false_accepts"] for row in bins) == results["false_accepts"] == 0
    assert results["false_accept_rate_percent"] == 0.0
    assert results["pitch_300_rejected_pairs"] == 941
    assert results["pitch_300_truly_different_percent"] == 0.0


def test_exact_public_identities_and_reproducibility_limit_are_bound() -> None:
    payload = _load_payload()

    assert payload["source_comment"]["id"] == 5394005579
    assert payload["result_repository"]["commit"] == (
        "9cfcaa8eb126264b85612f467ea20154a65dd7c3"
    )
    assert payload["harness_repository"]["commit"] == (
        "0c5ed007cf6c875c9712cec8677528952a64c14f"
    )
    assert payload["eight_cell_manifest_sha256"] == (
        "506f3353ba834bcdb839cf4000d2ec6b1a18e074ce1fb9c10dce24696e7299dd"
    )
    assert payload["local_verification"] == {
        "bin_total_arithmetic_checked": True,
        "external_scoring_rerun_locally": False,
        "file_hashes_checked_at_exact_commits": True,
        "full_pair_csvs_present_in_bound_public_result_directory": False,
        "repository_full_tests_passed": 480,
        "scientific_endpoints_reopened": False,
    }

    note = NOTE_PATH.read_text(encoding="utf-8")
    assert "does **not** turn Gap8 into a seven-gate success" in note
    assert "It did **not** rerun Jinho's scientific scoring" in note
    assert "No Progress Prize or other form was resubmitted" in note
