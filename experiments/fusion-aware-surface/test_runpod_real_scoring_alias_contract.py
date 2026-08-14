import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def canonical_sha256(payload: dict) -> str:
    content = dict(payload)
    content.pop("payload_sha256", None)
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_exact_cpu_launcher_alias_matches_public_execution_plan() -> None:
    plan = json.loads((HERE / "heldout_execution_plan.json").read_text())
    launcher = load("cpu_exact_alias", HERE / "one_shot_scoring_launcher_cpu_exact.py")
    assert plan["payload_sha256"] == canonical_sha256(plan)
    assert launcher.launcher_source_identity() == plan["one_shot_scoring_launcher"]
    correction = plan["result_blind_launcher_alias_contract_correction"]
    assert correction["correction"]["bytes_changed"] is False
    assert correction["scientific_gate"] == {
        "cache_or_result_opened": False,
        "model_metric_threshold_seed_panel_endpoint_gate_aggregation_or_claim_changed": False,
        "scorer_source_bytes_changed": False,
        "scientific_runtime_started_in_failed_attempt": False,
    }
    scorer = HERE / "score_real_test.py"
    assert hashlib.sha256(scorer.read_bytes()).hexdigest() == plan[
        "one_shot_scorers"
    ]["real"]["script"]["sha256"]


def test_runpod_alias_plan_preserves_scientific_contract() -> None:
    predecessor = json.loads((HERE / "runpod_real_scoring_plan.json").read_text())
    corrected = json.loads(
        (HERE / "runpod_real_scoring_alias_corrected_plan.json").read_text()
    )
    assert corrected["payload_sha256"] == canonical_sha256(corrected)
    ignored = {
        "payload_sha256",
        "embedded_real_launcher",
        "public_cache_delivery",
        "public_execution_plan",
        "result_blind_launcher_alias_contract_correction",
    }
    assert {k: v for k, v in corrected.items() if k not in ignored} == {
        k: v for k, v in predecessor.items() if k not in ignored
    }
    assert corrected["scientific_gate"] == predecessor["scientific_gate"]
