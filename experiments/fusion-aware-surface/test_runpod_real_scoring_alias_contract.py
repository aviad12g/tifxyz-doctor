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


def test_exact_source_runpod_plan_preserves_scientific_contract() -> None:
    predecessor = json.loads(
        (HERE / "runpod_real_scoring_alias_corrected_plan.json").read_text()
    )
    corrected = json.loads(
        (HERE / "runpod_real_scoring_exact_source_plan.json").read_text()
    )
    assert corrected["payload_sha256"] == canonical_sha256(corrected)
    ignored = {
        "payload_sha256",
        "embedded_real_launcher",
        "public_cache_delivery",
        "public_execution_plan",
        "result_blind_public_real_scorer_source_correction",
    }
    assert {k: v for k, v in corrected.items() if k not in ignored} == {
        k: v for k, v in predecessor.items() if k not in ignored
    }
    assert corrected["scientific_gate"] == predecessor["scientific_gate"]
    assert corrected["result_blind_public_real_scorer_source_correction"][
        "restored_scorer"
    ]["sha256"] == "3529b8213237a60d392ffec04efca602988b3242f6af8cadc87423e8e224bb79"


def test_publication_records_are_projection_metadata() -> None:
    source = (HERE / "stage_heldout_for_scoring.py").read_text()
    assert '"result_blind_launcher_alias_contract_correction"' in source
    assert '"result_blind_public_real_scorer_source_correction"' in source
    assert '"result_blind_scoring_publication_projection_correction"' in source
    plan = json.loads((HERE / "heldout_execution_plan.json").read_text())
    assert plan["one_shot_scoring_stager"] == {
        "file": "stage_heldout_for_scoring.py",
        "bytes": (HERE / "stage_heldout_for_scoring.py").stat().st_size,
        "sha256": hashlib.sha256(
            (HERE / "stage_heldout_for_scoring.py").read_bytes()
        ).hexdigest(),
    }


def test_projection_corrected_runpod_plan_preserves_scientific_contract() -> None:
    predecessor = json.loads(
        (HERE / "runpod_real_scoring_exact_source_plan.json").read_text()
    )
    corrected = json.loads(
        (HERE / "runpod_real_scoring_projection_corrected_plan.json").read_text()
    )
    assert corrected["payload_sha256"] == canonical_sha256(corrected)
    ignored = {
        "payload_sha256",
        "embedded_real_launcher",
        "public_cache_delivery",
        "public_execution_plan",
        "result_blind_scoring_publication_projection_correction",
    }
    assert {k: v for k, v in corrected.items() if k not in ignored} == {
        k: v for k, v in predecessor.items() if k not in ignored
    }
    assert corrected["scientific_gate"] == predecessor["scientific_gate"]


def test_pip_seeded_runpod_plan_preserves_scientific_contract() -> None:
    predecessor = json.loads(
        (HERE / "runpod_real_scoring_projection_corrected_plan.json").read_text()
    )
    corrected = json.loads(
        (HERE / "runpod_real_scoring_pip_seeded_plan.json").read_text()
    )
    assert corrected["payload_sha256"] == canonical_sha256(corrected)
    ignored = {
        "payload_sha256",
        "remote_executor",
        "result_blind_scoring_runtime_pip_seed_correction",
    }
    assert {k: v for k, v in corrected.items() if k not in ignored} == {
        k: v for k, v in predecessor.items() if k not in ignored
    }
    assert corrected["scientific_gate"] == predecessor["scientific_gate"]
    executor = (HERE / "runpod_execute_real_scoring.py").read_text()
    assert '"venv",\n                "--seed",' in executor


def test_metric_layout_runpod_plan_preserves_scientific_contract() -> None:
    predecessor = json.loads(
        (HERE / "runpod_real_scoring_pip_seeded_plan.json").read_text()
    )
    corrected = json.loads(
        (HERE / "runpod_real_scoring_metric_layout_plan.json").read_text()
    )
    assert corrected["payload_sha256"] == canonical_sha256(corrected)
    ignored = {
        "payload_sha256",
        "remote_executor",
        "result_blind_public_metric_layout_correction",
    }
    assert {k: v for k, v in corrected.items() if k not in ignored} == {
        k: v for k, v in predecessor.items() if k not in ignored
    }
    assert corrected["scientific_gate"] == predecessor["scientific_gate"]
    correction = corrected["result_blind_public_metric_layout_correction"]
    assert correction["scientific_gate"]["metric_source_bytes_changed"] is False
    executor = (HERE / "runpod_execute_real_scoring.py").read_text()
    assert "def materialize_public_metric_layout()" in executor
    assert 'Path("/workspace/real-scoring-public/metric-source")' in executor
    assert 'Path("/workspace/topological-metrics-kaggle")' in executor


def test_metric_runtime_path_plan_preserves_scientific_contract() -> None:
    predecessor = json.loads(
        (HERE / "runpod_real_scoring_metric_layout_plan.json").read_text()
    )
    corrected = json.loads(
        (HERE / "runpod_real_scoring_metric_runtime_path_plan.json").read_text()
    )
    assert corrected["payload_sha256"] == canonical_sha256(corrected)
    ignored = {
        "payload_sha256",
        "remote_executor",
        "result_blind_metric_runtime_path_correction",
    }
    assert {k: v for k, v in corrected.items() if k not in ignored} == {
        k: v for k, v in predecessor.items() if k not in ignored
    }
    assert corrected["scientific_gate"] == predecessor["scientific_gate"]
    correction = corrected["result_blind_metric_runtime_path_correction"]
    assert correction["scientific_gate"][
        "metric_runtime_wheel_or_version_changed"
    ] is False
    executor = (HERE / "runpod_execute_real_scoring.py").read_text()
    assert '"PATH": str(environment_root / "bin")' in executor


def test_isolated_input_plan_preserves_scientific_contract() -> None:
    predecessor = json.loads(
        (HERE / "runpod_real_scoring_metric_runtime_path_plan.json").read_text()
    )
    corrected = json.loads(
        (HERE / "runpod_real_scoring_isolated_input_plan.json").read_text()
    )
    assert corrected["payload_sha256"] == canonical_sha256(corrected)
    ignored = {
        "payload_sha256",
        "remote_executor",
        "result_blind_isolated_scoring_input_correction",
    }
    assert {k: v for k, v in corrected.items() if k not in ignored} == {
        k: v for k, v in predecessor.items() if k not in ignored
    }
    assert corrected["scientific_gate"] == predecessor["scientific_gate"]
    correction = corrected["result_blind_isolated_scoring_input_correction"]
    assert correction["scientific_gate"]["input_bytes_changed"] is False
    executor = (HERE / "runpod_execute_real_scoring.py").read_text()
    assert "def materialize_isolated_scoring_input(" in executor
    assert '"KAGGLE_INPUT_PATH": str(scoring_input_root)' in executor
    assert 'working_root / "input-view"' in executor


def test_metric_verifier_plan_preserves_scientific_contract() -> None:
    predecessor = json.loads(
        (HERE / "runpod_real_scoring_isolated_input_plan.json").read_text()
    )
    corrected = json.loads(
        (HERE / "runpod_real_scoring_metric_verifier_plan.json").read_text()
    )
    assert corrected["payload_sha256"] == canonical_sha256(corrected)
    ignored = {
        "payload_sha256",
        "public_metric_verifier",
        "remote_executor",
        "result_blind_public_metric_verifier_transport_correction",
    }
    assert {k: v for k, v in corrected.items() if k not in ignored} == {
        k: v for k, v in predecessor.items() if k not in ignored
    }
    assert corrected["scientific_gate"] == predecessor["scientific_gate"]
    assert corrected["public_metric_verifier"] == {
        "bytes": 822,
        "file": "verify_official_metric.py",
        "sha256": "09ba89028aa48405a3fc96390b76b455bd0b26d07c908b976f8d6fdfd1aa4e00",
    }
    executor = (HERE / "runpod_execute_real_scoring.py").read_text()
    assert 'parser.add_argument("--metric-verifier", type=Path, required=True)' in executor
    assert 'shutil.copyfile(metric_verifier, verifier_destination)' in executor
