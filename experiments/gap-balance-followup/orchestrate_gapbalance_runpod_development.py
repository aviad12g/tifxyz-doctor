#!/usr/bin/env python3
"""Resume, monitor, and stop the exact $25-capped RunPod campaign pod."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import time


def now() -> datetime:
    return datetime.now(timezone.utc)


def canonical(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_plan(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("payload_sha256", None)
    if canonical(body) != expected:
        raise RuntimeError("RunPod plan payload mismatch")
    return payload


def load_runpod():
    try:
        import runpod
    except ImportError as error:
        raise RuntimeError("controller requires runpod==1.9.0") from error
    return runpod


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.chmod(0o600)
    temp.replace(path)


def validate_pod(
    pod: dict,
    plan: dict,
    *,
    require_exited: bool = False,
    require_exact_reused_pod: bool = True,
    expected_gpu_count: int | None = None,
    expected_gpu_display: str = "RTX 4090",
    maximum_rate_usd: float | None = None,
) -> None:
    frozen = plan["execution"]
    if require_exact_reused_pod and (
        pod.get("id") != frozen["pod_id"] or pod.get("name") != frozen["pod_name"]
    ):
        raise RuntimeError("provider returned a different pod")
    expected_count = frozen["gpu_count"] if expected_gpu_count is None else expected_gpu_count
    if int(pod.get("gpuCount") or -1) != expected_count:
        raise RuntimeError("provider GPU count mismatch")
    if expected_gpu_display not in (pod.get("machine") or {}).get("gpuDisplayName", ""):
        raise RuntimeError("provider GPU type mismatch")
    if pod.get("imageName") != frozen["container_image"]:
        raise RuntimeError("provider image mismatch")
    rate = float(pod.get("costPerHr") or 0)
    rate_ceiling = (
        frozen["maximum_accepted_aggregate_hourly_rate_usd"]
        if maximum_rate_usd is None
        else maximum_rate_usd
    )
    if rate <= 0 or rate > rate_ceiling:
        raise RuntimeError(f"provider rate outside frozen ceiling: {rate}")
    if require_exited and pod.get("desiredStatus") != "EXITED":
        raise RuntimeError("exact reusable pod is not stopped")


def terminate_many(runpod, pod_ids: list[str]) -> None:
    for pod_id in pod_ids:
        try:
            runpod.terminate_pod(pod_id)
        except Exception:
            pass


def public_key_fingerprint(public_key: str) -> str:
    parts = public_key.strip().split()
    if len(parts) < 2 or parts[0] not in {"ssh-ed25519", "ssh-rsa", "ecdsa-sha2-nistp256"}:
        raise RuntimeError("unsupported SSH public key format")
    try:
        material = base64.b64decode(parts[1], validate=True)
    except Exception as error:
        raise RuntimeError("invalid SSH public key material") from error
    digest = base64.b64encode(hashlib.sha256(material).digest()).decode().rstrip("=")
    return f"SHA256:{digest}"


def registered_account_public_keys() -> list[str]:
    try:
        from runpod.api.graphql import run_graphql_query
    except ImportError as error:
        raise RuntimeError("controller requires runpod==1.9.0") from error
    response = run_graphql_query("query { myself { pubKey } }")
    registered = (
        response.get("data", {}).get("myself", {}).get("pubKey") or ""
    )
    return [line.strip() for line in registered.splitlines() if line.strip()]


def public_key_material(public_key: str) -> tuple[str, str]:
    parts = public_key.strip().split()
    if len(parts) < 2:
        raise RuntimeError("invalid SSH public key")
    return parts[0], parts[1]


def account_has_public_key(public_keys: list[str], expected: str) -> bool:
    expected_material = public_key_material(expected)
    for public_key in public_keys:
        try:
            if public_key_material(public_key) == expected_material:
                return True
        except RuntimeError:
            continue
    return False


def create_equivalent_pod(
    runpod,
    *,
    name: str,
    gpu_count: int,
    contract: dict,
    plan: dict,
    ssh_public_key: str | None = None,
    jupyter_password: str | None = None,
) -> dict:
    pod_environment = {
        "GAPBALANCE_DEVELOPMENT_PLAN_SHA256": plan["payload_sha256"],
        "GAPBALANCE_ROLE": "authoritative-development-cache-not-scored",
    }
    if ssh_public_key is not None:
        pod_environment["SSH_PUBLIC_KEY"] = ssh_public_key.strip()
    if jupyter_password is not None:
        pod_environment["JUPYTER_PASSWORD"] = jupyter_password
    created = runpod.create_pod(
        name=name,
        image_name=contract["container_image"],
        gpu_type_id=contract["gpu_type_id"],
        cloud_type="COMMUNITY",
        support_public_ip=True,
        start_ssh=True,
        gpu_count=gpu_count,
        volume_in_gb=contract["volume_in_gb_per_pod"],
        volume_mount_path="/workspace",
        container_disk_in_gb=contract["container_disk_in_gb_per_pod"],
        min_vcpu_count=max(8, gpu_count * 4),
        min_memory_in_gb=max(32, gpu_count * 16),
        ports="22/tcp,8888/http" if jupyter_password is not None else "22/tcp",
        env=pod_environment,
    )
    pod_id = created["id"]
    pod = None
    for _ in range(120):
        pod = runpod.get_pod(pod_id)
        if pod and pod.get("desiredStatus") == "RUNNING" and pod.get("runtime"):
            break
        time.sleep(5)
    else:
        runpod.terminate_pod(pod_id)
        raise RuntimeError(f"replacement pod did not become RUNNING: {pod_id}")
    try:
        validate_pod(
            pod,
            plan,
            require_exact_reused_pod=False,
            expected_gpu_count=gpu_count,
            expected_gpu_display=contract.get("gpu_display_name", "RTX 4090"),
            maximum_rate_usd=float(contract["per_gpu_hourly_ceiling_usd"]) * gpu_count,
        )
        per_gpu = float(pod["costPerHr"]) / gpu_count
        if per_gpu > float(contract["per_gpu_hourly_ceiling_usd"]):
            raise RuntimeError(f"per-GPU provider rate exceeds public ceiling: {per_gpu}")
    except Exception:
        runpod.terminate_pod(pod_id)
        raise
    return pod


def allocate_multi(
    args,
    plan: dict,
    *,
    stop_after_reservation: bool = False,
    use_hardware_substitution: bool = False,
) -> None:
    if args.receipt.exists():
        raise RuntimeError("provider receipt already exists; refusing duplicate allocation")
    if len(args.public_commit) != 40 or any(c not in "0123456789abcdef" for c in args.public_commit):
        raise RuntimeError("public commit must be an exact lowercase Git SHA")
    retry = load_plan(args.multi_pod_retry)
    reservation = None
    dynamic_egress = None
    ssh_retry = None
    account_ssh_retry = None
    explicit_ssh_retry = None
    readiness_retry = None
    jupyter_retry = None
    jupyter_terminal_retry = None
    jupyter_pid_retry = None
    croc_code_retry = None
    croc_room_retry = None
    croc_sender_ready_retry = None
    chunked_transfer_retry = None
    chunked_reallocation_retry = None
    ssh_public_key = None
    deployment_public_key = None
    if use_hardware_substitution:
        if args.hardware_substitution is None:
            raise RuntimeError("hardware substitution command requires the public substitution")
        reservation = load_plan(args.hardware_substitution)
        if (
            reservation.get("runpod_development_plan_payload_sha256") != plan["payload_sha256"]
            or reservation.get("runpod_multi_pod_retry_payload_sha256") != retry["payload_sha256"]
            or reservation.get("egress", {}).get("private_bundle_egress_to_replacements_permitted")
            is not False
            or reservation.get("numerical_reproducibility", {}).get(
                "all_12_jobs_must_use_one_uniform_selected_hardware_type"
            )
            is not True
        ):
            raise RuntimeError("hardware substitution changes a frozen result-blind or egress gate")
        if not stop_after_reservation:
            if args.dynamic_substitute_egress is None:
                raise RuntimeError("active substitute allocation requires dynamic egress approval")
            dynamic_egress = load_plan(args.dynamic_substitute_egress)
            destination = dynamic_egress.get("destination_contract", {})
            if (
                dynamic_egress.get("runpod_development_plan_payload_sha256") != plan["payload_sha256"]
                or dynamic_egress.get("hardware_substitution_payload_sha256")
                != reservation["payload_sha256"]
                or destination.get("allowed_gpu_type_ids")
                != [item["gpu_type_id"] for item in reservation["allowed_hardware_in_order"]]
                or destination.get("maximum_total_gpu_count") != 7
                or destination.get("maximum_aggregate_hourly_rate_usd")
                != plan["execution"]["maximum_accepted_aggregate_hourly_rate_usd"]
                or dynamic_egress.get("egress_exclusions", {}).get("pherc1218_included") is not False
                or dynamic_egress.get("egress_exclusions", {}).get(
                    "confirmation_seeds_500_504_included"
                )
                is not False
            ):
                raise RuntimeError("dynamic substitute egress changes the frozen provider or sealed gate")
            if (
                args.ssh_retry is not None
                or args.ssh_public_key is not None
                or args.account_ssh_retry is not None
                or args.explicit_ssh_retry is not None
                or args.deployment_public_key is not None
                or args.ssh_readiness_retry is not None
                or args.jupyter_croc_retry is not None
                or args.jupyter_terminal_retry is not None
                or args.jupyter_pid_retry is not None
                or args.croc_code_retry is not None
                or args.croc_room_retry is not None
                or args.croc_sender_ready_retry is not None
                or args.chunked_transfer_retry is not None
                or args.chunked_reallocation_retry is not None
            ):
                if args.ssh_retry is None or args.ssh_public_key is None:
                    raise RuntimeError("SSH injection retry requires both its public freeze and key path")
                ssh_retry = load_plan(args.ssh_retry)
                if (
                    ssh_retry.get("runpod_development_plan_payload_sha256") != plan["payload_sha256"]
                    or ssh_retry.get("dynamic_substitute_egress_payload_sha256")
                    != dynamic_egress["payload_sha256"]
                    or ssh_retry.get("retry", {}).get("private_key_leaves_local_mac") is not False
                    or ssh_retry.get("retry", {}).get("inject_public_key_environment_variable")
                    != "SSH_PUBLIC_KEY"
                ):
                    raise RuntimeError("SSH retry changes the frozen egress, key, or sealed gate")
                ssh_public_key = args.ssh_public_key.read_text(encoding="utf-8").strip()
                if public_key_fingerprint(ssh_public_key) != ssh_retry["retry"]["public_key_fingerprint"]:
                    raise RuntimeError("SSH public key fingerprint mismatch")
                if args.account_ssh_retry is not None:
                    account_ssh_retry = load_plan(args.account_ssh_retry)
                    registration = account_ssh_retry.get("account_key_registration", {})
                    if (
                        account_ssh_retry.get("runpod_development_plan_payload_sha256")
                        != plan["payload_sha256"]
                        or account_ssh_retry.get("ssh_injection_retry_payload_sha256")
                        != ssh_retry["payload_sha256"]
                        or registration.get("public_key_fingerprint")
                        != ssh_retry["retry"]["public_key_fingerprint"]
                        or registration.get("private_key_leaves_local_mac") is not False
                        or registration.get("registration_required_before_new_pod_creation")
                        is not True
                    ):
                        raise RuntimeError("account SSH retry changes the frozen key or sealed gate")
                if args.explicit_ssh_retry is not None or args.deployment_public_key is not None:
                    if (
                        args.explicit_ssh_retry is None
                        or args.deployment_public_key is None
                        or account_ssh_retry is None
                    ):
                        raise RuntimeError(
                            "explicit SSH identity retry requires its public freeze, public key, and account retry"
                        )
                    explicit_ssh_retry = load_plan(args.explicit_ssh_retry)
                    identity = explicit_ssh_retry.get("explicit_identity", {})
                    if (
                        explicit_ssh_retry.get("runpod_development_plan_payload_sha256")
                        != plan["payload_sha256"]
                        or explicit_ssh_retry.get("account_ssh_retry_payload_sha256")
                        != account_ssh_retry["payload_sha256"]
                        or identity.get("private_key_leaves_local_mac") is not False
                        or identity.get("identities_only") is not True
                        or identity.get("account_key_registered_before_new_pod_creation")
                        is not True
                    ):
                        raise RuntimeError("explicit SSH identity retry changes the frozen key or sealed gate")
                    deployment_public_key = args.deployment_public_key.read_text(
                        encoding="utf-8"
                    ).strip()
                    if public_key_fingerprint(deployment_public_key) != identity.get(
                        "public_key_fingerprint"
                    ):
                        raise RuntimeError("explicit deployment SSH key fingerprint mismatch")
                    if args.ssh_readiness_retry is not None:
                        readiness_retry = load_plan(args.ssh_readiness_retry)
                        readiness = readiness_retry.get("readiness_retry", {})
                        if (
                            readiness_retry.get("runpod_development_plan_payload_sha256")
                            != plan["payload_sha256"]
                            or readiness_retry.get(
                                "explicit_ssh_identity_retry_payload_sha256"
                            )
                            != explicit_ssh_retry["payload_sha256"]
                            or readiness.get("bundle_upload_starts_only_after_probe_success")
                            is not True
                            or readiness.get("stop_all_pods_if_probe_never_succeeds")
                            is not True
                            or int(readiness.get("maximum_elapsed_seconds_per_pod", 0))
                            > 300
                        ):
                            raise RuntimeError("SSH readiness retry changes a frozen transport or sealed gate")
                elif args.ssh_readiness_retry is not None:
                    raise RuntimeError("SSH readiness retry requires the explicit SSH identity retry")
                if args.jupyter_croc_retry is not None:
                    if readiness_retry is None:
                        raise RuntimeError("Jupyter/croc retry requires the SSH readiness retry")
                    jupyter_retry = load_plan(args.jupyter_croc_retry)
                    control = jupyter_retry.get("jupyter_control", {})
                    transfer = jupyter_retry.get("runpodctl_transfer", {})
                    payment = jupyter_retry.get("payment_authority", {})
                    if (
                        jupyter_retry.get("runpod_development_plan_payload_sha256")
                        != plan["payload_sha256"]
                        or jupyter_retry.get("ssh_readiness_retry_payload_sha256")
                        != readiness_retry["payload_sha256"]
                        or int(control.get("http_port", 0)) != 8888
                        or int(control.get("credential_entropy_bytes_per_pod", 0)) < 32
                        or int(control.get("maximum_readiness_seconds_per_pod", 0)) > 600
                        or control.get("credential_material_published_or_printed") is not False
                        or transfer.get("encrypted_croc_relay") is not True
                        or transfer.get("bundle_manifest_verified_before_executor") is not True
                        or transfer.get(
                            "local_bundle_symlinks_materialized_in_exact_tar_before_egress"
                        )
                        is not True
                        or payment.get("direct_credit_card_charge_permitted") is not False
                        or payment.get("runpod_auto_pay_verified_disabled") is not True
                    ):
                        raise RuntimeError("Jupyter/croc retry changes a frozen transport or sealed gate")
                if args.jupyter_terminal_retry is not None:
                    if jupyter_retry is None:
                        raise RuntimeError("Jupyter terminal retry requires the Jupyter/croc retry")
                    jupyter_terminal_retry = load_plan(args.jupyter_terminal_retry)
                    terminal_control = jupyter_terminal_retry.get("terminal_control", {})
                    payment = jupyter_terminal_retry.get("payment_authority", {})
                    if (
                        jupyter_terminal_retry.get("runpod_development_plan_payload_sha256")
                        != plan["payload_sha256"]
                        or jupyter_terminal_retry.get("jupyter_croc_retry_payload_sha256")
                        != jupyter_retry["payload_sha256"]
                        or terminal_control.get("websocket_path_template")
                        != "/terminals/websocket/{terminal_name}"
                        or terminal_control.get("delete_terminal_after_failed_handshake") is not True
                        or jupyter_terminal_retry.get("failed_deployment", {}).get(
                            "bundle_bytes_uploaded"
                        )
                        != 0
                        or payment.get("direct_credit_card_charge_permitted") is not False
                        or payment.get("runpod_auto_pay_verified_disabled") is not True
                    ):
                        raise RuntimeError(
                            "Jupyter terminal retry changes a frozen transport, budget, or sealed gate"
                        )
                if args.jupyter_pid_retry is not None:
                    if jupyter_terminal_retry is None:
                        raise RuntimeError("Jupyter PID retry requires the terminal-path retry")
                    jupyter_pid_retry = load_plan(args.jupyter_pid_retry)
                    terminal_control = jupyter_pid_retry.get("terminal_control", {})
                    payment = jupyter_pid_retry.get("payment_authority", {})
                    if (
                        jupyter_pid_retry.get("runpod_development_plan_payload_sha256")
                        != plan["payload_sha256"]
                        or jupyter_pid_retry.get("jupyter_terminal_retry_payload_sha256")
                        != jupyter_terminal_retry["payload_sha256"]
                        or terminal_control.get("background_pid_output_format") != "tagged"
                        or jupyter_pid_retry.get("failed_deployment", {}).get(
                            "bundle_bytes_uploaded"
                        )
                        != 0
                        or payment.get("direct_credit_card_charge_permitted") is not False
                        or payment.get("runpod_auto_pay_verified_disabled") is not True
                    ):
                        raise RuntimeError(
                            "Jupyter PID retry changes a frozen transport, budget, or sealed gate"
                        )
                if args.croc_code_retry is not None:
                    if jupyter_pid_retry is None:
                        raise RuntimeError("croc code retry requires the Jupyter PID retry")
                    croc_code_retry = load_plan(args.croc_code_retry)
                    code_contract = croc_code_retry.get("runpodctl_code_contract", {})
                    payment = croc_code_retry.get("payment_authority", {})
                    if (
                        croc_code_retry.get("runpod_development_plan_payload_sha256")
                        != plan["payload_sha256"]
                        or croc_code_retry.get("jupyter_pid_retry_payload_sha256")
                        != jupyter_pid_retry["payload_sha256"]
                        or code_contract.get("sender_starts_before_receiver") is not True
                        or int(code_contract.get("base_secret_entropy_bytes", 0)) < 24
                        or code_contract.get("capture_relay_qualified_code_from_sender_stdout")
                        is not True
                        or croc_code_retry.get("failed_deployment", {}).get(
                            "bundle_bytes_uploaded"
                        )
                        != 0
                        or payment.get("direct_credit_card_charge_permitted") is not False
                        or payment.get("runpod_auto_pay_verified_disabled") is not True
                    ):
                        raise RuntimeError(
                            "croc code retry changes a frozen transport, budget, or sealed gate"
                        )
                if args.croc_room_retry is not None:
                    if croc_code_retry is None:
                        raise RuntimeError("croc room retry requires the croc code retry")
                    croc_room_retry = load_plan(args.croc_room_retry)
                    receiver_retry = croc_room_retry.get("receiver_retry", {})
                    payment = croc_room_retry.get("payment_authority", {})
                    if (
                        croc_room_retry.get("runpod_development_plan_payload_sha256")
                        != plan["payload_sha256"]
                        or croc_room_retry.get("croc_code_retry_payload_sha256")
                        != croc_code_retry["payload_sha256"]
                        or receiver_retry.get("retry_room_not_ready") is not True
                        or receiver_retry.get("poll_rc_during_sender") is not True
                        or int(receiver_retry.get("maximum_retry_seconds", 0)) > 180
                        or croc_room_retry.get("failed_deployment", {}).get(
                            "bundle_bytes_uploaded"
                        )
                        != 0
                        or payment.get("direct_credit_card_charge_permitted") is not False
                        or payment.get("runpod_auto_pay_verified_disabled") is not True
                    ):
                        raise RuntimeError(
                            "croc room retry changes a frozen transport, budget, or sealed gate"
                        )
                if args.croc_sender_ready_retry is not None:
                    if croc_room_retry is None:
                        raise RuntimeError("croc sender readiness retry requires the room retry")
                    croc_sender_ready_retry = load_plan(args.croc_sender_ready_retry)
                    readiness = croc_sender_ready_retry.get("sender_readiness", {})
                    payment = croc_sender_ready_retry.get("payment_authority", {})
                    if (
                        croc_sender_ready_retry.get("runpod_development_plan_payload_sha256")
                        != plan["payload_sha256"]
                        or croc_sender_ready_retry.get("croc_room_retry_payload_sha256")
                        != croc_room_retry["payload_sha256"]
                        or readiness.get("require_post_hash_banner") is not True
                        or int(readiness.get("maximum_banner_seconds", 0)) > 120
                        or int(readiness.get("post_banner_delay_seconds", 0)) != 5
                        or croc_sender_ready_retry.get("failed_deployment", {}).get(
                            "bundle_bytes_uploaded"
                        )
                        != 0
                        or payment.get("direct_credit_card_charge_permitted") is not False
                        or payment.get("runpod_auto_pay_verified_disabled") is not True
                    ):
                        raise RuntimeError(
                            "croc sender readiness retry changes a frozen transport, budget, or sealed gate"
                        )
                if args.chunked_transfer_retry is not None:
                    if croc_sender_ready_retry is None:
                        raise RuntimeError(
                            "chunked transfer retry requires the sender readiness retry"
                        )
                    chunked_transfer_retry = load_plan(args.chunked_transfer_retry)
                    chunked = chunked_transfer_retry.get("retry_transport", {})
                    if (
                        chunked_transfer_retry.get(
                            "runpod_development_plan_payload_sha256"
                        )
                        != plan["payload_sha256"]
                        or chunked_transfer_retry.get(
                            "croc_sender_ready_retry_payload_sha256"
                        )
                        != croc_sender_ready_retry["payload_sha256"]
                        or int(chunked.get("chunk_bytes", 0)) != 128 << 20
                        or int(chunked.get("maximum_attempts_per_chunk", 0)) != 4
                        or chunked.get("restart_only_the_failed_chunk") is not True
                        or chunked_transfer_retry.get("sealed_gates", {}).get(
                            "scientific_endpoints_scored"
                        )
                        is not False
                    ):
                        raise RuntimeError(
                            "chunked transfer retry changes a frozen transport or sealed gate"
                        )
                if args.chunked_reallocation_retry is not None:
                    if chunked_transfer_retry is None:
                        raise RuntimeError(
                            "chunked reallocation retry requires the chunked transfer retry"
                        )
                    chunked_reallocation_retry = load_plan(
                        args.chunked_reallocation_retry
                    )
                    replacement = chunked_reallocation_retry.get(
                        "replacement_contract", {}
                    )
                    if (
                        chunked_reallocation_retry.get(
                            "runpod_development_plan_payload_sha256"
                        )
                        != plan["payload_sha256"]
                        or chunked_reallocation_retry.get(
                            "chunked_transfer_retry_payload_sha256"
                        )
                        != chunked_transfer_retry["payload_sha256"]
                        or chunked_reallocation_retry.get(
                            "hardware_substitution_payload_sha256"
                        )
                        != reservation["payload_sha256"]
                        or chunked_reallocation_retry.get(
                            "dynamic_substitute_egress_payload_sha256"
                        )
                        != dynamic_egress["payload_sha256"]
                        or replacement.get("maximum_total_gpu_count") != 7
                        or replacement.get("maximum_aggregate_hourly_rate_usd")
                        != plan["execution"][
                            "maximum_accepted_aggregate_hourly_rate_usd"
                        ]
                        or replacement.get(
                            "exact_new_pod_ids_recorded_privately_before_upload"
                        )
                        is not True
                        or chunked_reallocation_retry.get("billing", {}).get(
                            "additional_spend_from_failed_resume_usd"
                        )
                        != 0.0
                    ):
                        raise RuntimeError(
                            "chunked reallocation retry changes a frozen provider or budget gate"
                        )
    elif stop_after_reservation:
        if args.replacement_reservation is None:
            raise RuntimeError("reserve-multi requires the public replacement reservation")
        reservation = load_plan(args.replacement_reservation)
        if (
            reservation.get("runpod_development_plan_payload_sha256") != plan["payload_sha256"]
            or reservation.get("runpod_multi_pod_retry_payload_sha256") != retry["payload_sha256"]
            or reservation.get("authorization", {}).get("private_bundle_egress_to_replacements_permitted")
            is not False
            or reservation.get("provider_contract", {}).get("total_gpu_count") != 7
            or reservation.get("provider_contract", {}).get("aggregate_hourly_ceiling_usd")
            != plan["execution"]["maximum_accepted_aggregate_hourly_rate_usd"]
        ):
            raise RuntimeError("replacement reservation changes the frozen provider, budget, or egress gate")
    if retry.get("runpod_development_plan_payload_sha256") != plan["payload_sha256"]:
        raise RuntimeError("multi-pod retry is bound to another development plan")
    if (
        retry.get("budget") != plan["budget"]
        or retry.get("authority") != plan["authority"]
        or retry.get("sealed_gates") != plan["sealed_gates"]
    ):
        raise RuntimeError("multi-pod retry changes a frozen budget, authority, or sealed gate")
    contract = retry["provider_contract"]
    if (
        contract["total_gpu_count"] != 7
        or contract["aggregate_hourly_ceiling_usd"] != plan["execution"]["maximum_accepted_aggregate_hourly_rate_usd"]
        or contract["container_image"] != plan["execution"]["container_image"]
    ):
        raise RuntimeError("multi-pod provider contract differs from the primary plan")
    hardware_contracts = (
        [{**contract, **candidate} for candidate in reservation["allowed_hardware_in_order"]]
        if use_hardware_substitution
        else [contract]
    )
    for candidate in hardware_contracts:
        if (
            int(candidate["total_gpu_count"]) != 7
            or float(candidate["aggregate_hourly_ceiling_usd"])
            > float(plan["execution"]["maximum_accepted_aggregate_hourly_rate_usd"])
            or int(candidate.get("gpu_memory_gb", 24)) < 24
        ):
            raise RuntimeError("hardware substitution exceeds the frozen GPU, memory, or rate ceiling")
    runpod = load_runpod()
    if account_ssh_retry is not None:
        registered_keys = registered_account_public_keys()
        if not account_has_public_key(registered_keys, ssh_public_key):
            raise RuntimeError("frozen SSH public key is not registered on the RunPod account")
        if deployment_public_key is not None and not account_has_public_key(
            registered_keys, deployment_public_key
        ):
            raise RuntimeError("explicit deployment SSH key is not registered on the RunPod account")
    billing_started = now()
    errors = []
    accepted = None
    accepted_jupyter_access = None
    selected_layout = None
    selected_contract = None
    for hardware in hardware_contracts:
        for layout in retry["allowed_layouts_in_order"]:
            created = []
            created_jupyter_access = {}
            try:
                for index, partition in enumerate(layout["partitions"]):
                    jupyter_password = (
                        secrets.token_urlsafe(
                            int(jupyter_retry["jupyter_control"]["credential_entropy_bytes_per_pod"])
                        )
                        if jupyter_retry is not None
                        else None
                    )
                    pod = create_equivalent_pod(
                        runpod,
                        name=(
                            f"gapbalance-dev-{args.public_commit[:8]}-"
                            f"{hardware['gpu_display_name'].lower().replace(' ', '-')}-"
                            f"{layout['name']}-{index}"
                        ),
                        gpu_count=int(partition["gpu_count"]),
                        contract=hardware,
                        plan=plan,
                        ssh_public_key=ssh_public_key,
                        jupyter_password=jupyter_password,
                    )
                    created.append({
                        "id": pod["id"],
                        "name": pod["name"],
                        "gpu_count": int(pod["gpuCount"]),
                        "gpu_name": pod["machine"]["gpuDisplayName"],
                        "image_name": pod["imageName"],
                        "aggregate_hourly_rate_usd": float(pod["costPerHr"]),
                        "jobs": partition["jobs"],
                        "waves": partition["waves"],
                    })
                    if jupyter_password is not None:
                        created_jupyter_access[pod["id"]] = {
                            "base_url": f"https://{pod['id']}-8888.proxy.runpod.net",
                            "password": jupyter_password,
                        }
                rate = sum(item["aggregate_hourly_rate_usd"] for item in created)
                if rate <= 0 or rate > float(hardware["aggregate_hourly_ceiling_usd"]):
                    raise RuntimeError(f"aggregate provider rate exceeds public ceiling: {rate}")
                accepted = created
                accepted_jupyter_access = created_jupyter_access
                selected_layout = layout["name"]
                selected_contract = hardware
                break
            except Exception as error:
                terminate_many(runpod, [item["id"] for item in created])
                errors.append({
                    "gpu_type_id": hardware["gpu_type_id"],
                    "layout": layout["name"],
                    "error": (
                        error.__class__.__name__
                        if jupyter_retry is not None
                        else str(error)
                    ),
                })
        if accepted is not None:
            break
    if accepted is None:
        failure = {
            "schema_version": "1.0",
            "status": "MULTI_POD_ALLOCATION_FAILED_NO_ACTIVE_PODS",
            "plan_payload_sha256": plan["payload_sha256"],
            "multi_pod_retry_payload_sha256": retry["payload_sha256"],
            "attempt_started_at": billing_started.isoformat(),
            "layout_errors": errors,
            "scientific_endpoints_inspected": False,
            "confirmation_outputs_inspected": False,
        }
        atomic_json(args.receipt, failure)
        raise RuntimeError(f"no public multi-pod layout was available: {errors}")
    receipt = {
        "schema_version": "1.0",
        "status": "PODS_ALLOCATED_AWAITING_DEPLOYMENT",
        "allocation_route": (
            "result_blind_uniform_hardware_substitution"
            if use_hardware_substitution
            else "equivalent_multi_pod_after_two_zero_cost_seven_gpu_failures"
        ),
        "multi_pod_retry_payload_sha256": retry["payload_sha256"],
        "plan_payload_sha256": plan["payload_sha256"],
        "public_runpod_commit": args.public_commit,
        "billing_started_at": billing_started.isoformat(),
        "development_billing_cutoff_usd": plan["budget"]["development_billing_cutoff_usd"],
        "absolute_campaign_cap_usd": plan["budget"]["absolute_campaign_cap_usd"],
        "confirmation_reserve_usd": plan["budget"]["confirmation_reserve_usd"],
        "selected_layout": selected_layout,
        "selected_hardware": selected_contract,
        "layout_attempt_errors": errors,
        "pods": accepted,
        "total_hourly_rate_usd": sum(item["aggregate_hourly_rate_usd"] for item in accepted),
        "scientific_endpoints_inspected": False,
        "confirmation_outputs_inspected": False,
    }
    if dynamic_egress is not None:
        prior_spend = (
            chunked_reallocation_retry["billing"]["prior_conservative_development_spend_usd"]
            if chunked_reallocation_retry is not None
            else chunked_transfer_retry["billing"]["prior_conservative_development_spend_usd"]
            if chunked_transfer_retry is not None
            else croc_sender_ready_retry["billing"]["prior_conservative_development_spend_usd"]
            if croc_sender_ready_retry is not None
            else croc_room_retry["billing"]["prior_conservative_development_spend_usd"]
            if croc_room_retry is not None
            else croc_code_retry["billing"]["prior_conservative_development_spend_usd"]
            if croc_code_retry is not None
            else jupyter_pid_retry["billing"]["prior_conservative_development_spend_usd"]
            if jupyter_pid_retry is not None
            else jupyter_terminal_retry["billing"]["prior_conservative_development_spend_usd"]
            if jupyter_terminal_retry is not None
            else jupyter_retry["billing"]["prior_conservative_development_spend_usd"]
            if jupyter_retry is not None
            else readiness_retry["billing"]["prior_conservative_development_spend_usd"]
            if readiness_retry is not None
            else explicit_ssh_retry["billing"]["prior_conservative_development_spend_usd"]
            if explicit_ssh_retry is not None
            else account_ssh_retry["billing"]["prior_conservative_development_spend_usd"]
            if account_ssh_retry is not None
            else ssh_retry["billing"]["prior_conservative_development_spend_usd"]
            if ssh_retry is not None
            else dynamic_egress["billing"]["prior_conservative_development_spend_usd"]
        )
        receipt.update(
            dynamic_substitute_egress_payload_sha256=dynamic_egress["payload_sha256"],
            private_bundle_egress_permitted=True,
            prior_conservative_development_spend_usd=prior_spend,
            active_billing_started_at=billing_started.isoformat(),
        )
        if ssh_retry is not None:
            receipt.update(
                ssh_injection_retry_payload_sha256=ssh_retry["payload_sha256"],
                ssh_public_key_fingerprint=ssh_retry["retry"]["public_key_fingerprint"],
            )
        if account_ssh_retry is not None:
            receipt.update(
                account_ssh_retry_payload_sha256=account_ssh_retry["payload_sha256"],
                account_ssh_key_verified_before_allocation=True,
            )
        if explicit_ssh_retry is not None:
            receipt.update(
                explicit_ssh_retry_payload_sha256=explicit_ssh_retry["payload_sha256"],
                deployment_ssh_public_key_fingerprint=explicit_ssh_retry[
                    "explicit_identity"
                ]["public_key_fingerprint"],
                deployment_ssh_key_verified_before_allocation=True,
            )
        if readiness_retry is not None:
            receipt.update(
                ssh_readiness_retry_payload_sha256=readiness_retry["payload_sha256"],
                ssh_readiness_probe_required_before_bundle_upload=True,
            )
        if jupyter_retry is not None:
            receipt.update(
                jupyter_croc_retry_payload_sha256=jupyter_retry["payload_sha256"],
                jupyter_credentials_private=True,
                jupyter_access=accepted_jupyter_access,
            )
        if jupyter_terminal_retry is not None:
            receipt.update(
                jupyter_terminal_retry_payload_sha256=jupyter_terminal_retry[
                    "payload_sha256"
                ],
            )
        if jupyter_pid_retry is not None:
            receipt.update(
                jupyter_pid_retry_payload_sha256=jupyter_pid_retry["payload_sha256"],
            )
        if croc_code_retry is not None:
            receipt.update(
                croc_code_retry_payload_sha256=croc_code_retry["payload_sha256"],
            )
        if croc_room_retry is not None:
            receipt.update(
                croc_room_retry_payload_sha256=croc_room_retry["payload_sha256"],
            )
        if croc_sender_ready_retry is not None:
            receipt.update(
                croc_sender_ready_retry_payload_sha256=croc_sender_ready_retry[
                    "payload_sha256"
                ],
            )
        if chunked_transfer_retry is not None:
            receipt.update(
                chunked_transfer_retry_payload_sha256=chunked_transfer_retry[
                    "payload_sha256"
                ],
            )
        if chunked_reallocation_retry is not None:
            receipt.update(
                chunked_reallocation_retry_payload_sha256=chunked_reallocation_retry[
                    "payload_sha256"
                ],
            )
    if stop_after_reservation:
        try:
            for item in accepted:
                runpod.stop_pod(item["id"])
            for item in accepted:
                for _ in range(120):
                    pod = runpod.get_pod(item["id"])
                    if pod and pod.get("desiredStatus") == "EXITED":
                        break
                    time.sleep(5)
                else:
                    raise RuntimeError(f"replacement pod did not stop: {item['id']}")
        except Exception:
            terminate_many(runpod, [item["id"] for item in accepted])
            raise
        stopped_at = now()
        receipt.update(
            status="STOPPED_REPLACEMENTS_AWAITING_EXACT_EGRESS_APPROVAL",
            replacement_reservation_payload_sha256=reservation["payload_sha256"],
            provider_action_at=stopped_at.isoformat(),
            conservative_development_spend_usd=round(
                float(reservation["budget"]["prior_conservative_development_spend_usd"])
                + receipt["total_hourly_rate_usd"]
                * max(0.0, (stopped_at - billing_started).total_seconds())
                / 3600.0,
                6,
            ),
            private_bundle_egress_permitted=False,
        )
    atomic_json(args.receipt, receipt)
    printed_receipt = json.loads(json.dumps(receipt))
    for access in (printed_receipt.get("jupyter_access") or {}).values():
        access["password"] = "REDACTED"
    print(json.dumps(printed_receipt, indent=2, sort_keys=True))


def allocate(args, plan: dict) -> None:
    if args.receipt.exists():
        raise RuntimeError("provider receipt already exists; refusing duplicate allocation")
    if len(args.public_commit) != 40 or any(c not in "0123456789abcdef" for c in args.public_commit):
        raise RuntimeError("public commit must be an exact lowercase Git SHA")
    retry = load_plan(args.allocation_retry)
    if retry.get("runpod_development_plan_payload_sha256") != plan["payload_sha256"]:
        raise RuntimeError("allocation retry is bound to another development plan")
    authorized = retry.get("authorized_retry", {})
    if (
        authorized.get("route") != "create one equivalent replacement allocation"
        or authorized.get("gpu_count") != plan["execution"]["gpu_count"]
        or authorized.get("container_image") != plan["execution"]["container_image"]
        or authorized.get("maximum_accepted_aggregate_hourly_rate_usd")
        != plan["execution"]["maximum_accepted_aggregate_hourly_rate_usd"]
        or retry.get("budget") != plan["budget"]
        or retry.get("sealed_gates") != plan["sealed_gates"]
    ):
        raise RuntimeError("allocation retry changes a frozen execution or budget gate")
    runpod = load_runpod()
    created = runpod.create_pod(
        name=f"gapbalance-dev-{args.public_commit[:8]}-g7",
        image_name=authorized["container_image"],
        gpu_type_id=authorized["gpu_type_id"],
        cloud_type="COMMUNITY",
        support_public_ip=True,
        start_ssh=True,
        gpu_count=authorized["gpu_count"],
        volume_in_gb=authorized["volume_in_gb"],
        volume_mount_path="/workspace",
        container_disk_in_gb=authorized["container_disk_in_gb"],
        min_vcpu_count=28,
        min_memory_in_gb=112,
        ports="22/tcp",
        env={
            "GAPBALANCE_DEVELOPMENT_PLAN_SHA256": plan["payload_sha256"],
            "GAPBALANCE_ROLE": "authoritative-development-cache-not-scored",
        },
    )
    pod_id = created["id"]
    try:
        pod = None
        for _ in range(120):
            pod = runpod.get_pod(pod_id)
            if pod and pod.get("desiredStatus") == "RUNNING" and pod.get("runtime"):
                break
            time.sleep(5)
        else:
            raise RuntimeError("equivalent replacement pod did not become RUNNING")
        validate_pod(pod, plan, require_exact_reused_pod=False)
    except Exception:
        runpod.terminate_pod(pod_id)
        raise
    started = now()
    receipt = {
        "schema_version": "1.0",
        "status": "POD_RESUMED_AWAITING_DEPLOYMENT",
        "allocation_route": "new_equivalent_pod_after_result_blind_resume_failure",
        "allocation_retry_payload_sha256": retry["payload_sha256"],
        "plan_payload_sha256": plan["payload_sha256"],
        "public_runpod_commit": args.public_commit,
        "billing_started_at": started.isoformat(),
        "development_billing_cutoff_usd": plan["budget"]["development_billing_cutoff_usd"],
        "absolute_campaign_cap_usd": plan["budget"]["absolute_campaign_cap_usd"],
        "confirmation_reserve_usd": plan["budget"]["confirmation_reserve_usd"],
        "pod": {
            "id": pod["id"],
            "name": pod["name"],
            "gpu_count": int(pod["gpuCount"]),
            "gpu_name": pod["machine"]["gpuDisplayName"],
            "image_name": pod["imageName"],
            "aggregate_hourly_rate_usd": float(pod["costPerHr"]),
        },
        "scientific_endpoints_inspected": False,
        "confirmation_outputs_inspected": False,
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def resume_multi(args, plan: dict) -> None:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    approval = load_plan(args.egress_resume)
    if (
        receipt.get("status") != "STOPPED"
        or receipt.get("plan_payload_sha256") != plan["payload_sha256"]
        or approval.get("runpod_development_plan_payload_sha256") != plan["payload_sha256"]
        or approval.get("multi_pod_retry_payload_sha256") != receipt.get("multi_pod_retry_payload_sha256")
        or approval.get("resume", {}).get("prior_conservative_spend_usd")
        != receipt.get("conservative_development_spend_usd")
        or approval.get("budget") != plan["budget"]
        or approval.get("sealed_gates") != plan["sealed_gates"]
    ):
        raise RuntimeError("egress/resume approval is not bound to the stopped capped allocation")
    approved_pods = approval["resume"]["exact_pods_only"]
    if approved_pods != [
        {"id": pod["id"], "gpu_count": pod["gpu_count"], "jobs": pod["jobs"]}
        for pod in receipt["pods"]
    ]:
        raise RuntimeError("egress/resume approval pod set mismatch")
    runpod = load_runpod()
    resumed_ids = []
    try:
        for frozen in receipt["pods"]:
            pod = runpod.get_pod(frozen["id"])
            if pod is None or pod.get("desiredStatus") != "EXITED":
                raise RuntimeError(f"approved pod is not stopped: {frozen['id']}")
            runpod.resume_pod(frozen["id"], gpu_count=frozen["gpu_count"])
            resumed_ids.append(frozen["id"])
        for frozen in receipt["pods"]:
            for _ in range(120):
                pod = runpod.get_pod(frozen["id"])
                if pod and pod.get("desiredStatus") == "RUNNING" and pod.get("runtime"):
                    break
                time.sleep(5)
            else:
                raise RuntimeError(f"approved pod did not resume: {frozen['id']}")
            validate_pod(
                pod,
                plan,
                require_exact_reused_pod=False,
                expected_gpu_count=frozen["gpu_count"],
            )
            if float(pod["costPerHr"]) != float(frozen["aggregate_hourly_rate_usd"]):
                raise RuntimeError(f"resumed pod rate changed: {frozen['id']}")
    except Exception:
        for pod_id in resumed_ids:
            try:
                runpod.stop_pod(pod_id)
            except Exception:
                pass
        raise
    receipt.update(
        status="PODS_ALLOCATED_AWAITING_DEPLOYMENT",
        egress_resume_payload_sha256=approval["payload_sha256"],
        active_billing_started_at=now().isoformat(),
        prior_conservative_development_spend_usd=receipt["conservative_development_spend_usd"],
    )
    atomic_json(args.receipt, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "pod_ids": resumed_ids,
        "prior_conservative_development_spend_usd": receipt["prior_conservative_development_spend_usd"],
        "active_billing_started_at": receipt["active_billing_started_at"],
    }, indent=2, sort_keys=True))


def resume_substitute(args, plan: dict) -> None:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    approval = load_plan(args.substitute_egress_resume)
    if (
        receipt.get("status") != "STOPPED_REPLACEMENTS_AWAITING_EXACT_EGRESS_APPROVAL"
        or receipt.get("plan_payload_sha256") != plan["payload_sha256"]
        or approval.get("runpod_development_plan_payload_sha256") != plan["payload_sha256"]
        or approval.get("hardware_substitution_payload_sha256")
        != receipt.get("replacement_reservation_payload_sha256")
        or approval.get("billing", {}).get("prior_conservative_development_spend_usd")
        != receipt.get("conservative_development_spend_usd")
        or approval.get("billing", {}).get("development_cutoff_usd")
        != plan["budget"]["development_billing_cutoff_usd"]
        or approval.get("egress", {}).get("pherc1218_included") is not False
        or approval.get("egress", {}).get("confirmation_seeds_500_504_included") is not False
    ):
        raise RuntimeError("substitute egress approval is not bound to the stopped capped allocation")
    approved = approval["pods"]
    frozen = receipt["pods"]
    if approved != [
        {
            "id": pod["id"],
            "gpu_count": pod["gpu_count"],
            "gpu_name": pod["gpu_name"],
            "aggregate_hourly_rate_usd": pod["aggregate_hourly_rate_usd"],
            "jobs": pod["jobs"],
        }
        for pod in frozen
    ]:
        raise RuntimeError("substitute egress approval pod set mismatch")
    runpod = load_runpod()
    resumed_ids = []
    try:
        for pod_record in frozen:
            pod = runpod.get_pod(pod_record["id"])
            if pod is None or pod.get("desiredStatus") != "EXITED":
                raise RuntimeError(f"approved substitute pod is not stopped: {pod_record['id']}")
            runpod.resume_pod(pod_record["id"], gpu_count=pod_record["gpu_count"])
            resumed_ids.append(pod_record["id"])
        for pod_record in frozen:
            for _ in range(120):
                pod = runpod.get_pod(pod_record["id"])
                if pod and pod.get("desiredStatus") == "RUNNING" and pod.get("runtime"):
                    break
                time.sleep(5)
            else:
                raise RuntimeError(f"approved substitute pod did not resume: {pod_record['id']}")
            validate_pod(
                pod,
                plan,
                require_exact_reused_pod=False,
                expected_gpu_count=pod_record["gpu_count"],
                expected_gpu_display=pod_record["gpu_name"],
                maximum_rate_usd=pod_record["aggregate_hourly_rate_usd"],
            )
            if float(pod["costPerHr"]) != float(pod_record["aggregate_hourly_rate_usd"]):
                raise RuntimeError(f"substitute pod rate changed: {pod_record['id']}")
    except Exception:
        for pod_id in resumed_ids:
            try:
                runpod.stop_pod(pod_id)
            except Exception:
                pass
        raise
    receipt.update(
        status="PODS_ALLOCATED_AWAITING_DEPLOYMENT",
        substitute_egress_resume_payload_sha256=approval["payload_sha256"],
        active_billing_started_at=now().isoformat(),
        prior_conservative_development_spend_usd=receipt["conservative_development_spend_usd"],
        private_bundle_egress_permitted=True,
    )
    atomic_json(args.receipt, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "pod_ids": resumed_ids,
        "aggregate_hourly_rate_usd": receipt["total_hourly_rate_usd"],
        "prior_conservative_development_spend_usd": receipt[
            "prior_conservative_development_spend_usd"
        ],
        "active_billing_started_at": receipt["active_billing_started_at"],
    }, indent=2, sort_keys=True))


def resume_chunked(args, plan: dict) -> None:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    retry = load_plan(args.chunked_transfer_retry)
    if len(args.public_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.public_commit
    ):
        raise RuntimeError("public commit must be an exact lowercase Git SHA")
    frozen_pods = receipt.get("pods") or []
    prior = float(retry.get("billing", {}).get("prior_conservative_development_spend_usd", -1))
    if (
        receipt.get("status") != "STOPPED"
        or receipt.get("plan_payload_sha256") != plan["payload_sha256"]
        or retry.get("runpod_development_plan_payload_sha256") != plan["payload_sha256"]
        or [pod["id"] for pod in frozen_pods]
        != retry.get("failed_deployment", {}).get("pods_stopped")
        or retry.get("failed_deployment", {}).get("provider_status_after_stop")
        != ["EXITED"] * len(frozen_pods)
        or prior < float(receipt.get("conservative_development_spend_usd", 0.0))
        or prior >= float(plan["budget"]["development_billing_cutoff_usd"])
        or retry.get("payment_authority", {}).get("direct_credit_card_charge_permitted")
        is not False
        or retry.get("sealed_gates", {}).get("pherc1218_v2_opened") is not False
        or retry.get("sealed_gates", {}).get("scientific_endpoints_scored") is not False
    ):
        raise RuntimeError("chunked retry is not bound to the stopped sealed allocation")
    runpod = load_runpod()
    resumed_ids = []
    try:
        for pod_record in frozen_pods:
            pod = runpod.get_pod(pod_record["id"])
            if pod is None or pod.get("desiredStatus") != "EXITED":
                raise RuntimeError(f"chunked retry pod is not stopped: {pod_record['id']}")
            runpod.resume_pod(pod_record["id"], gpu_count=pod_record["gpu_count"])
            resumed_ids.append(pod_record["id"])
        for pod_record in frozen_pods:
            for _ in range(120):
                pod = runpod.get_pod(pod_record["id"])
                if pod and pod.get("desiredStatus") == "RUNNING" and pod.get("runtime"):
                    break
                time.sleep(5)
            else:
                raise RuntimeError(f"chunked retry pod did not resume: {pod_record['id']}")
            validate_pod(
                pod,
                plan,
                require_exact_reused_pod=False,
                expected_gpu_count=pod_record["gpu_count"],
                expected_gpu_display=pod_record["gpu_name"],
                maximum_rate_usd=pod_record["aggregate_hourly_rate_usd"],
            )
            if float(pod["costPerHr"]) != float(pod_record["aggregate_hourly_rate_usd"]):
                raise RuntimeError(f"chunked retry pod rate changed: {pod_record['id']}")
    except Exception:
        for pod_id in resumed_ids:
            try:
                runpod.stop_pod(pod_id)
            except Exception:
                pass
        raise
    receipt.update(
        status="PODS_ALLOCATED_AWAITING_DEPLOYMENT",
        public_runpod_commit=args.public_commit,
        chunked_transfer_retry_payload_sha256=retry["payload_sha256"],
        active_billing_started_at=now().isoformat(),
        prior_conservative_development_spend_usd=prior,
        conservative_development_spend_usd=prior,
        private_bundle_egress_permitted=True,
    )
    atomic_json(args.receipt, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "pod_ids": resumed_ids,
        "aggregate_hourly_rate_usd": receipt["total_hourly_rate_usd"],
        "prior_conservative_development_spend_usd": prior,
        "active_billing_started_at": receipt["active_billing_started_at"],
    }, indent=2, sort_keys=True))


def resume_verification(args, plan: dict) -> None:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    retry = load_plan(args.pycache_verification_retry)
    stale_retry = (
        load_plan(args.stale_rc_retry) if args.stale_rc_retry is not None else None
    )
    if len(args.public_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.public_commit
    ):
        raise RuntimeError("public commit must be an exact lowercase Git SHA")
    frozen_pods = receipt.get("pods") or []
    authority = stale_retry or retry
    prior = float(
        authority.get("billing", {}).get(
            "prior_conservative_development_spend_usd", -1
        )
    )
    failure = (
        stale_retry.get("failed_retry", {})
        if stale_retry is not None
        else retry.get("failed_deployment", {})
    )
    expected_status = (
        "BUNDLE_VERIFICATION_RETRY_ERROR_PODS_STOPPED"
        if stale_retry is not None
        else "JUPYTER_CROC_DEPLOYMENT_ERROR_PODS_STOPPED"
    )
    if (
        receipt.get("status") != expected_status
        or receipt.get("plan_payload_sha256") != plan["payload_sha256"]
        or retry.get("runpod_development_plan_payload_sha256") != plan["payload_sha256"]
        or retry.get("chunked_reallocation_retry_payload_sha256")
        != receipt.get("chunked_reallocation_retry_payload_sha256")
        or [pod["id"] for pod in frozen_pods] != failure.get("pods_stopped")
        or failure.get("provider_status_after_failure")
        != ["EXITED"] * len(frozen_pods)
        or (
            stale_retry is None
            and retry.get("failed_deployment", {}).get("bundle_extracted_on_both_pods")
            is not True
        )
        or retry.get("retry", {}).get("reupload_bundle") is not False
        or (
            stale_retry is not None
            and stale_retry.get("pycache_verification_retry_payload_sha256")
            != retry["payload_sha256"]
        )
        or (
            stale_retry is not None
            and stale_retry.get("retry", {}).get("reupload_bundle") is not False
        )
        or prior >= float(plan["budget"]["development_billing_cutoff_usd"])
        or retry.get("payment_authority", {}).get("direct_credit_card_charge_permitted")
        is not False
        or retry.get("sealed_gates", {}).get("scientific_endpoints_scored") is not False
    ):
        raise RuntimeError("verification retry is not bound to the stopped extracted bundles")
    runpod = load_runpod()
    resumed_ids = []
    try:
        for pod_record in frozen_pods:
            pod = runpod.get_pod(pod_record["id"])
            if pod is None or pod.get("desiredStatus") != "EXITED":
                raise RuntimeError(f"verification retry pod is not stopped: {pod_record['id']}")
            runpod.resume_pod(pod_record["id"], gpu_count=pod_record["gpu_count"])
            resumed_ids.append(pod_record["id"])
        for pod_record in frozen_pods:
            for _ in range(120):
                pod = runpod.get_pod(pod_record["id"])
                if pod and pod.get("desiredStatus") == "RUNNING" and pod.get("runtime"):
                    break
                time.sleep(5)
            else:
                raise RuntimeError(f"verification retry pod did not resume: {pod_record['id']}")
            validate_pod(
                pod,
                plan,
                require_exact_reused_pod=False,
                expected_gpu_count=pod_record["gpu_count"],
                expected_gpu_display=pod_record["gpu_name"],
                maximum_rate_usd=pod_record["aggregate_hourly_rate_usd"],
            )
    except Exception:
        for pod_id in resumed_ids:
            try:
                runpod.stop_pod(pod_id)
            except Exception:
                pass
        raise
    receipt.update(
        status="PODS_EXTRACTED_AWAITING_BLIND_VERIFICATION",
        public_runpod_commit=args.public_commit,
        pycache_verification_retry_payload_sha256=retry["payload_sha256"],
        stale_rc_retry_payload_sha256=(
            stale_retry["payload_sha256"] if stale_retry is not None else None
        ),
        active_billing_started_at=now().isoformat(),
        prior_conservative_development_spend_usd=prior,
        conservative_development_spend_usd=prior,
    )
    atomic_json(args.receipt, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "pod_ids": resumed_ids,
        "aggregate_hourly_rate_usd": receipt["total_hourly_rate_usd"],
        "prior_conservative_development_spend_usd": prior,
        "active_billing_started_at": receipt["active_billing_started_at"],
    }, indent=2, sort_keys=True))


def resume(args, plan: dict) -> None:
    if args.receipt.exists():
        raise RuntimeError("provider receipt already exists; refusing duplicate resume")
    if len(args.public_commit) != 40 or any(c not in "0123456789abcdef" for c in args.public_commit):
        raise RuntimeError("public commit must be an exact lowercase Git SHA")
    runpod = load_runpod()
    pod_id = plan["execution"]["pod_id"]
    pod = runpod.get_pod(pod_id)
    validate_pod(pod, plan, require_exited=True)
    runpod.resume_pod(pod_id, gpu_count=plan["execution"]["gpu_count"])
    resumed = None
    for _ in range(120):
        resumed = runpod.get_pod(pod_id)
        if resumed and resumed.get("desiredStatus") == "RUNNING" and resumed.get("runtime"):
            break
        time.sleep(5)
    else:
        runpod.stop_pod(pod_id)
        raise RuntimeError("exact RunPod pod did not become RUNNING")
    validate_pod(resumed, plan)
    started = now()
    receipt = {
        "schema_version": "1.0",
        "status": "POD_RESUMED_AWAITING_DEPLOYMENT",
        "plan_payload_sha256": plan["payload_sha256"],
        "public_runpod_commit": args.public_commit,
        "billing_started_at": started.isoformat(),
        "development_billing_cutoff_usd": plan["budget"]["development_billing_cutoff_usd"],
        "absolute_campaign_cap_usd": plan["budget"]["absolute_campaign_cap_usd"],
        "confirmation_reserve_usd": plan["budget"]["confirmation_reserve_usd"],
        "pod": {
            "id": resumed["id"],
            "name": resumed["name"],
            "gpu_count": int(resumed["gpuCount"]),
            "gpu_name": resumed["machine"]["gpuDisplayName"],
            "image_name": resumed["imageName"],
            "aggregate_hourly_rate_usd": float(resumed["costPerHr"]),
        },
        "scientific_endpoints_inspected": False,
        "confirmation_outputs_inspected": False,
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def report_or_watch(args, plan: dict, watch: bool) -> None:
    runpod = load_runpod()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if receipt.get("plan_payload_sha256") != plan["payload_sha256"]:
        raise RuntimeError("receipt is bound to another RunPod plan")
    started = datetime.fromisoformat(receipt.get("active_billing_started_at", receipt["billing_started_at"]))
    frozen_pods = receipt.get("pods") or [receipt["pod"]]
    rate = sum(float(item["aggregate_hourly_rate_usd"]) for item in frozen_pods)
    cutoff = float(receipt["development_billing_cutoff_usd"])
    while True:
        provider = []
        any_running = False
        for frozen in frozen_pods:
            pod = runpod.get_pod(frozen["id"])
            desired = "ABSENT" if pod is None else pod.get("desiredStatus")
            any_running = any_running or desired == "RUNNING"
            provider.append({"id": frozen["id"], "desired_status": desired})
        elapsed = max(0.0, (now() - started).total_seconds())
        prior_spend = float(receipt.get("prior_conservative_development_spend_usd", 0.0))
        conservative_spend = prior_spend + rate * elapsed / 3600.0
        guarded = conservative_spend + (rate * args.guard_seconds / 3600.0 if any_running else 0.0)
        stopped = False
        if args.enforce and any_running and guarded >= cutoff:
            for frozen in frozen_pods:
                runpod.stop_pod(frozen["id"])
            stopped = True
            receipt.update(
                status="DEVELOPMENT_BUDGET_CUTOFF_STOPPED",
                budget_stop_at=now().isoformat(),
                conservative_development_spend_usd=round(conservative_spend, 6),
            )
            atomic_json(args.receipt, receipt)
        report = {
            "provider": provider,
            "elapsed_seconds": round(elapsed, 3),
            "aggregate_hourly_rate_usd": rate,
            "conservative_development_spend_usd": round(conservative_spend, 6),
            "guarded_spend_usd": round(guarded, 6),
            "development_billing_cutoff_usd": cutoff,
            "stopped_at_budget_gate": stopped,
        }
        print(json.dumps(report, sort_keys=True), flush=True)
        if stopped or not watch or not any_running:
            return
        time.sleep(args.interval)


def stop_or_terminate(args, terminate: bool) -> None:
    runpod = load_runpod()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    frozen_pods = receipt.get("pods") or [receipt["pod"]]
    for frozen in frozen_pods:
        (runpod.terminate_pod if terminate else runpod.stop_pod)(frozen["id"])
    started = datetime.fromisoformat(receipt.get("active_billing_started_at", receipt["billing_started_at"]))
    rate = sum(float(item["aggregate_hourly_rate_usd"]) for item in frozen_pods)
    spend = float(receipt.get("prior_conservative_development_spend_usd", 0.0)) + rate * max(
        0.0, (now() - started).total_seconds()
    ) / 3600.0
    receipt.update(
        status="TERMINATED" if terminate else "STOPPED",
        provider_action_at=now().isoformat(),
        conservative_development_spend_usd=round(spend, 6),
    )
    atomic_json(args.receipt, receipt)
    print(receipt["status"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("resume", "resume-multi", "resume-substitute", "resume-chunked", "resume-verification", "allocate", "allocate-multi", "allocate-substitute", "reserve-multi", "reserve-substitute", "status", "watch", "stop", "terminate"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--public-commit", default="")
    parser.add_argument("--allocation-retry", type=Path)
    parser.add_argument("--multi-pod-retry", type=Path)
    parser.add_argument("--egress-resume", type=Path)
    parser.add_argument("--replacement-reservation", type=Path)
    parser.add_argument("--hardware-substitution", type=Path)
    parser.add_argument("--substitute-egress-resume", type=Path)
    parser.add_argument("--dynamic-substitute-egress", type=Path)
    parser.add_argument("--ssh-retry", type=Path)
    parser.add_argument("--account-ssh-retry", type=Path)
    parser.add_argument("--explicit-ssh-retry", type=Path)
    parser.add_argument("--ssh-readiness-retry", type=Path)
    parser.add_argument("--jupyter-croc-retry", type=Path)
    parser.add_argument("--jupyter-terminal-retry", type=Path)
    parser.add_argument("--jupyter-pid-retry", type=Path)
    parser.add_argument("--croc-code-retry", type=Path)
    parser.add_argument("--croc-room-retry", type=Path)
    parser.add_argument("--croc-sender-ready-retry", type=Path)
    parser.add_argument("--chunked-transfer-retry", type=Path)
    parser.add_argument("--chunked-reallocation-retry", type=Path)
    parser.add_argument("--pycache-verification-retry", type=Path)
    parser.add_argument("--stale-rc-retry", type=Path)
    parser.add_argument("--ssh-public-key", type=Path)
    parser.add_argument("--deployment-public-key", type=Path)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--guard-seconds", type=int, default=300)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.command == "resume":
        resume(args, plan)
    elif args.command == "resume-multi":
        if args.egress_resume is None:
            raise RuntimeError("resume-multi requires the public egress/resume approval")
        resume_multi(args, plan)
    elif args.command == "resume-substitute":
        if args.substitute_egress_resume is None:
            raise RuntimeError("resume-substitute requires the public substitute egress approval")
        resume_substitute(args, plan)
    elif args.command == "resume-chunked":
        if args.chunked_transfer_retry is None:
            raise RuntimeError("resume-chunked requires the public chunked retry")
        resume_chunked(args, plan)
    elif args.command == "resume-verification":
        if args.pycache_verification_retry is None:
            raise RuntimeError(
                "resume-verification requires the public pycache verification retry"
            )
        resume_verification(args, plan)
    elif args.command == "allocate":
        if args.allocation_retry is None:
            raise RuntimeError("allocate requires the public allocation retry")
        allocate(args, plan)
    elif args.command in {"allocate-multi", "allocate-substitute", "reserve-multi", "reserve-substitute"}:
        if args.multi_pod_retry is None:
            raise RuntimeError("allocate-multi requires the public multi-pod retry")
        if args.command in {"allocate-substitute", "reserve-substitute"} and args.hardware_substitution is None:
            raise RuntimeError("substitute allocation requires the public hardware substitution")
        if args.command == "allocate-substitute" and args.dynamic_substitute_egress is None:
            raise RuntimeError("allocate-substitute requires dynamic egress approval")
        allocate_multi(
            args,
            plan,
            stop_after_reservation=args.command in {"reserve-multi", "reserve-substitute"},
            use_hardware_substitution=args.command in {"allocate-substitute", "reserve-substitute"},
        )
    elif args.command in {"status", "watch"}:
        report_or_watch(args, plan, args.command == "watch")
    else:
        stop_or_terminate(args, args.command == "terminate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
