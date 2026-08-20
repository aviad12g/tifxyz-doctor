#!/usr/bin/env python3
"""Deploy the exact sealed bundle through authenticated Jupyter and runpodctl."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import time
from urllib.parse import urlparse
import uuid

import requests
from websockets.sync.client import connect as websocket_connect


def canonical(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("payload_sha256", None)
    if canonical(body) != expected:
        raise RuntimeError(f"payload SHA-256 mismatch: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_private_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.chmod(0o600)
    temp.replace(path)


def now() -> datetime:
    return datetime.now(timezone.utc)


def stop_all(runpod, pods: list[dict]) -> None:
    for pod in pods:
        try:
            runpod.stop_pod(pod["id"])
        except Exception:
            pass


def budget_deadline(receipt: dict, guard_seconds: int) -> float:
    started = datetime.fromisoformat(
        receipt.get("active_billing_started_at", receipt["billing_started_at"])
    )
    prior = float(receipt.get("prior_conservative_development_spend_usd", 0.0))
    rate = float(receipt["total_hourly_rate_usd"])
    cutoff = float(receipt["development_billing_cutoff_usd"])
    if rate <= 0 or prior >= cutoff:
        raise RuntimeError("provider receipt has no positive development budget")
    seconds = (cutoff - prior) * 3600.0 / rate - guard_seconds
    return started.timestamp() + seconds


def require_budget(deadline: float) -> None:
    if time.time() >= deadline:
        raise RuntimeError("development budget guard reached")


def tagged_pid(output: str, tag: str) -> int:
    matches = re.findall(re.escape(tag) + r":([0-9]+)", output)
    if not matches:
        raise RuntimeError("remote background task did not return a tagged PID")
    return int(matches[-1])


class JupyterTerminal:
    def __init__(self, base_url: str, password: str, session: requests.Session, websocket):
        self.base_url = base_url.rstrip("/")
        self.password = password
        self.session = session
        self.websocket = websocket

    @classmethod
    def open(
        cls,
        base_url: str,
        password: str,
        *,
        maximum_elapsed: int,
        interval: int,
        websocket_path_template: str,
    ) -> "JupyterTerminal":
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or parsed.path not in {"", "/"}:
            raise RuntimeError("invalid Jupyter proxy URL")
        session = requests.Session()
        deadline = time.monotonic() + maximum_elapsed
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            terminal_name: str | None = None
            try:
                login = session.get(base_url.rstrip("/") + "/login", timeout=20)
                login.raise_for_status()
                xsrf = session.cookies.get("_xsrf")
                if not xsrf:
                    match = re.search(r'name=["\']_xsrf["\'] value=["\']([^"\']+)', login.text)
                    xsrf = match.group(1) if match else None
                if not xsrf:
                    raise RuntimeError("Jupyter login did not supply XSRF material")
                authenticated = session.post(
                    base_url.rstrip("/") + "/login?next=%2Fapi%2Fterminals",
                    data={"_xsrf": xsrf, "password": password},
                    headers={"Referer": base_url.rstrip("/") + "/login"},
                    timeout=20,
                    allow_redirects=True,
                )
                authenticated.raise_for_status()
                check = session.get(base_url.rstrip("/") + "/api/status", timeout=20)
                if check.status_code != 200:
                    raise RuntimeError("Jupyter authentication was not accepted")
                xsrf = session.cookies.get("_xsrf") or xsrf
                created = session.post(
                    base_url.rstrip("/") + "/api/terminals",
                    json={},
                    headers={"X-XSRFToken": xsrf, "Referer": base_url.rstrip("/") + "/"},
                    timeout=20,
                )
                created.raise_for_status()
                terminal_name = str(created.json()["name"])
                cookie = "; ".join(
                    f"{item.name}={item.value}" for item in session.cookies
                )
                websocket_url = (
                    "wss://"
                    + parsed.netloc
                    + websocket_path_template.format(terminal_name=terminal_name)
                    + f"?session_id={uuid.uuid4()}"
                )
                websocket = websocket_connect(
                    websocket_url,
                    origin=base_url.rstrip("/"),
                    additional_headers={"Cookie": cookie},
                    open_timeout=20,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=1 << 20,
                )
                terminal = cls(base_url, password, session, websocket)
                terminal.run("unset HISTFILE; set +o history; stty -echo", timeout=20)
                return terminal
            except Exception as error:
                last_error = error
                if terminal_name is not None:
                    try:
                        session.delete(
                            base_url.rstrip("/") + f"/api/terminals/{terminal_name}",
                            headers={"X-XSRFToken": xsrf, "Referer": base_url.rstrip("/") + "/"},
                            timeout=20,
                        )
                    except Exception:
                        pass
                time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
        raise RuntimeError("Jupyter service did not become ready within the frozen bound") from last_error

    def run(self, command: str, *, timeout: int = 60) -> str:
        token = secrets.token_hex(16)
        marker = f"__GAPBALANCE_DONE_{token}__"
        wrapped = (
            f"( {command} ); __gapbalance_rc=$?; "
            f"printf '\\n{marker}:%s\\n' \"$__gapbalance_rc\"\r"
        )
        self.websocket.send(json.dumps(["stdin", wrapped]))
        deadline = time.monotonic() + timeout
        output = []
        total = 0
        pattern = re.compile(re.escape(marker) + r":([0-9]+)")
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            raw = self.websocket.recv(timeout=remaining)
            message = json.loads(raw)
            if not isinstance(message, list) or len(message) != 2:
                continue
            if message[0] not in {"stdout", "disconnect"}:
                continue
            text = str(message[1])
            total += len(text)
            if total > (1 << 20):
                raise RuntimeError("Jupyter command output exceeded the operational bound")
            output.append(text)
            joined = "".join(output)
            match = pattern.search(joined)
            if match:
                rc = int(match.group(1))
                before = joined[: match.start()].strip()
                if rc != 0:
                    raise RuntimeError("remote Jupyter command failed")
                return before
        raise RuntimeError("remote Jupyter command timed out")

    def start_background(self, body: str, *, log: str, rc_file: str) -> int:
        tag = "__GAPBALANCE_BACKGROUND_PID__"
        command = (
            f"nohup sh -lc {shlex.quote(body + '; __rc=$?; printf \'%s\\n\' \"$__rc\" > ' + shlex.quote(rc_file) + '; exit $__rc')} "
            f"> {shlex.quote(log)} 2>&1 < /dev/null & "
            f"printf '\\n{tag}:%s\\n' \"$!\""
        )
        output = self.run(command, timeout=30)
        return tagged_pid(output, tag)

    def wait_rc(self, rc_file: str, *, deadline: float, interval: int) -> None:
        while True:
            require_budget(deadline)
            output = self.run(
                f"if test -f {shlex.quote(rc_file)}; then cat {shlex.quote(rc_file)}; "
                "else printf 'WAIT\\n'; fi",
                timeout=30,
            ).strip()
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            if lines and lines[-1] == "0":
                return
            if lines and lines[-1] != "WAIT":
                raise RuntimeError("remote background task failed")
            time.sleep(interval)

    def close(self) -> None:
        try:
            self.websocket.close()
        except Exception:
            pass
        self.session.close()


def transfer_one(
    *,
    terminal: JupyterTerminal,
    pod: dict,
    pod_index: int,
    receipt: dict,
    archive: Path,
    archive_sha256: str,
    archive_bytes: int,
    runpodctl: Path,
    deadline: float,
    poll_interval: int,
) -> dict:
    require_budget(deadline)
    remote_root = (
        f"/workspace/gapbalance-development-{receipt['public_runpod_commit'][:8]}-p{pod_index}"
    )
    terminal.run(
        f"test ! -e {shlex.quote(remote_root)} && "
        f"mkdir -m 700 -p {shlex.quote(remote_root + '/status')}",
        timeout=30,
    )
    code = secrets.token_hex(24)
    receiver_log = remote_root + "/status/transfer.operational.log"
    receiver_rc = remote_root + "/status/transfer.rc"
    receiver_body = (
        f"cd {shlex.quote(remote_root)} && "
        f"runpodctl receive {shlex.quote(code)}"
    )
    receiver_pid = terminal.start_background(
        receiver_body, log=receiver_log, rc_file=receiver_rc
    )
    sender = subprocess.Popen(
        [str(runpodctl), "send", str(archive), "--code", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        while sender.poll() is None:
            require_budget(deadline)
            time.sleep(poll_interval)
        if sender.returncode != 0:
            raise RuntimeError("encrypted local bundle sender failed")
        terminal.wait_rc(receiver_rc, deadline=deadline, interval=poll_interval)
        terminal.run(f"rm -f {shlex.quote(receiver_log)}", timeout=30)
    except Exception:
        if sender.poll() is None:
            sender.terminate()
            try:
                sender.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sender.kill()
                sender.wait()
        raise
    remote_archive = remote_root + "/" + archive.name
    extract_log = remote_root + "/status/archive-extract.operational.log"
    extract_rc = remote_root + "/status/archive-extract.rc"
    extract_body = " && ".join(
        [
            f"test -f {shlex.quote(remote_archive)}",
            f"test \"$(stat -c %s {shlex.quote(remote_archive)})\" = {archive_bytes}",
            f"printf '%s  %s\\n' {shlex.quote(archive_sha256)} {shlex.quote(remote_archive)} | sha256sum -c -",
            f"test ! -e {shlex.quote(remote_root + '/bundle')}",
            f"mkdir {shlex.quote(remote_root + '/bundle.partial')}",
            f"tar -xf {shlex.quote(remote_archive)} -C {shlex.quote(remote_root + '/bundle.partial')}",
            f"mv {shlex.quote(remote_root + '/bundle.partial')} {shlex.quote(remote_root + '/bundle')}",
            f"rm -f {shlex.quote(remote_archive)}",
        ]
    )
    terminal.start_background(extract_body, log=extract_log, rc_file=extract_rc)
    terminal.wait_rc(extract_rc, deadline=deadline, interval=poll_interval)
    return {
        "pod_id": pod["id"],
        "jobs": pod["jobs"],
        "remote_root": remote_root,
        "receiver_pid": receiver_pid,
        "archive_sha256_verified_before_extraction": True,
    }


def verify_remote_bundle(
    terminal: JupyterTerminal,
    deployment: dict,
    manifest_payload_sha256: str,
    *,
    deadline: float,
    poll_interval: int,
) -> None:
    root = deployment["remote_root"]
    log = root + "/status/bundle-verification.operational.log"
    rc_file = root + "/status/bundle-verification.rc"
    verifier = root + "/bundle/controller/verify_gapbalance_runpod_development_bundle.py"
    plan = root + "/bundle/controller/GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json"
    manifest = root + "/bundle/bundle_manifest.json"
    command = shlex.join(
        [
            "python",
            verifier,
            "--plan",
            plan,
            "--bundle-root",
            root + "/bundle",
            "--bundle-manifest",
            manifest,
            "--expected-manifest-payload-sha256",
            manifest_payload_sha256,
        ]
    )
    terminal.start_background(command, log=log, rc_file=rc_file)
    terminal.wait_rc(rc_file, deadline=deadline, interval=poll_interval)


def start_executor(
    terminal: JupyterTerminal,
    deployment: dict,
    pod: dict,
    *,
    expected_gpu_name: str,
) -> int:
    root = deployment["remote_root"]
    command = [
        "python",
        root + "/bundle/controller/execute_gapbalance_runpod_development.py",
        "--plan",
        root + "/bundle/controller/GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json",
        "--bundle-root",
        root + "/bundle",
        "--bundle-manifest",
        root + "/bundle/bundle_manifest.json",
        "--input",
        root + "/bundle/input",
        "--launchers",
        root + "/bundle/launchers",
        "--wrapper",
        root + "/bundle/controller/run_gapbalance_runpod_development_job.py",
        "--working",
        root + "/output",
        "--temp",
        root + "/temp",
        "--status",
        root + "/status/status.json",
        "--allocation-layout",
        root + "/bundle/controller/GAPBALANCE_RUNPOD_MULTI_POD_RETRY.json",
        "--hardware-substitution",
        root + "/bundle/controller/GAPBALANCE_RUNPOD_HARDWARE_SUBSTITUTION.json",
        "--expected-gpu-name",
        expected_gpu_name,
    ]
    for job_id in pod["jobs"]:
        command.extend(["--job-id", job_id])
    shell = (
        "nohup "
        + shlex.join(command)
        + f" > {shlex.quote(root + '/status/controller.operational.log')} 2>&1 < /dev/null & "
        + "printf '\\n__GAPBALANCE_EXECUTOR_PID__:%s\\n' \"$!\""
    )
    output = terminal.run(shell, timeout=30)
    return tagged_pid(output, "__GAPBALANCE_EXECUTOR_PID__")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--provider-receipt", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-archive", type=Path, required=True)
    parser.add_argument("--jupyter-croc-retry", type=Path, required=True)
    parser.add_argument("--jupyter-terminal-retry", type=Path, required=True)
    parser.add_argument("--jupyter-pid-retry", type=Path, required=True)
    parser.add_argument("--bundle-egress", type=Path, required=True)
    parser.add_argument("--runpodctl", type=Path, required=True)
    args = parser.parse_args()

    plan = load_hashed(args.plan)
    manifest = load_hashed(args.bundle / "bundle_manifest.json")
    retry = load_hashed(args.jupyter_croc_retry)
    terminal_retry = load_hashed(args.jupyter_terminal_retry)
    pid_retry = load_hashed(args.jupyter_pid_retry)
    egress = load_hashed(args.bundle_egress)
    receipt = json.loads(args.provider_receipt.read_text(encoding="utf-8"))
    transfer = retry["runpodctl_transfer"]
    bundle_record = egress["bundle"]
    if (
        receipt.get("status") != "PODS_ALLOCATED_AWAITING_DEPLOYMENT"
        or receipt.get("plan_payload_sha256") != plan["payload_sha256"]
        or receipt.get("jupyter_croc_retry_payload_sha256") != retry["payload_sha256"]
        or receipt.get("jupyter_terminal_retry_payload_sha256")
        != terminal_retry["payload_sha256"]
        or receipt.get("jupyter_pid_retry_payload_sha256") != pid_retry["payload_sha256"]
        or receipt.get("private_bundle_egress_permitted") is not True
        or receipt.get("jupyter_credentials_private") is not True
        or retry.get("runpod_development_plan_payload_sha256") != plan["payload_sha256"]
        or terminal_retry.get("runpod_development_plan_payload_sha256")
        != plan["payload_sha256"]
        or terminal_retry.get("jupyter_croc_retry_payload_sha256")
        != retry["payload_sha256"]
        or pid_retry.get("runpod_development_plan_payload_sha256")
        != plan["payload_sha256"]
        or pid_retry.get("jupyter_terminal_retry_payload_sha256")
        != terminal_retry["payload_sha256"]
        or pid_retry.get("terminal_control", {}).get("background_pid_output_format")
        != "tagged"
        or terminal_retry.get("terminal_control", {}).get("websocket_path_template")
        != "/terminals/websocket/{terminal_name}"
        or egress.get("runpod_development_plan_payload_sha256") != plan["payload_sha256"]
        or egress.get("jupyter_croc_retry_payload_sha256") != retry["payload_sha256"]
        or manifest.get("plan_payload_sha256") != plan["payload_sha256"]
        or manifest.get("payload_sha256") != bundle_record.get("manifest_payload_sha256")
        or manifest.get("scientific_outputs_present") is not False
        or manifest.get("pherc1218_present") is not False
        or manifest.get("confirmation_seeds_500_504_present") is not False
        or egress.get("egress_exclusions", {}).get("pherc1218_included") is not False
        or egress.get("egress_exclusions", {}).get("confirmation_seeds_500_504_included")
        is not False
    ):
        raise RuntimeError("deployment inputs violate a frozen receipt, identity, or sealed gate")
    if (
        not args.bundle_archive.is_file()
        or args.bundle_archive.stat().st_size != int(bundle_record["archive_bytes"])
        or sha256_file(args.bundle_archive) != bundle_record["archive_sha256"]
        or args.runpodctl.stat().st_size <= 0
        or sha256_file(args.runpodctl) != transfer["darwin_arm64_sha256"]
    ):
        raise RuntimeError("local archive or runpodctl identity mismatch")
    version = subprocess.run(
        [str(args.runpodctl), "version"], capture_output=True, text=True, timeout=15
    )
    if version.returncode != 0 or transfer["version"] not in version.stdout + version.stderr:
        raise RuntimeError("local runpodctl version mismatch")
    pods = receipt["pods"]
    access = receipt.get("jupyter_access") or {}
    if set(access) != {pod["id"] for pod in pods}:
        raise RuntimeError("private Jupyter credential set does not match the allocated pods")
    if [job for pod in pods for job in pod["jobs"]] != [job["job_id"] for job in plan["jobs"]]:
        raise RuntimeError("provider partitions do not reconstruct the frozen job order")

    import runpod

    for pod in pods:
        provider = runpod.get_pod(pod["id"])
        if (
            provider is None
            or provider.get("desiredStatus") != "RUNNING"
            or int(provider.get("gpuCount") or -1) != int(pod["gpu_count"])
            or pod["gpu_name"] not in (provider.get("machine") or {}).get("gpuDisplayName", "")
            or float(provider.get("costPerHr") or 0) != float(pod["aggregate_hourly_rate_usd"])
        ):
            raise RuntimeError("provider allocation changed before bundle egress")
    guard_seconds = int(retry["billing"]["budget_guard_seconds"])
    deadline = budget_deadline(receipt, guard_seconds)
    poll_interval = int(retry["runpodctl_transfer"]["poll_interval_seconds"])
    terminals: dict[str, JupyterTerminal] = {}
    deployments: list[dict] = []
    try:
        for pod in pods:
            require_budget(deadline)
            record = access[pod["id"]]
            expected_url = f"https://{pod['id']}-8888.proxy.runpod.net"
            if record.get("base_url") != expected_url or not record.get("password"):
                raise RuntimeError("private Jupyter access record mismatch")
            terminals[pod["id"]] = JupyterTerminal.open(
                expected_url,
                record["password"],
                maximum_elapsed=int(retry["jupyter_control"]["maximum_readiness_seconds_per_pod"]),
                interval=int(retry["jupyter_control"]["readiness_poll_interval_seconds"]),
                websocket_path_template=terminal_retry["terminal_control"][
                    "websocket_path_template"
                ],
            )
            terminals[pod["id"]].run(
                "command -v runpodctl >/dev/null && command -v python >/dev/null && command -v tar >/dev/null",
                timeout=30,
            )
        with ThreadPoolExecutor(max_workers=len(pods)) as pool:
            futures = {
                pool.submit(
                    transfer_one,
                    terminal=terminals[pod["id"]],
                    pod=pod,
                    pod_index=index,
                    receipt=receipt,
                    archive=args.bundle_archive,
                    archive_sha256=bundle_record["archive_sha256"],
                    archive_bytes=int(bundle_record["archive_bytes"]),
                    runpodctl=args.runpodctl,
                    deadline=deadline,
                    poll_interval=poll_interval,
                ): index
                for index, pod in enumerate(pods)
            }
            indexed = {}
            for future in as_completed(futures):
                indexed[futures[future]] = future.result()
            deployments = [indexed[index] for index in range(len(pods))]
        for deployment in deployments:
            verify_remote_bundle(
                terminals[deployment["pod_id"]],
                deployment,
                manifest["payload_sha256"],
                deadline=deadline,
                poll_interval=poll_interval,
            )
            deployment["bundle_manifest_verified_before_executor"] = True
        expected_gpu_name = receipt["selected_hardware"]["gpu_display_name"]
        for deployment, pod in zip(deployments, pods, strict=True):
            require_budget(deadline)
            deployment["remote_executor_pid"] = start_executor(
                terminals[pod["id"]],
                deployment,
                pod,
                expected_gpu_name=expected_gpu_name,
            )
            deployment["jupyter_base_url"] = access[pod["id"]]["base_url"]
    except Exception:
        stop_all(runpod, pods)
        receipt.update(
            status="JUPYTER_CROC_DEPLOYMENT_ERROR_PODS_STOPPED",
            deployment_error_at=now().isoformat(),
            scientific_endpoints_inspected=False,
            confirmation_outputs_inspected=False,
        )
        atomic_private_json(args.provider_receipt, receipt)
        raise
    finally:
        for terminal in terminals.values():
            terminal.close()

    receipt["deployment"] = {
        "deployed_at": now().isoformat(),
        "transport": "authenticated_jupyter_https_wss_plus_encrypted_runpodctl_croc",
        "jupyter_croc_retry_payload_sha256": retry["payload_sha256"],
        "bundle_egress_payload_sha256": egress["payload_sha256"],
        "bundle_manifest_payload_sha256": manifest["payload_sha256"],
        "bundle_archive_sha256": bundle_record["archive_sha256"],
        "bundle_manifest_verified_before_executor": True,
        "ephemeral_transfer_codes_persisted": False,
        "pods": deployments,
    }
    receipt["status"] = "RUNPOD_DEVELOPMENT_EXECUTOR_RUNNING_NOT_SCORED"
    atomic_private_json(args.provider_receipt, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "pod_ids": [pod["id"] for pod in pods],
                "bundle_manifest_payload_sha256": manifest["payload_sha256"],
                "scientific_endpoints_scored": False,
                "confirmation_outputs_inspected": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
