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
import select
import shlex
import subprocess
import tempfile
import threading
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


def validated_full_croc_code(base_code: str, emitted_line: str) -> str:
    if not re.fullmatch(re.escape(base_code) + r"-[0-9]+", emitted_line):
        raise RuntimeError("runpodctl sender did not emit the expected relay-qualified code")
    return emitted_line


def start_local_sender(
    runpodctl: Path,
    archive: Path,
    base_code: str,
    *,
    ready_timeout: int,
    post_banner_delay: int,
) -> tuple[subprocess.Popen, str]:
    sender = subprocess.Popen(
        [str(runpodctl), "send", str(archive), "--code", base_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert sender.stdout is not None
    assert sender.stderr is not None
    try:
        deadline = time.monotonic() + ready_timeout
        buffer = b""
        while b"\n" not in buffer:
            if sender.poll() is not None:
                raise RuntimeError("runpodctl sender exited before emitting a code")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("runpodctl sender did not emit a code within the bound")
            ready, _, _ = select.select([sender.stdout], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(sender.stdout.fileno(), 4096)
            if not chunk:
                raise RuntimeError("runpodctl sender closed stdout before emitting a code")
            buffer += chunk
        first_line = buffer.split(b"\n", 1)[0].decode("utf-8", errors="strict").strip()
        full_code = validated_full_croc_code(base_code, first_line)
        stderr_buffer = b""
        while b"code is:" not in stderr_buffer:
            if sender.poll() is not None:
                raise RuntimeError("runpodctl sender exited before its readiness banner")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("runpodctl sender readiness banner exceeded the bound")
            ready, _, _ = select.select([sender.stderr], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(sender.stderr.fileno(), 1 << 16)
            if not chunk:
                raise RuntimeError("runpodctl sender closed stderr before its readiness banner")
            stderr_buffer = (stderr_buffer + chunk)[-(1 << 20) :]
        os.set_blocking(sender.stdout.fileno(), False)
        os.set_blocking(sender.stderr.fileno(), False)
        delay_deadline = time.monotonic() + post_banner_delay
        while time.monotonic() < delay_deadline:
            if sender.poll() is not None:
                raise RuntimeError("runpodctl sender exited during the post-banner delay")
            drain_sender_output(sender)
            time.sleep(min(0.2, delay_deadline - time.monotonic()))
        return sender, full_code
    except Exception:
        sender.terminate()
        try:
            sender.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sender.kill()
            sender.wait()
        raise


def drain_sender_output(sender: subprocess.Popen) -> None:
    for stream in (sender.stdout, sender.stderr):
        if stream is None:
            continue
        while True:
            try:
                if not os.read(stream.fileno(), 1 << 16):
                    break
            except BlockingIOError:
                break


def archive_chunk_spans(archive_bytes: int, chunk_bytes: int) -> list[tuple[int, int, int]]:
    if archive_bytes <= 0 or chunk_bytes <= 0:
        raise RuntimeError("archive and chunk sizes must be positive")
    return [
        (index, offset, min(chunk_bytes, archive_bytes - offset))
        for index, offset in enumerate(range(0, archive_bytes, chunk_bytes))
    ]


def materialize_archive_chunk(
    archive: Path, target: Path, *, offset: int, length: int
) -> str:
    if offset < 0 or length <= 0 or offset + length > archive.stat().st_size:
        raise RuntimeError("invalid archive chunk span")
    digest = hashlib.sha256()
    remaining = length
    with archive.open("rb") as source, target.open("xb") as destination:
        source.seek(offset)
        while remaining:
            block = source.read(min(8 << 20, remaining))
            if not block:
                raise RuntimeError("archive ended during chunk materialization")
            destination.write(block)
            digest.update(block)
            remaining -= len(block)
    if target.stat().st_size != length:
        raise RuntimeError("materialized archive chunk size mismatch")
    return digest.hexdigest()


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
            rc = self.read_rc(rc_file)
            if rc == 0:
                return
            if rc is not None:
                raise RuntimeError("remote background task failed")
            time.sleep(interval)

    def read_rc(self, rc_file: str) -> int | None:
        tag = "__GAPBALANCE_REMOTE_RC__"
        output = self.run(
            f"if test -f {shlex.quote(rc_file)}; then "
            f"printf '{tag}:%s\\n' \"$(cat {shlex.quote(rc_file)})\"; "
            f"else printf '{tag}:WAIT\\n'; fi",
            timeout=30,
        )
        matches = re.findall(re.escape(tag) + r":(WAIT|[0-9]+)", output)
        if not matches:
            raise RuntimeError("remote background status was not tagged")
        return None if matches[-1] == "WAIT" else int(matches[-1])

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
    sender_ready_timeout: int,
    post_banner_delay: int,
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
    base_code = secrets.token_hex(24)
    receiver_log = remote_root + "/status/transfer.operational.log"
    receiver_rc = remote_root + "/status/transfer.rc"
    sender, full_code = start_local_sender(
        runpodctl,
        archive,
        base_code,
        ready_timeout=sender_ready_timeout,
        post_banner_delay=post_banner_delay,
    )
    try:
        receiver_body = (
            f"cd {shlex.quote(remote_root)} && "
            "__gb_deadline=$(($(date +%s) + 180)); __gb_rc=1; "
            "while test \"$(date +%s)\" -lt \"$__gb_deadline\"; do "
            f"if runpodctl receive {shlex.quote(full_code)}; then __gb_rc=0; break; fi; "
            "sleep 2; done; test \"$__gb_rc\" -eq 0"
        )
        receiver_pid = terminal.start_background(
            receiver_body, log=receiver_log, rc_file=receiver_rc
        )
        while sender.poll() is None:
            drain_sender_output(sender)
            remote_rc = terminal.read_rc(receiver_rc)
            if remote_rc is not None and remote_rc != 0:
                raise RuntimeError("encrypted remote bundle receiver failed")
            require_budget(deadline)
            time.sleep(poll_interval)
        drain_sender_output(sender)
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
    finally:
        if sender.stdout is not None:
            sender.stdout.close()
        if sender.stderr is not None:
            sender.stderr.close()
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


def transfer_one_chunked(
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
    sender_ready_timeout: int,
    post_banner_delay: int,
    chunk_bytes: int,
    maximum_attempts: int,
    failed_parent_commit: str,
    cancel_event: threading.Event,
) -> dict:
    spans = archive_chunk_spans(archive_bytes, chunk_bytes)
    remote_root = (
        f"/workspace/gapbalance-development-{receipt['public_runpod_commit'][:8]}-p{pod_index}"
    )
    failed_root = f"/workspace/gapbalance-development-{failed_parent_commit[:8]}-p{pod_index}"
    terminal.run(
        f"rm -f {shlex.quote(failed_root + '/' + archive.name)} && "
        f"test ! -e {shlex.quote(remote_root)} && "
        f"mkdir -m 700 -p {shlex.quote(remote_root + '/status')} "
        f"{shlex.quote(remote_root + '/chunks')}",
        timeout=30,
    )
    chunk_names = [
        f"{archive.name}.part-{index:05d}-of-{len(spans):05d}"
        for index, _, _ in spans
    ]
    with tempfile.TemporaryDirectory(prefix=f"gapbalance-{pod['id']}-") as temp_dir:
        temp_root = Path(temp_dir)
        for (index, offset, length), chunk_name in zip(spans, chunk_names, strict=True):
            require_budget(deadline)
            if cancel_event.is_set():
                raise RuntimeError("peer chunk transfer failed")
            local_chunk = temp_root / chunk_name
            chunk_sha256 = materialize_archive_chunk(
                archive, local_chunk, offset=offset, length=length
            )
            succeeded = False
            try:
                for attempt in range(1, maximum_attempts + 1):
                    require_budget(deadline)
                    if cancel_event.is_set():
                        raise RuntimeError("peer chunk transfer failed")
                    base_code = secrets.token_hex(24)
                    receiver_log = (
                        f"{remote_root}/status/chunk-{index:05d}-attempt-{attempt}.operational.log"
                    )
                    receiver_rc = f"{remote_root}/status/chunk-{index:05d}-attempt-{attempt}.rc"
                    sender = None
                    receiver_pid = None
                    try:
                        sender, full_code = start_local_sender(
                            runpodctl,
                            local_chunk,
                            base_code,
                            ready_timeout=sender_ready_timeout,
                            post_banner_delay=post_banner_delay,
                        )
                        remote_chunk = remote_root + "/chunks/" + chunk_name
                        terminal.run(f"rm -f {shlex.quote(remote_chunk)}", timeout=30)
                        receiver_body = (
                            f"cd {shlex.quote(remote_root + '/chunks')} && "
                            "__gb_deadline=$(($(date +%s) + 180)); __gb_rc=1; "
                            "while test \"$(date +%s)\" -lt \"$__gb_deadline\"; do "
                            f"if runpodctl receive {shlex.quote(full_code)}; then __gb_rc=0; break; fi; "
                            "sleep 2; done; test \"$__gb_rc\" -eq 0"
                        )
                        receiver_pid = terminal.start_background(
                            receiver_body, log=receiver_log, rc_file=receiver_rc
                        )
                        while sender.poll() is None:
                            drain_sender_output(sender)
                            remote_rc = terminal.read_rc(receiver_rc)
                            if remote_rc is not None and remote_rc != 0:
                                raise RuntimeError("encrypted remote chunk receiver failed")
                            if cancel_event.is_set():
                                raise RuntimeError("peer chunk transfer failed")
                            require_budget(deadline)
                            time.sleep(poll_interval)
                        drain_sender_output(sender)
                        if sender.returncode != 0:
                            raise RuntimeError("encrypted local chunk sender failed")
                        terminal.wait_rc(
                            receiver_rc, deadline=deadline, interval=poll_interval
                        )
                        terminal.run(
                            f"test \"$(stat -c %s {shlex.quote(remote_chunk)})\" = {length} && "
                            f"printf '%s  %s\\n' {shlex.quote(chunk_sha256)} "
                            f"{shlex.quote(remote_chunk)} | sha256sum -c - >/dev/null",
                            timeout=60,
                        )
                        succeeded = True
                        break
                    except Exception:
                        if receiver_pid is not None:
                            try:
                                terminal.run(
                                    f"kill {receiver_pid} 2>/dev/null || true; "
                                    f"pkill -P {receiver_pid} 2>/dev/null || true; "
                                    f"rm -f {shlex.quote(remote_root + '/chunks/' + chunk_name)}",
                                    timeout=30,
                                )
                            except Exception:
                                pass
                        if attempt == maximum_attempts:
                            cancel_event.set()
                            raise
                        time.sleep(poll_interval)
                    finally:
                        if sender is not None and sender.poll() is None:
                            sender.terminate()
                            try:
                                sender.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                sender.kill()
                                sender.wait()
                        if sender is not None and sender.stdout is not None:
                            sender.stdout.close()
                        if sender is not None and sender.stderr is not None:
                            sender.stderr.close()
            finally:
                local_chunk.unlink(missing_ok=True)
            if not succeeded:
                cancel_event.set()
                raise RuntimeError("bounded chunk transfer attempts were exhausted")
    remote_archive = remote_root + "/" + archive.name
    extract_log = remote_root + "/status/archive-reassembly-extract.operational.log"
    extract_rc = remote_root + "/status/archive-reassembly-extract.rc"
    ordered_chunks = " ".join(
        shlex.quote(remote_root + "/chunks/" + name) for name in chunk_names
    )
    extract_body = " && ".join(
        [
            f"cat {ordered_chunks} > {shlex.quote(remote_archive + '.partial')}",
            f"mv {shlex.quote(remote_archive + '.partial')} {shlex.quote(remote_archive)}",
            f"test \"$(stat -c %s {shlex.quote(remote_archive)})\" = {archive_bytes}",
            f"printf '%s  %s\\n' {shlex.quote(archive_sha256)} {shlex.quote(remote_archive)} | sha256sum -c -",
            f"rm -f {ordered_chunks}",
            f"test ! -e {shlex.quote(remote_root + '/bundle')}",
            f"mkdir {shlex.quote(remote_root + '/bundle.partial')}",
            f"tar -xf {shlex.quote(remote_archive)} -C {shlex.quote(remote_root + '/bundle.partial')}",
            f"mv {shlex.quote(remote_root + '/bundle.partial')} {shlex.quote(remote_root + '/bundle')}",
            f"rm -f {shlex.quote(remote_archive)}",
        ]
    )
    receiver_pid = terminal.start_background(
        extract_body, log=extract_log, rc_file=extract_rc
    )
    terminal.wait_rc(extract_rc, deadline=deadline, interval=poll_interval)
    return {
        "pod_id": pod["id"],
        "jobs": pod["jobs"],
        "remote_root": remote_root,
        "receiver_pid": receiver_pid,
        "chunk_count": len(spans),
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
    terminal.run(
        f"rm -rf {shlex.quote(root + '/bundle/controller/__pycache__')}",
        timeout=30,
    )
    command = shlex.join(
        [
            "python",
            "-B",
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
    parser.add_argument("--croc-code-retry", type=Path, required=True)
    parser.add_argument("--croc-room-retry", type=Path, required=True)
    parser.add_argument("--croc-sender-ready-retry", type=Path, required=True)
    parser.add_argument("--chunked-transfer-retry", type=Path)
    parser.add_argument("--chunked-reallocation-retry", type=Path)
    parser.add_argument("--pycache-verification-retry", type=Path)
    parser.add_argument("--bundle-egress", type=Path, required=True)
    parser.add_argument("--runpodctl", type=Path, required=True)
    args = parser.parse_args()

    plan = load_hashed(args.plan)
    manifest = load_hashed(args.bundle / "bundle_manifest.json")
    retry = load_hashed(args.jupyter_croc_retry)
    terminal_retry = load_hashed(args.jupyter_terminal_retry)
    pid_retry = load_hashed(args.jupyter_pid_retry)
    code_retry = load_hashed(args.croc_code_retry)
    room_retry = load_hashed(args.croc_room_retry)
    sender_retry = load_hashed(args.croc_sender_ready_retry)
    chunked_retry = (
        load_hashed(args.chunked_transfer_retry)
        if args.chunked_transfer_retry is not None
        else None
    )
    reallocation_retry = (
        load_hashed(args.chunked_reallocation_retry)
        if args.chunked_reallocation_retry is not None
        else None
    )
    verification_retry = (
        load_hashed(args.pycache_verification_retry)
        if args.pycache_verification_retry is not None
        else None
    )
    egress = load_hashed(args.bundle_egress)
    receipt = json.loads(args.provider_receipt.read_text(encoding="utf-8"))
    transfer = retry["runpodctl_transfer"]
    bundle_record = egress["bundle"]
    expected_receipt_status = (
        "PODS_EXTRACTED_AWAITING_BLIND_VERIFICATION"
        if verification_retry is not None
        else "PODS_ALLOCATED_AWAITING_DEPLOYMENT"
    )
    if (
        receipt.get("status") != expected_receipt_status
        or receipt.get("plan_payload_sha256") != plan["payload_sha256"]
        or receipt.get("jupyter_croc_retry_payload_sha256") != retry["payload_sha256"]
        or receipt.get("jupyter_terminal_retry_payload_sha256")
        != terminal_retry["payload_sha256"]
        or receipt.get("jupyter_pid_retry_payload_sha256") != pid_retry["payload_sha256"]
        or receipt.get("croc_code_retry_payload_sha256") != code_retry["payload_sha256"]
        or receipt.get("croc_room_retry_payload_sha256") != room_retry["payload_sha256"]
        or receipt.get("croc_sender_ready_retry_payload_sha256")
        != sender_retry["payload_sha256"]
        or (
            chunked_retry is not None
            and receipt.get("chunked_transfer_retry_payload_sha256")
            != chunked_retry["payload_sha256"]
        )
        or (
            reallocation_retry is not None
            and receipt.get("chunked_reallocation_retry_payload_sha256")
            != reallocation_retry["payload_sha256"]
        )
        or (
            verification_retry is not None
            and receipt.get("pycache_verification_retry_payload_sha256")
            != verification_retry["payload_sha256"]
        )
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
        or code_retry.get("runpod_development_plan_payload_sha256")
        != plan["payload_sha256"]
        or code_retry.get("jupyter_pid_retry_payload_sha256") != pid_retry["payload_sha256"]
        or room_retry.get("runpod_development_plan_payload_sha256")
        != plan["payload_sha256"]
        or room_retry.get("croc_code_retry_payload_sha256") != code_retry["payload_sha256"]
        or sender_retry.get("runpod_development_plan_payload_sha256")
        != plan["payload_sha256"]
        or sender_retry.get("croc_room_retry_payload_sha256")
        != room_retry["payload_sha256"]
        or (
            chunked_retry is not None
            and chunked_retry.get("croc_sender_ready_retry_payload_sha256")
            != sender_retry["payload_sha256"]
        )
        or (
            chunked_retry is not None
            and chunked_retry.get("runpod_development_plan_payload_sha256")
            != plan["payload_sha256"]
        )
        or (
            reallocation_retry is not None
            and chunked_retry is None
        )
        or (
            reallocation_retry is not None
            and reallocation_retry.get("chunked_transfer_retry_payload_sha256")
            != chunked_retry["payload_sha256"]
        )
        or (
            reallocation_retry is not None
            and reallocation_retry.get("runpod_development_plan_payload_sha256")
            != plan["payload_sha256"]
        )
        or (
            verification_retry is not None
            and reallocation_retry is None
        )
        or (
            verification_retry is not None
            and verification_retry.get("chunked_reallocation_retry_payload_sha256")
            != reallocation_retry["payload_sha256"]
        )
        or (
            verification_retry is not None
            and verification_retry.get("runpod_development_plan_payload_sha256")
            != plan["payload_sha256"]
        )
        or (
            verification_retry is not None
            and verification_retry.get("retry", {}).get("reupload_bundle") is not False
        )
        or (
            verification_retry is not None
            and verification_retry.get("retry", {}).get(
                "delete_only_generated_controller_pycache_before_verification"
            )
            is not True
        )
        or (
            verification_retry is not None
            and verification_retry.get("retry", {}).get(
                "execute_verifier_with_python_dash_B"
            )
            is not True
        )
        or (
            verification_retry is not None
            and verification_retry.get("failed_deployment", {}).get("pods_stopped")
            != [pod["id"] for pod in receipt.get("pods", [])]
        )
        or (
            verification_retry is not None
            and verification_retry.get("failed_deployment", {}).get("bundle_commit")
            != verification_retry.get("public_parent_commit")
        )
        or (
            verification_retry is not None
            and verification_retry.get("sealed_gates", {}).get(
                "scientific_endpoints_scored"
            )
            is not False
        )
        or (
            chunked_retry is not None
            and int(chunked_retry.get("retry_transport", {}).get("archive_bytes", -1))
            != int(bundle_record["archive_bytes"])
        )
        or (
            chunked_retry is not None
            and chunked_retry.get("retry_transport", {}).get("archive_sha256")
            != bundle_record["archive_sha256"]
        )
        or (
            chunked_retry is not None
            and chunked_retry.get("retry_transport", {}).get(
                "reassembled_archive_sha256_verified_before_extraction"
            )
            is not True
        )
        or (
            chunked_retry is not None
            and len(
                archive_chunk_spans(
                    int(bundle_record["archive_bytes"]),
                    int(chunked_retry["retry_transport"]["chunk_bytes"]),
                )
            )
            != int(chunked_retry["retry_transport"]["chunk_count"])
        )
        or (
            chunked_retry is not None
            and chunked_retry.get("sealed_gates", {}).get("pherc1218_v2_opened")
            is not False
        )
        or (
            chunked_retry is not None
            and chunked_retry.get("sealed_gates", {}).get("scientific_endpoints_scored")
            is not False
        )
        or sender_retry.get("sender_readiness", {}).get("require_post_hash_banner") is not True
        or int(sender_retry.get("sender_readiness", {}).get("post_banner_delay_seconds", 0))
        != 5
        or room_retry.get("receiver_retry", {}).get("retry_room_not_ready") is not True
        or room_retry.get("receiver_retry", {}).get("poll_rc_during_sender") is not True
        or code_retry.get("runpodctl_code_contract", {}).get("sender_starts_before_receiver")
        is not True
        or int(code_retry.get("runpodctl_code_contract", {}).get("base_secret_entropy_bytes", 0))
        < 24
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
        if verification_retry is not None:
            bundle_commit = verification_retry["failed_deployment"]["bundle_commit"]
            deployments = [
                {
                    "pod_id": pod["id"],
                    "jobs": pod["jobs"],
                    "remote_root": (
                        f"/workspace/gapbalance-development-{bundle_commit[:8]}-p{index}"
                    ),
                    "reused_verified_extracted_bundle": True,
                    "archive_sha256_verified_before_extraction": True,
                }
                for index, pod in enumerate(pods)
            ]
        else:
            cancel_event = threading.Event()
            with ThreadPoolExecutor(max_workers=len(pods)) as pool:
                transfer_function = (
                    transfer_one_chunked if chunked_retry is not None else transfer_one
                )
                futures = {
                    pool.submit(
                        transfer_function,
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
                        sender_ready_timeout=int(
                            sender_retry["sender_readiness"]["maximum_banner_seconds"]
                        ),
                        post_banner_delay=int(
                            sender_retry["sender_readiness"]["post_banner_delay_seconds"]
                        ),
                        **(
                            {
                                "chunk_bytes": int(
                                    chunked_retry["retry_transport"]["chunk_bytes"]
                                ),
                                "maximum_attempts": int(
                                    chunked_retry["retry_transport"][
                                        "maximum_attempts_per_chunk"
                                    ]
                                ),
                                "failed_parent_commit": chunked_retry["public_parent_commit"],
                                "cancel_event": cancel_event,
                            }
                            if chunked_retry is not None
                            else {}
                        ),
                    ): index
                    for index, pod in enumerate(pods)
                }
                indexed = {}
                try:
                    for future in as_completed(futures):
                        indexed[futures[future]] = future.result()
                except Exception:
                    cancel_event.set()
                    for future in futures:
                        future.cancel()
                    raise
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
            status=(
                "BUNDLE_VERIFICATION_RETRY_ERROR_PODS_STOPPED"
                if verification_retry is not None
                else "JUPYTER_CROC_DEPLOYMENT_ERROR_PODS_STOPPED"
            ),
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
        "transport": (
            "reused_sha256_verified_extracted_bundle"
            if verification_retry is not None
            else "authenticated_jupyter_https_wss_plus_encrypted_runpodctl_croc"
        ),
        "jupyter_croc_retry_payload_sha256": retry["payload_sha256"],
        "bundle_egress_payload_sha256": egress["payload_sha256"],
        "bundle_manifest_payload_sha256": manifest["payload_sha256"],
        "bundle_archive_sha256": bundle_record["archive_sha256"],
        "bundle_manifest_verified_before_executor": True,
        "ephemeral_transfer_codes_persisted": False,
        "chunked_transfer_retry_payload_sha256": (
            chunked_retry["payload_sha256"] if chunked_retry is not None else None
        ),
        "pycache_verification_retry_payload_sha256": (
            verification_retry["payload_sha256"]
            if verification_retry is not None
            else None
        ),
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
