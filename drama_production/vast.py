"""Renting a GPU on Vast.ai for one job, and always giving it back.

Uses the official `vastai` command line with the VAST_API_KEY secret. A throwaway SSH key is
made for the run, registered with the account and attached to the machine; the job's files
go up by scp, the work runs detached on the machine (so a dropped connection loses nothing),
progress is read from its log, and the results come back by scp. The machine is destroyed in
a `finally`, the workflow's cleanup step destroys anything labelled with the run if the job
itself dies, and a deadline from the spending cap stops the work before it overspends.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
# 80 GB+ cards (H100, H200, RTX PRO 6000 class first; A100 80 GB as the fallback).
QUERIES = (
    "num_gpus=1 gpu_ram>=79 compute_cap>=900 verified=true rentable=true reliability>0.98 disk_space>=450 "
    "cpu_ram>=120 inet_down>=800 direct_port_count>=1 cuda_vers>=12.8",
    "num_gpus=1 gpu_ram>=79 compute_cap>=800 verified=true rentable=true reliability>0.97 disk_space>=450 "
    "cpu_ram>=100 inet_down>=500 direct_port_count>=1 cuda_vers>=12.8",
)


DOWNLOAD_GB = 240  # run 77 pulled 172 GB for pipeline 2; the drama job's models come to about 240 GB
UPLOAD_GB = 3  # the results brought back
HOURS = 2.5  # a typical run
# Run 77 paid $6.72 for downloads alone, so only hosts that charge nothing for downloads are
# rented. No 80 GB host was free both ways (run 82), and bringing the results back is a few GB,
# so an upload charge of up to ten cents is allowed. Raise these only on purpose.
MAX_DOWNLOAD_USD = 0.0
MAX_UPLOAD_USD = 0.10


def _per_tb(offer: dict[str, Any], tb_field: str, gb_field: str) -> float:
    per_tb = _num(offer.get(tb_field), -1)
    if per_tb < 0:
        per_tb = _num(offer.get(gb_field), 0.0) * 1000  # per GB
    return per_tb


def download_usd(offer: dict[str, Any], gb: float = DOWNLOAD_GB) -> float:
    """What the host charges for downloading the models (some hosts bill internet traffic per TB)."""
    return _per_tb(offer, "internet_down_cost_per_tb", "inet_down_cost") * gb / 1000


def upload_usd(offer: dict[str, Any], gb: float = UPLOAD_GB) -> float:
    return _per_tb(offer, "internet_up_cost_per_tb", "inet_up_cost") * gb / 1000


def download_hours(offer: dict[str, Any], gb: float = DOWNLOAD_GB) -> float:
    """Time spent downloading the models, which is rented time too (inet_down is in Mbit/s)."""
    mbps = max(_num(offer.get("inet_down"), 500.0), 100.0)
    return gb * 8000 / mbps / 3600 * 1.5  # Hugging Face rarely fills the line


def expected_cost(offer: dict[str, Any], hours: float = HOURS) -> float:
    """Rent for a typical run (download time included) plus traffic and storage: the true price."""
    return (_num(offer.get("dph_total"), 99) * (hours + download_hours(offer)) + download_usd(offer)
            + upload_usd(offer) + _num(offer.get("storage_total_cost"), 0.0))


class VastError(RuntimeError):
    pass


@dataclass
class Machine:
    id: int
    price: float  # USD per hour
    gpu: str
    host: str = ""
    port: int = 0
    started: float = 0.0

    def hours(self) -> float:
        return (time.time() - self.started) / 3600 if self.started else 0.0

    def cost(self) -> float:
        return self.hours() * self.price


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class Vast:
    def __init__(self, key: str | None = None, run=subprocess.run, label: str = "short-drama"):
        self.key = key or os.environ.get("VAST_API_KEY", "")
        if not self.key:
            raise VastError("Set VAST_API_KEY (a Vast.ai API key) as a GitHub Actions secret")
        self.run, self.label = run, label
        self.keydir = Path(tempfile.mkdtemp(prefix="vastkey-"))
        self.private = self.keydir / "id_ed25519"
        self.public = ""
        self.key_id: str | None = None

    # --- the command line -------------------------------------------------------------
    def cli(self, *args: str, raw: bool = True, timeout: int = 120) -> Any:
        cmd = ["vastai", *args, *(["--raw"] if raw else []), "--api-key", self.key]
        result = self.run(cmd, capture_output=True, text=True, timeout=timeout)
        shown = " ".join(a for a in cmd if a != self.key)
        if result.returncode != 0:
            raise VastError(f"{shown.replace('--api-key', '')}: {(result.stderr or result.stdout)[-500:]}")
        if not raw:
            return result.stdout
        text = result.stdout.strip()
        if not text and result.stderr.strip():
            raise VastError(f"vastai {args[0]} {args[1] if len(args) > 1 else ''}: {result.stderr.strip()[-500:]}")
        try:
            return json.loads(text) if text else {}
        except ValueError:
            match = re.search(r"[\[{].*[\]}]", text, re.S)  # some versions print a line before the JSON
            if match:
                return json.loads(match.group(0))
            raise VastError(f"unexpected output from vastai {args[0]}: {text[:300]}")

    def balance(self) -> float | None:
        user = self.cli("show", "user")
        for field in ("credit", "balance", "credit_balance"):
            if isinstance(user, dict) and field in user:
                return _num(user[field])
        return None

    def offers(self) -> list[dict[str, Any]]:
        for query in QUERIES:
            found = self.cli("search", "offers", query, "-o", "dph_total")
            found = found if isinstance(found, list) else found.get("offers", []) if isinstance(found, dict) else []
            # x86 machines only (DGX Spark / GB10 boxes are ARM and can't run the image).
            found = [o for o in found if str(o.get("cpu_arch", "amd64")).lower() in ("amd64", "x86_64")
                     and "GB10" not in str(o.get("gpu_name", "")) and download_usd(o) <= MAX_DOWNLOAD_USD
                     and upload_usd(o) <= MAX_UPLOAD_USD]
            if found:
                return sorted(found, key=expected_cost)
        return []

    # --- one machine --------------------------------------------------------------------
    def ssh_key(self) -> None:
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(self.private)], check=True)
        self.public = (self.keydir / "id_ed25519.pub").read_text().strip()
        self.cli("create", "ssh-key", self.public, raw=False)
        for key in self.cli("show", "ssh-keys") or []:
            if isinstance(key, dict) and str(key.get("public_key", key.get("ssh_key", ""))).strip() == self.public:
                self.key_id = str(key.get("id"))

    def rent(self, offer: dict[str, Any], disk: int = 300) -> Machine:
        made = self.cli("create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", str(disk), "--ssh",
                        "--direct", "--label", self.label)
        instance = made.get("new_contract") if isinstance(made, dict) else None
        if not instance:
            text = self.cli("create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", str(disk), "--ssh",
                            "--direct", "--label", self.label, raw=False) if not made else made
            match = re.search(r"new_contract['\"]?\s*[:=]\s*(\d+)", str(text))
            if not match:
                raise VastError(f"rent failed: {str(text)[:400]}")
            instance = match.group(1)
        machine = Machine(int(instance), _num(offer.get("dph_total")), str(offer.get("gpu_name", "?")))
        machine.started = time.time()
        log.info("Rented %s (instance %s) at $%.2f/h; internet traffic: download $%.2f, upload $%.2f",
                 machine.gpu, machine.id, machine.price, download_usd(offer), upload_usd(offer))
        return machine

    def wait(self, machine: Machine, timeout: int = 1500) -> None:
        deadline = time.time() + timeout
        status = None
        while time.time() < deadline:
            info = self.cli("show", "instance", str(machine.id))
            info = info[0] if isinstance(info, list) and info else info
            status = info.get("actual_status") if isinstance(info, dict) else None
            if status == "running":
                break
            if status in ("exited", "unknown", "offline"):
                raise VastError(f"instance {machine.id} stopped while starting ({status})")
            time.sleep(15)
        else:
            raise VastError(f"instance {machine.id} not running after {timeout} s (last status {status})")
        if self.public:
            try:
                self.cli("attach", "ssh", str(machine.id), self.public, raw=False)
            except VastError as exc:  # usually already attached from the account key
                log.info("attach ssh: %s", exc)
        url = str(self.cli("ssh-url", str(machine.id), raw=False)).strip()
        match = re.search(r"ssh://\w+@([^:\s]+):(\d+)", url)
        if not match:
            raise VastError(f"no SSH address for instance {machine.id}: {url[:200]}")
        machine.host, machine.port = match.group(1), int(match.group(2))
        for _ in range(40):  # sshd can lag behind "running"
            if self.ssh(machine, "true", timeout=30, check=False).returncode == 0:
                return
            time.sleep(15)
        raise VastError(f"could not log in to instance {machine.id}")

    def destroy(self, machine: Machine) -> None:
        for _ in range(3):
            try:
                self.cli("destroy", "instance", str(machine.id), "-y", raw=False)
                log.info("Destroyed instance %s after %.2f h (about $%.2f)", machine.id, machine.hours(), machine.cost())
                return
            except VastError as exc:
                log.warning("destroy failed (retrying): %s", exc)
                time.sleep(10)
        log.error("COULD NOT DESTROY instance %s: destroy it at cloud.vast.ai/instances", machine.id)

    def forget_key(self) -> None:
        if self.key_id:
            try:
                self.cli("delete", "ssh-key", self.key_id, raw=False)
            except VastError as exc:
                log.warning("could not remove the run's SSH key: %s", exc)

    def cleanup_label(self) -> int:
        """Destroy every instance carrying this run's label (the workflow's safety net)."""
        gone = 0
        for inst in self.cli("show", "instances") or []:
            if isinstance(inst, dict) and inst.get("label") == self.label:
                self.cli("destroy", "instance", str(inst.get("id")), "-y", raw=False)
                gone += 1
        return gone

    # --- talking to the machine ------------------------------------------------------------
    def _ssh_args(self, machine: Machine) -> list[str]:
        return ["-i", str(self.private), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20"]

    def ssh(self, machine: Machine, command: str, timeout: int = 600, check: bool = True):
        result = subprocess.run(["ssh", "-n", "-T", *self._ssh_args(machine), "-p", str(machine.port),
                                 f"root@{machine.host}", command], capture_output=True, text=True, timeout=timeout)
        if check and result.returncode != 0:
            raise VastError(f"ssh {command[:60]!r}: {result.stderr[-400:]}")
        return result

    def start(self, machine: Machine, folder: str, script: str, logfile: str) -> None:
        """Start the job fully detached. Some hosts keep the ssh channel open after the job is in the
        background (runs 77 and 83 timed out here), so a timeout is not a failure: the job counts as
        started once its log exists."""
        command = (f"cd {shlex.quote(folder)} && (setsid nohup bash {shlex.quote(script)} "
                   f"> {shlex.quote(logfile)} 2>&1 < /dev/null &) ; exit 0")
        try:
            self.ssh(machine, command, timeout=45, check=False)
        except subprocess.TimeoutExpired:
            log.info("The start command held the connection open; checking the job started")
        for _ in range(6):
            probe = self.ssh(machine, f"test -e {shlex.quote(folder + '/' + logfile)} && echo STARTED",
                             timeout=60, check=False)
            if "STARTED" in probe.stdout:
                log.info("Job started on the GPU")
                return
            time.sleep(10)
        raise VastError("the job did not start on the GPU (no log file appeared)")

    def put(self, machine: Machine, files: list[Path], remote: str) -> None:
        subprocess.run(["scp", *self._ssh_args(machine), "-P", str(machine.port), "-r", *map(str, files),
                        f"root@{machine.host}:{remote}"], check=True, capture_output=True, timeout=1800)

    def get(self, machine: Machine, remote: str, local: Path) -> None:
        local.mkdir(parents=True, exist_ok=True)
        subprocess.run(["scp", *self._ssh_args(machine), "-P", str(machine.port), "-r",
                        f"root@{machine.host}:{remote}", str(local)], check=True, capture_output=True, timeout=3600)

    def follow(self, machine: Machine, logfile: str, done: str, deadline: float, poll: int = 60) -> str:
        """Wait for the detached job, echoing new log lines. Returns the full log."""
        seen = 0
        text = ""
        while True:
            result = self.ssh(machine, f"cat {shlex.quote(logfile)} 2>/dev/null; test -e {shlex.quote(done)} && echo __DONE__",
                              timeout=120, check=False)
            text = result.stdout
            lines = text.splitlines()
            for line in lines[seen:]:
                if line != "__DONE__":
                    log.info("[gpu] %s", line[:2500] if "FAILED" in line or "Error" in line else line[:300])
            seen = len([x for x in lines if x != "__DONE__"])
            if "__DONE__" in lines:
                return text
            if time.time() > deadline:
                log.warning("Time limit reached (the spending cap): keeping what the GPU has made so far")
                return text + "\n__STOPPED_AT_LIMIT__"
            time.sleep(poll)


REMOTE_SCRIPT = """#!/usr/bin/env bash
# Runs detached on the rented machine; every step logs to run.log; DONE marks the end.
set -uo pipefail
cd /workspace/job
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
step() { echo "$(date +%H:%M:%S) == $*"; }
step installing
(apt-get update -qq && apt-get install -y -qq git ffmpeg) >/dev/null 2>&1
pip install -q -U "git+https://github.com/huggingface/diffusers" transformers accelerate sentencepiece \\
  hf_transfer ftfy imageio imageio-ffmpeg peft "huggingface_hub[hf_transfer]" 2>&1 | tail -n 3
step "downloading @PICTURE_MODEL@"
python -c "from huggingface_hub import snapshot_download as d; d('@PICTURE_MODEL@')" 2>&1 | tail -n 2
step "downloading @CLIP_MODEL@ in the background"
(python -c "from huggingface_hub import snapshot_download as d; d('@CLIP_MODEL@')" > clip_download.log 2>&1; touch CLIP_READY) &
step pictures
python vast_worker.py job.json pictures
step "waiting for the clip model"
while [ ! -e CLIP_READY ]; do sleep 10; done
tail -n 2 clip_download.log
step clips
python vast_worker.py job.json clips
step "installing MMAudio"
(git clone -q https://github.com/hkchengrex/MMAudio /workspace/MMAudio && pip install -q -e /workspace/MMAudio) 2>&1 | tail -n 2
step audio
python vast_worker.py job.json audio
step finished
touch /workspace/job/DONE
"""


def remote_script(job: dict[str, Any]) -> str:
    return REMOTE_SCRIPT.replace("@PICTURE_MODEL@", job["picture_model"]).replace("@CLIP_MODEL@", job["clip_model"])


def run_job(vast: Vast, job: dict[str, Any], worker: Path, local_out: Path, max_usd: float,
            disk: int = 300, max_hours: float = 3.4, script: str | None = None) -> dict[str, Any]:
    """Rent the cheapest suitable machine, run the job, bring back the results, destroy the machine."""
    credit = vast.balance()
    if credit is not None:
        if credit < 1.0:
            raise VastError(f"Vast.ai credit is ${credit:.2f}: add credit at cloud.vast.ai/billing, then run again")
        max_usd = min(max_usd, credit - 0.5)  # never plan to spend more than the account holds
    offers = vast.offers()
    if not offers:
        raise VastError("no suitable 80 GB GPU offer on Vast.ai right now")
    vast.ssh_key()
    machine = None
    for offer in offers[:6]:  # a cheap offer is often taken between the search and the rent
        price = _num(offer.get("dph_total"))
        if price * 0.5 > max_usd:
            continue
        try:
            machine = vast.rent(offer, disk)
            break
        except VastError as exc:
            log.warning("Offer %s (%s) not rented: %s", offer.get("id"), offer.get("gpu_name"), exc)
    if machine is None:
        vast.forget_key()
        raise VastError("none of the best offers could be rented right now; try again in a few minutes")
    price = machine.price
    hours = min(max_usd / price if price else 3.0, max_hours)
    log.info("Rented %s at $%.2f/h; time limit %.1f h from the $%.2f cap", machine.gpu, price, hours, max_usd)
    report: dict[str, Any] = {"gpu": machine.gpu, "price_per_hour": price, "instance": machine.id}
    try:
        vast.wait(machine)
        job = dict(job, out="/workspace/job/out")
        work = Path(tempfile.mkdtemp(prefix="vastjob-"))
        (work / "job.json").write_text(json.dumps(job))
        (work / "run.sh").write_text(script or remote_script(job))
        (work / "vast_worker.py").write_text(worker.read_text())
        vast.ssh(machine, "mkdir -p /workspace/job/out")
        vast.put(machine, [work / "job.json", work / "run.sh", work / "vast_worker.py"], "/workspace/job/")
        for name in job.get("uploads", []):
            vast.put(machine, [Path(name)], "/workspace/job/out/")
        vast.start(machine, "/workspace/job", "run.sh", "run.log")
        deadline = machine.started + hours * 3600 - 600  # keep 10 minutes for the download and teardown
        text = vast.follow(machine, "/workspace/job/run.log", "/workspace/job/DONE", deadline)
        report["stopped_at_limit"] = text.endswith("__STOPPED_AT_LIMIT__")
        report["log"] = text[-6000:]
    finally:
        try:
            if machine.host:
                vast.get(machine, "/workspace/job/out", local_out)  # whatever was made, even after a failure
        except Exception as exc:  # noqa: BLE001
            log.warning("could not fetch results: %s", exc)
        vast.destroy(machine)
        vast.forget_key()
        report["hours"] = round(machine.hours(), 2)
        report["usd"] = round(machine.cost(), 2)
    return report
