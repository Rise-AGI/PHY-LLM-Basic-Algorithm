"""
Autonomous SFT training monitor & retry loop.

Usage:
  python auto_sft.py check                        # cluster availability
  python auto_sft.py submit --gpus 6              # sync code + submit job
  python auto_sft.py submit --gpus 5              # sync code + submit 5-GPU
  python auto_sft.py status JOB_ID                # current job status
  python auto_sft.py metrics JOB_ID               # parse training metrics from logs
  python auto_sft.py logs JOB_ID [--tail 200]     # fetch recent logs
  python auto_sft.py loop --gpus 6                # full autonomous loop
  python auto_sft.py loop --continue              # resume previous loop

Output: all modes print JSON to stdout (one JSON object per call).
State file: train/auto_state.json
"""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import magnus

from config import (
    MAGNUS_ADDRESS, MAGNUS_TOKEN, SYSTEM_ENTRY_COMMAND,
    SFT_DATA_DIR, _ensure_record, check_model_version_exists,
    record_storage,
)

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(HERE, "auto_state.json")

# ── Config (mirrors submit_sft.py defaults) ─────────────────────

MODE = "sft"
MODEL_PATH = "/data/magnus/models/Qwen2.5-72B-Instruct"
BLUEPRINT_PATH = os.path.join(HERE, "blueprints", "OpenFundus_SFT_zyz.magnus")
GITHUB_REPO_PATH = os.path.join(HERE, "..", "PHY-LLM-Basic-Algorithm")
GITHUB_PATH = "PHY-LLM-Basic-Algorithm/sft_train.py"

# Hyperparams
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-5
MAX_LENGTH = 1024
NUM_WORKERS = 4

# Hardware
GPU_TYPE = "a100"
CPU_COUNT = 60
MEMORY = "160G"
STORAGE = "1024G"
PRIORITY = "A2"
CONTAINER_IMAGE = "docker://crpi-32rssczyu25r10yu.cn-beijing.personal.cr.aliyuncs.com/zyz25/sft-base:v2"

# Optimization
CPU_OFFLOAD = True
USE_8BIT_ADAM = True
BWD_PREFETCH = "post"

PROMPT_PREFIX = (
    '你是一位数学解题专家。请逐步推理并解答以下问题。\n'
    '\n'
    '输出格式：\n'
    '答案：[最终结果。多问时用 (1)...; (2)... 分别列出。数学公式使用 LaTeX $...$]\n'
    '\n'
    '解答：[完整推导过程。写明所用的定理、公式或变换方法，关键推导步骤不可省略]\n'
    '\n'
    '{instruction}'
)

# ── Issue detection patterns ────────────────────────────────────

FATAL_PATTERNS = {
    "nccL_timeout": re.compile(
        r"Watchdog caught collective operation timeout", re.I),
    "allgather_timeout": re.compile(
        r"_ALLGATHER_BASE.*timing out", re.I),
    "cuda_oom": re.compile(
        r"CUDA out of memory|out of memory.*cuda", re.I),
    "oom_killed": re.compile(
        r"Killed|Out of memory", re.I),
    "loss_nan": re.compile(
        r"loss.*NaN|NaN.*loss", re.I),
    "process_crash": re.compile(
        r"terminate called|SIGKILL|SIGTERM|signal", re.I),
    "nccl_error": re.compile(
        r"ncclSystemError|ncclInternalError|ncclInvalidUsage", re.I),
    "cuda_error": re.compile(
        r"CUDA error|CUDNN_STATUS", re.I),
}

WARN_PATTERNS = {
    "dist_barrier_timeout": re.compile(
        r"dist\.barrier.*timeout|barrier.*timed out", re.I),
    "slow_batch": re.compile(
        r"", re.I),  # detected via metrics, not regex
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── State management ────────────────────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "run_id": None,
        "goal": "solve_oom_and_slow_step",
        "gpu_count": 6,
        "attempt": 0,
        "max_attempts": 10,
        "phase": "idle",
        "job_id": None,
        "submit_time": None,
        "last_metrics": {},
        "last_log_length": 0,
        "fixes_applied": [],
        "history": [],
        "success": False,
        "started_at": None,
    }


def save_state(s: dict) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)


# ── Cluster checking ────────────────────────────────────────────

def cmd_check(gpu_min: int = 6) -> dict:
    """Check if cluster has enough free GPUs."""
    try:
        jobs_resp = magnus.list_jobs(limit=50)
        jobs = jobs_resp.get("items", []) if isinstance(jobs_resp, dict) else []
    except Exception as e:
        return {"available": False, "error": str(e), "free_gpus": 0}

    running = [j for j in jobs if j.get("status") == "Running"]
    pending = [j for j in jobs if j.get("status") == "Pending"]

    # Count GPUs in use by running jobs
    gpus_in_use = sum(
        j.get("gpu_count", 0) or 0 for j in running
    )

    # Assume cluster has at least 8 A100 GPUs (conservative estimate)
    # In practice, try to get cluster_stats
    cluster_total = 8
    try:
        stats = magnus.get_cluster_stats()
        # cluster_stats format varies; try to extract GPU count
        if isinstance(stats, dict):
            gpu_info = stats.get("gpu", {}) or stats.get("gpus", {})
            if isinstance(gpu_info, dict):
                total = gpu_info.get("total", 0)
                if total > 0:
                    cluster_total = total
    except Exception:
        pass

    free = max(0, cluster_total - gpus_in_use)

    result = {
        "available": free >= gpu_min,
        "free_gpus": free,
        "gpu_min": gpu_min,
        "cluster_total": cluster_total,
        "running_jobs": len(running),
        "pending_jobs": len(pending),
        "gpus_in_use": gpus_in_use,
        "timestamp": _now_iso(),
    }

    if not result["available"]:
        result["reason"] = f"need {gpu_min} free GPUs, have {free}"
        if pending:
            result["reason"] += f" ({len(pending)} jobs in queue)"

    return result


# ── Code sync ───────────────────────────────────────────────────

def cmd_sync() -> dict:
    """Copy sft_train.py to GitHub repo, commit & push if changed."""
    src = os.path.join(HERE, "sft_train.py")
    dst = os.path.join(GITHUB_REPO_PATH, "sft_train.py")

    if not os.path.exists(src):
        return {"synced": False, "error": f"source not found: {src}"}
    if not os.path.isdir(GITHUB_REPO_PATH):
        return {"synced": False, "error": f"repo not found: {GITHUB_REPO_PATH}"}

    try:
        shutil.copy2(src, dst)
    except Exception as e:
        return {"synced": False, "error": f"copy failed: {e}"}

    try:
        subprocess.run(
            ["git", "-C", GITHUB_REPO_PATH, "add", "sft_train.py"],
            check=True, capture_output=True, text=True)
        changed = subprocess.run(
            ["git", "-C", GITHUB_REPO_PATH, "diff", "--cached", "--quiet", "sft_train.py"],
            capture_output=True)
        if changed.returncode != 0:
            subprocess.run(
                ["git", "-C", GITHUB_REPO_PATH, "commit", "-m",
                 "[auto_sft] sync sft_train.py fixes"],
                check=True, capture_output=True, text=True)
            push = subprocess.run(
                ["git", "-C", GITHUB_REPO_PATH, "push"],
                check=True, capture_output=True, text=True)
            return {"synced": True, "pushed": True,
                    "commit": "latest", "repo": GITHUB_PATH}
        else:
            return {"synced": True, "pushed": False,
                    "reason": "no changes", "repo": GITHUB_PATH}
    except subprocess.CalledProcessError as e:
        return {"synced": False, "error": f"git failed: {e.stderr}"}


# ── Job submission ──────────────────────────────────────────────

def _model_short_name() -> str:
    return MODEL_PATH.rstrip("/").split("/")[-1]


def _auto_model_version() -> str:
    short = _model_short_name()
    record = _ensure_record()
    prefix = f"{short}-{MODE}-zyz-v"
    existing = [e for e in record.get("model-version", [])
                if e.get("model", "").startswith(prefix)]
    max_v = 0
    for e in existing:
        m = re.search(re.escape(prefix) + r"(\d+)$", e.get("model", ""))
        if m:
            v = int(m.group(1))
            if v > max_v:
                max_v = v
    return f"{prefix}{max_v + 1}"


def cmd_submit(gpu_count: int = 6) -> dict:
    """Sync code, save blueprint, launch job."""
    # 1. Sync code to GitHub
    sync_result = cmd_sync()
    if not sync_result["synced"]:
        return {"submitted": False, "error": f"sync failed: {sync_result.get('error')}",
                "sync": sync_result}

    # 2. Wait for GitHub CDN
    if sync_result.get("pushed"):
        time.sleep(10)

    # 3. Read blueprint
    if not os.path.exists(BLUEPRINT_PATH):
        return {"submitted": False, "error": f"blueprint not found: {BLUEPRINT_PATH}"}
    with open(BLUEPRINT_PATH, "r", encoding="utf-8") as f:
        bp_code = f.read()

    bp_id = os.path.splitext(os.path.basename(BLUEPRINT_PATH))[0]
    short_name = _model_short_name()
    model_version = _auto_model_version()
    out_dir = f"/data/magnus/models/{model_version}"

    if check_model_version_exists(model_version):
        return {"submitted": False,
                "error": f"model version already exists: {model_version}"}

    # 4. Configure Magnus
    magnus.configure(address=MAGNUS_ADDRESS, token=MAGNUS_TOKEN)

    # 5. Save blueprint
    try:
        magnus.save_blueprint(
            blueprint_id=bp_id,
            title=f"SFT-{bp_id}",
            description=f"Auto-saved SFT blueprint ({gpu_count} GPU, attempt from auto_sft)",
            code=bp_code,
        )
    except Exception as e:
        # continue with existing blueprint
        pass

    # 6. Build args
    bp_args = {
        "model_path": MODEL_PATH,
        "output_dir": out_dir,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "num_workers": NUM_WORKERS,
        "gpu_count": gpu_count,
        "gpu_type": GPU_TYPE,
        "cpu_count": CPU_COUNT,
        "memory_demand": MEMORY,
        "ephemeral_storage": STORAGE,
        "priority": PRIORITY,
        "container_image": CONTAINER_IMAGE,
        "cpu_offload": CPU_OFFLOAD,
        "use_8bit_adam": USE_8BIT_ADAM,
        "backward_prefetch": BWD_PREFETCH,
        "prompt_prefix": PROMPT_PREFIX,
        "prompt_prefix_b64": base64.b64encode(
            PROMPT_PREFIX.encode("utf-8")
        ).decode("ascii"),
    }

    # 7. Launch
    try:
        job_id = magnus.launch_blueprint(bp_id, args=bp_args)
    except Exception as e:
        return {"submitted": False, "error": f"launch failed: {e}"}

    return {
        "submitted": True,
        "job_id": job_id,
        "model_version": model_version,
        "output_dir": out_dir,
        "gpu_count": gpu_count,
        "blueprint_id": bp_id,
        "sync": sync_result,
        "timestamp": _now_iso(),
    }


# ── Status query ────────────────────────────────────────────────

def cmd_status(job_id: str) -> dict:
    """Get current job status."""
    try:
        job = magnus.get_job(job_id)
    except Exception as e:
        return {"job_id": job_id, "error": str(e)}

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "task_name": job.get("task_name", ""),
        "created_at": job.get("created_at", ""),
        "started_at": job.get("started_at", ""),
        "gpu_count": job.get("gpu_count", 0),
        "gpu_type": job.get("gpu_type", ""),
    }


# ── Log fetching ────────────────────────────────────────────────

def cmd_logs(job_id: str, tail: int = 200) -> dict:
    """Fetch logs, optionally tail only."""
    try:
        page = magnus.get_job_logs(job_id, page=0)
    except Exception as e:
        return {"job_id": job_id, "error": str(e), "logs": ""}

    text = page.get("logs", "") if isinstance(page, dict) else str(page)
    lines = text.splitlines()
    if tail and len(lines) > tail:
        text = "\n".join(lines[-tail:])

    return {"job_id": job_id, "logs": text, "total_lines": len(lines),
            "showing": min(len(lines), tail) if tail else len(lines)}


# ── Metrics parsing ─────────────────────────────────────────────

def cmd_metrics(job_id: str, state: dict = None) -> dict:
    """Parse training metrics from job logs."""
    if state is None:
        state = {}

    log_result = cmd_logs(job_id, tail=0)
    text = log_result.get("logs", "")
    lines = text.splitlines()
    last_log_len = state.get("last_log_length", 0)

    # Perf metrics: [diagnostic] perf.total_ms=12345 fwd=5000 bwd=4000 opt=2000 comm=500
    perf_re = re.compile(
        r"perf\.(\w+)=([\d.]+)", re.I)

    # Memory: GPU mem=XX.XGB (peak=XX.XGB)
    mem_re = re.compile(
        r"GPU mem=([\d.]+)GB\s*\(peak=([\d.]+)GB\)")

    # Loss: train.loss or loss= (written to JSONL)
    loss_re = re.compile(
        r"(?:train_loss|train\.loss|loss)[=:\s]+([\d.]+)", re.I)

    # Step diagnostic: >>> Step X/447 (global Y)
    step_re = re.compile(
        r">>> Step\s+(\d+)/(\d+)\s+\(global\s+(\d+)\)")

    # Epoch
    epoch_re = re.compile(
        r"\[Epoch\s+(\d+)/(\d+)\]")

    # FSDP strategy
    fsdp_re = re.compile(
        r"FSDP:\s+(\S+)")

    metrics = {
        "job_id": job_id,
        "timestamp": _now_iso(),
        "total_log_lines": len(lines),
        "new_lines": max(0, len(lines) - last_log_len),
        "steps_seen": 0,
        "steps": [],
        "epoch": None,
        "fsdp_strategy": None,
        "latest_memory_gb": None,
        "peak_memory_gb": None,
        "latest_loss": None,
        "perf_latest": {},
        "issues": [],
    }

    # Parse FSDP strategy
    for line in lines:
        m = fsdp_re.search(line)
        if m:
            metrics["fsdp_strategy"] = m.group(1)

    # Parse epoch
    for line in reversed(lines):
        m = epoch_re.search(line)
        if m:
            metrics["epoch"] = f"{m.group(1)}/{m.group(2)}"
            break

    # Parse step diagnostics & perf
    step_entries = []
    for line in lines:
        sm = step_re.search(line)
        if sm:
            step_entries.append({
                "step": int(sm.group(1)),
                "per_epoch": int(sm.group(2)),
                "global": int(sm.group(3)),
                "line": line.strip()[:200],
            })

    if step_entries:
        metrics["steps_seen"] = len(step_entries)
        metrics["latest_step"] = step_entries[-1]
        metrics["steps"] = step_entries[-20:]  # last 20 steps

    # Parse memory
    for line in reversed(lines):
        m = mem_re.search(line)
        if m:
            metrics["latest_memory_gb"] = float(m.group(1))
            metrics["peak_memory_gb"] = float(m.group(2))
            break

    # Parse perf from recent lines
    perf_vals = {}
    for line in reversed(lines[-500:]):
        for m in perf_re.finditer(line):
            key = m.group(1)
            val = float(m.group(2))
            if key not in perf_vals:
                perf_vals[key] = val
    if perf_vals:
        metrics["perf_latest"] = perf_vals
        # Derive step time
        if "total_ms" in perf_vals:
            metrics["step_time_s"] = round(perf_vals["total_ms"] / 1000, 1)
        # Check comm bottleneck
        if perf_vals.get("total_ms", 0) > 0:
            comm_pct = perf_vals.get("comm_ms", 0) / perf_vals["total_ms"] * 100
            metrics["comm_pct"] = round(comm_pct, 1)

    # Parse loss
    for line in reversed(lines):
        m = re.search(r'"train_loss":\s*([\d.]+)', line)
        if m:
            metrics["latest_loss"] = float(m.group(1))
            break
        m = re.search(r'training_log\.json.*loss.*?([\d.]+)', line)
        if m:
            metrics["latest_loss"] = float(m.group(1))
            break

    # Check for issues in NEW lines only
    new_text = "\n".join(lines[last_log_len:]) if last_log_len > 0 else text
    for name, pat in FATAL_PATTERNS.items():
        if pat.search(new_text):
            metrics["issues"].append({"severity": "FATAL", "type": name})

    # Staleness check
    if state.get("phase") == "running" and metrics["new_lines"] == 0:
        last_active = state.get("last_active_at")
        if last_active:
            try:
                last_dt = datetime.fromisoformat(last_active)
                idle_secs = (datetime.now(timezone.utc) - last_dt).total_seconds()
                if idle_secs > 2700:  # 45 min
                    metrics["issues"].append({
                        "severity": "FATAL",
                        "type": "stale_logs",
                        "idle_minutes": round(idle_secs / 60),
                    })
            except Exception:
                pass

    # Step time check
    step_s = metrics.get("step_time_s")
    if step_s and step_s > 300:
        metrics["issues"].append({
            "severity": "WARN", "type": "slow_step",
            "step_time_s": step_s,
        })

    # Memory check
    peak = metrics.get("peak_memory_gb")
    if peak and peak > 78:
        metrics["issues"].append({
            "severity": "WARN", "type": "high_memory",
            "peak_gb": peak,
        })

    return metrics


# ── Autonomous loop ─────────────────────────────────────────────

def cmd_loop(gpu_count: int = 6, continue_prev: bool = False,
             max_attempts: int = 10, wait_idle_min: int = 30) -> dict:
    """
    Full autonomous loop: check → submit → wait queue → monitor → fix → retry.

    This is designed to be called repeatedly by Claude. Each call
    reads state, advances one phase, and returns a directive for the
    next action. Claude reads the directive and calls again.

    Key: when a job fails with a known issue pattern, Claude modifies
    sft_train.py / blueprint, then calls `cmd_loop(continue_prev=True)`
    which will re-submit.
    """
    state = load_state()

    if not continue_prev:
        # Fresh start
        state["run_id"] = f"auto-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        state["gpu_count"] = gpu_count
        state["attempt"] = 1
        state["max_attempts"] = max_attempts
        state["phase"] = "checking"
        state["job_id"] = None
        state["submit_time"] = None
        state["last_metrics"] = {}
        state["last_log_length"] = 0
        state["fixes_applied"] = []
        state["history"] = []
        state["success"] = False
        state["started_at"] = _now_iso()
        state["last_active_at"] = _now_iso()
        save_state(state)
    else:
        # Continue: poll current job if active, otherwise start new attempt
        current_job = state.get("job_id")
        if current_job and state.get("phase") in ("queue_wait", "running"):
            # Job is still active — just poll it, don't start a new one
            pass  # fall through to the phase machine
        elif current_job:
            # Check if the job is still alive
            try:
                js = cmd_status(current_job)
                if js.get("status") in ("Pending", "Running"):
                    state["phase"] = "queue_wait" if js["status"] == "Pending" else "running"
                    # fall through
                else:
                    # Job ended — start new attempt
                    state["attempt"] += 1
                    state["phase"] = "checking"
                    state["job_id"] = None
                    state["last_log_length"] = 0
            except Exception:
                state["attempt"] += 1
                state["phase"] = "checking"
                state["job_id"] = None
                state["last_log_length"] = 0
        else:
            # No job — start new attempt
            state["attempt"] += 1
            state["phase"] = "checking"
            state["job_id"] = None
            state["last_log_length"] = 0
        save_state(state)

    directive = {
        "run_id": state["run_id"],
        "attempt": state["attempt"],
        "max_attempts": state["max_attempts"],
        "gpu_count": state["gpu_count"],
        "phase": None,
        "action": None,
        "job_id": state.get("job_id"),
        "metrics": None,
        "issues": [],
        "state": state,
    }

    # ── Phase: CHECK ──────────────────────────────────────────
    if state["phase"] == "checking":
        check = cmd_check(gpu_min=state["gpu_count"])

        if check["available"]:
            state["phase"] = "submitting"
            save_state(state)
            directive["phase"] = "submitting"
            directive["action"] = "submit_now"
            directive["cluster"] = check
        else:
            directive["phase"] = "waiting_cluster"
            directive["action"] = f"wait_{wait_idle_min}min"
            directive["cluster"] = check
            directive["reason"] = check.get("reason", "cluster busy")
            # Don't sleep here — Claude handles this in the skill
            return directive

    # ── Phase: SUBMIT ─────────────────────────────────────────
    if state["phase"] == "submitting":
        result = cmd_submit(gpu_count=state["gpu_count"])

        if not result.get("submitted"):
            state["phase"] = "checking"
            state["history"].append({
                "attempt": state["attempt"],
                "phase": "submit_failed",
                "error": result.get("error"),
                "time": _now_iso(),
            })
            save_state(state)
            directive["phase"] = "submit_failed"
            directive["action"] = "retry"
            directive["error"] = result.get("error")
            return directive

        state["job_id"] = result["job_id"]
        state["submit_time"] = _now_iso()
        state["phase"] = "queue_wait"
        state["last_active_at"] = _now_iso()
        directive["job_id"] = result["job_id"]
        directive["model_version"] = result.get("model_version")

        state["history"].append({
            "attempt": state["attempt"],
            "phase": "submitted",
            "job_id": result["job_id"],
            "model_version": result.get("model_version"),
            "time": _now_iso(),
        })
        save_state(state)
        directive["phase"] = "queue_wait"
        directive["action"] = "wait_for_running"
        return directive

    # ── Phase: QUEUE WAIT ─────────────────────────────────────
    if state["phase"] == "queue_wait":
        job_id = state["job_id"]
        job_status = cmd_status(job_id)
        status = job_status.get("status", "Unknown")

        if status == "Running":
            state["phase"] = "running"
            state["last_active_at"] = _now_iso()
            save_state(state)
            directive["phase"] = "running"
            directive["action"] = "monitor"
            directive["job_status"] = job_status
            return directive
        elif status in ("Failed", "Terminated", "Success"):
            state["phase"] = "checking"
            state["history"].append({
                "attempt": state["attempt"],
                "phase": f"queue_ended_{status}",
                "job_id": job_id,
                "status": status,
                "time": _now_iso(),
            })
            save_state(state)
            directive["phase"] = f"job_{status.lower()}"
            directive["action"] = "analyze_and_retry"
            directive["job_status"] = job_status
            return directive
        elif status == "Pending":
            # Check how long we've been waiting
            submit_ts = state.get("submit_time", "")
            wait_minutes = 0
            if submit_ts:
                try:
                    submit_dt = datetime.fromisoformat(submit_ts)
                    wait_minutes = (datetime.now(timezone.utc) - submit_dt).total_seconds() / 60
                except Exception:
                    pass
            if wait_minutes > 120:
                directive["action"] = "queue_timeout_2h"
                directive["warning"] = f"waiting {wait_minutes:.0f} min in queue"
            else:
                directive["action"] = "keep_waiting"
            directive["phase"] = "queue_wait"
            directive["job_status"] = job_status
            directive["wait_minutes"] = round(wait_minutes, 1)
            return directive
        else:
            directive["phase"] = "queue_wait"
            directive["action"] = "keep_waiting"
            directive["job_status"] = job_status
            return directive

    # ── Phase: RUNNING ────────────────────────────────────────
    if state["phase"] == "running":
        job_id = state["job_id"]
        state["last_active_at"] = _now_iso()

        # Get status + metrics
        job_status = cmd_status(job_id)
        status = job_status.get("status", "Unknown")
        m = cmd_metrics(job_id, state)
        state["last_metrics"] = m
        state["last_log_length"] = m.get("total_log_lines", 0)

        directive["metrics"] = m
        directive["job_status"] = job_status

        if status == "Running":
            if m.get("issues"):
                # Has issues — decide whether to terminate or wait
                fatal = [i for i in m["issues"] if i["severity"] == "FATAL"]
                if fatal:
                    state["phase"] = "checking"
                    state["history"].append({
                        "attempt": state["attempt"],
                        "phase": "running_fatal",
                        "job_id": job_id,
                        "issues": fatal,
                        "time": _now_iso(),
                    })
                    save_state(state)
                    directive["phase"] = "fatal_detected"
                    directive["action"] = "terminate_and_fix"
                    directive["fatal_issues"] = fatal
                    return directive
                else:
                    directive["phase"] = "running_warn"
                    directive["action"] = "continue_monitoring"
                    directive["warnings"] = [i for i in m["issues"] if i["severity"] == "WARN"]
                    save_state(state)
                    return directive
            else:
                directive["phase"] = "running_healthy"
                directive["action"] = "continue_monitoring"
                save_state(state)
                return directive
        elif status in ("Success", "Failed", "Terminated"):
            state["phase"] = "completed"
            state["history"].append({
                "attempt": state["attempt"],
                "phase": f"job_{status.lower()}",
                "job_id": job_id,
                "status": status,
                "metrics_summary": {
                    "steps_seen": m.get("steps_seen"),
                    "step_time_s": m.get("step_time_s"),
                    "peak_memory_gb": m.get("peak_memory_gb"),
                    "latest_loss": m.get("latest_loss"),
                },
                "time": _now_iso(),
            })
            if status == "Success":
                state["success"] = True
            save_state(state)
            directive["phase"] = f"job_{status.lower()}"
            directive["action"] = "report_and_decide"
            return directive
        else:
            directive["phase"] = "running"
            directive["action"] = "continue_monitoring"
            save_state(state)
            return directive

    # ── Phase: COMPLETED ──────────────────────────────────────
    if state["phase"] == "completed":
        if state.get("success"):
            # Record model version
            last_hist = state.get("history", [])[-1] if state.get("history") else {}
            record_storage("model-version", {
                "time": _now_iso(),
                "model": last_hist.get("model_version", "unknown"),
                "local_path": f"/data/magnus/models/{last_hist.get('model_version', '')}",
                "base_model": MODEL_PATH,
                "status": "success",
            })
            directive["phase"] = "goal_achieved"
            directive["action"] = "exit_success"
            directive["summary"] = {
                "attempts": state["attempt"],
                "fixes_applied": state["fixes_applied"],
                "history": state["history"],
            }
        else:
            directive["phase"] = "job_failed"
            directive["action"] = "analyze_and_retry"
            state["phase"] = "checking"
        save_state(state)
        return directive

    directive["phase"] = state["phase"]
    directive["action"] = "unknown"
    return directive


# ── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="auto SFT training monitor")
    sub = parser.add_subparsers(dest="command")

    # check
    p_check = sub.add_parser("check", help="check cluster availability")
    p_check.add_argument("--gpus", type=int, default=6)

    # submit
    p_sub = sub.add_parser("submit", help="sync code & submit job")
    p_sub.add_argument("--gpus", type=int, default=6)

    # status
    p_st = sub.add_parser("status", help="query job status")
    p_st.add_argument("job_id")

    # logs
    p_log = sub.add_parser("logs", help="fetch job logs")
    p_log.add_argument("job_id")
    p_log.add_argument("--tail", type=int, default=200)

    # metrics
    p_met = sub.add_parser("metrics", help="parse training metrics from logs")
    p_met.add_argument("job_id")

    # loop
    p_loop = sub.add_parser("loop", help="autonomous loop")
    p_loop.add_argument("--gpus", type=int, default=6)
    p_loop.add_argument("--continue", dest="continue_prev",
                        action="store_true", default=False)
    p_loop.add_argument("--max-attempts", type=int, default=10)

    args = parser.parse_args()

    # Configure magnus for all commands
    magnus.configure(address=MAGNUS_ADDRESS, token=MAGNUS_TOKEN)

    if args.command == "check":
        result = cmd_check(gpu_min=args.gpus)
    elif args.command == "submit":
        result = cmd_submit(gpu_count=args.gpus)
    elif args.command == "status":
        result = cmd_status(args.job_id)
    elif args.command == "logs":
        result = cmd_logs(args.job_id, tail=args.tail)
    elif args.command == "metrics":
        result = cmd_metrics(args.job_id)
    elif args.command == "loop":
        result = cmd_loop(
            gpu_count=args.gpus,
            continue_prev=args.continue_prev,
            max_attempts=args.max_attempts,
        )
    else:
        result = {"error": f"unknown command: {args.command}"}

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
