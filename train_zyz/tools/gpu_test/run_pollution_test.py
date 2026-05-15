"""
GPU 污染验证编排器 — 在本地运行，通过 Magnus API 提交/终止 job。

验证流程：
    Phase 1: 提交 gpu_check → 获取 baseline
    Phase 2: 提交 gpu_nccl_stress → 等待 30s → SIGKILL 强杀
    Phase 3: 提交 gpu_check → 获取 post
    Phase 4: 自身对比 → 报告污染

用法：
    cd train
    python tools/gpu_test/run_pollution_test.py
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import magnus

from config import (
    MAGNUS_ADDRESS,
    MAGNUS_TOKEN,
    SYSTEM_ENTRY_COMMAND,
    wait_for_job,
    parse_scan_json,
)

# ── constants ──────────────────────────────────────────────────

HERE = Path(__file__).parent
GPU_CHECK_SOURCE = (HERE / "gpu_check.py").read_text(encoding="utf-8")
GPU_STRESS_SOURCE = (HERE / "gpu_nccl_stress.py").read_text(encoding="utf-8")

GPU_COUNT = 6
GPU_TYPE = "a100"
STRESS_WAIT_SECONDS = 30
SENTINEL = "GPU_CHECK_JSON"

JOB_COMMON = {
    "namespace": "Rise-AGI",
    "repo_name": "OpenFundus",
    "gpu_count": GPU_COUNT,
    "gpu_type": GPU_TYPE,
    "cpu_count": 8,
    "memory_demand": "64G",
    "ephemeral_storage": "10G",
    "job_type": "A2",
    "system_entry_command": SYSTEM_ENTRY_COMMAND,
}


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _submit_gpu_check(task_name: str) -> str:
    """Submit a gpu_check job, return job_id."""
    entry_command = f"""python3 <<'PYEOF'
{GPU_CHECK_SOURCE}
PYEOF"""
    print(f"[{_ts()}] Submitting {task_name} ({GPU_COUNT}x {GPU_TYPE})...")
    job_id = magnus.submit_job(
        task_name=task_name,
        entry_command=entry_command,
        **JOB_COMMON,
    )
    print(f"[{_ts()}] {task_name} job_id: {job_id[:12]}...")
    return job_id


def _submit_stress(task_name: str) -> str:
    """Submit gpu_nccl_stress job, return job_id."""
    entry_command = f"""cat > /tmp/gpu_nccl_stress.py <<'PYEOF'
{GPU_STRESS_SOURCE}
PYEOF
export STRESS_DURATION=60
torchrun --nproc_per_node={GPU_COUNT} --rdzv_endpoint=127.0.0.1:29500 --rdzv_backend=c10d --max_restarts=0 /tmp/gpu_nccl_stress.py"""
    print(f"[{_ts()}] Submitting {task_name} ({GPU_COUNT}x {GPU_TYPE}, NCCL stress)...")
    job_id = magnus.submit_job(
        task_name=task_name,
        entry_command=entry_command,
        **JOB_COMMON,
    )
    print(f"[{_ts()}] {task_name} job_id: {job_id[:12]}...")
    return job_id


def _get_snapshot(job_id: str, phase_name: str) -> Optional[Dict[str, Any]]:
    """Wait for gpu_check job, parse sentinel JSON from logs, return snapshot."""
    print(f"\n[{_ts()}] [{phase_name}] Waiting for job {job_id[:12]}...")
    job = wait_for_job(job_id, poll_interval=15)

    status = job.get("status", "Unknown")
    print(f"[{_ts()}] [{phase_name}] Job status: {status}")

    # Get full logs and parse
    try:
        page = magnus.get_job_logs(job_id, page=0)
        logs = page.get("logs", "")
    except Exception as e:
        print(f"[{_ts()}] [{phase_name}] ERROR reading logs: {e}")
        return None

    snapshot_list = parse_scan_json(logs, sentinel=SENTINEL)
    if snapshot_list is None:
        print(f"[{_ts()}] [{phase_name}] ERROR: No {SENTINEL} sentinel in logs")
        print(f"[{_ts()}] [{phase_name}] Log tail:\n{logs[-2000:]}")
        return None

    # parse_scan_json returns json.loads() result — may be dict or list
    if isinstance(snapshot_list, dict):
        snapshot = snapshot_list
    elif isinstance(snapshot_list, list) and snapshot_list:
        snapshot = snapshot_list[0]
    else:
        print(f"[{_ts()}] [{phase_name}] ERROR: Unexpected JSON type: {type(snapshot_list)}")
        return None
    print(f"[{_ts()}] [{phase_name}] Snapshot parsed: "
          f"gpus={len(snapshot.get('gpus', {}))}, "
          f"cuda_context={snapshot.get('cuda_context', {})}")
    return snapshot


def _confirm_stress_running(job_id: str, phase_name: str) -> bool:
    """Poll logs until we see 'Iter 100' or timeout after 120s."""
    print(f"\n[{_ts()}] [{phase_name}] Waiting for stress to start (looking for 'Iter 100')...")
    start = time.time()
    while time.time() - start < 120:
        try:
            page = magnus.get_job_logs(job_id, page=0)
            logs = page.get("logs", "")
        except Exception:
            time.sleep(5)
            continue

        if "Iter 100" in logs:
            print(f"[{_ts()}] [{phase_name}] Stress confirmed running (Iter 100 seen)")
            return True
        if "NCCL init FAILED" in logs or "STRESS_COMPLETE" in logs:
            print(f"[{_ts()}] [{phase_name}] Stress ended unexpectedly:\n{logs[-500:]}")
            return False
        if "FAILED" in logs or "Error" in logs:
            # Could be pre-start info, don't bail yet
            pass

        job = magnus.get_job(job_id)
        status = job.get("status", "")
        if status in ("Failed", "Terminated", "Success"):
            print(f"[{_ts()}] [{phase_name}] Job ended before stress started: {status}")
            print(f"Logs:\n{logs[-1000:]}")
            return False

        time.sleep(5)

    print(f"[{_ts()}] [{phase_name}] TIMEOUT: stress did not start in 120s")
    return False


def _compare_snapshots(baseline: Dict, post: Dict):
    """Analyze differences between baseline and post-kill GPU snapshots."""
    print("\n" + "=" * 70)
    print("  PHASE 4: ANALYSIS — Baseline vs Post-Kill Comparison")
    print("=" * 70)

    issues = []

    # ── 1. GPU identity check ──
    base_gpus = baseline.get("gpus", {})
    post_gpus = post.get("gpus", {})
    base_ids = {idx: g.get("pci_bus_id", "") for idx, g in base_gpus.items()}
    post_ids = {idx: g.get("pci_bus_id", "") for idx, g in post_gpus.items()}

    # Map post indices → baseline indices by PCI bus ID
    id_to_post_idx = {pci: idx for idx, pci in post_ids.items()}
    matched = 0
    for base_idx, pci in base_ids.items():
        post_idx = id_to_post_idx.get(pci)
        if post_idx is not None:
            matched += 1

    print(f"\n─ GPU identity: {matched}/{GPU_COUNT} matched by PCI bus ID")
    if matched < GPU_COUNT:
        issues.append(f"GPU identity mismatch: only {matched}/{GPU_COUNT} matched")

    # ── 2. Memory comparison ──
    print(f"\n─ GPU Memory (used/total MiB):")
    memory_anomalies = 0
    for base_idx, base_g in base_gpus.items():
        pci = base_ids.get(base_idx, "")
        post_idx = id_to_post_idx.get(pci, "?")
        base_used = base_g.get("memory_used_mib", "?")
        post_g = post_gpus.get(post_idx, {})
        post_used = post_g.get("memory_used_mib", "?")

        flag = ""
        try:
            diff = int(post_used) - int(base_used)
            if diff > 100:
                flag = " *** HIGH (possible leak)"
                memory_anomalies += 1
            elif diff > 10:
                flag = " * elevated"
            elif diff < 0:
                flag = " (post < baseline)"
        except (ValueError, TypeError):
            flag = " (unparseable)"
        print(f"  GPU {base_idx} (post={post_idx}): {base_used} → {post_used}  Δ={flag.strip() if flag else '0'}")

    if memory_anomalies > 0:
        issues.append(f"Memory anomaly on {memory_anomalies} GPU(s): >100MB residual after kill")

    # ── 3. CUDA context test ──
    print(f"\n─ CUDA Context per GPU:")
    base_cuda = baseline.get("cuda_context", {})
    post_cuda = post.get("cuda_context", {})
    cuda_fails = 0
    for base_idx in sorted(base_cuda.keys(), key=int):
        pci = base_ids.get(base_idx, "")
        post_idx = id_to_post_idx.get(pci, base_idx)
        base_r = base_cuda.get(base_idx, {}).get("status", "?")
        post_r = post_cuda.get(post_idx, {}).get("status", "?")

        flag = ""
        if base_r == "OK" and post_r != "OK":
            flag = " *** POLLUTION DETECTED"
            cuda_fails += 1
        elif base_r == "OK" and post_r == "OK":
            flag = " OK"
        else:
            flag = f" (base={base_r}, post={post_r})"

        print(f"  GPU {base_idx} (post={post_idx}): {base_r} → {post_r}{flag}")

    if cuda_fails > 0:
        issues.append(f"CUDA context failure on {cuda_fails}/{GPU_COUNT} GPU(s) after kill")

    # ── 4. NCCL test (MOST CRITICAL) ──
    print(f"\n─ NCCL allreduce test:")
    base_nccl = baseline.get("nccl_test", {})
    post_nccl = post.get("nccl_test", {})

    base_nccl_status = base_nccl.get("status", "?")
    post_nccl_status = post_nccl.get("status", "?")

    print(f"  Baseline: {base_nccl_status}")
    print(f"  Post-kill: {post_nccl_status}")

    if base_nccl_status == "OK" and post_nccl_status != "OK":
        print(f"\n  *** NCCL POLLUTION CONFIRMED ***")
        print(f"  Post-kill stdout: {post_nccl.get('stdout', '')[:500]}")
        print(f"  Post-kill stderr: {post_nccl.get('stderr', '')[:500]}")
        issues.append("NCCL allreduce FAILED after kill — GPU state pollution confirmed")
    elif base_nccl_status == "OK" and post_nccl_status == "OK":
        print(f"  NCCL test passed both pre and post. May indicate SLURM epilog GPU reset exists.")

    # ── 5. /dev/shm NCCL orphan files ──
    print(f"\n─ /dev/shm nccl-* orphan files:")
    base_shm = baseline.get("dev_shm_nccl_files", [])
    post_shm = post.get("dev_shm_nccl_files", [])

    print(f"  Baseline: {len(base_shm)} file(s)")
    for f in base_shm:
        print(f"    {f}")
    print(f"  Post-kill: {len(post_shm)} file(s)")
    for f in post_shm:
        print(f"    {f}")

    new_orphans = len(post_shm) - len(base_shm)
    if new_orphans > 0:
        issues.append(f"NCCL /dev/shm orphan files: {new_orphans} new file(s) after kill")

    # ── 6. ECC errors ──
    print(f"\n─ ECC errors:")
    base_ecc = baseline.get("ecc_errors", {})
    post_ecc = post.get("ecc_errors", {})
    ecc_changes = 0
    for idx in sorted(base_ecc.keys(), key=int):
        base_v = base_ecc.get(idx, {}).get("Volatile", {}).get("Single Bit", "0") if isinstance(base_ecc.get(idx), dict) else "?"
        post_v = post_ecc.get(idx, {}).get("Volatile", {}).get("Single Bit", "0") if isinstance(post_ecc.get(idx), dict) else "?"
        if base_v != post_v:
            print(f"  GPU {idx}: {base_v} → {post_v} *** CHANGE")
            ecc_changes += 1
        else:
            print(f"  GPU {idx}: {base_v} (unchanged)")
    if ecc_changes > 0:
        issues.append(f"ECC error count changed on {ecc_changes} GPU(s)")

    # ── 7. SUMMARY ──
    print("\n" + "=" * 70)
    if issues:
        print("  RESULT: POLLUTION DETECTED")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("  RESULT: NO POLLUTION DETECTED")
        print("  (SLURM epilog GPU reset may be active, or GPU state was clean after SIGKILL)")
    print("=" * 70)

    return issues


def main():
    print("=" * 70)
    print("  GPU Pollution Test")
    print(f"  Cluster: {MAGNUS_ADDRESS}")
    print(f"  GPUs: {GPU_COUNT}x {GPU_TYPE}")
    print(f"  Stress duration: {STRESS_WAIT_SECONDS}s before SIGKILL")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    magnus.configure(address=MAGNUS_ADDRESS, token=MAGNUS_TOKEN)

    # ═══════════════════════════════════════════════════════════
    # Phase 1: BASELINE
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*50}\n  Phase 1: BASELINE\n{'─'*50}")
    baseline_job_id = _submit_gpu_check("GPU-Pollution-Baseline")
    baseline = _get_snapshot(baseline_job_id, "BASELINE")
    if baseline is None:
        print("FATAL: Cannot get baseline snapshot. Aborting.")
        sys.exit(1)
    baseline_ts = datetime.now()

    # ═══════════════════════════════════════════════════════════
    # Phase 2: STRESS + KILL
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*50}\n  Phase 2: NCCL STRESS + KILL\n{'─'*50}")
    stress_job_id = _submit_stress("GPU-Pollution-Stress")

    # Wait for stress to actually start
    running = _confirm_stress_running(stress_job_id, "STRESS")
    if not running:
        print("WARNING: Stress job did not start properly. Proceeding anyway...")

    # Wait STRESS_WAIT_SECONDS from now (stress has been running during our polling)
    print(f"\n[{_ts()}] [STRESS] Waiting {STRESS_WAIT_SECONDS}s before SIGKILL...")
    for remaining in range(STRESS_WAIT_SECONDS, 0, -10):
        print(f"[{_ts()}] [STRESS] {remaining}s remaining...")
        time.sleep(min(10, remaining))

    # Kill
    print(f"\n[{_ts()}] [STRESS] Sending SIGKILL via magnus.terminate_job()...")
    try:
        result = magnus.terminate_job(stress_job_id)
        print(f"[{_ts()}] [STRESS] Terminated: {result}")
    except Exception as e:
        print(f"[{_ts()}] [STRESS] Terminate error: {e}")

    kill_ts = datetime.now()
    print(f"[{_ts()}] [STRESS] Killed at {kill_ts.strftime('%H:%M:%S')}")

    # Short wait to let CG phase complete
    print(f"[{_ts()}] [STRESS] Waiting 10s for SLURM CG/epilog...")
    time.sleep(10)

    # ═══════════════════════════════════════════════════════════
    # Phase 3: POST-CHECK
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*50}\n  Phase 3: POST-CHECK\n{'─'*50}")
    post_job_id = _submit_gpu_check("GPU-Pollution-PostCheck")
    post = _get_snapshot(post_job_id, "POST")
    if post is None:
        print("FATAL: Cannot get post-kill snapshot.")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════
    # Phase 4: ANALYSIS
    # ═══════════════════════════════════════════════════════════
    issues = _compare_snapshots(baseline, post)

    # Print job IDs for manual inspection
    print(f"\nJob IDs for manual inspection:")
    print(f"  Baseline:  {baseline_job_id}")
    print(f"  Stress:    {stress_job_id}")
    print(f"  Post-check:{post_job_id}")

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
