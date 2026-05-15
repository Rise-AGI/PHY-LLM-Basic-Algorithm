"""
GPU 状态快照脚本 — 在 Magnus 容器内运行（嵌入 entry_command）。
输出 Sentinel JSON：=== GPU_CHECK_JSON === ... === END_GPU_CHECK_JSON ===
"""
import json
import os
import subprocess
import sys
import time


def run(cmd, timeout=30):
    """Run shell command, return (rc, stdout, stderr)."""
    try:
        p = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT ({timeout}s)"
    except Exception as e:
        return -1, "", str(e)


def check_nvidia_smi():
    """Query GPU hardware inventory."""
    rc, out, err = run(
        "nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.total,"
        "utilization.gpu,temperature.gpu,power.draw,pcie.link.gen.current,"
        "clocks.sm,clocks.mem --format=csv,noheader 2>&1"
    )
    gpus = {}
    if rc != 0:
        return {"error": f"nvidia-smi failed: {err}"}, gpus

    headers = [
        "index", "name", "pci_bus_id", "memory_used_mib", "memory_total_mib",
        "utilization_pct", "temperature_c", "power_draw_w",
        "pcie_gen", "clock_sm_mhz", "clock_mem_mhz",
    ]
    for line in out.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(headers):
            continue
        gpu = dict(zip(headers, parts))
        idx = gpu["index"]
        gpus[idx] = gpu
    return None, gpus


def check_compute_processes():
    """Query running compute processes on GPUs."""
    rc, out, err = run(
        "nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_bus_id "
        "--format=csv,noheader 2>&1"
    )
    if rc != 0:
        return []
    procs = []
    for line in out.split("\n"):
        line = line.strip()
        if line and not line.startswith("No running"):
            procs.append(line)
    return procs


def check_ecc_errors(gpu_count):
    """Query ECC error counters per GPU."""
    ecc = {}
    for i in range(gpu_count):
        rc, out, err = run(f"nvidia-smi -q -i {i} 2>&1 | grep -A 10 'ECC Errors'", timeout=15)
        if rc == 0 and out:
            lines = out.strip().split("\n")
            parsed = {}
            for line in lines:
                line = line.strip()
                if ":" in line:
                    k, v = line.split(":", 1)
                    parsed[k.strip()] = v.strip()
            ecc[str(i)] = parsed
        else:
            ecc[str(i)] = {"error": err or "no ECC info"}
    return ecc


def check_cuda_context(gpu_count):
    """Per-GPU CUDA context creation + tensor allocation test."""
    import torch

    results = {}
    for i in range(gpu_count):
        try:
            with torch.cuda.device(i):
                t = torch.zeros(1, device="cuda")
                # 做一次计算验证不仅仅是 alloc
                t += 1
                del t
            torch.cuda.synchronize(i)
            results[str(i)] = {"status": "OK"}
        except Exception as e:
            results[str(i)] = {"status": "FAIL", "error": str(e)[:200]}
    return results


def check_dev_shm_nccl():
    """List /dev/shm NCCL orphan files."""
    rc, out, err = run("ls -la /dev/shm/nccl-* 2>/dev/null || echo 'NONE'", timeout=10)
    if "NONE" in out or "No such file" in err:
        return []
    files = []
    for line in out.split("\n"):
        line = line.strip()
        if line and not line.startswith("total") and "nccl-" in line:
            files.append(line)
    return files


def check_gpu_topo():
    """GPU topology map."""
    rc, out, err = run("nvidia-smi topo -m 2>&1", timeout=15)
    if rc != 0:
        return {"error": err}
    return {"topology": out}


def check_nccl_allreduce(gpu_count):
    """NCCL allreduce test in a SUBPROCESS to avoid polluting this process's CUDA context."""
    nccl_test_script = f'''import os
import sys
import time
import torch
import torch.distributed as dist

def main():
    try:
        dist.init_process_group(backend="nccl", init_method="env://", timeout=torch.distributed.timedelta(seconds=25))
    except Exception as e:
        print(f"NCCL_INIT_FAIL: {{e}}")
        sys.exit(1)

    rank = dist.get_rank()
    device = torch.device(f"cuda:{{rank}}")

    try:
        t = torch.randn(128, 128, device=device)
        start = time.time()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        dist.barrier()
        elapsed = time.time() - start
        if rank == 0:
            print(f"NCCL_TEST_OK: allreduce+barr {{elapsed*1000:.1f}}ms")
    except Exception as e:
        if rank == 0:
            print(f"NCCL_TEST_FAIL: {{e}}")
        sys.exit(1)
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass

if __name__ == "__main__":
    main()
'''

    # Write temp script
    tmp_path = "/tmp/_gpu_check_nccl_test.py"
    try:
        with open(tmp_path, "w") as f:
            f.write(nccl_test_script)
    except Exception as e:
        return {"status": "FAIL", "error": f"cannot write tmp script: {e}"}

    # Run via torchrun
    rc, out, err = run(
        f"torchrun --nproc_per_node={gpu_count} --rdzv_endpoint=127.0.0.1:29500 "
        f"--rdzv_backend=c10d --max_restarts=0 {tmp_path} 2>&1",
        timeout=35,
    )

    # Cleanup
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    result = {"status": "UNKNOWN", "rc": rc, "stdout": out[:2000], "stderr": err[:2000]}
    if "NCCL_TEST_OK" in out:
        result["status"] = "OK"
    elif "NCCL_TEST_FAIL" in out or "NCCL_INIT_FAIL" in out:
        result["status"] = "FAIL"
    elif "ncclSystemError" in err or "ncclSystemError" in out:
        result["status"] = "FAIL (ncclSystemError)"
    elif "unhandled cuda error" in (out + err).lower():
        result["status"] = "FAIL (unhandled CUDA error)"
    elif rc == -1:
        result["status"] = "TIMEOUT"
    elif rc != 0:
        result["status"] = f"FAIL (rc={rc})"

    return result


def main():
    import torch

    print("=" * 60)
    print("  GPU State Check")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Host: {os.uname().nodename}")
    print("=" * 60)

    gpu_count = torch.cuda.device_count()
    print(f"\nCUDA device_count: {gpu_count}")

    if gpu_count == 0:
        result = {"error": "No CUDA devices found"}
        print(f"\n=== GPU_CHECK_JSON ===\n{json.dumps(result, indent=2)}\n=== END_GPU_CHECK_JSON ===")
        return

    snapshot = {
        "timestamp": time.time(),
        "hostname": os.uname().nodename,
        "cuda_device_count": gpu_count,
        "cuda_available": torch.cuda.is_available(),
    }

    # 1. nvidia-smi
    print("\n[1/7] nvidia-smi hardware inventory...")
    err, gpus = check_nvidia_smi()
    snapshot["nvidia_smi_error"] = err
    snapshot["gpus"] = gpus

    # 2. Compute processes
    print("[2/7] Compute processes...")
    snapshot["compute_processes"] = check_compute_processes()

    # 3. ECC errors
    print("[3/7] ECC errors...")
    snapshot["ecc_errors"] = check_ecc_errors(gpu_count)

    # 4. CUDA context per GPU
    print("[4/7] Per-GPU CUDA context test...")
    snapshot["cuda_context"] = check_cuda_context(gpu_count)

    # 5. Empty CUDA cache before NCCL test
    print("[5/7] Emptying CUDA cache...")
    torch.cuda.empty_cache()

    # 6. NCCL allreduce test (SUB PROCESS)
    print("[6/7] NCCL allreduce test (subprocess)...")
    snapshot["nccl_test"] = check_nccl_allreduce(gpu_count)

    # 7. /dev/shm
    print("[7/7] /dev/shm NCCL files...")
    snapshot["dev_shm_nccl_files"] = check_dev_shm_nccl()

    # Bonus: topology
    snapshot["topo"] = check_gpu_topo()

    # Per-GPU summary
    summary = {}
    for i in range(gpu_count):
        idx = str(i)
        gpu_info = gpus.get(idx, {})
        cuda_r = snapshot["cuda_context"].get(idx, {})
        summary[idx] = {
            "name": gpu_info.get("name", "?"),
            "memory_used_mib": gpu_info.get("memory_used_mib", "?"),
            "cuda_context": cuda_r.get("status", "?"),
            "cuda_error": cuda_r.get("error", ""),
        }
    snapshot["per_gpu_summary"] = summary

    # Output sentinel JSON
    print(f"\n=== GPU_CHECK_JSON ===")
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    print(f"=== END_GPU_CHECK_JSON ===")


if __name__ == "__main__":
    main()
