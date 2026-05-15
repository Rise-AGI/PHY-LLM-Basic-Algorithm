"""
6-GPU NCCL allreduce 密集通信压力测试 — 由 torchrun 启动 (--nproc_per_node=6)。

用法（容器内）：
    export STRESS_DURATION=60
    torchrun --nproc_per_node=6 gpu_nccl_stress.py

不注册 SIGTERM handler —— 验证默认行为下被 kill 的 GPU 状态残留。
"""
import os
import sys
import time
import torch
import torch.distributed as dist


def main():
    duration = int(os.environ.get("STRESS_DURATION", "60"))

    # Init NCCL
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=torch.distributed.timedelta(seconds=max(30, duration + 15)),
        )
    except Exception as e:
        print(f"[rank-?] NCCL init FAILED: {e}", flush=True)
        sys.exit(1)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"NCCL Stress: world_size={world_size}, duration={duration}s", flush=True)
        print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
        for i in range(world_size):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)", flush=True)

    # Allocate ~2GB per GPU
    tensor_size_mb = 2048
    elem_size = 4  # float32
    num_elements = (tensor_size_mb * 1024 * 1024) // elem_size
    # Make it roughly square for reasonable memory layout
    side = int(num_elements ** 0.5)
    num_elements = side * side

    try:
        tensor = torch.randn(side, side, device=device)
    except RuntimeError as e:
        if rank == 0:
            print(f"Tensor allocation failed ({side}x{side} float32 ≈ {num_elements*4/1024/1024:.0f}MB): {e}", flush=True)
        dist.destroy_process_group()
        sys.exit(1)

    if rank == 0:
        print(f"Allocated {num_elements * 4 / 1024 / 1024:.0f}MB per GPU", flush=True)

    # Barrier: all ranks ready before timing
    dist.barrier()
    if rank == 0:
        print("All ranks ready. Starting stress loop...", flush=True)

    t_start = time.time()
    iteration = 0

    try:
        while True:
            elapsed = time.time() - t_start
            if elapsed >= duration:
                break

            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            iteration += 1

            if rank == 0 and iteration % 100 == 0:
                elapsed = time.time() - t_start
                bw = (num_elements * 4 * iteration * 2) / (elapsed * 1e9)  # GB/s (x2 for send+recv)
                print(
                    f"[iter {iteration:6d}] elapsed={elapsed:6.1f}s  "
                    f"bw={bw:6.1f}GB/s  "
                    f"loop_time={(time.time()-t_start)*1000/iteration:.2f}ms/iter",
                    flush=True,
                )
    except KeyboardInterrupt:
        if rank == 0:
            print("Interrupted.", flush=True)
    except Exception as e:
        if rank == 0:
            print(f"Stress loop error: {e}", flush=True)
    finally:
        elapsed = time.time() - t_start
        if rank == 0:
            print(f"\nStress finished: {iteration} allreduces in {elapsed:.1f}s", flush=True)
            print("STRESS_COMPLETE", flush=True)

        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
