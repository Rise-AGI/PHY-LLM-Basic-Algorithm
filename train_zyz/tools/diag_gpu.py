"""
GPU 诊断工具 — 在 Magnus 集群上提交 job 逐卡验证 GPU 状态。

用法：
    python tools/diag_gpu.py                          # 默认 6 卡
    python tools/diag_gpu.py --gpu-count 8            # 指定 GPU 数量
    python tools/diag_gpu.py --gpu-count 5 --timeout 120  # 自定义超时

用途：
    - GPU 故障后确认驱动状态是否恢复正常
    - nvidia-smi -r 或节点重启后验证修复生效
    - 排查 CUDA context 创建失败 / NCCL 不可用等问题
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import magnus

from config import (
    auto_source,
    notify_exe,
    SYSTEM_ENTRY_COMMAND,
    wait_for_job,
    MAGNUS_ADDRESS,
    MAGNUS_TOKEN,
)


GPU_CHECK_SCRIPT = r"""
set -e

echo "============================================"
echo "  GPU 诊断 v1.0"
echo "  $(date -Iseconds)"
echo "============================================"
echo ""

# ── Layer 0: nvidia-smi 基础信息 ──────────────────────────────
echo "=== [0/7] nvidia-smi 硬件清单 ==="
nvidia-smi --query-gpu=index,name,driver_version,pcie.link.gen.current,pcie.link.width.current,memory.total,temperature.gpu,power.draw --format=csv 2>&1 || echo "ERR: nvidia-smi 不可用"
echo ""

echo "=== [0/7] nvidia-smi 进程信息 ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || echo "ERR: 无法查询进程"
echo ""

echo "=== [0/7] nvidia-smi 拓扑 ==="
nvidia-smi topo -m 2>&1 || echo "ERR: 无法查询拓扑"
echo ""

# ── Layer 1: GSP-RM firmware ──────────────────────────────────
echo "=== [1/7] GSP-RM Firmware 状态 ==="
GSP_LINE=$(nvidia-smi -q 2>/dev/null | grep -i 'GSP' | head -1 || echo 'unknown')
echo "GSP: ${GSP_LINE:-unknown}"

# 同时检查内核参数
if [ -f /proc/driver/nvidia/params ]; then
    GSP_PARAM=$(grep GpuFirmware /proc/driver/nvidia/params 2>/dev/null || echo 'not found')
    echo "内核参数: ${GSP_PARAM}"
else
    echo "内核参数: /proc/driver/nvidia/params 不可用（容器内预期行为）"
fi
echo ""

# ── Layer 2: PyTorch CUDA 可用性 ──────────────────────────────
echo "=== [2/7] PyTorch CUDA 基础检测 ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA 编译版本: {torch.version.cuda}')
print(f'device_count(): {torch.cuda.device_count()}')
print(f'is_available(): {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'当前设备: {torch.cuda.current_device()}')
    print(f'设备名: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: torch.cuda.is_available() = False — CUDA context 创建失败！')
    print('这通常意味着 NVIDIA 驱动内部状态不一致（GPU context 残留）。')
"
echo ""

# ── Layer 3: 逐卡 CUDA context 创建 ───────────────────────────
echo "=== [3/7] 逐卡 CUDA context 创建测试 ==="
python3 -c "
import torch
import sys

total = torch.cuda.device_count()
if total == 0:
    print('FATAL: device_count() = 0，无 GPU 可用')
    sys.exit(0)

results = []
all_ok = True
for i in range(total):
    try:
        with torch.cuda.device(i):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            print(f'GPU {i} context: OK | {name} | CC {cap[0]}.{cap[1]}')
            results.append({'gpu': i, 'status': 'OK', 'name': name})
    except Exception as e:
        all_ok = False
        print(f'GPU {i} context: FAIL | {e}')
        results.append({'gpu': i, 'status': 'FAIL', 'error': str(e)})

print(f'')
print(f'逐卡 context 结果: {sum(1 for r in results if r[\"status\"]==\"OK\")}/{total} 可用')
if not all_ok:
    print('诊断: 部分/全部 GPU CUDA context 创建失败 → 驱动内部状态不一致')
    print('      需要宿主机 root 执行: nvidia-smi -r')
    sys.exit(1)
"
echo ""

# ── Layer 4: 逐卡 tensor 分配 + 小运算 ────────────────────────
echo "=== [4/7] 逐卡 CUDA tensor 分配 + 计算 ==="
python3 -c "
import torch
total = torch.cuda.device_count()
all_ok = True
for i in range(total):
    try:
        with torch.cuda.device(i):
            t = torch.randn(1024, 1024, device=f'cuda:{i}')
            s = t.sum().item()
            del t
            torch.cuda.synchronize(i)
            allocated = torch.cuda.memory_allocated(i) // (1024**2)
            print(f'GPU {i} tensor: OK | sum={s:.2f} | alloc={allocated} MiB')
    except Exception as e:
        all_ok = False
        print(f'GPU {i} tensor: FAIL | {e}')
print(f'')
print(f'逐卡 tensor 结果: {\"ALL OK\" if all_ok else \"PARTIAL/ALL FAIL\"} ')
"
echo ""

# ── Layer 5: NCCL 通信测试 ────────────────────────────────────
echo "=== [5/7] NCCL 通信测试 ==="
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")

if [ "$GPU_COUNT" -ge 2 ]; then
    # 简单双卡 all_reduce：超时 30s 防止卡死
    timeout 60 python3 -c "
import os
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '30'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29555'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'

import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl', init_method='env://',
                        world_size=1, rank=0)
for i in range(1, min($GPU_COUNT, 2)):
    # 单进程内多卡 NCCL 操作
    t0 = torch.randn(1024, device=f'cuda:0')
    t1 = torch.randn(1024, device=f'cuda:{i}')
    print(f'NCCL all_reduce GPU 0 <-> GPU {i}: OK (sum={t0.sum().item():.2f})')
dist.destroy_process_group()
print('NCCL 通信测试: PASS')
" 2>&1 || echo "NCCL 通信测试: FAIL (超时或错误，可能 NCCL 不可用)"
else
    echo "NCCL 通信测试: SKIP (可用 GPU < 2)"
fi
echo ""

# ── Layer 6: 僵尸进程检查 ─────────────────────────────────────
echo "=== [6/7] /dev/nvidia* 持有检查 ==="
if command -v fuser &> /dev/null; then
    NVIDIA_USERS=$(fuser /dev/nvidia* 2>/dev/null || echo 'none')
    echo "持有 /dev/nvidia* 的 PID: ${NVIDIA_USERS}"
    if [ "$NVIDIA_USERS" != "none" ] && [ -n "$NVIDIA_USERS" ]; then
        echo "WARNING: 仍有进程持有 GPU 设备文件！"
        ps -p $(echo $NVIDIA_USERS | tr ' ' '\n' | head -5) 2>/dev/null || true
    else
        echo "无进程持有 GPU 设备文件 ✓"
    fi
else
    echo "fuser 不可用，跳过"
fi
echo ""

# ── Layer 7: CUDA_VISIBLE_DEVICES ─────────────────────────────
echo "=== [7/7] 环境变量 ==="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-未设置}"
echo ""

# ── 最终摘要 ──────────────────────────────────────────────────
echo "============================================"
echo "  诊断完成"
echo "============================================"
python3 -c "
import torch
total = torch.cuda.device_count()
avail = torch.cuda.is_available()
ok_count = 0
if avail:
    for i in range(total):
        try:
            with torch.cuda.device(i):
                t = torch.zeros(1, device=f'cuda:{i}')
                del t
            torch.cuda.synchronize(i)
            ok_count += 1
        except:
            pass

print(f'总体: device_count={total}, is_available={avail}, 可用卡数={ok_count}')
if not avail:
    print('结论: GPU 驱动状态异常，CUDA context 无法创建')
    print('      需要宿主机 root 执行: nvidia-smi -r')
elif ok_count < total:
    print(f'结论: 部分 GPU 不可用 ({ok_count}/{total})')
else:
    print(f'结论: 全部 {total} 张 GPU 正常可用')
"
"""


def main():
    parser = argparse.ArgumentParser(
        description="GPU 诊断工具 — 在 Magnus 集群上提交 job 逐卡验证 GPU 状态",
    )
    parser.add_argument("--address", default=MAGNUS_ADDRESS)
    parser.add_argument("--token", default=MAGNUS_TOKEN)
    parser.add_argument(
        "--gpu-count", type=int, default=6,
        help="请求的 GPU 数量 (默认 6)",
    )
    parser.add_argument(
        "--gpu-type", default="a100",
        help="GPU 类型 (默认 a100)",
    )
    args = parser.parse_args()

    magnus.configure(address=args.address, token=args.token)

    source = auto_source()
    task_name = f"GPU-Diag-{source}" if source else "GPU-Diag"

    print(f"=== GPU 诊断工具 ===")
    print(f"  GPU: {args.gpu_count}×{args.gpu_type}")
    print(f"  Magnus: {args.address}")
    print(f"")

    print(f"[1/2] 提交 GPU 诊断 job...")
    job_id = magnus.submit_job(
        task_name=task_name,
        description=f"逐卡诊断 {args.gpu_count}×{args.gpu_type} GPU 状态",
        entry_command=GPU_CHECK_SCRIPT,
        system_entry_command=SYSTEM_ENTRY_COMMAND,
        namespace="Rise-AGI",
        repo_name="OpenFundus",
        gpu_count=args.gpu_count,
        gpu_type=args.gpu_type,
        cpu_count=8,
        memory_demand="32G",
        ephemeral_storage="10G",
        job_type="A2",
    )
    print(f"  Job ID: {job_id}")
    print(f"")

    notify_exe(job_id=job_id, task_name=task_name)

    print(f"[2/2] 等待诊断完成...")
    print(f"")
    job = wait_for_job(job_id, poll_interval=30)

    status = job.get("status", "Unknown")
    print(f"")
    print(f"=== 诊断 job 状态: {status} ===")

    if status == "Success":
        print(f"日志中查看 [7/7] 部分的最终结论。")
    else:
        print(f"诊断 job 未成功完成，请检查日志排查。")
        if status == "Failed":
            print(f"可能原因: GPU 全部不可用 / NCCL 死锁超时 / 镜像问题")


if __name__ == "__main__":
    main()
