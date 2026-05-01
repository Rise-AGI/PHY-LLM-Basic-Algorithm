"""
在 Magnus 集群上预热 SFT pip 包到持久存储 /data/。

用法：
    python warmup_packages.py
    python warmup_packages.py --address http://xxx:3011/ --token sk-xxx

原理：提交一个作业下载 wheel 包到 /tmp/ → cp -r 到 /data/<用户名>/pip-cache/wheels/。
后续蓝图作业通过 PIP_FIND_LINKS 使用本地缓存，零网络下载。
"""

import argparse
import os
import sys
from datetime import datetime

import magnus

from monitor import Monitor, record_storage, auto_source, notify_exe, SYSTEM_ENTRY_COMMAND

DEFAULT_ADDRESS = "http://162.105.151.134:3011/"
DEFAULT_TOKEN   = "sk-xxx"

# 预下载的包列表（SFT 常用依赖）
# 注意：accelerate 不在此列表，单独用 --no-deps 下载以避免拖入 torch + nvidia/CUDA (~3GB)
PACKAGES = [
    "transformers", "datasets", "pandas",
    "einops", "sentencepiece", "protobuf",
    "tokenizers", "safetensors", "numpy", "scipy", "pyarrow", "jinja2",
    "huggingface-hub", "tiktoken", "modelscope",
    "wandb", "psutil", "pyyaml", "click", "scikit-learn",
    "matplotlib", "seaborn",
    "requests",
]


def main():
    parser = argparse.ArgumentParser(description="预热 pip 包到集群持久存储")
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--token",   default=DEFAULT_TOKEN)
    args = parser.parse_args()

    magnus.configure(address=args.address, token=args.token)

    # 逐包下载：优先二进制 wheel，失败回退源码，再失败跳过
    download_cmds = "\n".join(
        f'echo "--- {pkg} ---" && '
        f'(pip download --only-binary :all: -d /tmp/pip-wheels {pkg} 2>/dev/null || '
        f'pip download -d /tmp/pip-wheels {pkg} 2>/dev/null || '
        f'echo "[跳过] {pkg}: 无法获取")'
        for pkg in PACKAGES
    )

    # accelerate 单独下载 -- 必须 --no-deps，否则会拖入 torch + 20+ nvidia/CUDA 包 (~3GB)
    # 容器镜像 pytorch:2.5.1-cuda12.4 已自带 torch/CUDA，accelerate 的其他依赖均已在 PACKAGES 中
    accelerate_cmd = (
        'echo "--- accelerate (--no-deps) ---" && '
        '(pip download --only-binary :all: --no-deps -d /tmp/pip-wheels accelerate 2>/dev/null || '
        'pip download --no-deps -d /tmp/pip-wheels accelerate 2>/dev/null || '
        'echo "[跳过] accelerate: 无法获取")'
    )

    entry_command = f"""
set -e

echo "=== 当前用户 ==="
whoami
USERNAME=$(whoami)
SAVE_DIR="/data/$USERNAME/pip-cache/wheels"
echo "缓存目录: $SAVE_DIR"

# 仅当完整标记文件存在时才跳过
if [ -f "$SAVE_DIR/.warmup_complete" ]; then
    echo "[跳过] 缓存已完整: $SAVE_DIR（$(ls "$SAVE_DIR"/*.whl 2>/dev/null | wc -l) 个 wheel）"
    exit 0
fi

# 清理上次失败留下的不完整缓存
if [ -d "$SAVE_DIR" ]; then
    echo "[清理] 删除上次不完整缓存: $SAVE_DIR"
    rm -rf "$SAVE_DIR"
fi

mkdir -p /tmp/pip-wheels "$SAVE_DIR"

echo "=== 开始下载 wheel 包到 /tmp/pip-wheels ==="
{download_cmds}
{accelerate_cmd}

# 清除 torch/nvidia/cuda/triton -- 容器镜像 (pytorch:2.5.1-cuda12.4) 已自带
# （安全网：accelerate 已用 --no-deps 跳过，但仍有其他包可能间接引入）
echo "=== 清理容器自带的大包 ==="
rm -f /tmp/pip-wheels/torch-*.whl \\
    /tmp/pip-wheels/nvidia_*.whl \\
    /tmp/pip-wheels/nvidia-*.whl \\
    /tmp/pip-wheels/triton-*.whl \\
    /tmp/pip-wheels/cuda_*.whl \\
    /tmp/pip-wheels/cuda-*.whl
echo "保留: $(find /tmp/pip-wheels -name '*.whl' 2>/dev/null | wc -l) 个 wheel"

echo "=== 磁盘用量 ==="
du -sh /tmp/pip-wheels 2>/dev/null || true
df -h /data 2>/dev/null | tail -1 || true

echo "=== 移动到持久目录 ==="
find /tmp/pip-wheels -name '*.whl' -exec cp {{}} "$SAVE_DIR/" \\;
echo "已复制 $(find "$SAVE_DIR" -name '*.whl' 2>/dev/null | wc -l) 个 wheel"

# 写入完整标记
echo "$(date -Iseconds)" > "$SAVE_DIR/.warmup_complete"
echo "=== 完成！缓存已保存到: $SAVE_DIR ==="
ls "$SAVE_DIR"/*.whl 2>/dev/null || echo "(无 wheel 文件)"
"""

    print(f"[1/3] 提交预热作业...")
    print(f"      包数量: {len(PACKAGES)} + accelerate(--no-deps)")
    print(f"      目标  : /data/<用户名>/pip-cache/wheels/")
    print()

    job_id = magnus.submit_job(
        task_name         = "Warmup-Pip-Cache",
        description       = f"预热 {len(PACKAGES)} 个 pip 包到持久存储",
        entry_command     = entry_command,
        system_entry_command = SYSTEM_ENTRY_COMMAND,
        namespace         = "Rise-AGI",
        repo_name         = "OpenFundus",
        gpu_count         = 0,
        gpu_type          = "cpu",
        cpu_count         = 4,
        memory_demand     = "80G",
        ephemeral_storage = "20G",
        job_type          = "A2",
    )
    print(f"      Job ID: {job_id}")

    notify_exe(job_id=job_id)
    Monitor(poll_interval=60, source=auto_source()).add(job_id).run()

    # 成功后记录持久存储
    job = magnus.get_job(job_id)
    if job.get("status") == "Success":
        record_storage("pip", {
            "time": datetime.now().isoformat(),
            "target": "/data/<用户名>/pip-cache/wheels",
            "packages": len(PACKAGES),
            "status": "success",
        })
        print(f"\n{'=' * 60}")
        print(f"预热完成！蓝图可通过 PIP_FIND_LINKS 使用本地缓存。")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
