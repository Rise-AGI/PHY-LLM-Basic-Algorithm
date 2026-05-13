"""
查询 Magnus 集群上的训练集数据，展示完整样本内容。

用法:
    python inspect_storage.py                           # 默认 3 条
    python inspect_storage.py --samples 5               # 显示 5 条
    python inspect_storage.py --data-path /data/magnus/training_data/train.json
    python inspect_storage.py --address http://xxx:3011/ --token sk-xxx

原理：提交一个低优先级 CPU 作业到集群，由容器内 Python 读取 JSON 数据文件，
返回完整样本内容到日志。本地轮询完成后打印结果。
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import magnus
from config import notify_exe, SYSTEM_ENTRY_COMMAND, MAGNUS_ADDRESS, MAGNUS_TOKEN

# 默认按优先级查找的训练数据路径列表
DEFAULT_DATA_PATHS = [
    "/data/magnus/training_data/train.json",
    "/data/magnus/training_data/test.json",
    "/data/magnus/training_data/train.jsonl",
    "/data/magnus/training_data/test.jsonl",
    "/data/magnus/training_data/*.parquet",
]

# 要查询的样本数
DEFAULT_SAMPLES = 3


INSPECT_PY = r'''
import json
import os
import glob as _glob
import random

# ── 以下变量由 entry_command 注入 ──
DATA_PATHS = []
NUM_SAMPLES = 3


def _fmt_len(obj, field):
    """安全获取字段长度"""
    val = obj.get(field, "")
    if val is None:
        return 0
    return len(str(val))


def _preview(text, max_chars=120):
    """截断预览"""
    t = str(text) if text else ""
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + f"...({len(t)} chars total)"


def inspect_file(path, num_samples):
    """读取并展示一个数据文件的完整样本。"""
    print()
    print("=" * 70)
    print(f"  文件: {path}")
    print("=" * 70)

    # ── 加载 ──
    if not os.path.exists(path):
        print(f"  [不存在]")
        return

    fsize = os.path.getsize(path)
    print(f"  文件大小: {fsize:,} bytes ({fsize/1024/1024:.1f} MB)")

    samples = []
    if path.endswith(".parquet"):
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            samples = df.to_dict(orient="records")
        except Exception as e:
            print(f"  [parquet 读取失败: {e}]")
            return
    elif path.endswith(".jsonl") or path.endswith(".jsonlines"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    else:  # .json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict):
            # 可能是 {"key": [...]} 或单条
            for v in data.values():
                if isinstance(v, list):
                    samples = v
                    break
            if not samples:
                samples = [data]

    print(f"  总样本数: {len(samples)}")

    if not samples:
        print("  [无样本]")
        return

    # ── 全局统计 ──
    fields = set()
    for s in samples[:500]:
        fields.update(k for k in s.keys())
    print(f"  字段: {', '.join(sorted(fields))}")

    # 统计各字段长度
    for fld in sorted(fields):
        lengths = [_fmt_len(s, fld) for s in samples[:500]]
        if lengths:
            avg_l = sum(lengths) / len(lengths)
            max_l = max(lengths)
            min_l = min(lengths)
            print(f"    {fld}: avg={avg_l:.0f}  min={min_l}  max={max_l} chars")

    # ── 选取样本 ──
    if num_samples >= len(samples):
        indices = list(range(len(samples)))
    else:
        # 均匀采样 + 首尾
        step = max(1, len(samples) // num_samples)
        indices = list(range(0, len(samples), step))[:num_samples]
        # 确保包含最后一条
        if indices[-1] != len(samples) - 1:
            indices[-1] = len(samples) - 1

    # ── 完整展示样本 ──
    for idx in indices:
        s = samples[idx]
        print()
        print(f"  {'─' * 60}")
        print(f"  [样本 #{idx}]  (共 {len(samples)} 条)")
        print(f"  {'─' * 60}")
        for k in sorted(s.keys()):
            val = s[k]
            if val is None:
                print(f"  {k}: (null)")
            elif isinstance(val, (int, float, bool)):
                print(f"  {k}: {val}")
            else:
                val_str = str(val)
                print(f"  {k} ({len(val_str)} chars):")
                # 完整输出，不截断
                print(f"  {val_str}")
        print()


def main():
    # 展开 glob 模式
    resolved_paths = []
    for p in DATA_PATHS:
        if "*" in p or "?" in p:
            resolved_paths.extend(sorted(_glob.glob(p)))
        else:
            resolved_paths.append(p)

    if not resolved_paths:
        print("[无数据路径配置]")
        return

    print("=" * 70)
    print("  训练数据检查")
    print(f"  检查 {len(resolved_paths)} 个文件，每文件展示 {NUM_SAMPLES} 条完整样本")
    print("=" * 70)

    for fp in resolved_paths:
        inspect_file(fp, NUM_SAMPLES)

    print()
    print("=" * 70)
    print("  检查完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
'''


def _build_entry_command(data_paths: list[str], num_samples: int) -> str:
    """构建 entry_command，注入数据路径和样本数到 INSPECT_PY。"""
    paths_str = json.dumps(data_paths)
    filled = INSPECT_PY.replace(
        "DATA_PATHS = []",
        f"DATA_PATHS = {paths_str}"
    ).replace(
        "NUM_SAMPLES = 3",
        f"NUM_SAMPLES = {num_samples}"
    )

    return (
        "set -e\n"
        "_log() { echo \"[$(date '+%Y-%m-%d %H:%M:%S')] $*\"; }\n\n"
        "_log \"开始检查训练数据...\"\n"
        "cat > /tmp/inspect_data.py << 'PYEOF'\n"
        + filled
        + "\nPYEOF\n"
        "python3 /tmp/inspect_data.py 2>&1\n"
        "echo '---DONE---'\n"
    )


def _poll_status(job_id: str, poll_interval: int = 30) -> dict:
    """轮询任务状态。"""
    last_status = None
    task_name = job_id[:8]

    while True:
        try:
            job = magnus.get_job(job_id)
        except Exception as exc:
            print(f"[{_ts()}] [{task_name}] 查询失败: {exc}")
            time.sleep(poll_interval)
            continue

        status: str = job.get("status", "Unknown")
        name: str = job.get("task_name", task_name)
        task_name = name

        if status != last_status:
            if last_status is None:
                print(f"[{_ts()}] [{task_name}] [{status}]")
            else:
                print(f"[{_ts()}] [{task_name}] [{last_status}] -> [{status}]")
            last_status = status
        else:
            if status not in ("Success", "Failed", "Terminated"):
                print(f"[{_ts()}] [{task_name}] [{status}] ...")

        if status in ("Success", "Failed", "Terminated"):
            return job

        time.sleep(poll_interval)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(
        description="查询 Magnus 集群上的训练集数据样本"
    )
    parser.add_argument("--address", default=MAGNUS_ADDRESS)
    parser.add_argument("--token", default=MAGNUS_TOKEN)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"每个文件展示的完整样本数 (default: {DEFAULT_SAMPLES})")
    parser.add_argument("--data-path", action="append", default=None,
                        help="指定数据文件路径（可多次使用）。默认: 自动探测常见路径")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="轮询间隔秒 (default: 30)")
    args = parser.parse_args()

    data_paths = args.data_path or DEFAULT_DATA_PATHS

    # ── 1. 配置 ──
    print(f"[1/3] 配置 Magnus 连接...")
    print(f"      地址: {args.address}")
    print(f"      Token: {args.token[:8]}...{args.token[-4:]}")
    magnus.configure(address=args.address, token=args.token)

    # ── 2. 提交 ──
    entry_command = _build_entry_command(data_paths, args.samples)
    print(f"[2/3] 提交训练数据检查作业...")
    print(f"      数据路径: {data_paths}")
    print(f"      每文件展示: {args.samples} 条完整样本")
    print()

    job_id = magnus.submit_job(
        task_name         = "Inspect-Training-Data",
        description       = f"查询训练数据: {data_paths}",
        entry_command     = entry_command,
        system_entry_command = SYSTEM_ENTRY_COMMAND,
        namespace         = "Rise-AGI",
        repo_name         = "OpenFundus",
        gpu_count         = 0,
        gpu_type          = "cpu",
        cpu_count         = 2,
        memory_demand     = "4G",
        ephemeral_storage = "10G",
        job_type          = "B2",
    )

    # ── 3. 通知 + 轮询 ──
    print(f"[3/3] 提交成功，Job ID: {job_id}")
    notify_exe(job_id=job_id)
    print(f"      每 {args.poll_interval}s 轮询状态...")
    print()
    job = _poll_status(job_id, args.poll_interval)
    print()

    # ── 4. 打印结果 ──
    if job.get("status") == "Success":
        print("=" * 60)
        print("  训练数据检查完成，完整样本如下：")
        print("=" * 60)
        all_logs = []
        page_num = 0
        while True:
            try:
                page = magnus.get_job_logs(job_id, page=page_num)
                text = page.get("logs", "").strip()
                if not text:
                    break
                all_logs.append(text)
                page_num += 1
            except Exception:
                break
        if all_logs:
            print("\n".join(all_logs))
        else:
            result = magnus.get_job_result(job_id)
            if result:
                print(result)
    else:
        print(f"任务未成功结束 (status={job.get('status')})")
        try:
            log_page = magnus.get_job_logs(job_id, page=0)
            text = log_page.get("logs", "").strip()
            if text:
                print(f"最后日志:\n{text[:3000]}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
