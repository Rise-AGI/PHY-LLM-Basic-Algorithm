"""
Download training results from a completed Magnus job.

Usage:
    python download_results.py <job_id>
    python download_results.py <job_id> --output ./my_results/
    python download_results.py <job_id> --address http://... --token sk-...

Downloads:
    1. training_log.json       — 训练 loss 日志
    2. eval_results_initial.json — 训练前生成式评估 (v6)
    3. eval_results_final.json   — 训练后生成式评估 (v6)
    4. eval_results.json         — eval-only 推理结果
"""

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import magnus

from config import MAGNUS_ADDRESS, MAGNUS_TOKEN
HERE = Path(__file__).resolve().parent


def extract_secrets(text: str) -> dict:
    """Extract all magnus receive commands and their secrets from log text."""
    secrets = {}
    for m in re.finditer(
        r"magnus receive\s+(\S+)\s+-o\s+(\S+)",
        text,
    ):
        secret = m.group(1)
        path_hint = m.group(2)
        name = Path(path_hint).name
        secrets[name] = secret
    return secrets


def download_all(job_id: str, output_dir: Path, address: str, token: str):
    magnus.configure(address=address, token=token)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch job info
    print(f"[1/4] 查询任务 {job_id} ...")
    job = magnus.get_job(job_id)
    status = job.get("status", "Unknown")
    task_name = job.get("task_name", job_id[:8])
    print(f"       任务: {task_name}")
    print(f"       状态: {status}")

    # 2. Fetch logs to extract secrets
    print(f"[2/4] 获取日志，提取下载凭证...")
    found = {}
    for page in range(5):  # scan last 5 pages
        try:
            log_data = magnus.get_job_logs(job_id, page=page)
            text = log_data.get("logs", "")
            if text:
                secrets = extract_secrets(text)
                found.update(secrets)
        except Exception:
            break

    if found:
        print(f"       发现 {len(found)} 个文件:")
        for name, secret in found.items():
            print(f"         {name}  ->  {secret[:40]}...")
    else:
        print("       未在日志中找到 magnus receive 命令")
        print("       尝试从 job result 获取...")

    # 3. Download each file
    print(f"[3/4] 下载文件到 {output_dir}/ ...")
    downloaded = []

    for name, secret in found.items():
        dest = output_dir / name
        try:
            print(f"       下载 {name} ...")
            magnus.download_file(secret, str(dest))
            size = dest.stat().st_size
            print(f"         ✓ {name} ({size:,} bytes)")
            downloaded.append(name)
        except Exception as e:
            print(f"         ✗ {name}: {e}")

    # 4. Try job result (fallback)
    if not downloaded:
        print(f"[4/4] 尝试 job result...")
        try:
            result = magnus.get_job_result(job_id)
            if result:
                # result may be JSON or a magnus-secret
                if isinstance(result, str) and result.startswith("magnus-secret:"):
                    dest = output_dir / "job_result"
                    magnus.download_file(result, str(dest))
                    print(f"        ✓ job_result ({dest.stat().st_size:,} bytes)")
                    downloaded.append("job_result")
                else:
                    dest = output_dir / "job_result.json"
                    dest.write_text(str(result), encoding="utf-8")
                    print(f"        ✓ job_result.json")
                    downloaded.append("job_result.json")
        except Exception as e:
            print(f"        ✗ {e}")

    # Summary
    print()
    if downloaded:
        print(f"完成! 已下载 {len(downloaded)} 个文件到 {output_dir}/")
        for n in downloaded:
            print(f"  {output_dir / n}")
    else:
        print("未能下载任何文件。")
        print()
        print("可能原因:")
        print("  1. 任务尚未完成 (当前状态: {})".format(status))
        print("  2. 日志已被滚动清除 (容器已被回收)")
        print("  3. _on_exit trap 未执行 (任务被 Terminated)")
        print()
        print("如果文件在集群 /data 路径下，可以尝试直接访问:")
        print("  ssh 到集群节点后 cp 文件到可访问位置")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="下载 Magnus 训练任务的输出文件"
    )
    parser.add_argument("job_id", help="Magnus job ID")
    parser.add_argument("--output", "-o", default=None,
                        help="下载目录 (默认: ./downloads/<job_id>/)")
    parser.add_argument("--address", default=MAGNUS_ADDRESS)
    parser.add_argument("--token", default=MAGNUS_TOKEN)
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path(HERE) / "downloads" / args.job_id[:12]
    download_all(args.job_id, output_dir, args.address, args.token)


if __name__ == "__main__":
    main()
