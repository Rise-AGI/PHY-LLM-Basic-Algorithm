"""
删除 Magnus 集群长期存储中的文件/目录。

原理：提交一个轻量 job，通过 system_entry_command 挂载 /data/，
然后在容器内执行 rm -rf 删除指定路径。

用法：
    python remove_storage.py /data/magnus/models/old-model
    python remove_storage.py /data/magnus/pip-cache --yes
    python remove_storage.py /data/magnus/models/Qwen2.5-72B-Instruct -y
"""

import argparse
import sys
from datetime import datetime

import magnus

from monitor import Monitor, auto_source, notify_exe, SYSTEM_ENTRY_COMMAND

DEFAULT_ADDRESS = "http://162.105.151.134:3011/"
DEFAULT_TOKEN   = "sk-xxx"


def main():
    parser = argparse.ArgumentParser(
        description="删除 Magnus 集群长期存储中的文件/目录",
        epilog="示例: python remove_storage.py /data/magnus/models/old-model -y"
    )
    parser.add_argument("target", help="要删除的路径（如 /data/magnus/models/xxx）")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认，直接删除")
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--token",   default=DEFAULT_TOKEN)
    args = parser.parse_args()

    target = args.target.rstrip("/")
    if not target.startswith("/data/"):
        print(f"[错误] 只能删除 /data/ 下的路径: {target}")
        sys.exit(1)

    # ── 确认 ──────────────────────────────────────────────
    if not args.yes:
        print(f"即将删除: {target}")
        print(f"此操作不可逆！")
        confirm = input("确认删除？[y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("已取消。")
            return

    # ── 提交删除作业 ─────────────────────────────────────
    magnus.configure(address=args.address, token=args.token)

    entry_command = f"""set -e
TARGET="{target}"

echo "=== 删除目标 ==="
echo "路径: $TARGET"

if [ ! -e "$TARGET" ]; then
    echo "[跳过] 路径不存在: $TARGET"
    exit 0
fi

if [ -f "$TARGET" ]; then
    TYPE="文件"
    SIZE="$(ls -lh "$TARGET" | awk '{{print $5}}')"
    rm -f "$TARGET"
elif [ -d "$TARGET" ]; then
    TYPE="目录"
    SIZE="$(du -sh "$TARGET" 2>/dev/null | cut -f1)"
    rm -rf "$TARGET"
else
    echo "[错误] 无法判断类型: $TARGET"
    exit 1
fi

if [ -e "$TARGET" ]; then
    echo "[失败] 删除失败: $TARGET"
    exit 1
else
    echo "[成功] 已删除: $TARGET (类型=${{TYPE}}, 大小=${{SIZE}})"
fi
"""

    print(f"提交删除作业...")
    print(f"  目标: {target}")
    print()

    job_id = magnus.submit_job(
        task_name         = f"Remove-{target.split('/')[-1]}",
        description       = f"删除: {target}",
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
    print(f"  Job ID: {job_id}")

    notify_exe(job_id=job_id)
    Monitor(poll_interval=30, source=auto_source()).add(job_id).run()

    job = magnus.get_job(job_id)
    if job.get("status") == "Success":
        print(f"\n已删除: {target}")
    else:
        print(f"\n删除失败，查看日志: {job_id}")


if __name__ == "__main__":
    main()
