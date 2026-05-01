"""
轻量级持久存储连通性测试。

提交两个作业到 Magnus 集群：
  1. write_job:  在 /data/ 下写入一个标记文件
  2. read_job:   在 新作业 中检查该标记文件是否存在

目的：验证容器退出后 /data/ 的写入是否持久化（NFS 挂载正常）。
"""

import argparse
import sys
import time
from datetime import datetime

import magnus

from monitor import Monitor, auto_source, notify_exe, SYSTEM_ENTRY_COMMAND

DEFAULT_ADDRESS = "http://162.105.151.134:3011/"
DEFAULT_TOKEN   = "sk-xxx"


def main():
    parser = argparse.ArgumentParser(description="持久存储连通性测试")
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--token",   default=DEFAULT_TOKEN)
    args = parser.parse_args()

    magnus.configure(address=args.address, token=args.token)

    # ── Step 1: 写入一个标记文件 ──
    print("=" * 60)
    print("Step 1/2: 提交写入作业（写标记文件到 /data/）")
    print("=" * 60)

    write_cmd = """set -e
USERNAME=$(whoami)
MARKER_DIR="/data/$USERNAME/persist-test"
MARKER_FILE="$MARKER_DIR/hello.txt"
mkdir -p "$MARKER_DIR"
echo "Hello from job $(date -Iseconds)" > "$MARKER_FILE"
echo "=== 写入完成 ==="
echo "用户: $(whoami)"
echo "文件: $MARKER_FILE"
echo "内容: $(cat $MARKER_FILE)"
echo "---DONE---"
"""
    write_job_id = magnus.submit_job(
        task_name         = "PersistTest-Write",
        description       = "标记文件写入，验证 NFS 持久性",
        entry_command     = write_cmd,
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
    print(f"写入作业 ID: {write_job_id}")

    notify_exe(job_id=write_job_id)
    mon = Monitor(poll_interval=30, source=auto_source())
    mon.add(write_job_id)
    mon.run()

    write_job = magnus.get_job(write_job_id)
    if write_job.get("status") != "Success":
        print("[失败] 写入作业未成功结束")
        sys.exit(1)
    print("[OK] 写入作业成功\n")

    # ── Step 2: 在新作业中读取 ──
    print("=" * 60)
    print("Step 2/2: 提交读取作业（验证标记是否持久）")
    print("=" * 60)

    read_cmd = """set -e
USERNAME=$(whoami)
MARKER_FILE="/data/$USERNAME/persist-test/hello.txt"
echo "=== 检查标记文件 ==="
echo "路径: $MARKER_FILE"
if [ -f "$MARKER_FILE" ]; then
    echo "结果: 文件存在 — 持久存储正常工作！"
    echo "内容: $(cat $MARKER_FILE)"
else
    echo "结果: 文件不存在 — 持久存储可能有问题！"
fi
echo "=== /data/ 顶层面板 ==="
ls -la /data/$USERNAME/ 2>/dev/null || echo "(无 /data/ 目录或无权访问)"
echo "---DONE---"
"""
    read_job_id = magnus.submit_job(
        task_name         = "PersistTest-Read",
        description       = "验证标记文件是否跨作业持久",
        entry_command     = read_cmd,
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
    print(f"读取作业 ID: {read_job_id}")

    notify_exe(job_id=read_job_id)
    mon2 = Monitor(poll_interval=30, source=auto_source())
    mon2.add(read_job_id)
    mon2.run()

    read_job = magnus.get_job(read_job_id)
    if read_job.get("status") != "Success":
        print("[失败] 读取作业未成功结束")
        sys.exit(1)

    # 打印读取作业的结果
    result = magnus.get_job_result(read_job_id)
    if result:
        print(result)
    else:
        page = magnus.get_job_logs(read_job_id, page=0)
        print(page.get("logs", ""))

    # 判断最终结果（检查 result 和 logs，因为作业可能不显式设置 MAGNUS_RESULT）
    logs = magnus.get_job_logs(read_job_id, page=0).get("logs", "")
    persist_ok = (result and "文件存在" in result) or ("文件存在" in logs)
    if persist_ok:
        print("\n✅✅✅ 结论：持久存储正常工作！可以放心运行 warmup_packages.py")
    else:
        print("\n❌❌❌ 结论：持久存储有问题，需要调查后再运行 warmup_packages.py")


if __name__ == "__main__":
    main()
