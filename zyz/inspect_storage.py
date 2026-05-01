"""
检查 Magnus 集群的长期和短期存储目录结构。

用法:
    python inspect_storage.py
    python inspect_storage.py --address http://xxx:3011/ --token sk-xxx

原理：提交一个低优先级作业到集群，用 Python os.walk 遍历目录
并以树形图输出持久存储 (/data/) 和作业临时空间的文件布局。
本地每分钟轮询一次状态，完成后打印树形结果。
"""

import argparse
import time
from datetime import datetime

import magnus


DEFAULT_ADDRESS = "http://162.105.151.134:3011/"
DEFAULT_TOKEN   = "sk-xxx"  # 需修改 ===============

# ═══ 要检查的目录（按需修改）══════════════════════════════════
TREE_ROOTS = ["/data", "/tmp"]


TREE_PY = r'''
import os
import json
import stat as s

TREE_ROOTS = ["/data", "/tmp"]

def fmt_size(size):
    for unit in ("B", "K", "M", "G"):
        if size < 1024:
            return f"{size:.1f}{unit}" if unit != "B" else f"{size}B"
        size /= 1024
    return f"{size:.1f}T"

def walk_tree(root, prefix="", max_depth=4, max_children=50):
    """以树形格式打印目录结构，输出到 stdout（会被 Magnus 捕获）。"""
    if max_depth < 0:
        print(f"{prefix}└── ...")
        return
    try:
        names = sorted(os.listdir(root))
    except PermissionError:
        print(f"{prefix}[权限不足]")
        return
    except FileNotFoundError:
        print(f"{prefix}[不存在]")
        return

    # 先统计汇总信息
    total_size = 0
    file_count = 0
    dir_count = 0
    for name in names:
        fp = os.path.join(root, name)
        try:
            st = os.lstat(fp)
            if s.S_ISDIR(st.st_mode):
                dir_count += 1
            else:
                file_count += 1
                total_size += st.st_size
            if s.S_ISLNK(st.st_mode):
                pass  # 符号链接
        except OSError:
            pass

    summary = f"  [{dir_count} dirs, {file_count} files, {fmt_size(total_size)}]"
    print(f"{os.path.basename(root) or root}{summary}")

    names = names[:max_children]
    for i, name in enumerate(names):
        is_last = (i == len(names) - 1)
        connector = "└── " if is_last else "├── "
        fp = os.path.join(root, name)
        try:
            st = os.lstat(fp)
            if s.S_ISLNK(st.st_mode):
                # 符号链接
                link_target = os.readlink(fp)
                print(f"{prefix}{connector}{name} -> {link_target}")
            elif s.S_ISDIR(st.st_mode):
                print(f"{prefix}{connector}{name}/")
                ext = "    " if is_last else "│   "
                walk_tree(fp, prefix + ext, max_depth - 1, max_children)
            else:
                size_str = fmt_size(st.st_size)
                print(f"{prefix}{connector}{name}  ({size_str})")
        except OSError as e:
            print(f"{prefix}{connector}{name}  [错误: {e.strerror}]")

# ── 开始遍历 ──
print("=" * 60)
print("  集群存储树形结构")
print("=" * 60)

for r in TREE_ROOTS:
    print()
    walk_tree(r)
    print()

print("=" * 60)
print("  检查完成")
print("=" * 60)
'''


def _build_entry_command() -> str:
    """构建 entry_command，嵌入 TREE_PY 并输出树形结果。"""
    # 构造 TREE_ROOTS 的 Python 列表字符串
    roots_str = ", ".join(repr(p) for p in TREE_ROOTS)
    tree_py_filled = TREE_PY.replace('TREE_ROOTS = ["/data", "/tmp"]', f"TREE_ROOTS = [{roots_str}]")

    return (
        "set -e\n"
        "_log() { echo \"[$(date '+%Y-%m-%d %H:%M:%S')] $*\"; }\n\n"
        "_log \"开始存储检查...\"\n"
        "cat > /tmp/tree_walk.py << 'TREEOF'\n"
        + tree_py_filled
        + "\nTREEOF\n"
        "python3 /tmp/tree_walk.py 2>&1\n"
        "echo '---DONE---'\n"
    )


def _poll_status(job_id: str, poll_interval: int = 60) -> dict:
    """简单轮询任务状态，每分钟输出心跳，不打印日志。"""
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
                print(f"[{_ts()}] [{task_name}] [{last_status}] → [{status}]")
            last_status = status
        else:
            # 每分钟心跳
            if status not in ("Success", "Failed", "Terminated"):
                print(f"[{_ts()}] [{task_name}] [{status}] (运行中...)")

        if status in ("Success", "Failed", "Terminated"):
            return job

        time.sleep(poll_interval)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="检查 Magnus 集群存储目录")
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="轮询间隔（秒）")
    args = parser.parse_args()

    # ── 1. 配置 ──
    print(f"[1/3] 配置 Magnus 连接...")
    print(f"      地址: {args.address}")
    print(f"      Token: {args.token[:8]}...{args.token[-4:]}")
    magnus.configure(address=args.address, token=args.token)

    # ── 2. 提交 ──
    entry_command = _build_entry_command()
    print(f"[2/3] 提交存储检查作业...")
    print(f"      检查目录: {TREE_ROOTS}")
    print()

    job_id = magnus.submit_job(
        task_name         = "Inspect-Storage",
        description       = f"检查集群存储: {TREE_ROOTS}",
        entry_command     = entry_command,
        namespace         = "Rise-AGI",
        repo_name         = "OpenFundus",
        gpu_count         = 0,
        gpu_type          = "cpu",
        cpu_count         = 2,
        memory_demand     = "4G",
        ephemeral_storage = "10G",
        job_type          = "B2",
    )

    # ── 3. 通知 EXE + 轮询 ──
    print(f"[3/3] 提交成功，Job ID: {job_id}")
    print(f"      每 {args.poll_interval}s 轮询状态...")
    print()
    job = _poll_status(job_id, args.poll_interval)
    print()

    # ── 4. 完成后打印树形结果 ──
    if job.get("status") == "Success":
        print("=" * 60)
        print("  存储检查完成，树形结构如下：")
        print("=" * 60)
        # 获取所有日志页面
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
            # 退回到 result
            result = magnus.get_job_result(job_id)
            if result:
                print(result)
    else:
        print(f"任务未成功结束 (status={job.get('status')})")
        # 即使失败也打印日志
        try:
            log_page = magnus.get_job_logs(job_id, page=0)
            text = log_page.get("logs", "").strip()
            if text:
                print(f"最后日志:\n{text[:2000]}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
