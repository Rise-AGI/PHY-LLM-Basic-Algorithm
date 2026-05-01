"""
Magnus 任务监控模块

可独立运行:
    python monitor.py <job_id1> [job_id2 ...]
    python monitor.py --address http://xxx:3011/ --token sk-xxx <job_id>

也可作为模块导入:
    from monitor import Monitor

    monitor = Monitor(poll_interval=60, source="wp")  # source 用于日志文件命名
    monitor.add("abc123")
    monitor.run()
"""

import io
import os
import re
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Optional, Any

import magnus


HERE = os.path.dirname(os.path.abspath(__file__))
DATA1_DIR = os.path.join(HERE, "data1")          # 日志文件
DATA2_DIR = os.path.join(HERE, "data2")          # 时间线 + 记录
STORAGE_RECORD_PATH = os.path.join(DATA2_DIR, "storage_record.json")
SFT_DATA_DIR = os.path.join(HERE, "SFT_data")

# Magnus 平台更新后不再默认挂载，需显式通过 system_entry_command 声明
# 格式: "宿主机路径:容器路径"，用 APPTAINER_BIND 传给 Apptainer
# system_entry_command 在宿主机上、容器启动前执行；如集群有默认值则 job 级覆盖
SYSTEM_ENTRY_COMMAND = """mounts=(
    "/home:/home"
    "/data:/data"
)
export APPTAINER_BIND=$(IFS=,; echo "${mounts[*]}")
export MAGNUS_HOME=/magnus
unset -f nvidia-smi
unset VIRTUAL_ENV SSL_CERT_FILE
"""


# ── 持久存储记录读写 ─────────────────────────────────────

def _ensure_record() -> dict:
    """读取 storage_record.json，不存在则返回空结构。"""
    if not os.path.exists(STORAGE_RECORD_PATH):
        return {"pip": [], "modelscope": [], "model-version": []}
    with open(STORAGE_RECORD_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_record(record: dict) -> None:
    os.makedirs(DATA2_DIR, exist_ok=True)
    with open(STORAGE_RECORD_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def record_storage(category: str, entry: dict) -> None:
    """追加一条存储记录。category: pip / modelscope / model-version"""
    record = _ensure_record()
    if category not in record:
        record[category] = []
    record[category].append(entry)
    _save_record(record)


def check_model_version_exists(model_version: str) -> bool:
    """检查 model-version 是否已存在 (用于提交前去重)。"""
    record = _ensure_record()
    for entry in record.get("model-version", []):
        if entry.get("model") == model_version:
            return True
    return False


# ── 提交者文件缩写对照表 ─────────────────────────────────────

SOURCE_ABBR = {
    "warmup_packages.py": "wp",
    "download_model_auto.py": "dma",
    "submit_sft.py": "ss",
    "magnus_sft.py": "ms",
    "monitor.py": "mo",
    "inspect_storage.py": "is",
    "remove_storage.py": "rs",
}


def auto_source() -> Optional[str]:
    """根据调用者文件名自动返回缩写，未匹配则返回 None。"""
    import inspect
    caller = inspect.getframeinfo(inspect.stack()[1].frame)
    return SOURCE_ABBR.get(os.path.basename(caller.filename))


# ── 默认连接 ─────────────────────────────────────────────

DEFAULT_ADDRESS = "http://162.105.151.134:3011/"
DEFAULT_TOKEN   = "sk-xxx"


def notify_exe(job_id, task_name=None, submitter=None, address=None, token=None):
    """通知 Magnus Monitor EXE 跟踪一个新 job。

    submit_job() / launch_blueprint() 成功后立即调用。
    自动推断 submitter（inspect 调用栈）、address/token（模块默认值）。
    EXE 不在线 → 写入 data2/incoming/*.job.json 作为 fallback。
    """
    import inspect

    caller_file = inspect.getframeinfo(inspect.stack()[1].frame).filename
    caller_name = os.path.basename(caller_file)

    payload = {
        "job_id": job_id,
        "task_name": task_name or job_id[:12],
        "submitter": submitter or caller_name,
        "address": address or DEFAULT_ADDRESS,
        "token": token or DEFAULT_TOKEN,
    }

    try:
        import urllib.request, urllib.error
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            "http://localhost:9876/api/job",
            data=data, headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
        return
    except Exception:
        pass

    incoming_dir = os.path.join(HERE, "data2", "incoming")
    os.makedirs(incoming_dir, exist_ok=True)
    safe_name = re.sub(r'[^\w-]', '_', payload["task_name"])[:40]
    fpath = os.path.join(incoming_dir, f"{job_id[:12]}-{safe_name}.job.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"[notify_exe] EXE 未运行，已保存到 {fpath}")


class Monitor:
    """监控一个或多个 Magnus 任务的状态和日志输出。"""

    _STATUS_LETTER = {
        "Success": "s",
        "Failed": "f",
        "Terminated": "t",
    }
    _global_seq = 0  # 全局序列号，跨 Monitor 实例递增

    def __init__(self, poll_interval: int = 60, source: Optional[str] = None):
        self.poll_interval = poll_interval
        self.source = source
        self._jobs: Dict[str, dict] = {}
        self._start_time = time.time()
        self._buffer = io.StringIO()
        # log_cache: {job_id -> 上一轮 page 0 原文}，用于增量 diff
        self._log_cache: Dict[str, str] = {}

    @classmethod
    def _next_seq(cls) -> int:
        cls._global_seq += 1
        return cls._global_seq

    @staticmethod
    def _format_header(filename, time_str, task_name, source, job_id, log_type,
                       status, created_at, seq, submitter="") -> str:
        """生成 BibTeX 风格的元数据头，供 data1/data2 文件使用。"""
        return (
            f"@{{{filename},\n"
            f"  time = {time_str},\n"
            f"  name = {task_name},\n"
            f"  submitter = {submitter or source},\n"
            f"  job_id = {job_id},\n"
            f"  type = {log_type},\n"
            f"  status = {status},\n"
            f"  source_abbr = {source},\n"
            f"  seq = {seq:03d},\n"
            f"  created_at = {created_at},\n"
            f"  updated_at = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')},\n"
            f"}}\n"
        )

    # ── 公开接口 ──────────────────────────────────────────

    def add(self, job_id: str) -> "Monitor":
        if job_id not in self._jobs:
            self._jobs[job_id] = dict(
                last_status=None,
                last_pages=0,
                task_name="",
            )
        return self

    def add_many(self, *job_ids: str) -> "Monitor":
        for jid in job_ids:
            self.add(jid)
        return self

    def run(self) -> None:
        self._print(f"[监控] poll_interval={self.poll_interval}s | 跟踪 {len(self._jobs)} 个任务")
        self._print(f"[监控] {'─' * 50}")
        try:
            while True:
                alive = self._poll_once()
                if not alive:
                    break
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            self._print(f"\n[监控] 用户中断 (已运行 {self._elapsed():.0f}s)")
        self._print(f"[监控] 结束 (共运行 {self._elapsed():.0f}s)")
        self._save_logs()

    # ── 内部 ──────────────────────────────────────────────

    def _elapsed(self) -> float:
        return time.time() - self._start_time

    def _print(self, *args, **kwargs) -> None:
        print(*args, **kwargs)
        print(*args, file=self._buffer, **kwargs)

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _save_logs(self) -> None:
        if not self.source:
            return
        content = self._buffer.getvalue().strip()
        if not content:
            return

        os.makedirs(DATA1_DIR, exist_ok=True)

        for job_id, state in self._jobs.items():
            task_name = state.get("task_name", job_id[:8])
            safe = re.sub(r'[^\w-]', '_', task_name)

            # 使用任务提交时间 (created_at)，fallback 到当前时间
            created_raw = state.get("created_at", "")
            if created_raw:
                try:
                    dt = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                    ts = dt.strftime("%Y%m%d-%H%M%S")
                    time_friendly = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    time_friendly = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                time_friendly = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            letter = state.get("final_status_letter", "u")  # s/f/t 或 u(未知)
            status_name = {v: k for k, v in self._STATUS_LETTER.items()}.get(letter, "Unknown")
            seq = self._next_seq()
            filename = f"{ts}-{letter}-{self.source}-{safe}-{seq:03d}.data1"
            filepath = os.path.join(DATA1_DIR, filename)

            header = self._format_header(
                filename=filename,
                time_str=time_friendly,
                task_name=task_name,
                source=self.source,
                job_id=job_id,
                log_type="日志",
                status=status_name,
                created_at=created_raw or "N/A",
                seq=seq,
            )
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(header + "\n")
                f.write(content)
            self._print(f"[监控] 日志保存: {filepath}")
            break  # 只存第一份

    def _poll_once(self) -> bool:
        still_running = False

        for job_id in list(self._jobs.keys()):
            state = self._jobs[job_id]
            ts = self._ts()

            try:
                job = magnus.get_job(job_id)
            except Exception as exc:
                self._print(f"[{ts}] [{job_id[:8]}] 查询失败: {exc}")
                still_running = True
                continue

            status: str = job.get("status", "Unknown")
            task_name: str = job.get("task_name", job_id[:8])
            state["task_name"] = task_name
            # 保留 created_at 用于日志文件命名
            state["created_at"] = state.get("created_at") or job.get("created_at", "")

            # ── 状态输出 ──
            old = state["last_status"]
            if status != old:
                if old is None:
                    self._print(f"[{ts}] [{task_name}] [{status}]")
                else:
                    self._print(f"[{ts}] [{task_name}] [{old}] → [{status}]")
                state["last_status"] = status
            else:
                # 状态未变时也输出心跳，避免用户以为卡死
                if status not in self._STATUS_LETTER:  # 非终态才心跳
                    self._print(f"[{ts}] [{task_name}] [{status}] (运行中...)")

            # ── 增量日志：每轮只读 page 0（最新日志页），与前轮对比输出新增 ──
            if status not in self._STATUS_LETTER:  # 仅在运行中读取日志
                try:
                    page = magnus.get_job_logs(job_id, page=0)
                    text = page.get("logs", "").strip()
                    if text:
                        prev_text = self._log_cache.get(job_id, "")
                        if prev_text and text.startswith(prev_text):
                            # 正常追加：输出新增部分
                            new_part = text[len(prev_text):].strip()
                            if new_part:
                                self._print(f"[{ts}] [{task_name}] --- 新增日志 ---")
                                for line in new_part.splitlines():
                                    self._print(f"  {line}")
                                self._print()
                        elif prev_text and not text.startswith(prev_text):
                            # 日志滚动（旧内容被移出 page 0），输出全部
                            self._print(f"[{ts}] [{task_name}] --- 日志滚动，最新内容 ---")
                            for line in text.splitlines():
                                self._print(f"  {line}")
                            self._print()
                        else:
                            # 首次读取
                            self._print(f"[{ts}] [{task_name}] --- 日志 ---")
                            for line in text.splitlines():
                                self._print(f"  {line}")
                            self._print()
                        self._log_cache[job_id] = text
                except Exception as exc:
                    self._print(f"[{ts}] [{task_name}] 日志获取失败: {exc}")

            # ── 终态处理 ──
            if status == "Success":
                self._print(f"[{ts}] [{task_name}] ✅ 成功!")
                result = None
                try:
                    result = magnus.get_job_result(job_id)
                except Exception:
                    pass
                if result:
                    self._print(f"[{ts}] [{task_name}] 结果: {result[:300]}")
                state["done"] = True
                state["final_status_letter"] = self._STATUS_LETTER["Success"]

            elif status in ("Failed", "Terminated"):
                self._print(f"[{ts}] [{task_name}] ❌ {status}")
                result = None
                try:
                    result = magnus.get_job_result(job_id)
                except Exception:
                    pass
                if result:
                    self._print(f"[{ts}] [{task_name}] 结果: {result[:300]}")
                state["done"] = True
                state["final_status_letter"] = self._STATUS_LETTER.get(status, "f")

            else:
                still_running = True

        return still_running


# ── 便捷函数 ─────────────────────────────────────────────

def monitor_jobs(*job_ids: str, poll_interval: int = 60) -> None:
    Monitor(poll_interval=poll_interval).add_many(*job_ids).run()


# ── 独立入口 ─────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="监控 Magnus 任务状态和日志")
    p.add_argument("job_ids", nargs="+", help="任务 ID（至少一个）")
    p.add_argument("--address", default=DEFAULT_ADDRESS)
    p.add_argument("--token", default=DEFAULT_TOKEN)
    p.add_argument("--poll-interval", type=int, default=60,
                   help="轮询间隔（秒）")
    return p.parse_args()


def main():
    args = _parse_args()
    magnus.configure(address=args.address, token=args.token)
    print(f"[配置] address={args.address}, token={args.token[:8]}...")
    Monitor(poll_interval=args.poll_interval).add_many(*args.job_ids).run()


if __name__ == "__main__":
    main()
