"""
Magnus Job Monitor — Web 版
启动后浏览器打开 http://localhost:9876

功能:
    - HTTP 服务器接收 Python 脚本的 job 通知
    - 三层轮询：快速状态(5s) + 全量日志(60s) + 自动发现(60s)
    - Web 前端：左栏任务列表 + 上栏操作按钮 + 主日志区
    - 内嵌 HTML/CSS/JS，无需外部文件
    - 启动自动打开浏览器
    - data1/ + data2/ 日志持久化 + @{} 元数据头
    - 重启自动恢复：从 Magnus 服务器同步全部日志和时间线
"""

import os
import re
import sys
import io
import json
import time
import socket
import threading
import webbrowser
from datetime import datetime, timezone, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Optional
from urllib.parse import urlparse, parse_qs

import magnus

if getattr(sys, 'frozen', False):
    HERE = os.path.dirname(sys.executable)
else:
    HERE = os.path.dirname(os.path.abspath(__file__))
DATA1_DIR = os.path.join(HERE, "data1")
DATA2_DIR = os.path.join(HERE, "data2")
CONFIG_PATH = os.path.join(DATA2_DIR, "config.json")
JOBS_PATH = os.path.join(DATA2_DIR, "jobs.json")

STATUS_LETTER = {"Success": "s", "Failed": "f", "Terminated": "t"}
LETTER_STATUS = {"s": "Success", "f": "Failed", "t": "Terminated", "u": "Unknown"}
STATUS_COLORS = {
    "Pending": "#9E9E9E", "Preparing": "#FF9800", "Running": "#2196F3",
    "Paused": "#FF9800", "Success": "#4CAF50", "Failed": "#F44336",
    "Terminated": "#616161",
}
SOURCE_ABBR = {
    "warmup_packages.py": "wp", "download_model_auto.py": "dma",
    "submit_sft.py": "ss", "magnus_sft.py": "ms", "monitor.py": "mo",
    "inspect_storage.py": "is", "warmup_test.py": "wt",
}

os.makedirs(DATA1_DIR, exist_ok=True)
os.makedirs(DATA2_DIR, exist_ok=True)


def _scan_max_seq():
    max_seq = 0
    _pat = re.compile(r'.*-(\d{3})\.data1$')
    if os.path.isdir(DATA1_DIR):
        for fname in os.listdir(DATA1_DIR):
            m = _pat.search(fname)
            if m:
                try:
                    max_seq = max(max_seq, int(m.group(1)))
                except Exception:
                    pass
    return max_seq


# ── 配置管理 ─────────────────────────────────────────────

class Config:
    def __init__(self):
        self.address = "http://162.105.151.134:3011/"
        self.token = "sk-xxx"
        self.poll_interval = 60
        self.auto_discover = True
        self.discover_limit = 50
        self.minimize_to_tray = True
        self.auto_start = False
        self._load()
        if self.auto_start:
            self._set_auto_start(True)

    def _load(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.address = d.get("address", self.address)
                self.token = d.get("token", self.token)
                self.poll_interval = d.get("poll_interval", self.poll_interval)
                self.auto_discover = d.get("auto_discover", self.auto_discover)
                self.discover_limit = d.get("discover_limit", self.discover_limit)
                self.minimize_to_tray = d.get("minimize_to_tray", self.minimize_to_tray)
                self.auto_start = d.get("auto_start", self.auto_start)
            except Exception:
                pass

    def save(self):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "address": self.address, "token": self.token,
                "poll_interval": self.poll_interval,
                "auto_discover": self.auto_discover,
                "discover_limit": self.discover_limit,
                "minimize_to_tray": self.minimize_to_tray,
                "auto_start": self.auto_start,
            }, f, ensure_ascii=False, indent=2)

    def _set_auto_start(self, enable):
        if sys.platform != "win32":
            return
        import winreg
        run_key = r"Software\Microsoft\Windows\CurrentVersion\Run"
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key, 0,
                                 winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE)
        except OSError:
            try:
                key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, run_key)
            except OSError:
                return
        try:
            if enable:
                exe_path = sys.executable
                if "python" in os.path.basename(exe_path).lower():
                    script = os.path.abspath(sys.argv[0]) if sys.argv else __file__
                    exe_path = f'"{exe_path}" "{script}"'
                winreg.SetValueEx(key, "MagnusMonitor", 0, winreg.REG_SZ, exe_path)
            else:
                try:
                    winreg.DeleteValue(key, "MagnusMonitor")
                except OSError:
                    pass
        finally:
            winreg.CloseKey(key)


# ── 任务管理器 ──────────────────────────────────────────

class JobManager:
    def __init__(self, config: Config):
        self.config = config
        self.jobs: Dict[str, dict] = {}
        self._log_cache: Dict[str, str] = {}
        self._listeners: list = []
        self._magnus_configured = False
        self._force_refresh = False
        self._snapshot_counter = _scan_max_seq()
        self._snapshot_saved: set = set()
        self._minute_buffer: Dict[str, list] = {}
        self._minute_jobs: Dict[str, set] = {}
        self._ensure_magnus_config()
        self._load()

    def _ensure_magnus_config(self):
        if not self._magnus_configured:
            try:
                magnus.configure(address=self.config.address, token=self.config.token)
                self._magnus_configured = True
            except Exception:
                pass

    def _load(self):
        if os.path.exists(JOBS_PATH):
            try:
                with open(JOBS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.jobs = data.get("jobs", {})
                for jid in self.jobs:
                    self._snapshot_saved.add(jid)
            except Exception:
                pass

    def save(self):
        os.makedirs(DATA1_DIR, exist_ok=True)
        os.makedirs(DATA2_DIR, exist_ok=True)
        with open(JOBS_PATH, "w", encoding="utf-8") as f:
            json.dump({"updated_at": datetime.now().isoformat(), "jobs": self.jobs},
                      f, ensure_ascii=False, indent=2)

    def add_job(self, job_id, task_name=None, submitter="unknown",
                address=None, token=None):
        if not job_id or job_id in self.jobs:
            return False
        self.jobs[job_id] = {
            "job_id": job_id,
            "task_name": task_name or job_id[:12],
            "submitter": submitter,
            "address": address or self.config.address,
            "token": token or self.config.token,
            "status": "Pending",
            "last_log_ts": "",
            "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "discovered": False,
        }
        self._snapshot_saved.discard(job_id)
        self.save()
        return True

    def remove_job(self, job_id):
        self.jobs.pop(job_id, None)
        self._log_cache.pop(job_id, None)
        self._snapshot_saved.discard(job_id)
        self.save()

    def poll_status(self):
        changed = False
        self._ensure_magnus_config()
        for job_id, info in self.jobs.items():
            old_status = info.get("status", "Unknown")
            if old_status in ("Success", "Failed", "Terminated"):
                continue
            try:
                magnus.configure(address=info["address"], token=info["token"])
                job = magnus.get_job(job_id)
                new_status = job.get("status", "Unknown")
                task_name = job.get("task_name", "")
                if not info.get("created_at"):
                    info["created_at"] = job.get("created_at", "")
                info["gpu_count"] = job.get("gpu_count", "N/A")
                info["gpu_type"] = job.get("gpu_type", "N/A")
                if task_name and task_name != job_id[:12]:
                    if info["task_name"] == job_id[:12] or info["task_name"] != task_name:
                        info["task_name"] = task_name
                if new_status != old_status:
                    info["status"] = new_status
                    changed = True
                    if new_status == "Running" and old_status in ("Pending", "Preparing", "Unknown"):
                        self._save_timeline(job_id,
                            f"[任务开始] {info.get('created_at', self._ts())}", info)
                    if new_status in STATUS_LETTER:
                        self._save_timeline(job_id,
                            f"[任务结束] {new_status} {self._ts()}", info)
                    if new_status in STATUS_LETTER:
                        self._save_snapshot(job_id, info)
            except Exception:
                pass
        if changed:
            self.save()

    def _ts_minute(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    def poll_logs(self):
        self._ensure_magnus_config()
        force = self._force_refresh
        terminal = ("Success", "Failed", "Terminated")
        current_minute = self._ts_minute()

        for job_id, info in list(self.jobs.items()):
            if not force and job_id in self._snapshot_saved:
                continue
            try:
                magnus.configure(address=info["address"], token=info["token"])
                page = magnus.get_job_logs(job_id, page=0)
                text = page.get("logs", "").strip()
                if text:
                    prev = self._log_cache.get(job_id, "")
                    if text.startswith(prev):
                        new_part = text[len(prev):].strip()
                    else:
                        new_part = text
                    self._log_cache[job_id] = text
                else:
                    new_part = ""
            except Exception as e:
                new_part = ""
                text = f"[{self._ts()}] [查询失败] {e}"

            status = info.get("status", "Unknown")
            if new_part or status in terminal or not info.get("_last_full_log"):
                self._save_full_log(job_id, text, info)

            if new_part:
                buf = self._minute_buffer.setdefault(current_minute, [])
                buf.append((job_id, new_part, info))
                self._minute_jobs.setdefault(current_minute, set()).add(job_id)
            elif status not in terminal and not info.get("_last_timeline_log"):
                self._save_timeline(job_id, "(运行中, 无新日志)", info)

            if status in terminal and job_id not in self._snapshot_saved:
                self._save_snapshot(job_id, info)

        stale = [m for m in self._minute_buffer if m != current_minute]
        for minute_key in stale:
            self._flush_minute(minute_key)

        self._force_refresh = False

    def _flush_minute(self, minute_key):
        buf = self._minute_buffer.pop(minute_key, [])
        jobs = self._minute_jobs.pop(minute_key, set())
        if not buf:
            return
        by_job = {}
        for job_id, new_part, info in buf:
            by_job.setdefault(job_id, []).append((new_part, info))
        for job_id, parts in by_job.items():
            combined = "\n".join(p[0] for p in parts)
            last_info = parts[-1][1]
            self._save_timeline(job_id, combined, last_info, header_ts=minute_key)

    def force_refresh(self):
        self._force_refresh = True

    def recover(self):
        """全量同步 Magnus 服务器。"""
        self._ensure_magnus_config()
        terminal = ("Success", "Failed", "Terminated")

        if self.config.auto_discover:
            try:
                result = magnus.list_jobs(limit=self.config.discover_limit)
                for item in result.get("items", []):
                    jid = item.get("id", "")
                    if not jid or jid in self.jobs:
                        continue
                    self.jobs[jid] = {
                        "job_id": jid,
                        "task_name": item.get("task_name", jid[:12]),
                        "submitter": "magnus",
                        "address": self.config.address,
                        "token": self.config.token,
                        "status": item.get("status", "Unknown"),
                        "last_log_ts": "",
                        "created_at": item.get("created_at", ""),
                        "gpu_count": item.get("gpu_count", "N/A"),
                        "gpu_type": item.get("gpu_type", "N/A"),
                        "discovered": True,
                    }
                    self._log_cache[jid] = ""
                    self._snapshot_saved.discard(jid)
            except Exception:
                pass

        for job_id, info in self.jobs.items():
            try:
                magnus.configure(address=info["address"], token=info["token"])
                job = magnus.get_job(job_id)
                new_status = job.get("status", "Unknown")
                task_name = job.get("task_name", "")
                if not info.get("created_at") or info.get("created_at") == "":
                    info["created_at"] = job.get("created_at", "")
                info["gpu_count"] = job.get("gpu_count", "N/A")
                info["gpu_type"] = job.get("gpu_type", "N/A")
                if task_name and task_name != job_id[:12]:
                    info["task_name"] = task_name
                if new_status != info.get("status"):
                    info["status"] = new_status
            except Exception:
                pass

        for job_id, info in list(self.jobs.items()):
            if self._has_finished_marker(job_id, info):
                self._snapshot_saved.add(job_id)
                continue
            try:
                magnus.configure(address=info["address"], token=info["token"])
                page = magnus.get_job_logs(job_id, page=0)
                text = page.get("logs", "").strip()
                if text:
                    self._log_cache[job_id] = text
                    self._save_full_log(job_id, text, info)
                    self._rebuild_timeline_from_log(job_id, text, info)
            except Exception:
                pass
            self._ensure_timeline_bookends(job_id, info)
            if info.get("status") in terminal:
                self._save_snapshot(job_id, info)

        self.save()

    def _has_finished_marker(self, job_id, info):
        full_path = info.get("_last_full_log", "")
        if not full_path or not os.path.exists(full_path):
            legacy = os.path.join(DATA1_DIR, f"{job_id[:12]}.data1")
            if os.path.exists(legacy):
                full_path = legacy
            else:
                return False
        meta = self._parse_header(full_path)
        return bool(meta and "finished" in meta)

    def _rebuild_timeline_from_log(self, job_id, text, info):
        if not text or not text.strip():
            return
        os.makedirs(DATA2_DIR, exist_ok=True)
        fname = f"{job_id[:12]}.data2"
        fpath = os.path.join(DATA2_DIR, fname)
        minute_groups = {}
        for line in text.strip().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            m = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}):\d{2}\]', stripped)
            if m:
                minute_key = m.group(1)
                minute_groups.setdefault(minute_key, []).append(stripped)
            else:
                minute_groups.setdefault("unknown", []).append(stripped)
        try:
            header = self._make_header(fname, info, "时间线", 0)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(header + "\n")
            for minute_key in sorted(minute_groups.keys()):
                lines = minute_groups[minute_key]
                status = info.get("status", "Unknown")
                if minute_key != "unknown":
                    f.write(f"\n[{minute_key}] [{status}]\n")
                for l in lines:
                    f.write(f"  {l}\n")
            info["_last_timeline_log"] = fpath
        except Exception:
            pass

    def _ensure_timeline_bookends(self, job_id, info):
        tline_path = info.get("_last_timeline_log", "")
        if not tline_path or not os.path.exists(tline_path):
            created = info.get("created_at", "")
            self._save_timeline(job_id, f"[任务开始] {created}", info)
            status = info.get("status", "Unknown")
            if status in ("Success", "Failed", "Terminated"):
                self._save_timeline(job_id, f"[任务结束] {status} {self._ts()}", info)
            return
        try:
            with open(tline_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return
        body = self._strip_header(content)
        if "[任务开始]" not in body:
            created = info.get("created_at", "")
            self._save_timeline(job_id, f"[任务开始] {created}", info)
        status = info.get("status", "Unknown")
        if status in ("Success", "Failed", "Terminated") and "[任务结束]" not in body:
            self._save_timeline(job_id, f"[任务结束] {status} {self._ts()}", info)

    def _ts(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _abbr(self, submitter):
        return SOURCE_ABBR.get(submitter, submitter.replace(".py", "")[:3])

    def _make_header(self, filename, info, log_type, seq=0):
        return (
            f"@{{{filename},\n"
            f"  time = {info.get('created_at', self._ts())},\n"
            f"  name = {info.get('task_name', '')},\n"
            f"  submitter = {info.get('submitter', 'unknown')},\n"
            f"  job_id = {info.get('job_id', '')},\n"
            f"  type = {log_type},\n"
            f"  status = {info.get('status', 'Unknown')},\n"
            f"  source_abbr = {self._abbr(info.get('submitter', ''))},\n"
            f"  seq = {seq:03d},\n"
            f"  created_at = {info.get('created_at', 'N/A')},\n"
            f"  updated_at = {self._ts()},\n"
            f"  gpu_count = {info.get('gpu_count', 'N/A')},\n"
            f"  gpu_type = {info.get('gpu_type', 'N/A')},\n"
            f"}}\n"
        )

    @staticmethod
    def _parse_header(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                first = f.readline().strip()
                if not first.startswith("@{"):
                    return None
                meta = {}
                for line in f:
                    stripped = line.strip()
                    if stripped == "}":
                        break
                    m = re.match(r'\s*(\w+)\s*=\s*(.+?)\s*,?\s*$', stripped)
                    if m:
                        meta[m.group(1)] = m.group(2).strip('"')
            return meta
        except Exception:
            return None

    @staticmethod
    def _strip_header(content):
        if content.startswith("@{"):
            idx = content.find("\n}")
            if idx != -1:
                return content[idx + 2:].lstrip("\n")
        return content

    def _save_full_log(self, job_id, text, info):
        os.makedirs(DATA1_DIR, exist_ok=True)
        fname = f"{job_id[:12]}.data1"
        fpath = os.path.join(DATA1_DIR, fname)
        header = self._make_header(fname, info, "日志", 0)
        content = f"{header}\n[{self._ts()}] [{info.get('task_name','')}] [{info.get('status','')}]\n{text or '(无日志内容)'}\n"
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            pass
        info["_last_full_log"] = fpath

    def _save_timeline(self, job_id, new_part, info, header_ts=None):
        if not new_part:
            return
        os.makedirs(DATA2_DIR, exist_ok=True)
        fname = f"{job_id[:12]}.data2"
        fpath = os.path.join(DATA2_DIR, fname)
        ts = header_ts or self._ts()
        status = info.get("status", "Unknown")
        lines = new_part.strip().splitlines()
        if not lines:
            return
        entries = f"\n[{ts}] [{status}]\n"
        entries += "".join(f"  {l}\n" for l in lines)
        try:
            if not os.path.exists(fpath):
                header = self._make_header(fname, info, "时间线", 0)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(header + "\n")
            with open(fpath, "a", encoding="utf-8") as f:
                f.write(entries)
        except Exception:
            pass
        info["_last_timeline_log"] = fpath

    def _save_snapshot(self, job_id, info):
        status = info.get("status", "")
        letter = STATUS_LETTER.get(status)
        if not letter or job_id in self._snapshot_saved:
            return
        created = info.get("created_at", "")
        if created:
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                ts = dt.strftime("%Y%m%d-%H%M%S")
            except Exception:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        task_name = info.get("task_name", job_id[:12])
        safe = re.sub(r'[^\w-]', '_', task_name)
        submitter = info.get("submitter", "unknown")
        abbr = self._abbr(submitter)

        full_path = info.get("_last_full_log", "")
        full_content = ""
        if full_path and os.path.exists(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                full_content = self._strip_header(raw)
            except Exception:
                full_content = ""
        if not full_content.strip():
            return

        self._snapshot_counter += 1
        seq = self._snapshot_counter
        self._snapshot_saved.add(job_id)
        base = f"{ts}-{letter}-{abbr}-{safe}-{seq:03d}"

        fname1 = f"{base}.data1"
        path1 = os.path.join(DATA1_DIR, fname1)
        header1 = self._make_header(fname1, info, "日志", seq)
        try:
            with open(path1, "w", encoding="utf-8") as f:
                f.write(header1 + "\n")
                f.write(full_content)
        except Exception:
            pass

        tline_path = info.get("_last_timeline_log", "")
        fname2 = f"{base}.data2"
        path2 = os.path.join(DATA2_DIR, fname2)
        header2 = self._make_header(fname2, info, "时间线", seq)
        try:
            with open(path2, "w", encoding="utf-8") as f:
                f.write(header2 + "\n")
            if tline_path and os.path.exists(tline_path):
                with open(tline_path, "r", encoding="utf-8") as src:
                    for line in src:
                        if line.strip() == "}":
                            break
                    with open(path2, "a", encoding="utf-8") as dst:
                        for line in src:
                            dst.write(line)
        except Exception:
            pass

    def discover_jobs(self):
        if not self.config.auto_discover:
            return
        self._ensure_magnus_config()
        try:
            result = magnus.list_jobs(limit=self.config.discover_limit)
            for item in result.get("items", []):
                jid = item.get("id", "")
                if not jid or jid in self.jobs:
                    continue
                self.jobs[jid] = {
                    "job_id": jid,
                    "task_name": item.get("task_name", jid[:12]),
                    "submitter": "magnus",
                    "address": self.config.address,
                    "token": self.config.token,
                    "status": item.get("status", "Unknown"),
                    "last_log_ts": "",
                    "created_at": item.get("created_at", ""),
                    "gpu_count": item.get("gpu_count", "N/A"),
                    "gpu_type": item.get("gpu_type", "N/A"),
                    "discovered": True,
                }
                self._log_cache[jid] = ""
                self._snapshot_saved.discard(jid)
            if result.get("items"):
                self.save()
        except Exception:
            pass

    def process_incoming(self):
        incoming_dir = os.path.join(DATA2_DIR, "incoming")
        if not os.path.isdir(incoming_dir):
            return
        for fname in os.listdir(incoming_dir):
            if not fname.endswith(".job.json"):
                continue
            fpath = os.path.join(incoming_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.add_job(
                    job_id=data.get("job_id", ""),
                    task_name=data.get("task_name"),
                    submitter=data.get("submitter", "incoming"),
                    address=data.get("address"),
                    token=data.get("token"),
                )
                os.remove(fpath)
            except Exception:
                pass
        self._normalize_filenames()

    def _normalize_filenames(self):
        _STD = re.compile(r'^\d{8}-\d{6}-[sftu]-[a-z0-9]+-.+-\d{3}\.data[12]$')
        for dir_path in (DATA1_DIR, DATA2_DIR):
            if not os.path.isdir(dir_path):
                continue
            for fname in list(os.listdir(dir_path)):
                if not fname.endswith((".data1", ".data2")):
                    continue
                if _STD.match(fname):
                    continue
                fpath = os.path.join(dir_path, fname)
                try:
                    meta = self._parse_header(fpath)
                    if not meta:
                        continue
                    raw = meta.get("created_at", "")
                    if raw and raw != "N/A":
                        try:
                            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                            ts = dt.strftime("%Y%m%d-%H%M%S")
                        except Exception:
                            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    else:
                        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    letter = STATUS_LETTER.get(meta.get("status", ""), "u")
                    abbr = meta.get("source_abbr", "unk")
                    safe_name = re.sub(r'[^\w-]', '_', meta.get("name", "unknown"))
                    seq = str(int(meta.get("seq", 0))).zfill(3)
                    ext = fname[-5:]
                    new_name = f"{ts}-{letter}-{abbr}-{safe_name}-{seq}{ext}"
                    new_path = os.path.join(dir_path, new_name)
                    if new_path != fpath and not os.path.exists(new_path):
                        os.rename(fpath, new_path)
                except Exception:
                    pass

    def terminate_job(self, job_id):
        info = self.jobs.get(job_id)
        if not info:
            return
        try:
            magnus.configure(address=info["address"], token=info["token"])
            magnus.terminate_job(job_id)
            info["status"] = "Terminated"
            self.save()
        except Exception as e:
            raise e

    def get_log_content(self, job_id, timeline=False):
        info = self.jobs.get(job_id)
        if not info:
            return "(无任务信息)"
        key = "_last_timeline_log" if timeline else "_last_full_log"
        fpath = info.get(key, "")
        if fpath and os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    return self._strip_header(f.read())
            except Exception:
                pass
        fallback = "_last_full_log" if timeline else "_last_timeline_log"
        fpath = info.get(fallback, "")
        if fpath and os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    return self._strip_header(f.read())
            except Exception:
                pass
        return "(暂无日志)"

    def get_job_list(self):
        """Return jobs as a list suitable for JSON serialization."""
        result = []
        for jid, info in self.jobs.items():
            created = info.get("created_at", "")
            try:
                ts = created.replace("Z", "+00:00")
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    # naive → assume UTC+8 (Beijing time)
                    dt_cn = dt.replace(tzinfo=timezone(timedelta(hours=8)))
                else:
                    # timezone-aware → convert to UTC+8
                    dt_cn = dt.astimezone(timezone(timedelta(hours=8)))
                time_str = dt_cn.strftime("%H:%M")
            except Exception:
                time_str = created[11:16] if len(created) > 16 else ""
            result.append({
                "job_id": jid,
                "task_name": info.get("task_name", ""),
                "submitter": info.get("submitter", ""),
                "status": info.get("status", "Unknown"),
                "created_at": created,
                "time_str": time_str,
                "discovered": info.get("discovered", False),
                "gpu_count": info.get("gpu_count", "N/A"),
                "gpu_type": info.get("gpu_type", "N/A"),
            })
        result.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return result


# ── Web 服务器 ─────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Magnus Job Monitor</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#1a1a2e;color:#e0e0e0;display:flex;flex-direction:column;height:100vh;overflow:hidden}
/* toolbar */
#toolbar{display:flex;align-items:center;gap:4px;padding:4px 8px;background:#16213e;border-bottom:1px solid #0f3460;flex-shrink:0;flex-wrap:wrap}
#toolbar button{padding:5px 12px;border:1px solid #0f3460;border-radius:4px;background:#1a1a2e;color:#e0e0e0;cursor:pointer;font-size:13px;white-space:nowrap}
#toolbar button:hover{background:#0f3460}
#toolbar button.danger{color:#f44336;border-color:#f44336}
#toolbar button.danger:hover{background:#4a1515}
#toolbar .sep{width:1px;height:20px;background:#0f3460;margin:0 4px}
#toolbar .spacer{flex:1}
#toolbar select{padding:4px 8px;border:1px solid #0f3460;border-radius:4px;background:#1a1a2e;color:#e0e0e0;font-size:13px}
/* edit dropdown */
.dropdown{position:relative;display:inline-block}
.dropdown-content{display:none;position:absolute;top:100%;left:0;background:#1a1a2e;border:1px solid #0f3460;border-radius:4px;z-index:100;min-width:200px;box-shadow:0 4px 12px rgba(0,0,0,.5)}
.dropdown-content button{display:block;width:100%;text-align:left;border:none;border-radius:0;padding:8px 14px}
.dropdown-content button:hover{background:#0f3460}
.dropdown-content .sep2{height:1px;background:#0f3460;margin:2px 0}
.dropdown.open .dropdown-content{display:block}
/* main */
#main{display:flex;flex:1;overflow:hidden}
/* sidebar */
#sidebar{width:300px;min-width:200px;background:#16213e;border-right:1px solid #0f3460;overflow-y:auto;flex-shrink:0}
#sidebar .group-hdr{padding:6px 10px;background:#0f3460;cursor:pointer;font-size:13px;font-weight:bold;user-select:none;display:flex;justify-content:space-between;align-items:center}
#sidebar .group-hdr:hover{background:#1a4080}
#sidebar .card{padding:8px 12px;cursor:pointer;border-bottom:1px solid #0f3460;transition:background .15s}
#sidebar .card:hover{background:#1a3050}
#sidebar .card.selected{background:#1a4080;border-left:3px solid #4CAF50}
#sidebar .card .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
#sidebar .card .time{color:#888;font-size:11px}
#sidebar .card .name{font-size:13px;font-weight:bold;margin:2px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#sidebar .card .sub{color:#888;font-size:11px}
#sidebar .card .status{font-size:11px;font-weight:bold}
/* main area */
#content-area{flex:1;display:flex;flex-direction:column;overflow:hidden}
#log-viewer{flex:1;background:#0d1117;color:#c9d1d9;font-family:'Consolas','Cascadia Code','Courier New',monospace;font-size:13px;padding:12px;overflow:auto;white-space:pre-wrap;word-break:break-all;line-height:1.5}
#log-viewer::selection{background:#264f78}
/* status bar */
#statusbar{padding:4px 12px;background:#16213e;border-top:1px solid #0f3460;font-size:12px;color:#888;flex-shrink:0;display:flex;justify-content:space-between}
/* resizer */
#resizer{width:4px;background:#0f3460;cursor:col-resize;flex-shrink:0}
#resizer:hover{background:#1a4080}
/* settings modal */
.modal-overlay{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.6);z-index:200;justify-content:center;align-items:center}
.modal-overlay.show{display:flex}
.modal{background:#1a1a2e;border:1px solid #0f3460;border-radius:8px;padding:20px;min-width:420px;max-width:90vw;box-shadow:0 8px 32px rgba(0,0,0,.5)}
.modal h3{margin-bottom:16px;color:#e0e0e0}
.modal label{display:block;margin:10px 0 4px;color:#aaa;font-size:13px}
.modal input[type=text],.modal input[type=number]{width:100%;padding:8px;border:1px solid #0f3460;border-radius:4px;background:#0d1117;color:#e0e0e0;font-size:13px}
.modal .chk-row{display:flex;align-items:center;gap:8px;margin:8px 0}
.modal .chk-row input[type=checkbox]{accent-color:#4CAF50}
.modal .btn-row{display:flex;justify-content:flex-end;gap:8px;margin-top:16px}
.modal button{padding:6px 16px;border:1px solid #0f3460;border-radius:4px;background:#1a1a2e;color:#e0e0e0;cursor:pointer;font-size:13px}
.modal button.primary{background:#0f3460;border-color:#1a4080}
.modal button:hover{background:#0f3460}
/* toast */
#toast{position:fixed;bottom:40px;left:50%;transform:translateX(-50%);background:#333;color:#fff;padding:8px 20px;border-radius:20px;font-size:13px;z-index:300;opacity:0;transition:opacity .3s}
#toast.show{opacity:1}
/* loading */
.loading{text-align:center;padding:40px;color:#888}
/* scrollbar */
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:#0d1117}
::-webkit-scrollbar-thumb{background:#0f3460;border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:#1a4080}
</style>
</head>
<body>

<div id="toolbar">
  <button onclick="terminateJob()" class="danger" title="终止选中的任务">■ 终止任务</button>
  <button onclick="openBrowser()" title="在 Magnus 平台打开">🌐 打开</button>
  <span class="sep"></span>
  <button onclick="doRefresh()">⟳ 刷新</button>
  <button onclick="doForceRefresh()">⟳ 强制刷新</button>
  <span class="sep"></span>
  <button onclick="clearDone()">✕ 清除已完成</button>
  <span class="sep"></span>
  <div class="dropdown" id="editDropdown">
    <button onclick="toggleDropdown()">编辑 ▾</button>
    <div class="dropdown-content">
      <button onclick="copyLogContent()">复制日志内容</button>
      <button onclick="copyLogPath()">复制日志文件路径</button>
      <button onclick="copyTimelineContent()">复制时间线内容</button>
      <button onclick="copyTimelinePath()">复制时间线文件路径</button>
      <div class="sep2"></div>
      <button onclick="copyJobUrl()">复制任务网址</button>
      <button onclick="copyJobId()">复制 Job ID</button>
      <button onclick="copyTaskName()">复制任务名称</button>
    </div>
  </div>
  <span class="spacer"></span>
  <label style="font-size:12px;color:#888">显示:</label>
  <select id="viewMode" onchange="switchView()">
    <option value="full">📄 完整日志</option>
    <option value="timeline">⏱ 时间线</option>
  </select>
  <button onclick="openSettings()" title="设置">⚙</button>
</div>

<div id="main">
  <div id="sidebar"><div class="loading">加载中...</div></div>
  <div id="resizer"></div>
  <div id="content-area">
    <div id="log-viewer">← 在左侧选择一个任务查看日志</div>
  </div>
</div>

<div id="statusbar">
  <span id="status-left">就绪</span>
  <span id="status-right"></span>
</div>

<div id="toast"></div>

<div class="modal-overlay" id="settingsModal">
  <div class="modal">
    <h3>设置</h3>
    <label>Magnus 地址</label>
    <input type="text" id="cfgAddress">
    <label>API Token</label>
    <input type="text" id="cfgToken">
    <label>轮询间隔 (秒)</label>
    <input type="number" id="cfgInterval" min="15">
    <div class="chk-row">
      <input type="checkbox" id="cfgDiscover">
      <label style="margin:0">启用自动发现</label>
    </div>
    <div class="chk-row">
      <input type="checkbox" id="cfgStartup">
      <label style="margin:0">开机自启</label>
    </div>
    <div class="btn-row">
      <button onclick="closeSettings()">取消</button>
      <button class="primary" onclick="saveSettings()">保存</button>
    </div>
  </div>
</div>

<script>
let selectedJobId = null;
let sidebarTimer = null;
let logTimer = null;
let config = {};

// ── init ──
async function init(){
  await loadConfig();
  await refreshSidebar();
  sidebarTimer = setInterval(refreshSidebar, 60000);
  logTimer = setInterval(refreshLog, 60000);
  document.getElementById('resizer').addEventListener('mousedown', initResize);
  document.addEventListener('click', e => {
    const dd = document.getElementById('editDropdown');
    if(!dd.contains(e.target)) dd.classList.remove('open');
  });
}
init();

// ── config ──
async function loadConfig(){
  const r = await fetch('/api/config');
  config = await r.json();
}
async function saveConfig(){
  await fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify(config)});
}

// ── sidebar ──
async function refreshSidebar(){
  try{
    const r = await fetch('/api/jobs');
    const jobs = await r.json();
    renderSidebar(jobs);
  }catch(e){}
}

function getGroupKey(created){
  if(!created) return ['其他','9'];
  try{
    const dt = new Date(created.replace('Z','+00:00'));
    const now = new Date();
    const today = new Date(now.getFullYear(),now.getMonth(),now.getDate());
    const jd = new Date(dt.getFullYear(),dt.getMonth(),dt.getDate());
    const diff = (today-jd)/86400000;
    if(diff===0) return ['今天','1'];
    if(diff===1) return ['昨天','2'];
    if(diff<=7) return ['本周','3'];
    if(dt.getMonth()===now.getMonth() && dt.getFullYear()===now.getFullYear()) return ['本月','4'];
    return ['更早','5'];
  }catch(e){return ['其他','9']}
}

function renderSidebar(jobs){
  const sidebar = document.getElementById('sidebar');
  const groups = {};
  for(const j of jobs){
    const [label, pri] = getGroupKey(j.created_at);
    if(!groups[label]) groups[label] = {pri, items:[]};
    groups[label].items.push(j);
  }
  let html = '';
  const sorted = Object.entries(groups).sort((a,b)=>a[1].pri.localeCompare(b[1].pri));
  for(const [label, g] of sorted){
    html += `<div class="group-hdr" onclick="toggleGroup(this)">▼ ${label} (${g.items.length})</div>`;
    html += '<div class="group-body">';
    for(const j of g.items){
      const sel = j.job_id===selectedJobId?' selected':'';
      const color = statusColor(j.status);
      html += `<div class="card${sel}" data-jid="${j.job_id}" onclick="selectJob('${j.job_id}')">
        <div><span class="dot" style="background:${color}"></span><span class="time">${j.time_str}</span></div>
        <div class="name" title="${escHtml(j.task_name)}">${escHtml(j.task_name).substring(0,40)}</div>
        <div class="sub">${j.discovered?'⚡ magnus':escHtml(j.submitter)}</div>
        <div class="status" style="color:${color}">${j.status}</div>
      </div>`;
    }
    html += '</div>';
  }
  sidebar.innerHTML = html || '<div class="loading">暂无任务</div>';
  document.getElementById('status-left').textContent =
    `共 ${jobs.length} 个任务 | ${jobs.filter(j=>!['Success','Failed','Terminated'].includes(j.status)).length} 个活跃 | 上次更新: ${new Date().toLocaleTimeString('zh-CN',{hour12:false})}`;
}

function toggleGroup(hdr){
  const body = hdr.nextElementSibling;
  const hidden = body.style.display==='none';
  body.style.display = hidden?'':'none';
  hdr.textContent = (hidden?'▼':'▶') + hdr.textContent.substring(2);
}

function statusColor(s){
  const m={Pending:'#9E9E9E',Preparing:'#FF9800',Running:'#2196F3',Paused:'#FF9800',Success:'#4CAF50',Failed:'#F44336',Terminated:'#616161'};
  return m[s]||'#999';
}
function escHtml(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}

// ── job selection ──
function selectJob(jid){
  selectedJobId = jid;
  document.querySelectorAll('#sidebar .card').forEach(c=>c.classList.remove('selected'));
  const card = document.querySelector(`.card[data-jid="${jid}"]`);
  if(card) card.classList.add('selected');
  refreshLog();
}

// ── log viewer ──
async function refreshLog(){
  if(!selectedJobId) return;
  const mode = document.getElementById('viewMode').value;
  const url = mode==='timeline'?`/api/jobs/${selectedJobId}/timeline`:`/api/jobs/${selectedJobId}/log`;
  try{
    const r = await fetch(url);
    const text = await r.text();
    document.getElementById('log-viewer').textContent = text;
    document.getElementById('log-viewer').scrollTop = document.getElementById('log-viewer').scrollHeight;
  }catch(e){}
}
function switchView(){ refreshLog(); }

// ── actions ──
async function terminateJob(){
  if(!selectedJobId) return;
  if(!confirm(`确定终止任务 ${selectedJobId.substring(0,12)}？`)) return;
  try{
    await fetch(`/api/jobs/${selectedJobId}/terminate`, {method:'POST'});
    toast('已发送终止指令');
    setTimeout(refreshSidebar, 2000);
  }catch(e){toast('终止失败: '+e)}
}
function openBrowser(){
  if(!selectedJobId) return;
  window.open(config.address+'jobs/'+selectedJobId, '_blank');
}
async function doRefresh(){
  await fetch('/api/refresh', {method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({force:false})});
  await refreshSidebar();
  if(selectedJobId) refreshLog();
  toast('刷新完成');
}
async function doForceRefresh(){
  document.title = 'Magnus Job Monitor — 强制刷新中...';
  await fetch('/api/refresh', {method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({force:true})});
  document.title = 'Magnus Job Monitor';
  await refreshSidebar();
  if(selectedJobId) refreshLog();
  toast('强制刷新完成');
}
async function clearDone(){
  await fetch('/api/clear-done', {method:'POST'});
  selectedJobId = null;
  await refreshSidebar();
  document.getElementById('log-viewer').textContent = '← 在左侧选择一个任务查看日志';
  toast('已清除已完成任务');
}

// ── copy ──
async function copyText(txt, label){
  try{
    await navigator.clipboard.writeText(txt);
    toast(`已复制${label} (${txt.length} 字符)`);
  }catch(e){
    // fallback
    const ta = document.createElement('textarea');
    ta.value = txt; ta.style.position='fixed'; ta.style.left='-9999px';
    document.body.appendChild(ta); ta.select();
    document.execCommand('copy'); document.body.removeChild(ta);
    toast(`已复制${label}`);
  }
}
async function copyLogContent(){
  if(!selectedJobId) return;
  const r = await fetch(`/api/jobs/${selectedJobId}/log`);
  copyText(await r.text(), '日志内容');
}
async function copyLogPath(){
  if(!selectedJobId) return;
  const r = await fetch(`/api/jobs/${selectedJobId}/paths`);
  const p = await r.json();
  copyText(p.log_path||'', '日志文件路径');
}
async function copyTimelineContent(){
  if(!selectedJobId) return;
  const r = await fetch(`/api/jobs/${selectedJobId}/timeline`);
  copyText(await r.text(), '时间线内容');
}
async function copyTimelinePath(){
  if(!selectedJobId) return;
  const r = await fetch(`/api/jobs/${selectedJobId}/paths`);
  const p = await r.json();
  copyText(p.timeline_path||'', '时间线文件路径');
}
function copyJobUrl(){
  if(!selectedJobId) return;
  copyText(config.address+'jobs/'+selectedJobId, '任务网址');
}
function copyJobId(){
  if(!selectedJobId) return;
  copyText(selectedJobId, 'Job ID');
}
function copyTaskName(){
  if(!selectedJobId) return;
  const card = document.querySelector(`.card[data-jid="${selectedJobId}"]`);
  const name = card ? card.querySelector('.name').getAttribute('title')||card.querySelector('.name').textContent : '';
  copyText(name, '任务名称');
}

// ── dropdown ──
function toggleDropdown(){
  document.getElementById('editDropdown').classList.toggle('open');
}

// ── settings ──
function openSettings(){
  document.getElementById('cfgAddress').value = config.address||'';
  document.getElementById('cfgToken').value = config.token||'';
  document.getElementById('cfgInterval').value = config.poll_interval||60;
  document.getElementById('cfgDiscover').checked = config.auto_discover!==false;
  document.getElementById('cfgStartup').checked = config.auto_start===true;
  document.getElementById('settingsModal').classList.add('show');
}
function closeSettings(){
  document.getElementById('settingsModal').classList.remove('show');
}
async function saveSettings(){
  config.address = document.getElementById('cfgAddress').value;
  config.token = document.getElementById('cfgToken').value;
  config.poll_interval = Math.max(15, parseInt(document.getElementById('cfgInterval').value)||60);
  config.auto_discover = document.getElementById('cfgDiscover').checked;
  config.auto_start = document.getElementById('cfgStartup').checked;
  await saveConfig();
  closeSettings();
  toast('设置已保存');
}

// ── resize ──
function initResize(e){
  document.addEventListener('mousemove', doResize);
  document.addEventListener('mouseup', stopResize);
}
function doResize(e){
  const sidebar = document.getElementById('sidebar');
  sidebar.style.width = Math.max(180, Math.min(500, e.clientX)) + 'px';
}
function stopResize(){
  document.removeEventListener('mousemove', doResize);
  document.removeEventListener('mouseup', stopResize);
}

// ── toast ──
function toast(msg){
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  clearTimeout(t._tid); t._tid = setTimeout(()=>t.classList.remove('show'), 2000);
}

// ── keyboard ──
document.addEventListener('keydown', e=>{
  if(e.ctrlKey && e.key==='r'){ e.preventDefault(); doRefresh(); }
});
</script>
</body>
</html>"""


class WebHandler(BaseHTTPRequestHandler):
    jm: JobManager = None
    config_ref: Config = None

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/status":
            self._json({
                "job_count": len(self.jm.jobs),
                "active": sum(1 for j in self.jm.jobs.values()
                              if j.get("status") not in ("Success", "Failed", "Terminated")),
            })
        elif path == "/api/jobs":
            self._json(self.jm.get_job_list())
        elif path == "/api/config":
            self._json({
                "address": self.config_ref.address,
                "token": self.config_ref.token,
                "poll_interval": self.config_ref.poll_interval,
                "auto_discover": self.config_ref.auto_discover,
                "auto_start": self.config_ref.auto_start,
            })
        elif path == "/api/restore":
            self._json({"status": "ok"})
        elif path.startswith("/api/jobs/") and path.endswith("/log"):
            jid = path.split("/")[-2]
            content = self.jm.get_log_content(jid, timeline=False)
            self._text(content)
        elif path.startswith("/api/jobs/") and path.endswith("/timeline"):
            jid = path.split("/")[-2]
            content = self.jm.get_log_content(jid, timeline=True)
            self._text(content)
        elif path.startswith("/api/jobs/") and path.endswith("/paths"):
            jid = path.split("/")[-2]
            info = self.jm.jobs.get(jid, {})
            log_path = info.get("_last_full_log", "") or os.path.join(DATA1_DIR, f"{jid[:12]}.data1")
            tl_path = info.get("_last_timeline_log", "") or os.path.join(DATA2_DIR, f"{jid[:12]}.data2")
            self._json({"log_path": log_path, "timeline_path": tl_path})
        else:
            self._json({"status": "error", "message": "not found"}, 404)

    def do_POST(self):
        body = self._read_body()
        data = {}
        if body:
            try:
                data = json.loads(body)
            except Exception:
                pass

        path = self.path.split("?")[0]

        if path == "/api/job":
            self.jm.add_job(
                job_id=data.get("job_id", ""),
                task_name=data.get("task_name"),
                submitter=data.get("submitter", "unknown"),
                address=data.get("address"),
                token=data.get("token"),
            )
            self._json({"status": "ok", "message": "已添加"})

        elif path == "/api/terminate":
            try:
                self.jm.terminate_job(data.get("job_id", ""))
                self._json({"status": "ok", "message": "已终止"})
            except Exception as e:
                self._json({"status": "error", "message": str(e)}, 500)

        elif path == "/api/config":
            for k in ("address", "token", "poll_interval", "auto_discover", "auto_start"):
                if k in data:
                    setattr(self.config_ref, k, data[k])
            self.config_ref.save()
            self.jm._magnus_configured = False
            self._json({"status": "ok"})

        elif path.startswith("/api/jobs/") and path.endswith("/terminate"):
            jid = path.split("/")[-2]
            try:
                self.jm.terminate_job(jid)
                self._json({"status": "ok", "message": "已终止"})
            except Exception as e:
                self._json({"status": "error", "message": str(e)}, 500)

        elif path == "/api/refresh":
            force = data.get("force", False)
            if force:
                self.jm.force_refresh()
                # flush minute buffers
                self.jm._flush_minute(self.jm._ts_minute())
                stale = list(self.jm._minute_buffer.keys())
                for m in stale:
                    self.jm._flush_minute(m)
                self.jm.recover()
                # mark finished
                terminal = ("Success", "Failed", "Terminated")
                for jid, info in self.jm.jobs.items():
                    if info.get("status") in terminal:
                        self._mark_finished(jid, info)
            else:
                self.jm.poll_status()
                self.jm.poll_logs()
                self.jm.discover_jobs()
                self.jm.process_incoming()
            self._json({"status": "ok"})

        elif path == "/api/clear-done":
            terminal = ("Success", "Failed", "Terminated")
            to_remove = [jid for jid, info in self.jm.jobs.items()
                         if info.get("status") in terminal]
            for jid in to_remove:
                self.jm.remove_job(jid)
            self._json({"status": "ok", "removed": len(to_remove)})

        else:
            self._json({"status": "error", "message": "not found"}, 404)

    def _mark_finished(self, job_id, info):
        full_path = info.get("_last_full_log", "")
        if not full_path or not os.path.exists(full_path):
            return
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            if "\nfinished\n" in content or content.rstrip().endswith("finished"):
                return
            with open(full_path, "a", encoding="utf-8") as f:
                f.write("finished\n")
            header_end = content.find("\n}")
            if header_end != -1:
                new_header = content[:header_end] + f",\n  finished = {self.jm._ts()}\n" + content[header_end:]
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(new_header)
        except Exception:
            pass

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def _json(self, obj, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(obj, ensure_ascii=False).encode())

    def _text(self, text, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def log_message(self, fmt, *args):
        pass


# ── 轮询线程 ────────────────────────────────────────────

def poll_loop(job_manager, stop_event):
    counter = 0
    while not stop_event.is_set():
        try:
            job_manager.poll_status()
            counter += 1
            if counter % 12 == 0:
                job_manager.poll_logs()
                job_manager.discover_jobs()
                job_manager.process_incoming()
        except Exception:
            pass
        stop_event.wait(5)


# ── 入口 ───────────────────────────────────────────────

def main():
    config = Config()
    job_manager = JobManager(config)

    job_manager.process_incoming()
    job_manager._normalize_filenames()

    # 启动时后台全量恢复
    def _recover_async():
        try:
            job_manager.recover()
        except Exception:
            pass
    threading.Thread(target=_recover_async, daemon=True).start()

    # 轮询线程
    stop_event = threading.Event()
    poll_thread = threading.Thread(target=poll_loop,
                                   args=(job_manager, stop_event),
                                   daemon=True)
    poll_thread.start()

    # Web 服务器
    WebHandler.jm = job_manager
    WebHandler.config_ref = config
    server = HTTPServer(("127.0.0.1", 9876), WebHandler)

    # 自动打开浏览器
    def _open_browser():
        time.sleep(1.0)
        webbrowser.open("http://localhost:9876")
    threading.Thread(target=_open_browser, daemon=True).start()

    print(f"Magnus Monitor Web 已启动: http://localhost:9876")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.shutdown()


if __name__ == "__main__":
    main()
