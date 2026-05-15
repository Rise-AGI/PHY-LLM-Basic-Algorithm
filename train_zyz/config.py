"""
Shared configuration for train/ scripts.
Loads secrets from secret.json, provides Magnus connection and utilities.
"""
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Load secrets ────────────────────────────────────────────

with open(os.path.join(os.path.dirname(HERE), "secret.json"), "r", encoding="utf-8") as f:
    _secrets = json.load(f)

MAGNUS_ADDRESS = _secrets["magnus_address"]
MAGNUS_TOKEN   = _secrets["magnus_token"]
API_TOKENS     = set(_secrets.get("api_tokens", [])) | {MAGNUS_TOKEN}

# ── Paths ───────────────────────────────────────────────────

DATA1_DIR = os.path.join(HERE, "data1")
DATA2_DIR = os.path.join(HERE, "data2")
SFT_DATA_DIR = os.path.join(HERE, "SFT_data")
STORAGE_RECORD_PATH = os.path.join(DATA2_DIR, "storage_record.json")

# ── Magnus Apptainer bind mounts ─────────────────────────────

SYSTEM_ENTRY_COMMAND = """mounts=(
    "/home:/home"
    "/data:/data"
)
export APPTAINER_BIND=$(IFS=,; echo "${mounts[*]}")
export MAGNUS_HOME=/magnus
unset -f nvidia-smi
unset VIRTUAL_ENV SSL_CERT_FILE

# ── GPU 预检（仅 GPU job 触发，CPU job 零开销跳过）──────────
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "NONE" ]; then
    echo "=== [Magnus GPU 预检] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>&1 || echo "[Magnus GPU 预检] WARNING: nvidia-smi 不可用"
    python3 -c "
import torch, sys
total = torch.cuda.device_count()
avail = torch.cuda.is_available()
print(f'[Magnus GPU 预检] device_count={total}, is_available={avail}')
if not avail:
    print('[Magnus GPU 预检] FATAL: CUDA context 不可用 — GPU 驱动状态可能损坏')
    sys.exit(0)
ok = 0
for i in range(total):
    try:
        with torch.cuda.device(i):
            t = torch.zeros(1, device='cuda')
            del t
        torch.cuda.synchronize(i)
        ok += 1
    except Exception as e:
        print(f'[Magnus GPU 预检] GPU {i}: FAIL — {e}')
torch.cuda.empty_cache()
if ok == total:
    print(f'[Magnus GPU 预检] 逐卡验证: {ok}/{total} 可用')
else:
    print(f'[Magnus GPU 预检] WARNING: 逐卡验证 {ok}/{total} 可用 — 部分 GPU 异常')
" 2>&1 || echo "[Magnus GPU 预检] WARNING: PyTorch GPU 检测失败"
    echo "=== [Magnus GPU 预检] 完成 ==="
fi
"""

# ── Source abbreviations ─────────────────────────────────────

SOURCE_ABBR = {
    "warmup_packages.py": "wp",
    "download_model_auto.py": "dma",
    "submit_sft.py": "ss",
    "magnus_sft.py": "ms",
    "inspect_storage.py": "is",
    "remove_storage.py": "rs",
    "auto_grade.py": "ag",
    "eval_baseline.py": "eb",
    "serve_model.py": "sm",
    "download_results.py": "dr",
    "warmup_test.py": "wt",
    "run_sft_blueprint.py": "rsb",
    "diag_gpu.py": "dg",
}


def auto_source() -> Optional[str]:
    """返回调用者文件名对应的缩写，未匹配则返回 None。"""
    import inspect
    caller = inspect.getframeinfo(inspect.stack()[1].frame)
    return SOURCE_ABBR.get(os.path.basename(caller.filename))


# ── notify_exe (simplified — no GUI EXE anymore) ─────────────

def notify_exe(job_id: str, task_name: str = None, submitter: str = None,
               address: str = None, token: str = None) -> None:
    """Log job submission. Formerly notified Magnus Monitor EXE."""
    print(f"[notify] job submitted: {job_id}")


# ── wait_for_job (replaces Monitor class) ────────────────────

def wait_for_job(job_id: str, poll_interval: int = 60) -> dict:
    """轮询 Magnus 任务状态直到完成，打印状态变化和日志。

    Args:
        job_id: Magnus job ID
        poll_interval: 轮询间隔秒数

    Returns:
        最终的 job dict
    """
    import magnus

    _STATUS_TERMINAL = {"Success", "Failed", "Terminated"}

    last_status = None
    task_name = job_id[:8]
    log_len = 0  # 已打印日志字符数，用于增量

    print(f"[wait] polling every {poll_interval}s for job {job_id[:12]}...")
    print(f"[wait] {'─' * 50}")

    try:
        while True:
            try:
                job = magnus.get_job(job_id)
            except Exception as exc:
                print(f"[{_ts()}] [{task_name}] query failed: {exc}")
                time.sleep(poll_interval)
                continue

            status = job.get("status", "Unknown")
            task_name = job.get("task_name", task_name)

            # Status changes
            if status != last_status:
                if last_status is None:
                    print(f"[{_ts()}] [{task_name}] [{status}]")
                else:
                    print(f"[{_ts()}] [{task_name}] [{last_status}] -> [{status}]")
                last_status = status
            else:
                if status not in _STATUS_TERMINAL:
                    print(f"[{_ts()}] [{task_name}] [{status}] ...")

            # Incremental logs
            if status not in _STATUS_TERMINAL:
                try:
                    page = magnus.get_job_logs(job_id, page=0)
                    text = page.get("logs", "").strip()
                    if text and len(text) > log_len:
                        new_text = text[log_len:]
                        for line in new_text.splitlines():
                            print(f"  {line}")
                        log_len = len(text)
                except Exception:
                    pass

            if status in _STATUS_TERMINAL:
                print(f"[{_ts()}] [{task_name}] done ({status})")
                return job

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print(f"\n[wait] interrupted (job {job_id[:12]} still running)")
        return magnus.get_job(job_id)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Storage record helpers ───────────────────────────────────

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
    """检查 model-version 是否已存在（提交前去重）。"""
    record = _ensure_record()
    for entry in record.get("model-version", []):
        if entry.get("model") == model_version:
            return True
    return False


# ── Scan result parsing (for eval_baseline.py) ──────────────────

def parse_scan_json(logs_text: str, sentinel: str = "SCAN_RESULT_JSON") -> list[dict] | None:
    """从容器 job 日志中解析 sentinel JSON 块。

    日志格式:
        === {sentinel} ===
        [{...}, ...]
        === END_{sentinel} ===
    """
    start_marker = f"=== {sentinel} ==="
    end_marker = f"=== END_{sentinel} ==="
    lines = logs_text.splitlines()
    capture = False
    json_lines = []
    for line in lines:
        if line.strip() == start_marker:
            capture = True
            continue
        if line.strip() == end_marker:
            break
        if capture:
            json_lines.append(line)
    if not json_lines:
        return None
    try:
        return json.loads("\n".join(json_lines))
    except json.JSONDecodeError:
        return None


def detect_latest_sft_version(versions: list[dict], base_model: str) -> int:
    """从扫描结果中找到指定 base_model 的最新 SFT 版本号。"""
    max_v = 0
    suffix = "-sft-zyz-v"
    for v in versions:
        name = v.get("name", "")
        if name.startswith(base_model + suffix):
            try:
                num = int(name.split(suffix)[-1])
                if num > max_v:
                    max_v = num
            except ValueError:
                pass
    return max_v
