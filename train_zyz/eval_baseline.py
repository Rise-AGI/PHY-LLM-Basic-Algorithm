"""
自动化评估启动器：扫描 Magnus 服务器上的模型 → 选择模型 → 上传测试集 → 提交评估 job → 输出 file secret。

支持:
  - Base models (ModelScope 预热) 和 SFT 权重版本 (自动检测 -sft-zyz-v{N} 编号)
  - 两份测试集 (test.json / test_2.json)
  - 交互式模型选择 / 命令行指定

用法:
    python eval_baseline.py                           # 交互式选择
    python eval_baseline.py --scan                    # 仅扫描
    python eval_baseline.py --model Qwen2.5-72B-Instruct
    python eval_baseline.py --model Qwen2.5-72B-Instruct-sft-zyz-v1 --test-set test_2.json
    python eval_baseline.py --model Qwen2.5-72B-Instruct --all-versions
    python eval_baseline.py --test-secret magnus-secret:XXXX-xxxx-xxxx-xxxx
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Optional

import magnus

from config import (
    auto_source, notify_exe, wait_for_job,
    MAGNUS_ADDRESS, MAGNUS_TOKEN, SYSTEM_ENTRY_COMMAND,
    parse_scan_json, detect_latest_sft_version,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
#  CONFIG — 默认值
# ============================================================

DEFAULT_TEST_SET  = os.path.join(HERE, "test.json")
DEFAULT_TEST_SET2 = os.path.join(HERE, "test_2.json")
SCAN_CACHE_PATH   = os.path.join(HERE, "json", "scan_cache.json")

# -- 推理参数 --
MAX_LENGTH     = 2048
MAX_NEW_TOKENS = 512

# -- 评估 job 硬件默认 (会按模型大小自动调整) --
GPU_COUNT_DEFAULT = 1
GPU_TYPE      = "a100"
CPU_COUNT     = 16
MEMORY        = "128G"
STORAGE       = "200G"
PRIORITY      = "A2"
CONTAINER_IMAGE = "docker://crpi-32rssczyu25r10yu.cn-beijing.personal.cr.aliyuncs.com/zyz25/sft-base:v2"

# ============================================================
#  容器内模型扫描脚本
# ============================================================

_SCAN_MODELS_PY = r'''
import json, os, glob as _glob, re

SFT_VERSION_RE = re.compile(r'^(.+)-(sft|lora)-zyz-v(\d+)$')

def scan():
    """扫描 /data/*/models/ 和 /data/magnus/models/ 下的所有模型。"""
    base_models = []
    sft_versions = []
    seen = set()

    search_roots = []
    for d in _glob.glob("/data/*/models"):
        if os.path.isdir(d):
            search_roots.append(d)
    magnus_root = "/data/magnus/models"
    if os.path.isdir(magnus_root):
        search_roots.append(magnus_root)

    for root in search_roots:
        try:
            names = sorted(os.listdir(root))
        except (PermissionError, OSError):
            continue
        for name in names:
            full = os.path.join(root, name)
            if not os.path.isdir(full) or name in seen:
                continue
            seen.add(name)

            config_json = os.path.join(full, "config.json")
            if not os.path.isfile(config_json):
                continue

            # 读 config 获取模型信息
            try:
                with open(config_json, "r") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}

            # 检查权重文件
            has_weights = bool(
                _glob.glob(os.path.join(full, "*.safetensors")) or
                _glob.glob(os.path.join(full, "pytorch_model*.bin"))
            )

            # 估算参数量
            params_b = None
            for k in ("num_parameters", "num_params"):
                if k in cfg and isinstance(cfg[k], (int, float)):
                    params_b = round(cfg[k] / 1e9, 2)
                    break
            if params_b is None:
                hs = cfg.get("hidden_size")
                nl = cfg.get("num_hidden_layers")
                if hs and nl:
                    params_b = round(12 * nl * (hs ** 2) / 1e9, 2)

            entry = {
                "name": name,
                "path": full,
                "model_type": cfg.get("model_type", "unknown"),
                "estimated_params_b": params_b,
                "has_weights": has_weights,
            }

            m = SFT_VERSION_RE.match(name)
            if m and os.path.isdir(os.path.join(full, "final")):
                entry["base_model"] = m.group(1)
                entry["mode"] = m.group(2)
                entry["version_num"] = int(m.group(3))
                sft_versions.append(entry)
            else:
                base_models.append(entry)

    return {"base_models": base_models, "sft_versions": sft_versions}


if __name__ == "__main__":
    results = scan()
    print(f"=== 扫描完成: {len(results['base_models'])} base + {len(results['sft_versions'])} SFT ===")
    print("=== SCAN_RESULT_JSON ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("=== END_SCAN_RESULT_JSON ===")
'''


def _build_scan_entry() -> str:
    """构建容器扫描 entry_command。"""
    scan_b64 = base64.b64encode(_SCAN_MODELS_PY.encode("utf-8")).decode("ascii")
    return fr"""set -e
echo "=== 开始扫描 Magnus 存储模型 ==="
echo "{scan_b64}" | base64 -d > /tmp/scan_models.py
python3 /tmp/scan_models.py
echo "---DONE---"
"""


# ============================================================
#  容器内推理脚本 (保留原有逻辑, 参数化)
# ============================================================

_EVAL_INFER_PY = r'''
import json, os, re, time as _time
from datetime import datetime

def log(msg): print(f"[{_time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

MODEL_PATH     = os.getenv("EVAL_MODEL_PATH", "")
TEST_FILE      = os.getenv("EVAL_TEST_FILE", "/tmp/test.json")
OUTPUT_DIR     = os.getenv("EVAL_OUTPUT_DIR", "")
MAX_LENGTH     = int(os.getenv("EVAL_MAX_LENGTH", "2048"))
MAX_NEW_TOKENS = int(os.getenv("EVAL_MAX_NEW_TOKENS", "512"))

def parse_answer_solution(text):
    if not text: return "", ""
    parts = re.split(r'\n?\s*解答\s*[：:]\s*', text, maxsplit=1)
    if len(parts) >= 2:
        ans_part, sol = parts[0], parts[1].strip()
        m = re.search(r'答案\s*[：:]\s*(.*?)$', ans_part, re.DOTALL)
        ans = m.group(1).strip() if m else ans_part.strip()
        return ans, sol
    m_ans = re.search(r'答案\s*[：:]\s*(.*?)(?:\n\n|\n解答|$)', text, re.DOTALL)
    m_sol = re.search(r'解答\s*[：:]\s*(.*?)$', text, re.DOTALL)
    ans = m_ans.group(1).strip() if m_ans else ""
    sol = m_sol.group(1).strip() if m_sol else ""
    if not ans and not sol: return text.strip(), ""
    return ans, sol

log("[1/5] 读取测试集...")
with open(TEST_FILE, "r", encoding="utf-8") as f:
    raw = f.read().strip()
samples = json.loads(raw) if raw.startswith("[") else [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
log(f"[1/5] 已加载 {len(samples)} 条测试样本")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log(f"[2/5] 加载 tokenizer: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
        "{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\nAssistant: ' }}"
        "{% elif message['role'] == 'assistant' %}{{ message['content'] + '\n\n' }}"
        "{% endif %}{% endfor %}"
    )
log(f"[2/5] tokenizer 加载完成 | vocab_size={tokenizer.vocab_size}")

device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"[3/5] 加载模型 ({'bf16' if device == 'cuda' else 'fp32'})...")
t0 = _time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
)
if device == "cpu":
    model = model.to(device)
model.eval()
log(f"[3/5] 模型加载完成 ({_time.time() - t0:.1f}s)")

if device == "cuda":
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        log(f"  GPU {i}: 空闲 {free/1e9:.1f}GB / 总计 {total/1e9:.1f}GB")

log(f"[4/5] 开始生成式推理 ({len(samples)} 条)...")
results = []
for i, sample in enumerate(samples):
    instruction = sample.get("instruction", "")
    extra       = sample.get("input", "")
    gt_out      = sample.get("output", "")
    user_content = instruction + ("\n" + extra if extra else "")
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    new_ids  = out_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    gt_ans, gt_sol = parse_answer_solution(gt_out)
    md_ans, md_sol = parse_answer_solution(response)
    results.append({
        "id": sample.get("id", i),
        "question": user_content,
        "gt_full": gt_out,
        "gt_answer": gt_ans,
        "gt_solution": gt_sol,
        "model_full": response,
        "model_answer": md_ans,
        "model_solution": md_sol,
    })
    if (i + 1) % 10 == 0 or (i + 1) == len(samples):
        log(f"  [{i+1}/{len(samples)}]")

log(f"[5/5] 保存结果...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
summary = {
    "total": len(results), "eval_type": "baseline",
    "model_path": MODEL_PATH, "generated_at": datetime.now().isoformat(),
}
out_full = os.path.join(OUTPUT_DIR, "eval_results_initial.json")
with open(out_full, "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
log(f"[5/5] 完整结果: {out_full}")
out_summary = os.path.join(OUTPUT_DIR, "eval_results_summary.json")
with open(out_summary, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
log(f"[5/5] 摘要: {out_summary}")
print()
print("=" * 60)
print("  评估完成")
print(f"  样本数:     {len(results)}")
print(f"  输出目录:   {OUTPUT_DIR}")
print("=" * 60)
'''


def _build_eval_entry(
    model_path: str,
    test_secret: str,
    output_dir: str,
    model_display_name: str,
) -> str:
    """构建容器评估 entry_command。"""
    eval_b64 = base64.b64encode(_EVAL_INFER_PY.encode("utf-8")).decode("ascii")

    # 截断 secret prefix 用于 URL 下载
    test_token = test_secret.replace("magnus-secret:", "") if test_secret.startswith("magnus-secret:") else test_secret

    return fr"""set -e

echo "============================================"
echo "  Eval: {model_display_name}"
echo "  启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

echo "{eval_b64}" | base64 -d > /tmp/eval_infer.py
python3 -c "compile(open('/tmp/eval_infer.py').read(), '/tmp/eval_infer.py', 'exec'); print('[check] eval_infer.py syntax OK')"

MODEL_DIR="{model_path}"
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "[FATAL] config.json not found in $MODEL_DIR"
    ls "$MODEL_DIR" 2>/dev/null || echo "(directory missing)"
    exit 1
fi
echo "[check] model OK ($(du -sh "$MODEL_DIR" | cut -f1))"

OUTPUT_DIR="{output_dir}"
mkdir -p "$OUTPUT_DIR"

echo "[check] 拉取测试数据..."
python3 -c "
import urllib.request
url = '{MAGNUS_ADDRESS}api/files/download/' + '{test_token}'
urllib.request.urlretrieve(url, '/tmp/test.json')
"
echo "[check] test data downloaded ($(wc -c < /tmp/test.json) bytes)"

export EVAL_MODEL_PATH="$MODEL_DIR"
export EVAL_TEST_FILE="/tmp/test.json"
export EVAL_OUTPUT_DIR="$OUTPUT_DIR"
export EVAL_MAX_LENGTH="{MAX_LENGTH}"
export EVAL_MAX_NEW_TOKENS="{MAX_NEW_TOKENS}"
export MAGNUS_ADDRESS="{MAGNUS_ADDRESS}"
export MAGNUS_TOKEN="{MAGNUS_TOKEN}"

echo ""
echo "=== starting eval ==="
python3 /tmp/eval_infer.py
EVAL_EXIT=$?

echo ""
echo "=== 上传结果到 Magnus ==="
python3 << 'PYEOF'
import json, os, sys

out_dir = os.environ.get("EVAL_OUTPUT_DIR", "")
magnus_addr = os.environ.get("MAGNUS_ADDRESS", "")
magnus_token = os.environ.get("MAGNUS_TOKEN", "")
files = [
    ("eval_results_initial.json", "完整推理结果"),
    ("eval_results_summary.json", "摘要统计"),
]
CRLF = chr(13) + chr(10)

def upload_file(filepath, url, token):
    import urllib.request, uuid
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        content = f.read()
    boundary = "----" + uuid.uuid4().hex
    body = b""
    body += ("--" + boundary + CRLF).encode()
    body += ('Content-Disposition: form-data; name="file"; filename="' + filename + '"' + CRLF).encode()
    body += ("Content-Type: application/octet-stream" + CRLF + CRLF).encode()
    body += content + CRLF.encode()
    body += ("--" + boundary + CRLF).encode()
    body += ("Content-Disposition: form-data; name=\"expire_minutes\"" + CRLF + CRLF + "1440" + CRLF).encode()
    body += ("--" + boundary + "--" + CRLF).encode()
    req = urllib.request.Request(url, data=body)
    req.add_header("Content-Type", "multipart/form-data; boundary=" + boundary)
    req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())

for fname, label in files:
    fpath = os.path.join(out_dir, fname) if out_dir else fname
    if not os.path.exists(fpath):
        print(f"  [{{label}}] file not found: {{fpath}}")
        continue
    try:
        url = magnus_addr.rstrip("/") + "/api/files/upload"
        result = upload_file(fpath, url, magnus_token)
        secret = result.get("file_secret", "")
        print(f"  [{{label}}]")
        print(f"  magnus receive {{secret}} -o ./{{fname}}")
    except Exception as e:
        print(f"  [{{label}}] upload failed: {{e}}")
PYEOF

exit $EVAL_EXIT
"""


# ============================================================
#  扫描 & 模型选择
# ============================================================

def _submit_scan_job(poll_interval: int = 30) -> list[dict]:
    """提交扫描 job 并解析结果。返回 [{"name": ..., "path": ..., ...}] 统一列表。"""
    print("[1/6] 提交模型扫描 job (CPU, B2)...")
    magnus.configure(address=MAGNUS_ADDRESS, token=MAGNUS_TOKEN)

    entry = _build_scan_entry()
    job_id = magnus.submit_job(
        task_name          = "Eval-Scan-Models",
        description        = "扫描 Magnus 存储中的 base models 和 SFT 版本",
        entry_command      = entry,
        system_entry_command = SYSTEM_ENTRY_COMMAND,
        namespace          = "Rise-AGI",
        repo_name          = "OpenFundus",
        gpu_count          = 0,
        gpu_type           = "cpu",
        cpu_count          = 2,
        memory_demand      = "4G",
        ephemeral_storage  = "10G",
        job_type           = "B2",
    )
    print(f"      Job ID: {job_id}")

    notify_exe(job_id=job_id)
    print(f"[2/6] 等待扫描完成 (poll={poll_interval}s)...")
    job = wait_for_job(job_id, poll_interval=poll_interval)

    if job.get("status") != "Success":
        print(f"[WARN] 扫描 job 未成功 (status={job.get('status')})，尝试解析已有日志...")

    # 收集日志
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
    full_logs = "\n".join(all_logs)

    results = parse_scan_json(full_logs)
    if results is None:
        print("[FATAL] 未能从扫描日志中解析结果")
        sys.exit(1)

    # 合并为统一列表
    all_models = []
    for m in results.get("base_models", []):
        m["kind"] = "base"
        all_models.append(m)
    for m in results.get("sft_versions", []):
        m["kind"] = "sft"
        all_models.append(m)

    # 缓存
    os.makedirs(os.path.dirname(SCAN_CACHE_PATH), exist_ok=True)
    with open(SCAN_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "scanned_at": datetime.now().isoformat(),
            "job_id": job_id,
            "models": all_models,
        }, f, ensure_ascii=False, indent=2)
    print(f"      缓存: {SCAN_CACHE_PATH}")

    return all_models


def _load_cached_scan() -> Optional[list[dict]]:
    """加载缓存的扫描结果。"""
    if not os.path.exists(SCAN_CACHE_PATH):
        return None
    try:
        with open(SCAN_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("models", [])
    except Exception:
        return None


def _print_model_list(models: list[dict]) -> None:
    """分组打印模型列表。"""
    base = [m for m in models if m.get("kind") == "base"]
    sft  = [m for m in models if m.get("kind") == "sft"]

    print()
    print("=" * 60)
    print("  Base Models (ModelScope 预热)")
    print("=" * 60)
    for i, m in enumerate(base):
        p = m.get("estimated_params_b")
        params_str = f" ~{p}B" if p else ""
        print(f"  [{i}] {m['name']}{params_str}")

    print()
    print("=" * 60)
    print("  SFT 权重版本")
    print("=" * 60)
    # 按 base_model 分组
    groups: dict[str, list[dict]] = {}
    for m in sft:
        bm = m.get("base_model", "?")
        groups.setdefault(bm, []).append(m)
    for bm, versions in sorted(groups.items()):
        versions.sort(key=lambda x: x.get("version_num", 0))
        for v in versions:
            p = v.get("estimated_params_b")
            params_str = f" ~{p}B" if p else ""
            print(f"  [{base.__len__() + sft.index(v)}] {v['name']}{params_str}  ({v.get('mode', '?')} v{v.get('version_num', '?')})")

    print()
    print(f"  共 {len(models)} 个模型 ({len(base)} base + {len(sft)} sft)")


def _select_model_interactive(models: list[dict]) -> dict:
    """交互式选择模型。"""
    _print_model_list(models)

    while True:
        try:
            choice = input("\n  选择模型编号 (或输入名称, q 退出): ").strip()
            if choice.lower() == "q":
                sys.exit(0)
            # 尝试编号
            if choice.isdigit():
                idx = int(choice)
                if 0 <= idx < len(models):
                    return models[idx]
            # 尝试名称匹配
            for m in models:
                if m["name"] == choice:
                    return m
            print(f"  无效选择: {choice}")
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)


def _auto_gpu_config(estimated_params_b: Optional[float]) -> int:
    """根据参数量自动选择 GPU 数量。"""
    if estimated_params_b is None:
        return GPU_COUNT_DEFAULT
    if estimated_params_b <= 7:
        return 1
    if estimated_params_b <= 32:
        return 1
    if estimated_params_b <= 72:
        return 2
    return 4


# ============================================================
#  main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="自动化评估启动器：扫描模型 → 选择 → 提交评估 → 输出 file secret")
    parser.add_argument("--scan", action="store_true",
                        help="仅扫描模型列表，不提交评估")
    parser.add_argument("--model", default=None,
                        help="指定模型名称 (如 Qwen2.5-72B-Instruct 或 Qwen2.5-72B-Instruct-sft-zyz-v1)")
    parser.add_argument("--all-versions", action="store_true",
                        help="对指定 base model 的所有 SFT 版本逐一评估")
    parser.add_argument("--test-set", default=None,
                        help=f"本地测试集 JSON 路径 (默认: {DEFAULT_TEST_SET})")
    parser.add_argument("--test-secret", default=None,
                        help="已上传到 Magnus 的测试集 secret (跳过本地上传)")
    parser.add_argument("--no-scan", action="store_true",
                        help="跳过扫描，使用缓存 (需先运行过 --scan)")
    parser.add_argument("--address", default=MAGNUS_ADDRESS)
    parser.add_argument("--token",   default=MAGNUS_TOKEN)
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="监控轮询间隔秒数 (0=跳过监控)")
    args = parser.parse_args()

    magnus.configure(address=args.address, token=args.token)

    # ── 0. 获取模型列表 ──
    if args.no_scan:
        models = _load_cached_scan()
        if models is None:
            print("[FATAL] 无缓存且 --no-scan 指定。请先运行 --scan")
            sys.exit(1)
        print(f"[0/6] 从缓存加载 {len(models)} 个模型")
    else:
        models = _submit_scan_job(poll_interval=args.poll_interval)

    if not models:
        print("[FATAL] 未找到任何模型")
        sys.exit(1)

    if args.scan:
        _print_model_list(models)
        return

    # ── 1. 确定评估目标 ──
    if args.model:
        targets = [m for m in models if m["name"] == args.model]
        if not targets:
            # 模糊匹配
            targets = [m for m in models if args.model in m["name"]]
        if not targets:
            print(f"[FATAL] 未找到模型: {args.model}")
            print(f"       可用模型: {', '.join(m['name'] for m in models)}")
            sys.exit(1)

        if args.all_versions and targets[0].get("kind") == "base":
            base_name = targets[0]["name"]
            targets = [m for m in models
                       if m.get("kind") == "sft" and m.get("base_model") == base_name]
            targets.sort(key=lambda x: x.get("version_num", 0))
            if not targets:
                print(f"[FATAL] base model '{base_name}' 无 SFT 版本")
                sys.exit(1)
            print(f"[3/6] 将对 {len(targets)} 个版本逐一评估: {', '.join(t['name'] for t in targets)}")
        elif args.all_versions:
            print(f"[FATAL] --all-versions 仅对 base model 有效, 当前选择是 {targets[0].get('kind')} model")
            sys.exit(1)
        else:
            target = targets[0]
            targets = [target]
            print(f"[3/6] 选定模型: {target['name']} ({target.get('kind', '?')})")
    else:
        target = _select_model_interactive(models)
        targets = [target]

    # ── 2. 确定测试集 ──
    if args.test_secret:
        test_secret = args.test_secret
        if not test_secret.startswith("magnus-secret:"):
            test_secret = f"magnus-secret:{test_secret}"
        print(f"[4/6] 使用已有测试集 secret: {test_secret[:40]}...")
    else:
        test_path = args.test_set or DEFAULT_TEST_SET
        if not os.path.exists(test_path):
            # 尝试 test_2.json 作为备选
            alt = args.test_set or DEFAULT_TEST_SET2
            if os.path.exists(alt):
                test_path = alt
            else:
                print(f"[FATAL] 测试集文件不存在: {test_path}")
                print(f"       请用 --test-set 指定路径 或 --test-secret 传入已上传的 secret")
                sys.exit(1)
        print(f"[4/6] 上传测试集: {test_path}")
        test_secret = magnus.custody_file(test_path, expire_minutes=1440)
        print(f"      secret: {test_secret}")

    # ── 3. 提交评估 job(s) ──
    for target in targets:
        model_path = target["path"]
        model_name = target["name"]
        kind = target.get("kind", "?")
        params_b = target.get("estimated_params_b")
        gpu_count = _auto_gpu_config(params_b)
        output_dir = f"/data/${{whoami}}/eval_{model_name}"

        print()
        print(f"[5/6] 提交评估 job: {model_name}")
        print(f"      模型路径: {model_path}")
        print(f"      硬件: {gpu_count}x {GPU_TYPE} | CPU={CPU_COUNT} | MEM={MEMORY}")

        entry = _build_eval_entry(
            model_path=model_path,
            test_secret=test_secret,
            output_dir=output_dir,
            model_display_name=model_name,
        )

        job_id = magnus.submit_job(
            task_name          = f"Eval-{model_name[:40]}",
            description        = f"评估 {model_name} ({kind}) | {target.get('estimated_params_b', '?')}B",
            entry_command      = entry,
            system_entry_command = SYSTEM_ENTRY_COMMAND,
            namespace          = "Rise-AGI",
            repo_name          = "OpenFundus",
            gpu_count          = gpu_count,
            gpu_type           = GPU_TYPE,
            cpu_count          = CPU_COUNT,
            memory_demand      = MEMORY,
            ephemeral_storage  = STORAGE,
            job_type           = PRIORITY,
            container_image    = CONTAINER_IMAGE,
        )
        print(f"      Job ID: {job_id}")

        notify_exe(job_id=job_id)

        if args.poll_interval > 0:
            print(f"[6/6] 监控 (poll={args.poll_interval}s, Ctrl+C 退出)")
            print(f"{'='*60}")
            wait_for_job(job_id, poll_interval=args.poll_interval)

        # 打印结果
        print()
        print("=" * 60)
        print(f"  评估完成: {model_name}")
        print(f"  Job ID: {job_id}")
        job = magnus.get_job(job_id)
        print(f"  状态: {job.get('status', 'unknown')}")

        # 尝试获取 file secret
        try:
            result = magnus.get_job_result(job_id)
            if result:
                print(f"\n  === 下载结果 ===")
                print(f"  magnus receive {result} -o ./eval_results_{model_name}.json")
            else:
                print(f"\n  (无 file secret，请查看日志中的 magnus receive 命令)")
        except Exception:
            print(f"\n  (无法获取结果，查看日志: magnus logs {job_id})")

        print(f"\n  输出目录 (Magnus): {output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
