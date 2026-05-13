"""
Baseline eval: 上传测试集到 Magnus，用原始模型做生成式推理，产出 eval_results_initial.json

与 SFT eval (sft_train.py --eval-only) 输出格式完全一致，可直接喂给 auto_grade.py 批改。

用法:
    python eval_baseline.py
    python eval_baseline.py --test_path sft/test.json
    python eval_baseline.py --model_path /data/magnus/models/Qwen2.5-72B-Instruct --test_path /path/to/test.json
"""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime

import magnus

from config import auto_source, notify_exe, SYSTEM_ENTRY_COMMAND, wait_for_job, MAGNUS_ADDRESS, MAGNUS_TOKEN

# ============================================================
#  CONFIG — 修改这里然后运行: python eval_baseline.py
# ============================================================

# -- 模型路径（Magnus 集群上的路径）--
MODEL_PATH = "/data/$(whoami)/models/Qwen2.5-72B-Instruct"

# -- 本地待上传的测试集 --
TEST_PATH = "sft/test.json"

# -- 推理参数 --
MAX_LENGTH     = 2048
MAX_NEW_TOKENS = 512

# -- 硬件资源 --
GPU_COUNT = 1
GPU_TYPE  = "a100"
CPU_COUNT = 16
MEMORY    = "128G"
STORAGE   = "200G"
PRIORITY  = "A2"
CONTAINER_IMAGE = "docker://crpi-32rssczyu25r10yu.cn-beijing.personal.cr.aliyuncs.com/zyz25/sft-base:v2"

# -- 输出目录（Magnus 集群上）--
OUTPUT_DIR = "/data/$(whoami)/eval_baseline"

# ============================================================


# ── Magnus 端推理脚本 ────────────────────────────────────────────
_EVAL_BASELINE_PY = r'''
import json
import os
import re
import time as _time
from datetime import datetime

def log(msg: str) -> None:
    print(f"[{_time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ── 环境变量 ──
MODEL_PATH     = os.getenv("EVAL_MODEL_PATH", "")
TEST_FILE      = os.getenv("EVAL_TEST_FILE", "/tmp/test.json")
OUTPUT_DIR     = os.getenv("EVAL_OUTPUT_DIR", "")
MAX_LENGTH     = int(os.getenv("EVAL_MAX_LENGTH", "2048"))
MAX_NEW_TOKENS = int(os.getenv("EVAL_MAX_NEW_TOKENS", "512"))

log(f"[baseline_eval] model_path={MODEL_PATH}")
log(f"[baseline_eval] output_dir={OUTPUT_DIR}")

# ── 解析 答案/解答 ──
def parse_answer_solution(text: str):
    """将 '答案：...\\n\\n解答：...' 拆分为 (answer, solution)。"""
    if not text:
        return "", ""
    parts = re.split(r'\n?\s*解答\s*[：:]\s*', text, maxsplit=1)
    if len(parts) >= 2:
        ans_part = parts[0]
        sol = parts[1].strip()
        m = re.search(r'答案\s*[：:]\s*(.*?)$', ans_part, re.DOTALL)
        ans = m.group(1).strip() if m else ans_part.strip()
        return ans, sol
    m_ans = re.search(r'答案\s*[：:]\s*(.*?)(?:\n\n|\n解答|$)', text, re.DOTALL)
    m_sol = re.search(r'解答\s*[：:]\s*(.*?)$', text, re.DOTALL)
    ans = m_ans.group(1).strip() if m_ans else ""
    sol = m_sol.group(1).strip() if m_sol else ""
    if not ans and not sol:
        return text.strip(), ""
    return ans, sol

# ── 读取测试集 ──
log("[1/5] 读取测试集...")
with open(TEST_FILE, "r", encoding="utf-8") as f:
    raw = f.read().strip()
samples = json.loads(raw) if raw.startswith("[") else [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
log(f"[1/5] 已加载 {len(samples)} 条测试样本")

# ── 加载模型 ──
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

# ── 逐样本推理 ──
log(f"[4/5] 开始生成式推理 ({len(samples)} 条)...")
results = []

for i, sample in enumerate(samples):
    instruction = sample.get("instruction", "")
    extra       = sample.get("input", "")
    gt_out      = sample.get("output", "")

    user_content = instruction + ("\n" + extra if extra else "")

    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
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

# ── 统计 ──
log(f"[5/5] 保存结果...")
total = len(results)

summary = {
    "total": total,
    "eval_type": "baseline",
    "model_path": MODEL_PATH,
    "generated_at": datetime.now().isoformat(),
}

# ── 保存 ──
os.makedirs(OUTPUT_DIR, exist_ok=True)

out_full = os.path.join(OUTPUT_DIR, "eval_results_initial.json")
with open(out_full, "w", encoding="utf-8") as f:
    json.dump({
        "summary": summary,
        "results": results,
    }, f, ensure_ascii=False, indent=2)
log(f"[5/5] 完整结果: {out_full}")

out_summary = os.path.join(OUTPUT_DIR, "eval_results_summary.json")
with open(out_summary, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
log(f"[5/5] 摘要: {out_summary}")

print()
print("=" * 60)
print("  基线评估完成")
print(f"  样本数:     {total}")
print(f"  输出目录:   {OUTPUT_DIR}")
print("=" * 60)
print()
print("  === 输出文件 ===")
print(f"  完整推理结果: {OUTPUT_DIR}/eval_results_initial.json")
print(f"  摘要统计:     {OUTPUT_DIR}/eval_results_summary.json")
print()
print("  === 创建下载链接 ===")
print("  请从 Magnus 作业日志末尾查看 magnus receive 命令")
print()
'''


def _build_entry_command(secret_id: str) -> str:
    """构建容器入口命令。"""
    eval_b64 = base64.b64encode(_EVAL_BASELINE_PY.encode("utf-8")).decode("ascii")

    return fr"""set -e

echo "============================================"
echo "  Baseline Eval: 原始模型生成式推理"
echo "  启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# ── 解码脚本 ──
echo "{eval_b64}" | base64 -d > /tmp/eval_baseline.py
python3 -c "compile(open('/tmp/eval_baseline.py').read(), '/tmp/eval_baseline.py', 'exec'); print('[check] eval_baseline.py syntax OK')"

# ── 解析模型路径 ──
MODEL_DIR="{MODEL_PATH}"
MODEL_DIR=$(eval echo "$MODEL_DIR")
echo "[check] model dir: $MODEL_DIR"
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "[FATAL] config.json not found in $MODEL_DIR"
    ls "$MODEL_DIR" 2>/dev/null || echo "(directory missing)"
    exit 1
fi
echo "[check] model OK ($(du -sh "$MODEL_DIR" | cut -f1))"

OUTPUT_DIR="{OUTPUT_DIR}"
OUTPUT_DIR=$(eval echo "$OUTPUT_DIR")
mkdir -p "$OUTPUT_DIR"

# ── 拉取测试数据 ──
echo "[check] 拉取测试数据: {secret_id}"
python3 -c "
import urllib.request
url = '{MAGNUS_ADDRESS}api/files/download/' + '{secret_id}'.replace('magnus-secret:', '')
urllib.request.urlretrieve(url, '/tmp/test.json')
"
echo "[check] test data downloaded ($(wc -c < /tmp/test.json) bytes)"

# ── 导出环境变量 ──
export EVAL_MODEL_PATH="$MODEL_DIR"
export EVAL_TEST_FILE="/tmp/test.json"
export EVAL_OUTPUT_DIR="$OUTPUT_DIR"
export EVAL_MAX_LENGTH="{MAX_LENGTH}"
export EVAL_MAX_NEW_TOKENS="{MAX_NEW_TOKENS}"
export MAGNUS_ADDRESS="{MAGNUS_ADDRESS}"
export MAGNUS_TOKEN="{MAGNUS_TOKEN}"

# ── 执行推理 ──
echo ""
echo "=== starting baseline eval ==="
python3 /tmp/eval_baseline.py
EVAL_EXIT=$?

echo ""
echo "=== 上传推理结果到 Magnus ==="
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


def _read_and_count(test_path: str) -> int:
    """读取本地测试集 JSON 并返回样本数。"""
    if not os.path.exists(test_path):
        print(f"[错误] 文件不存在: {test_path}")
        raise SystemExit(1)

    with open(test_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    data = json.loads(raw) if raw.startswith("[") else [json.loads(ln) for ln in raw.splitlines() if ln.strip()]

    return len(data)


def main():
    parser = argparse.ArgumentParser(
        description="Baseline eval: 上传测试集到 Magnus 并用原始模型做推理")
    parser.add_argument("--test_path", default=TEST_PATH,
                        help=f"本地测试集 JSON 文件 (default: {TEST_PATH})")
    parser.add_argument("--model_path", default=MODEL_PATH,
                        help="Magnus 集群上的模型路径")
    parser.add_argument("--address", default=MAGNUS_ADDRESS)
    parser.add_argument("--token",   default=MAGNUS_TOKEN)
    parser.add_argument("--poll_interval", type=int, default=30,
                        help="监控轮询间隔秒数 (0=跳过监控)")
    args = parser.parse_args()

    # ── 1. 读取本地文件 ──
    print(f"[1/6] 读取本地测试集: {args.test_path}")
    sample_count = _read_and_count(args.test_path)
    print(f"      样本数: {sample_count}")

    # ── 2. configure Magnus ──
    print(f"[2/6] 配置 Magnus 连接: {args.address}")
    magnus.configure(address=args.address, token=args.token)

    # ── 3. 上传测试集到 Magnus ──
    print(f"[3/6] 上传测试集到 Magnus...")
    secret_id = magnus.custody_file(args.test_path, expire_minutes=1440)
    print(f"      magnus-secret: {secret_id}")

    # ── 4. submit job ──
    print(f"[4/6] 提交基线评估任务...")
    print(f"      模型: {args.model_path}")
    print(f"      硬件: {GPU_COUNT}x {GPU_TYPE} | CPU={CPU_COUNT} | MEM={MEMORY}")

    entry = _build_entry_command(secret_id)
    job_id = magnus.submit_job(
        task_name          = "BaselineEval",
        description        = f"Baseline model inference on test set ({sample_count} samples)",
        entry_command      = entry,
        system_entry_command = SYSTEM_ENTRY_COMMAND,
        namespace          = "Rise-AGI",
        repo_name          = "OpenFundus",
        gpu_count          = GPU_COUNT,
        gpu_type           = GPU_TYPE,
        cpu_count          = CPU_COUNT,
        memory_demand      = MEMORY,
        ephemeral_storage  = STORAGE,
        job_type           = PRIORITY,
        container_image    = CONTAINER_IMAGE if CONTAINER_IMAGE else None,
    )
    print(f"      Job ID: {job_id}")

    # ── 5. notify + monitor ──
    print(f"[5/6] 等待模型加载和推理执行...")
    notify_exe(job_id=job_id)

    if args.poll_interval > 0:
        print(f"[6/6] 监控日志 (Ctrl+C 退出，任务继续)")
        print(f"{'='*60}")
        wait_for_job(job_id, poll_interval=args.poll_interval)

    # ── 打印结果 ──
    print()
    print("=" * 60)
    print("  Baseline Eval 任务已提交")
    print(f"  Job ID: {job_id}")
    job = magnus.get_job(job_id)
    print(f"  状态: {job.get('status', 'unknown')}")
    print()
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"    eval_results_initial.json  — 完整推理结果（未批改）")
    print(f"    eval_results_summary.json  — 摘要统计")
    print()
    print(f"  查看日志: magnus logs {job_id}")
    print(f"  下载结果: 查看上方 Magnus 日志末尾的 magnus receive 命令")
    print()
    print(f"  下一步: 将 eval_results_initial.json 复制/重命名为")
    print(f"          eval_results_final.json 后运行 python auto_grade.py 批改")
    print("=" * 60)


if __name__ == "__main__":
    main()
