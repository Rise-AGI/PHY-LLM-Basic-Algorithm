"""
Auto-grader: 上传 eval_results 到 Magnus，用 LLM 逐条批改，生成批改报告。

工作流程:
  1. 读取本地 eval_results_final.json
  2. 通过 magnus custody 上传到 Magnus
  3. 提交 Magnus job（容器内通过 magnus receive 拉取数据）
  4. 加载已有模型，逐条批改
  5. 每条加入 result (true/false), process_score (0-100), comment (简短评语)
  6. 末尾输出正确率和平均过程分

用法:
    python auto_grade.py
    python auto_grade.py --input train/eval_results_final.json
    python train/auto_grade.py --model_path /data/magnus/models/Qwen2.5-Math-7B-Instruct
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
#  CONFIG — 修改这里然后运行: python auto_grade.py
# ============================================================

# -- 模型路径（Magnus 集群上的路径）--
MODEL_PATH = "/data/$(whoami)/models/Qwen2.5-72B-Instruct"

# -- 量化设置 --
QUANTIZATION = "4bit"  # "4bit" 用 bitsandbytes NF4, "none" 用 bf16

# -- 本地待上传的 eval 结果文件 --
INPUT_FILE = "sft/eval_results_final.json"

# -- 批改提示词模板 --
# 占位符: {question} {gt_output} {model_output}
GRADING_PROMPT = """你是一个严格的数学/逻辑批改老师。请比较"学生解答"与"标准答案"，判断是否正确。

【题目】
{question}

【标准答案】
{gt_output}

【学生解答】
{model_output}

请严格按以下JSON格式输出（只输出JSON，不要任何额外文字）：
{{"result": true或false, "process_score": 0到100的整数, "comment": "简短中文评语, 不超过30字"}}"""

# -- 推理参数 --
MAX_NEW_TOKENS  = 256
MAX_INPUT_LENGTH = 2048

# -- 硬件资源（72B 4-bit ≈ 36GB VRAM/卡，1×A100 80G 足够）--
GPU_COUNT = 1
GPU_TYPE  = "a100"
CPU_COUNT = 16
MEMORY    = "128G"
STORAGE   = "200G"
PRIORITY  = "A2"
CONTAINER_IMAGE = "docker://crpi-32rssczyu25r10yu.cn-beijing.personal.cr.aliyuncs.com/zyz25/sft-base:v2"

# -- 输出目录（Magnus 集群上）--
OUTPUT_DIR = "/data/$(whoami)/auto_grade"

# ============================================================


# ── Magnus 端批改脚本（base64 注入容器）────────────────────
_GRADER_PY = r'''
import base64
import json
import os
import re
import sys
import time as _time
from datetime import datetime

def log(msg: str) -> None:
    print(f"[{_time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ── 环境变量 ──
MODEL_PATH         = os.getenv("GRADER_MODEL_PATH", "")
INPUT_FILE         = os.getenv("GRADER_INPUT_FILE", "/tmp/input.b64")
OUTPUT_DIR         = os.getenv("GRADER_OUTPUT_DIR", "")
MAX_NEW_TOKENS     = int(os.getenv("GRADER_MAX_NEW_TOKENS", "256"))
MAX_INPUT_LENGTH   = int(os.getenv("GRADER_MAX_INPUT_LENGTH", "2048"))
PROMPT_TEMPLATE_B64 = os.getenv("GRADER_PROMPT_TEMPLATE_B64", "")
PROMPT_TEMPLATE    = base64.b64decode(PROMPT_TEMPLATE_B64).decode("utf-8") if PROMPT_TEMPLATE_B64 else ""

log(f"[auto_grade] model_path={MODEL_PATH}")
log(f"[auto_grade] output_dir={OUTPUT_DIR}")

# ── 读取输入文件（由 entry_command 通过 magnus receive 拉取）──
log("[1/6] 读取输入文件...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    samples = json.load(f)
log(f"[1/6] 已加载 {len(samples)} 条待批改样本")

# ── 加载模型 ──
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

QUANTIZATION = os.getenv("GRADER_QUANTIZATION", "none")

log(f"[2/6] 加载 tokenizer: {MODEL_PATH}")
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
log(f"[2/6] tokenizer 加载完成 | vocab_size={tokenizer.vocab_size}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 量化配置 ──
if QUANTIZATION == "4bit" and device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    load_kwargs = dict(quantization_config=bnb_config, device_map="auto")
    log(f"[3/6] 加载模型到 {device} (4-bit QLoRA / NF4)...")
else:
    bnb_config = None
    load_kwargs = dict(
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    log(f"[3/6] 加载模型到 {device} (bf16)...")

t0 = _time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    **load_kwargs,
)
if device == "cpu":
    model = model.to(device)
model.eval()
log(f"[3/6] 模型加载完成 ({_time.time() - t0:.1f}s)")

if device == "cuda":
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        log(f"  GPU {i}: 空闲 {free/1e9:.1f}GB / 总计 {total/1e9:.1f}GB")

# ── JSON 解析辅助 ──
def _extract_json(text: str) -> dict | None:
    """从模型输出中提取 JSON 对象。"""
    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # 尝试匹配 ```json ... ``` 代码块
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 尝试匹配第一个 {...}
    m = re.search(r'\{[^{}]*"result"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # 更宽松的匹配：任意 {...}
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "result" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _parse_grading(raw_response: str) -> dict:
    """从模型原始输出中解析批改结果，始终返回有效 dict。"""
    parsed = _extract_json(raw_response)
    if parsed is not None:
        result = parsed.get("result", None)
        if isinstance(result, str):
            result = result.lower() in ("true", "yes", "1", "正确")
        elif isinstance(result, bool):
            pass
        else:
            result = None

        score = parsed.get("process_score", None)
        if isinstance(score, str):
            try:
                score = int(score)
            except ValueError:
                score = None
        elif isinstance(score, (int, float)):
            score = int(score)
        else:
            score = None

        comment = parsed.get("comment", "")
        if not isinstance(comment, str):
            comment = str(comment) if comment else ""

        return {
            "result": result if result is not None else False,
            "process_score": max(0, min(100, score if score is not None else 0)),
            "comment": comment[:100] if comment else "（无法解析评语）",
            "raw_response": raw_response,
            "parse_ok": True,
        }

    # 完全无法解析 JSON——启发式判断
    text_lower = raw_response.lower()
    likely_correct = any(w in text_lower for w in ["正确", "true", "correct", "一致", "相同"])
    likely_wrong   = any(w in text_lower for w in ["错误", "false", "incorrect", "不一致", "不同"])

    if likely_correct and not likely_wrong:
        result = True
    elif likely_wrong and not likely_correct:
        result = False
    else:
        result = False  # 保守默认

    return {
        "result": result,
        "process_score": 50,
        "comment": "（模型输出格式异常，启发式判断）",
        "raw_response": raw_response,
        "parse_ok": False,
    }


# ── 错误分类 ──
def _classify_error(comment: str, result: bool, parse_ok: bool) -> str:
    """根据评语和结果分类错误类型。"""
    if result:
        return "完全正确"

    if not parse_ok:
        return "格式异常"

    c = comment

    # 部分正确：部分题对 / 符号错误但过程对
    if any(w in c for w in ["部分", "第2题", "第(2)", "其余正确", "基本正确", "符号错误", "最后一步推导不完整"]):
        return "部分正确"

    # 方法/定理错误
    if any(w in c for w in ["定理", "公式应用", "方法错误", "格林公式", "斯托克斯", "高斯散度",
                              "奥氏公式", "帕斯卡", "使用错误", "应用不当"]):
        return "方法错误"

    # 推导不完整
    if any(w in c for w in ["推导不完整", "推导不完全", "未给出详细推导", "缺少关键步骤",
                              "缺少对三维", "未求出", "仅给出微分", "未明确指出", "推导过程不严谨",
                              "需重新审视", "推导过程冗余"]):
        return "推导不完整"

    # 计算错误（含偏导数/积分/梯度等具体错误）
    if any(w in c for w in ["计算", "积分", "偏导数", "梯度", "旋度", "参数方程",
                              "表达式错误", "常数项", "理解不准确", "理解偏差", "积分式",
                              "积分性质", "积分上下限", "积分区域", "积分路径", "路径",
                              "体积积分", "侧表面", "底面通量", "通量"]):
        return "计算错误"

    # 题目理解错误（答非所问）
    if any(w in c for w in ["题目要求", "而非", "理解有误", "理解错误", "计算了错误的表达式"]):
        return "题目理解错误"

    # 答案形式问题
    if any(w in c for w in ["形式不", "需简化", "冗余", "不完全等价", "标准形式"]):
        return "答案形式问题"

    return "其他错误"


def _grade_one(sample: dict, idx: int, total: int) -> dict:
    """批改单条样本。"""
    question     = sample.get("question", sample.get("instruction", ""))
    gt_output    = sample.get("gt_output") or sample.get("gt_full") or sample.get("output") or sample.get("gt_answer") or ""
    model_output = sample.get("model_output") or sample.get("model_full") or sample.get("model_answer") or ""

    prompt_text = PROMPT_TEMPLATE.replace("{question}", question)
    prompt_text = prompt_text.replace("{gt_output}", gt_output)
    prompt_text = prompt_text.replace("{model_output}", model_output)

    messages = [{"role": "user", "content": prompt_text}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = f"User: {prompt_text}\n\nAssistant: "

    inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                       max_length=MAX_INPUT_LENGTH).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids  = out_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    grading = _parse_grading(response)

    graded = dict(sample)
    graded["grading"] = {
        "result":        grading["result"],
        "process_score": grading["process_score"],
        "comment":       grading["comment"],
    }
    graded["grading_raw"] = response

    if (idx + 1) % 5 == 0 or (idx + 1) == total:
        r = "OK" if grading["result"] else "WRONG"
        log(f"  [{idx+1}/{total}] {r} | score={grading['process_score']} | {grading['comment'][:40]}")

    return graded


# ── 主流程 ──
log(f"[4/6] 开始批改 ({len(samples)} 条)...")
graded_results = []
for i, sample in enumerate(samples):
    graded = _grade_one(sample, i, len(samples))
    graded_results.append(graded)

# ── 统计 ──
log(f"[5/6] 计算统计...")
total       = len(graded_results)
correct     = sum(1 for g in graded_results if g["grading"]["result"])
scores      = [g["grading"]["process_score"] for g in graded_results]
accuracy    = correct / total if total > 0 else 0.0
avg_score   = sum(scores) / total if total > 0 else 0.0
parse_ok_n  = sum(1 for g in graded_results if "grading_raw" in g and True)
# 统计原始响应中可成功解析 JSON 的数量
json_ok      = sum(1 for g in graded_results
                   if _extract_json(g.get("grading_raw", "")) is not None)


# ── 错误分类统计 ──
from collections import Counter
error_categories = Counter()
for g in graded_results:
    parse_ok = g.get("grading_raw", "") and _extract_json(g.get("grading_raw", "")) is not None
    cat = _classify_error(g["grading"]["comment"], g["grading"]["result"], parse_ok)
    error_categories[cat] += 1

summary = {
    "total":            total,
    "correct":          correct,
    "wrong":            total - correct,
    "accuracy":         round(accuracy, 4),
    "accuracy_pct":     f"{accuracy*100:.1f}%",
    "avg_process_score": round(avg_score, 2),
    "min_score":         min(scores) if scores else 0,
    "max_score":         max(scores) if scores else 0,
    "json_parse_ok":    json_ok,
    "json_parse_fail":  total - json_ok,
    "error_categories": {cat: count for cat, count in error_categories.most_common()},
    "graded_at":        datetime.now().isoformat(),
}

log(f"[5/6] 正确率: {summary['accuracy_pct']}  ({correct}/{total})")
log(f"[5/6] 平均过程分: {summary['avg_process_score']}/100")
log(f"[5/6] JSON 解析成功: {json_ok}/{total}")
log(f"[5/6] 错误分类:")
for cat, count in error_categories.most_common():
    pct = count / total * 100 if total > 0 else 0
    log(f"        {cat}: {count} ({pct:.1f}%)")

# ── 保存 ──
log(f"[6/6] 保存批改报告...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
	
	# 完整报告（含每条 grading）
out_full = os.path.join(OUTPUT_DIR, "eval_results_graded.json")
with open(out_full, "w", encoding="utf-8") as f:
    json.dump({
        "summary": summary,
        "results": graded_results,
    }, f, ensure_ascii=False, indent=2)
log(f"[6/6] 完整报告: {out_full}")

# 摘要单独保存
out_summary = os.path.join(OUTPUT_DIR, "eval_results_summary.json")
with open(out_summary, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
log(f"[6/6] 摘要报告: {out_summary}")

# 打印最终摘要
print()
print("=" * 60)
print("  批改完成")
print(f"  样本数:     {total}")
print(f"  正确率:     {summary['accuracy_pct']}  ({correct}/{total})")
print(f"  平均过程分: {summary['avg_process_score']:.1f}/100")
print(f"  最低/最高:  {summary['min_score']}/{summary['max_score']}")
print(f"  JSON 解析:  {json_ok}/{total} 成功")
print(f"  输出目录:   {OUTPUT_DIR}")
print("=" * 60)
print()
print()
print("  === 输出文件 ===")
print(f"  完整批改报告: {OUTPUT_DIR}/eval_results_graded.json")
print(f"  摘要统计:     {OUTPUT_DIR}/eval_results_summary.json")
print()
print("  === 创建下载链接 ===")
print("  请从 Magnus 作业日志末尾查看 magnus receive 命令")
print()'''


def _build_entry_command(secret_id: str) -> str:
    """构建容器入口命令。"""
    grader_b64 = base64.b64encode(_GRADER_PY.encode("utf-8")).decode("ascii")
    prompt_b64 = base64.b64encode(GRADING_PROMPT.encode("utf-8")).decode("ascii")

    return fr"""set -e

echo "============================================"
echo "  Auto-Grader: LLM 批改 eval_results"
echo "  启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# ── 解码脚本 ──
echo "{grader_b64}" | base64 -d > /tmp/auto_grade.py
python3 -c "compile(open('/tmp/auto_grade.py').read(), '/tmp/auto_grade.py', 'exec'); print('[check] auto_grade.py syntax OK')"

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

# ── 拉取输入数据（Python urllib 下载，标准库始终可用）──
echo "[check] 拉取输入数据: {secret_id}"
python3 -c "
import urllib.request
url = '{MAGNUS_ADDRESS}api/files/download/' + '{secret_id}'.replace('magnus-secret:', '')
urllib.request.urlretrieve(url, '/tmp/input.json')
"
echo "[check] input data downloaded ($(wc -c < /tmp/input.json) bytes)"

# ── 导出环境变量 ──
export GRADER_MODEL_PATH="$MODEL_DIR"
export GRADER_INPUT_FILE="/tmp/input.json"
export GRADER_OUTPUT_DIR="$OUTPUT_DIR"
export GRADER_QUANTIZATION="{QUANTIZATION}"
export GRADER_MAX_NEW_TOKENS="{MAX_NEW_TOKENS}"
export GRADER_MAX_INPUT_LENGTH="{MAX_INPUT_LENGTH}"
export GRADER_PROMPT_TEMPLATE_B64="{prompt_b64}"
export MAGNUS_ADDRESS="{MAGNUS_ADDRESS}"
export MAGNUS_TOKEN="{MAGNUS_TOKEN}"

# ── 执行批改 ──
echo ""
echo "=== starting auto-grader ==="
python3 /tmp/auto_grade.py
GRADER_EXIT=$?

echo ""
echo "=== 上传批改结果到 Magnus ==="
python3 << 'PYEOF'
import json, os, sys

out_dir = os.environ.get("GRADER_OUTPUT_DIR", "")
magnus_addr = os.environ.get("MAGNUS_ADDRESS", "")
magnus_token = os.environ.get("MAGNUS_TOKEN", "")
files = [
    ("eval_results_graded.json", "完整批改报告"),
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

exit $GRADER_EXIT
"""


def _read_and_count(input_path: str) -> int:
    """读取本地 JSON 文件并返回样本数。"""
    if not os.path.exists(input_path):
        print(f"[错误] 文件不存在: {input_path}")
        raise SystemExit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[错误] JSON 顶层应为数组，实际为: {type(data).__name__}")
        raise SystemExit(1)

    return len(data)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-grader: 上传 eval_results 到 Magnus 并用 LLM 批改")
    parser.add_argument("--input", default=INPUT_FILE,
                        help=f"本地 eval_results JSON 文件 (default: {INPUT_FILE})")
    parser.add_argument("--model_path", default=MODEL_PATH,
                        help="Magnus 集群上的模型路径")
    parser.add_argument("--address", default=MAGNUS_ADDRESS)
    parser.add_argument("--token",   default=MAGNUS_TOKEN)
    parser.add_argument("--poll_interval", type=int, default=30,
                        help="监控轮询间隔秒数 (0=跳过监控)")
    args = parser.parse_args()

    # ── 1. 读取本地文件 ──
    print(f"[1/6] 读取本地文件: {args.input}")
    sample_count = _read_and_count(args.input)
    print(f"      样本数: {sample_count}")

    # ── 2. configure Magnus ──
    print(f"[2/6] 配置 Magnus 连接: {args.address}")
    magnus.configure(address=args.address, token=args.token)

    # ── 3. 上传到 Magnus ──
    print(f"[3/6] 上传 eval_results 到 Magnus...")
    secret_id = magnus.custody_file(args.input, expire_minutes=1440)
    print(f"      magnus-secret: {secret_id}")

    # ── 4. submit job ──
    print(f"[4/6] 提交批改任务...")
    print(f"      模型: {args.model_path}")
    print(f"      硬件: {GPU_COUNT}x {GPU_TYPE} | CPU={CPU_COUNT} | MEM={MEMORY}")

    entry = _build_entry_command(secret_id)
    job_id = magnus.submit_job(
        task_name          = "AutoGrade-eval",
        description        = f"LLM auto-grading of eval_results ({sample_count} samples)",
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
    print(f"[5/6] 等待模型加载和批改执行...")
    notify_exe(job_id=job_id)

    if args.poll_interval > 0:
        print(f"[6/6] 监控日志 (Ctrl+C 退出，任务继续)")
        print(f"{'='*60}")
        wait_for_job(job_id, poll_interval=args.poll_interval)

    # ── 打印结果 ──
    print()
    print("=" * 60)
    print("  Auto-Grade 任务已提交")
    print(f"  Job ID: {job_id}")
    job = magnus.get_job(job_id)
    print(f"  状态: {job.get('status', 'unknown')}")
    print()
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"    eval_results_graded.json   — 完整批改报告")
    print(f"    eval_results_summary.json  — 摘要统计")
    print()
    print(f"  查看日志: magnus logs {job_id}")
    print(f"  下载结果: 查看上方 Magnus 日志末尾的 magnus receive 命令")
    print("=" * 60)


if __name__ == "__main__":
    main()
