"""
Submit an OpenAI-compatible API service to Magnus (long-running).

API 格式与 OpenAI 官方完全兼容，可用 openai Python SDK 直接调用:
    from openai import OpenAI
    client = OpenAI(base_url="https://xxx.ngrok-free.app/v1", api_key="sk-xxx")
    client.chat.completions.create(model="...", messages=[...])

公网隧道:
  - ngrok (推荐, 免费注册 30s): 在下方设置 NGROK_AUTH_TOKEN
  - 如不填 ngrok token，仅在 Magnus 集群内网可访问

用法:
    python serve_model.py                          # 使用下方 CONFIG
    python serve_model.py --model_dir /data/...    # 覆盖模型路径
    python serve_model.py --address http://...     # 覆盖 Magnus 地址

API 端点:
    POST /v1/chat/completions   (OpenAI 兼容)
    POST /v1/completions        (OpenAI 兼容)
    GET  /v1/models
    GET  /health
"""

import argparse
import base64
import os
import sys
import time
from datetime import datetime

import magnus

from config import auto_source, notify_exe, SYSTEM_ENTRY_COMMAND, wait_for_job, MAGNUS_ADDRESS, MAGNUS_TOKEN, API_TOKENS

# ============================================================
#  CONFIG — 修改这里然后运行: python serve_model.py
# ============================================================

# -- 模型路径 --
# download_model_auto.py 下载后模型在 /data/<username>/models/<model_name>/
# 留空 = 自动: /data/$(whoami)/models/{MODEL_NAME}
MODEL_DIR  = ""
MODEL_NAME = "Qwen2.5-Math-7B-Instruct"

# -- 推理默认值 --
DEFAULT_MAX_TOKENS       = 2048
DEFAULT_TEMPERATURE      = 0.7
DEFAULT_TOP_P            = 0.9
DEFAULT_SYSTEM_PROMPT    = "你是一个有用的AI助手，请认真回答用户的问题。"
# 留空 = 不加默认 system prompt
# DEFAULT_SYSTEM_PROMPT    = ""

# -- 服务配置 --
SERVE_PORT      = 7860
TASK_NAME       = ""            # 留空 = 自动: ServeAPI-{MODEL_NAME}
NGROK_AUTH_TOKEN = ""           # ngrok 免费隧道 auth token (https://dashboard.ngrok.com/get-started/your-authtoken)
                                # 留空 = 仅内网，不创建公网隧道

# -- 硬件资源 --
GPU_COUNT = 1
GPU_TYPE  = "a100"              # "a100" / "v100" / "cpu"
CPU_COUNT = 8
MEMORY    = "32G"
STORAGE   = "100G"
PRIORITY  = "A2"
CONTAINER_IMAGE = "docker://crpi-32rssczyu25r10yu.cn-beijing.personal.cr.aliyuncs.com/zyz25/sft-base:v2"

# ============================================================


# ── 服务端 Python 脚本（base64 注入容器）────────────────────
_SERVE_PY = r'''
import os, sys, json, time as _time
from datetime import datetime

# ── 环境变量 ──────────────────────────────────────────────
MODEL_DIR      = os.getenv("SERVE_MODEL_DIR")
PORT           = int(os.getenv("SERVE_PORT", "7860"))
HOST           = os.getenv("SERVE_HOST", "0.0.0.0")
MAX_TOKENS     = int(os.getenv("SERVE_MAX_TOKENS", "2048"))
TEMPERATURE    = float(os.getenv("SERVE_TEMPERATURE", "0.7"))
TOP_P          = float(os.getenv("SERVE_TOP_P", "0.9"))
SYS_PROMPT     = os.getenv("SERVE_SYSTEM_PROMPT", "")
API_TOKENS     = set(t.strip() for t in os.getenv("SERVE_API_TOKENS", "").split(",") if t.strip())
NGROK_TOKEN    = os.getenv("SERVE_NGROK_TOKEN", "")
MODEL_ID       = os.path.basename(MODEL_DIR.rstrip("/"))

print(f"[serve] model={MODEL_ID}  port={PORT}  api_tokens={len(API_TOKENS)}", flush=True)
if not API_TOKENS:
    print("[serve] WARNING: no API tokens configured, all requests will be rejected!", flush=True)

# ── 加载模型 ──────────────────────────────────────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"[serve] loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[serve] loading model to {device}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None, trust_remote_code=True)
if device == "cpu":
    model = model.to(device)
model.eval()
print(f"[serve] model loaded on {device}", flush=True)
if device == "cuda":
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"[serve] GPU {i}: free={free/1e9:.1f}GB / total={total/1e9:.1f}GB", flush=True)

# ── FastAPI 应用 ──────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Union
import uvicorn

app = FastAPI(title=f"LLM-API-{MODEL_ID}", version="1.0.0")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: Union[str, List[str]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False

# ── 鉴权 ──────────────────────────────────────────────────
def _verify(auth: str) -> str:
    if not auth:
        raise HTTPException(401, "Missing Authorization header")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Use: Bearer sk-xxx")
    token = auth[7:].strip()
    if token not in API_TOKENS:
        tail = token[-6:] if len(token) >= 6 else token
        raise HTTPException(401, f"Invalid API key: ...{tail}")
    return token

# ── 请求日志 ──────────────────────────────────────────────
def _log(tok: str, ok: bool, in_n: int, out_n: int, ela: float, msg: str = ""):
    tail  = tok[-6:] if len(tok) >= 6 else tok
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    flag  = "OK" if ok else "FAIL"
    line  = f"[api] ...{tail} | {stamp} | {flag:4s} | in={in_n} out={out_n} | {ela:.1f}s"
    if msg:
        line += f" | {msg}"
    print(line, flush=True)

# ── 核心生成 ──────────────────────────────────────────────
@torch.no_grad()
def _generate_chat(messages: List[dict], temperature: float,
                   max_tokens: int, top_p: float) -> tuple:
    # 注入默认 system prompt
    msgs = list(messages)
    if SYS_PROMPT and not any(m["role"] == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": SYS_PROMPT})

    try:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        for m in msgs:
            r = m["role"]
            c = m["content"]
            if r == "system":
                parts.append(c + "\n")
            elif r == "user":
                parts.append(f"User: {c}\n\nAssistant: ")
            elif r == "assistant":
                parts.append(f"{c}\n\n")
        text = "".join(parts)
        if not text.endswith("Assistant: "):
            text += "Assistant: "

    inputs   = tokenizer(text, return_tensors="pt").to(device)
    in_count = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response  = tokenizer.decode(outputs[0][in_count:], skip_special_tokens=True).strip()
    out_count = outputs[0].shape[0] - in_count
    return response, in_count, out_count

@torch.no_grad()
def _generate_text(prompt: str, temperature: float,
                   max_tokens: int, top_p: float) -> tuple:
    inputs   = tokenizer(prompt, return_tensors="pt").to(device)
    in_count = inputs.input_ids.shape[1]
    outputs  = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response  = tokenizer.decode(outputs[0][in_count:], skip_special_tokens=True).strip()
    out_count = outputs[0].shape[0] - in_count
    return response, in_count, out_count

# ── POST /v1/chat/completions ─────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    token = _verify(request.headers.get("Authorization", ""))
    t0    = _time.time()
    temp  = req.temperature if req.temperature is not None else TEMPERATURE
    maxt  = req.max_tokens  if req.max_tokens  is not None else MAX_TOKENS
    topp  = req.top_p       if req.top_p       is not None else TOP_P

    try:
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        text, in_n, out_n = _generate_chat(msgs, temp, maxt, topp)
        elapsed = _time.time() - t0
        _log(token, True, in_n, out_n, elapsed)
        return {
            "id": f"chatcmpl-{int(t0*1000)}",
            "object": "chat.completion",
            "created": int(t0),
            "model": req.model or MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": in_n,
                "completion_tokens": out_n,
                "total_tokens": in_n + out_n,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        elapsed = _time.time() - t0
        _log(token, False, 0, 0, elapsed, str(e)[:80])
        raise HTTPException(500, f"generation failed: {e}")

# ── POST /v1/completions ──────────────────────────────────
@app.post("/v1/completions")
async def completions(req: CompletionRequest, request: Request):
    token = _verify(request.headers.get("Authorization", ""))
    t0    = _time.time()
    temp  = req.temperature if req.temperature is not None else TEMPERATURE
    maxt  = req.max_tokens  if req.max_tokens  is not None else MAX_TOKENS
    topp  = req.top_p       if req.top_p       is not None else TOP_P
    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]

    try:
        text, in_n, out_n = _generate_text(prompt, temp, maxt, topp)
        elapsed = _time.time() - t0
        _log(token, True, in_n, out_n, elapsed)
        return {
            "id": f"cmpl-{int(t0*1000)}",
            "object": "text_completion",
            "created": int(t0),
            "model": req.model or MODEL_ID,
            "choices": [{
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": in_n,
                "completion_tokens": out_n,
                "total_tokens": in_n + out_n,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        elapsed = _time.time() - t0
        _log(token, False, 0, 0, elapsed, str(e)[:80])
        raise HTTPException(500, f"generation failed: {e}")

# ── GET /v1/models ────────────────────────────────────────
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": 0,
            "owned_by": "user",
        }],
    }

# ── GET /health ───────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "device": device}

# ── 全局异常处理 ──────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "invalid_request_error",
                            "code": exc.status_code}},
    )

@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    print(f"[serve] unhandled error: {exc}", flush=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "server_error", "code": 500}},
    )

# ── 启动入口 ──────────────────────────────────────────────
def _start():
    public_url = None

    if NGROK_TOKEN:
        try:
            from pyngrok import ngrok, conf
            conf.get_default().auth_token = NGROK_TOKEN
            tunnel = ngrok.connect(PORT, "http",
                                   options={"log_format": "json", "log_level": "error"})
            public_url = tunnel.public_url
            print(f"[serve] ngrok tunnel active: {public_url} -> localhost:{PORT}", flush=True)
        except Exception as e:
            print(f"[serve] ngrok failed: {e}", flush=True)
            print(f"[serve] continuing without public tunnel...", flush=True)

    print("=" * 60, flush=True)
    print(f"[serve] FastAPI server: http://{HOST}:{PORT}", flush=True)
    print(f"[serve] Endpoints:", flush=True)
    print(f"         POST /v1/chat/completions", flush=True)
    print(f"         POST /v1/completions", flush=True)
    print(f"         GET  /v1/models", flush=True)
    print(f"         GET  /health", flush=True)
    if public_url:
        print(f"", flush=True)
        print(f"  PUBLIC API (ngrok): {public_url}", flush=True)
        print(f"", flush=True)
        print(f"  # 测试命令:", flush=True)
        print(f'  curl {public_url}/v1/chat/completions \\', flush=True)
        print(f'    -H "Authorization: Bearer <api-token>" \\', flush=True)
        print(f'    -H "Content-Type: application/json" \\', flush=True)
        print(f'    -d \'{{"messages":[{{"role":"user","content":"Hello"}}]}}\'', flush=True)
        print(f"", flush=True)
        print(f"  # Python SDK:", flush=True)
        print(f"  from openai import OpenAI", flush=True)
        print(f"  client = OpenAI(base_url='{public_url}', api_key='<api-token>')", flush=True)
        print(f"  r = client.chat.completions.create(model='{MODEL_ID}',", flush=True)
        print(f"        messages=[{{'role':'user','content':'Hello'}}])", flush=True)
    else:
        print(f"", flush=True)
        print(f"  [NOTE] 未配置 NGROK_AUTH_TOKEN，仅内网可访问", flush=True)
        print(f"  免费注册 ngrok: https://dashboard.ngrok.com/signup", flush=True)
    print("=" * 60, flush=True)

    uvicorn.run(app, host=HOST, port=PORT, log_level="warning", access_log=False)

if __name__ == "__main__":
    _start()
'''


def _build_entry_command(model_dir: str) -> str:
    """构建容器入口命令。"""
    serve_b64 = base64.b64encode(_SERVE_PY.encode("utf-8")).decode("ascii")
    api_tokens_str = ",".join(API_TOKENS)

    return fr"""set -e

echo "============================================"
echo "  Magnus Model Serving (OpenAI-compatible API)"
echo "  启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# ── 安装依赖 ──
echo "[setup] installing fastapi / uvicorn / ngrok..."
pip install -q fastapi uvicorn pyngrok 2>&1 | tail -3

# ── 检查模型 ──
MODEL_DIR="{model_dir}"
echo "[check] model dir: $MODEL_DIR"
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "[FATAL] config.json not found in $MODEL_DIR"
    ls "$MODEL_DIR" 2>/dev/null || echo "(directory missing)"
    exit 1
fi
echo "[check] model OK ($(du -sh "$MODEL_DIR" | cut -f1))"

# ── 解码服务脚本 ──
echo "{serve_b64}" | base64 -d > /tmp/serve.py
python3 -c "compile(open('/tmp/serve.py').read(), '/tmp/serve.py', 'exec'); print('[check] serve.py syntax OK')"

# ── 导出环境变量 ──
export SERVE_MODEL_DIR="$MODEL_DIR"
export SERVE_PORT="{SERVE_PORT}"
export SERVE_HOST="0.0.0.0"
export SERVE_MAX_TOKENS="{DEFAULT_MAX_TOKENS}"
export SERVE_TEMPERATURE="{DEFAULT_TEMPERATURE}"
export SERVE_TOP_P="{DEFAULT_TOP_P}"
export SERVE_SYSTEM_PROMPT="{DEFAULT_SYSTEM_PROMPT}"
export SERVE_API_TOKENS="{api_tokens_str}"
export SERVE_NGROK_TOKEN="{NGROK_AUTH_TOKEN}"

# ── 启动 API 服务（前台，持续运行）──
echo ""
echo "=== starting API server (OpenAI-compatible) ==="
echo "    port: {SERVE_PORT}"
echo "    ngrok: {'ON' if NGROK_AUTH_TOKEN else 'OFF'}"
echo ""
exec python3 /tmp/serve.py
"""


def _resolve_model_dir() -> str:
    if MODEL_DIR:
        return MODEL_DIR.rstrip("/")
    return f"/data/$(whoami)/models/{MODEL_NAME}"


def _resolve_task_name() -> str:
    if TASK_NAME:
        return TASK_NAME
    name = os.path.basename(MODEL_DIR.rstrip("/")) if MODEL_DIR else MODEL_NAME
    return f"ServeAPI-{name}"


def main():
    parser = argparse.ArgumentParser(
        description="Submit an OpenAI-compatible API service to Magnus")
    parser.add_argument("--address", default=MAGNUS_ADDRESS)
    parser.add_argument("--token",   default=MAGNUS_TOKEN)
    parser.add_argument("--model_dir", default="",
                        help="model path in /data (default: auto)")
    parser.add_argument("--port", type=int, default=SERVE_PORT)
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="monitor poll interval (0=skip monitor)")
    args = parser.parse_args()

    model_dir = args.model_dir or _resolve_model_dir()
    task_name = _resolve_task_name()
    has_ngrok = bool(NGROK_AUTH_TOKEN)

    print(f"[*] Magnus Model Serving (OpenAI-compatible API)")
    print(f"    server     : {args.address}")
    print(f"    task       : {task_name}")
    print(f"    model      : {model_dir}")
    print(f"    port       : {args.port}")
    print(f"    public URL : {'ngrok (auto)' if has_ngrok else '内网 only'}")
    print(f"    API tokens : {len(API_TOKENS)} configured")
    print(f"    GPU        : {GPU_COUNT} x {GPU_TYPE}")
    print()

    # ── 1. configure ──
    magnus.configure(address=args.address, token=args.token)

    # ── 2. submit ──
    entry = _build_entry_command(model_dir)
    print(f"[1/3] submitting job...")

    job_id = magnus.submit_job(
        task_name          = task_name,
        description        = f"OpenAI-compatible API for {MODEL_NAME or model_dir}",
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
    )
    print(f"    Job ID: {job_id}")
    print()

    # ── 3. monitor ──
    print(f"[2/3] 模型加载中（约 2-5 分钟），等待服务就绪...")
    print(f"[3/3] 监控日志 (Ctrl+C 退出，服务继续运行)")
    print(f"{'='*60}")

    notify_exe(job_id=job_id)

    if args.poll_interval > 0:
        wait_for_job(job_id, poll_interval=args.poll_interval)

    # ── 提示 ──
    print()
    print("=" * 60)
    print("  服务已提交到 Magnus")
    print()
    job = magnus.get_job(job_id)
    print(f"  Job ID       : {job_id}")
    print(f"  状态         : {job.get('status', 'unknown')}")
    if has_ngrok:
        print(f"  公网 API URL : 查看下方日志中的 'PUBLIC API' 行")
    else:
        print(f"  访问方式     : Magnus 集群内网 {args.port} 端口")
    print()
    print("  获取公网 URL:")
    print(f"    magnus logs {job_id} | grep -E 'PUBLIC API|ngrok-free'")
    print()
    print("  测试 API (获取到 URL 后):")
    print(f'    curl <PUBLIC_URL>/v1/chat/completions \\')
    print(f'      -H "Authorization: Bearer <your-token>" \\')
    print(f'      -H "Content-Type: application/json" \\')
    print(f'      -d \'{{"messages":[{{"role":"user","content":"Hello"}}]}}\'')
    print()
    print("  停止服务:")
    print(f"    magnus cancel {job_id}")
    print("=" * 60)

    # ── 额外：打印 API token 提示 ──
    print()
    print("配置的 API tokens (用于 Authorization: Bearer):")
    for t in sorted(API_TOKENS):
        print(f"    ...{t[-6:]}")


if __name__ == "__main__":
    main()
