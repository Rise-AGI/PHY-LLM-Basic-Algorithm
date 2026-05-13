"""
Submit SFT training via Magnus blueprint (save to server + launch).

Workflow: read .magnus blueprint → save to server → wait 30s → launch → monitor

Usage:
    python submit_sft.py                          # use config section below
    python submit_sft.py --address http://...     # override connection only
"""

import argparse
import base64
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional

import magnus

from config import (
    auto_source, notify_exe, wait_for_job,
    record_storage, check_model_version_exists,
    SFT_DATA_DIR, _ensure_record, MAGNUS_ADDRESS, MAGNUS_TOKEN,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
#  CONFIG — edit here then run: python submit_sft.py
# ============================================================

# ── 模式 + 模型路径（放在最前面，输出目录/任务名自动从此推导）──
MODE        = "sft"          # "sft" = 全参微调, "lora" = LoRA/QLoRA
MODEL_PATH  = "/data/magnus/models/Qwen2.5-Math-7B-Instruct"

# ── 以下参数通常无需手动修改（留空则自动推导） ───────────────
# 蓝图文件（留空=根据 MODE 自动选择 .magnus）
BLUEPRINT_FILE  = ""         # 留空自动

# 输出目录（留空=自动: {model_short_name}-{mode}-v{version}）
OUTPUT_DIR      = ""         # 留空自动

# 模型版本  # None = 自动递增
MODEL_VERSION   = None
# 训练数据集  # None = 使用模拟数据
TRAIN_DATA      = None
# 测试数据集
TEST_DATA       = None

# -- 超参数 --
# 训练轮次
EPOCHS          = 3
# 批次大小
BATCH_SIZE      = 1
# 梯度累积步数
GRAD_ACCUM      = 8
# 学习率
LEARNING_RATE   = 2e-5
# 最大序列长度
MAX_LENGTH      = 1024
# DataLoader worker 进程数（0=主进程加载，2-4 可加速）
NUM_WORKERS     = 4
# -- 硬件资源与任务调度 --
# GPU数量
GPU_COUNT       = 2
# GPU型号  # "a100"（英伟达A100显卡） 或 "cpu"（仅CPU）
GPU_TYPE        = "a100"        
# CPU核心数
CPU_COUNT       = 40
# 内存大小
MEMORY          = "160G"
# 存储容量
STORAGE         = "1024G"
# 任务优先级  # A1 / A2 / B1 / B2（优先级依次降低）
PRIORITY        = "A2"          
# 容器镜像地址  # 预装pip依赖环境
CONTAINER_IMAGE = "docker://crpi-32rssczyu25r10yu.cn-beijing.personal.cr.aliyuncs.com/zyz25/sft-base:v2"  
# 断点续训路径  # None = 从头训练
RESUME_FROM     = None
# 统一提示词前缀  # None = 不修改原始 instruction；支持 {instruction} 占位符
PROMPT_PREFIX   = (
    '你是一位数学解题专家。请逐步推理并解答以下问题。\n'
    '\n'
    '输出格式：\n'
    '答案：[最终结果。多问时用 (1)...; (2)... 分别列出。数学公式使用 LaTeX $...$]\n'
    '\n'
    '解答：[完整推导过程。写明所用的定理、公式或变换方法，关键推导步骤不可省略]\n'
    '\n'
    '{instruction}'
)

# -- GitHub 同步 --
GITHUB_REPO_PATH = os.path.join(HERE, "..", "PHY-LLM-Basic-Algorithm")
GITHUB_PATH      = "PHY-LLM-Basic-Algorithm/sft_train.py"

# ============================================================



def _auto_model_version(short_name: str) -> str:
    record = _ensure_record()
    existing = [e for e in record.get("model-version", [])
                if e.get("model", "").startswith(short_name + "-v")]
    max_v = 0
    for e in existing:
        m = re.search(r'-v(\d+)$', e.get("model", ""))
        if m:
            v = int(m.group(1))
            if v > max_v:
                max_v = v
    return f"{short_name}-v{max_v + 1}"


def _download_report(model_version: str, secret_or_text: str) -> Optional[str]:
    os.makedirs(SFT_DATA_DIR, exist_ok=True)
    dest = os.path.join(SFT_DATA_DIR, model_version)
    if secret_or_text.startswith("magnus-secret:"):
        try:
            magnus.download_file(secret_or_text, dest)
            return dest
        except Exception:
            pass
    try:
        with open(dest, "w", encoding="utf-8") as f:
            f.write(secret_or_text)
        return dest
    except Exception:
        return None


# ── 自动推导辅助 ──────────────────────────────────────────

def _model_short_name() -> str:
    return MODEL_PATH.rstrip("/").split("/")[-1]


def _resolve_blueprint() -> str:
    if BLUEPRINT_FILE:
        return BLUEPRINT_FILE
    if MODE == "lora":
        return os.path.join(HERE, "blueprints", "LoRA_zyz.magnus")
    return os.path.join(HERE, "blueprints", "OpenFundus_SFT_zyz.magnus")


def _resolve_output_dir(model_version: str) -> str:
    if OUTPUT_DIR:
        return OUTPUT_DIR
    short = _model_short_name()
    version_suffix = f"-v{model_version.split('-v')[-1]}" if '-v' in model_version else ""
    return f"/data/magnus/models/{short}-{MODE}{version_suffix}"


def _task_prefix() -> str:
    return "LoRA" if MODE == "lora" else "SFT"


def _build_bp_args(resolved_output_dir: str) -> dict:
    """Map local CONFIG to blueprint parameter names."""
    args = {
        "model_path":        MODEL_PATH,
        "output_dir":        resolved_output_dir,
        "epochs":            EPOCHS,
        "batch_size":        BATCH_SIZE,
        "grad_accum":        GRAD_ACCUM,
        "learning_rate":     LEARNING_RATE,
        "max_length":        MAX_LENGTH,
        "num_workers":       NUM_WORKERS,
        "gpu_count":         GPU_COUNT,
        "gpu_type":          GPU_TYPE,
        "cpu_count":         CPU_COUNT,
        "memory_demand":     MEMORY,
        "ephemeral_storage": STORAGE,
        "priority":          PRIORITY,
    }
    if TRAIN_DATA:
        args["train_data"] = TRAIN_DATA
    if TEST_DATA:
        args["test_data"] = TEST_DATA
    if CONTAINER_IMAGE:
        args["container_image"] = CONTAINER_IMAGE
    if RESUME_FROM:
        args["resume_from"] = RESUME_FROM
    if PROMPT_PREFIX:
        args["prompt_prefix"] = PROMPT_PREFIX
        args["prompt_prefix_b64"] = base64.b64encode(
            PROMPT_PREFIX.encode("utf-8")
        ).decode("ascii")
    return args


def main():
    parser = argparse.ArgumentParser(description="submit SFT training via Magnus blueprint")
    parser.add_argument("--address", default=MAGNUS_ADDRESS)
    parser.add_argument("--token",   default=MAGNUS_TOKEN)
    parser.add_argument("--blueprint", default="",
                        help="path to .magnus blueprint file (default: auto from MODE)")
    parser.add_argument("--model-version", default=MODEL_VERSION,
                        help="override model version (default: auto-increment)")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="monitor poll interval in seconds (default: 60)")
    args = parser.parse_args()

    # -- 0. resolve auto-config --
    bp_path   = args.blueprint or _resolve_blueprint()
    short_name = _model_short_name()
    model_version = args.model_version or _auto_model_version(short_name)
    out_dir   = _resolve_output_dir(model_version)
    task_prefix = _task_prefix()

    print(f"[0/5] mode={MODE}, model={short_name}")
    print(f"       blueprint={os.path.basename(bp_path)}")
    print(f"       model version: {model_version}")
    print(f"       output dir: {out_dir}")
    if check_model_version_exists(model_version):
        print(f"[FATAL] model version '{model_version}' already exists.")
        print(f"       use --model-version to specify a new one")
        sys.exit(1)
    print(f"       version check passed")

    # -- 1. configure --
    magnus.configure(address=args.address, token=args.token)
    print(f"[1/5] config: {args.address}")
    print(f"       token: {args.token[:8]}...{args.token[-4:]}")

    # -- 2. save blueprint to server --
    if not os.path.exists(bp_path):
        print(f"[FATAL] blueprint file not found: {bp_path}")
        sys.exit(1)

    with open(bp_path, "r", encoding="utf-8") as f:
        blueprint_code = f.read()

    # blueprint ID = .magnus filename (convention: all _zyz suffix)
    bp_id = os.path.splitext(os.path.basename(bp_path))[0]
    bp_title = f"{task_prefix}-{bp_id}"

    print(f"[2/5] save blueprint to server: {bp_id}")
    try:
        magnus.save_blueprint(
            blueprint_id=bp_id,
            title=bp_title,
            description=f"{task_prefix} training blueprint (synced from {bp_path})",
            code=blueprint_code,
        )
        print(f"       saved successfully")
    except Exception as e:
        print(f"       save returned: {e}")
        print(f"       continuing with existing server version...")

    # -- 2.5. sync sft_train.py to GitHub --
    print(f"[2.5/5] 同步 sft_train.py 到 GitHub...")
    src = os.path.join(HERE, "sft_train.py")
    dst = os.path.join(GITHUB_REPO_PATH, "sft_train.py")
    try:
        shutil.copy2(src, dst)
        subprocess.run(["git", "-C", GITHUB_REPO_PATH, "add", "sft_train.py"],
                       check=True, capture_output=True)
        # only commit/push when there are actual changes
        changed = subprocess.run(
            ["git", "-C", GITHUB_REPO_PATH, "diff", "--cached", "--quiet", "sft_train.py"],
            capture_output=True)
        if changed.returncode != 0:
            subprocess.run(["git", "-C", GITHUB_REPO_PATH, "commit", "-m", "Auto sync sft_train.py"],
                           check=True, capture_output=True)
            subprocess.run(["git", "-C", GITHUB_REPO_PATH, "push"],
                           check=True, capture_output=True)
            print(f"       已推送到 github.com/Rise-AGI/{GITHUB_PATH}")
        else:
            print(f"       sft_train.py 无变化，跳过提交")
    except Exception as e:
        print(f"       同步失败: {e}")
        print(f"       继续执行...（GitHub 上可能不是最新版本）")

    # -- 3. wait 10s (让 GitHub CDN 传播) --
    print(f"[3/5] waiting 10s for GitHub CDN propagation...")
    time.sleep(10)

    # -- 4. launch blueprint --
    bp_args = _build_bp_args(out_dir)
    print(f"[4/5] launching blueprint: {bp_id}")
    for k, v in bp_args.items():
        print(f"       {k}: {v}")

    job_id = magnus.launch_blueprint(bp_id, args=bp_args)
    print(f"       Job ID: {job_id}")
    print()

    # -- 5. monitor + post-process --
    print(f"[5/5] monitoring (poll={args.poll_interval}s)")
    print("-" * 60)
    notify_exe(job_id=job_id)
    wait_for_job(job_id, poll_interval=args.poll_interval)

    job = magnus.get_job(job_id)
    if job.get("status") == "Success":
        result = None
        try:
            result = magnus.get_job_result(job_id)
        except Exception:
            pass

        if result:
            local_path = _download_report(model_version, result)
            if local_path:
                print(f"       report downloaded: {local_path}")
            else:
                print(f"       report download failed, raw result: {result[:200]}")

        record_storage("model-version", {
            "time": datetime.now().isoformat(),
            "model": model_version,
            "local_path": out_dir,
            "base_model": MODEL_PATH,
            "status": "success",
        })
        print(f"       model-version recorded: {model_version}")
    else:
        print(f"       job did not succeed, skipping record")


if __name__ == "__main__":
    main()
