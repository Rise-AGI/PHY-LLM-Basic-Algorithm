#!/bin/bash

# ========================================
# Magnus 平台入口脚本
# 在 Magnus 平台上传此脚本作为 Job 入口
# ========================================

set -e

echo "========================================="
echo "Magnus Job 启动"
echo "当前目录: $(pwd)"
echo "时间: $(date)"
echo "========================================="

# 检查 Magnus 环境变量
echo "环境检查:"
echo "  MAGNUS_WORKSPACE: ${MAGNUS_WORKSPACE:-未设置}"
echo "  MAGNUS_JOB_ID: ${MAGNUS_JOB_ID:-未设置}"
echo "  MAGNUS_ACTION: ${MAGNUS_ACTION:-未设置}"
echo ""

# 工作目录配置
WORK_DIR="${MAGNUS_WORKSPACE:-/magnus/workspace}"
PROJECT_DIR="$WORK_DIR/post-train"

# 自动定位 post-train 目录
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    echo "切换到项目目录: $PROJECT_DIR"
elif [ -d "$WORK_DIR" ]; then
    cd "$WORK_DIR"
    echo "工作目录: $WORK_DIR"
else
    cd "$(pwd)"
    echo "当前目录: $(pwd)"
fi

SCRIPT_DIR="$(pwd)"
echo ""
echo "当前目录文件:"
ls -la
echo ""

# ========================================
# 配置参数（直接在此设置，无需外部配置文件）
# ========================================
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/shared/models}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"
export FINAL_MODEL_DIR="${FINAL_MODEL_DIR:-/shared/trained_models/qwen_integral_$(date +%Y%m%d_%H%M%S)}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export USE_4BIT="${USE_4BIT:-false}"

# ========================================
# 文件路径配置（自动检测 post-train 目录）
# ========================================
TRAIN_PY="train.py"
GENERATE_DATA_PY="generate_data.py"
INFERENCE_PY="inference.py"
TRAIN_JSON="train.json"
VAL_JSON="val.json"

# 如果当前目录不是 post-train，尝试在子目录查找
if [ ! -f "$TRAIN_PY" ]; then
    if [ -f "post-train/train.py" ]; then
        TRAIN_PY="post-train/train.py"
        GENERATE_DATA_PY="post-train/generate_data.py"
        INFERENCE_PY="post-train/inference.py"
        TRAIN_JSON="post-train/train.json"
        VAL_JSON="post-train/val.json"
        echo "检测到 post-train 子目录，使用相对路径"
    fi
fi

# ========================================
# 检查必要文件
# ========================================
echo "检查必要文件..."

# 检查 train.py
if [ ! -f "$TRAIN_PY" ]; then
    echo "错误: 未找到 $TRAIN_PY"
    echo "请确保已上传以下文件到 Magnus:"
    echo "  - post-train/train.py"
    echo "  - post-train/generate_data.py (可选，用于生成训练数据)"
    echo "  - post-train/inference.py (可选，用于测试)"
    exit 1
fi
echo "✓ $TRAIN_PY 存在"

# 检查训练数据
if [ ! -f "$TRAIN_JSON" ] || [ ! -f "$VAL_JSON" ]; then
    echo "未找到训练数据，尝试生成..."
    if [ -f "$GENERATE_DATA_PY" ]; then
        python "$GENERATE_DATA_PY"
    else
        echo "错误: 未找到 $GENERATE_DATA_PY，无法生成训练数据"
        echo "请上传 post-train/train.json 和 post-train/val.json 文件"
        exit 1
    fi
fi
echo "✓ 训练数据就绪"

# ========================================
# 检查/下载模型
# ========================================
echo ""
echo "========================================="
echo "步骤1: 检查模型"
echo "========================================="

MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_PATH="${MODEL_CACHE_DIR}/Qwen2.5-1.5B-Instruct"

# 检查模型是否已缓存
if [ -d "$MODEL_PATH" ] && ls "$MODEL_PATH"/*.safetensors 1> /dev/null 2>&1; then
    echo "模型已缓存: $MODEL_PATH"
else
    echo "模型未缓存，开始下载..."
    mkdir -p "$MODEL_CACHE_DIR"
    
    python3 << 'DOWNLOAD_SCRIPT'
import os
import sys
from pathlib import Path

model_cache = os.environ.get("MODEL_CACHE_DIR", "/shared/models")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
local_path = Path(model_cache) / "Qwen2.5-1.5B-Instruct"

if local_path.exists() and any(local_path.glob("*.safetensors")):
    print(f"模型已存在: {local_path}")
    sys.exit(0)

local_path.mkdir(parents=True, exist_ok=True)

print(f"下载模型到: {local_path}")
try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_path),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("模型下载完成!")
except Exception as e:
    print(f"下载失败: {e}")
    sys.exit(1)
DOWNLOAD_SCRIPT

    if [ $? -ne 0 ]; then
        echo "错误: 模型下载失败"
        exit 1
    fi
fi

echo "模型路径: $MODEL_PATH"

# ========================================
# 开始训练
# ========================================
echo ""
echo "========================================="
echo "步骤2: 开始训练"
echo "========================================="

CMD="python $TRAIN_PY \
    --model_name $MODEL_PATH \
    --train_file $TRAIN_JSON \
    --val_file $VAL_JSON \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100"

if [ "$USE_4BIT" = "true" ]; then
    CMD="$CMD --use_4bit"
fi

echo "执行: $CMD"
echo ""
$CMD

# ========================================
# 保存结果
# ========================================
echo ""
echo "========================================="
echo "步骤3: 保存结果"
echo "========================================="

mkdir -p "$FINAL_MODEL_DIR"
cp -r "$OUTPUT_DIR"/* "$FINAL_MODEL_DIR/"

# 保存训练信息
cat > "${FINAL_MODEL_DIR}/training_info.json" << EOF
{
    "base_model": "$MODEL_NAME",
    "model_path": "$MODEL_PATH",
    "batch_size": $BATCH_SIZE,
    "num_epochs": $NUM_EPOCHS,
    "learning_rate": $LEARNING_RATE,
    "use_4bit": $USE_4BIT,
    "train_time": "$(date -Iseconds)",
    "magnus_job_id": "${MAGNUS_JOB_ID:-unknown}",
    "output_dir": "$FINAL_MODEL_DIR"
}
EOF

echo "模型已保存到: $FINAL_MODEL_DIR"

# ========================================
# Magnus 导出（可选）
# ========================================
echo ""
echo "========================================="
echo "步骤4: Magnus 导出（可选）"
echo "========================================="

if [ -n "$MAGNUS_ACTION" ]; then
    echo "检测到 Magnus 环境"
    
    if command -v magnus &> /dev/null; then
        echo "使用 Magnus CLI 上传..."
        SECRET=$(magnus send "$FINAL_MODEL_DIR" 2>/dev/null || echo "")
        if [ -n "$SECRET" ]; then
            echo "$SECRET" > "$MAGNUS_ACTION"
            echo "模型已上传到 Magnus File Custody"
            echo "Secret: $SECRET"
        fi
    else
        echo "Magnus CLI 不可用，尝试 Python SDK..."
        python3 << 'UPLOAD_SCRIPT'
import os
try:
    from magnus import send_file
    result = send_file(os.environ.get("FINAL_MODEL_DIR"))
    if result:
        with open(os.environ.get("MAGNUS_ACTION"), "w") as f:
            f.write(result)
        print(f"模型已上传，Secret: {result}")
except ImportError:
    print("Magnus SDK 未安装，跳过导出")
except Exception as e:
    print(f"导出失败: {e}")
UPLOAD_SCRIPT
    fi
else
    echo "非 Magnus 环境，跳过导出"
fi

# ========================================
# 完成
# ========================================
echo ""
echo "========================================="
echo "训练完成!"
echo "========================================="
echo ""
echo "输出位置:"
echo "  - 训练输出: $OUTPUT_DIR"
echo "  - 最终模型: $FINAL_MODEL_DIR"
echo ""
echo "测试命令:"
echo "  python $INFERENCE_PY --model_path $FINAL_MODEL_DIR"
echo ""