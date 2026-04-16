#!/bin/bash

set -e

# ========================================
# Magnus 平台 Qwen1.5B 微调启动脚本
# 功能：下载模型 → 微调训练 → 固化结果
# ========================================

# 获取脚本所在目录（兼容 Magnus 环境和普通环境）
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [ -n "$0" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
else
    SCRIPT_DIR="${MAGNUS_WORKSPACE:-$(pwd)}"
fi
cd "$SCRIPT_DIR"

# 尝试加载配置文件（可选，不存在则跳过）
if [ -f "$SCRIPT_DIR/magnus_config.env" ]; then
    source "$SCRIPT_DIR/magnus_config.env"
    echo "已加载配置文件: magnus_config.env"
else
    echo "未找到配置文件，使用默认配置"
fi

# 模型存储路径（集群共享存储，持久化）
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/shared/models}"
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_LOCAL_PATH="${MODEL_CACHE_DIR}/Qwen2.5-1.5B-Instruct"

# 训练输出路径
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
FINAL_MODEL_DIR="${FINAL_MODEL_DIR:-/shared/trained_models/qwen_integral_$(date +%Y%m%d_%H%M%S)}"

# 数据路径
TRAIN_FILE="${TRAIN_FILE:-train.json}"
VAL_FILE="${VAL_FILE:-val.json}"

# Magnus 环境变量
MAGNUS_WORKSPACE="${MAGNUS_WORKSPACE:-/magnus/workspace}"
MAGNUS_ACTION="${MAGNUS_ACTION:-/magnus/workspace/.magnus_action}"

echo "========================================="
echo "Magnus 平台 Qwen 1.5B 微调训练"
echo "========================================="
echo "工作目录: $SCRIPT_DIR"
echo "模型缓存: $MODEL_CACHE_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "最终存储: $FINAL_MODEL_DIR"
echo "========================================="

# ========================================
# 步骤1：下载/检查模型
# ========================================
echo ""
echo "[步骤1] 检查并下载模型..."
echo "----------------------------------------"

download_model() {
    echo "方法A: 使用 Magnus File Custody (如果有 MODEL_SECRET)"
    if [ -n "$MODEL_SECRET" ]; then
        echo "检测到 MODEL_SECRET，使用 Magnus 下载..."
        if command -v magnus &> /dev/null; then
            magnus receive "$MODEL_SECRET" --output "$MODEL_LOCAL_PATH"
            echo "模型已下载到: $MODEL_LOCAL_PATH"
            return 0
        else
            echo "Magnus CLI 不可用，尝试 Python SDK..."
            python3 << EOF
from magnus import download_file
import os
download_file(os.environ.get("MODEL_SECRET"), "$MODEL_LOCAL_PATH")
EOF
            if [ $? -eq 0 ]; then
                return 0
            fi
        fi
    fi
    
    echo "方法B: 使用 HuggingFace Hub 下载到共享存储..."
    python3 << 'PYEOF'
import os
import sys
from pathlib import Path

model_cache = os.environ.get("MODEL_CACHE_DIR", "/shared/models")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
local_path = Path(model_cache) / "Qwen2.5-1.5B-Instruct"

# 检查模型是否已存在
if local_path.exists() and any(local_path.glob("*.safetensors")):
    print(f"模型已存在: {local_path}")
    sys.exit(0)

# 创建目录
local_path.mkdir(parents=True, exist_ok=True)

# 下载模型
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
PYEOF
}

# 检查模型是否存在
if [ -d "$MODEL_LOCAL_PATH" ] && [ "$(ls -A $MODEL_LOCAL_PATH/*.safetensors 2>/dev/null)" ]; then
    echo "模型已缓存: $MODEL_LOCAL_PATH"
else
    mkdir -p "$MODEL_CACHE_DIR"
    download_model
fi

# 验证模型
echo "验证模型文件..."
if [ ! -d "$MODEL_LOCAL_PATH" ]; then
    echo "错误: 模型目录不存在"
    exit 1
fi

# ========================================
# 步骤2：准备训练环境
# ========================================
echo ""
echo "[步骤2] 准备训练环境..."
echo "----------------------------------------"

# 激活 conda 环境
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "qwen_integral" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate qwen_integral
fi

# 检查训练数据
if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    echo "生成训练数据..."
    python generate_data.py
fi

# ========================================
# 步骤3：开始微调训练
# ========================================
echo ""
echo "[步骤3] 开始微调训练..."
echo "----------------------------------------"
echo "模型路径: $MODEL_LOCAL_PATH"
echo "训练数据: $TRAIN_FILE, $VAL_FILE"
echo ""

# 训练参数
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
USE_4BIT="${USE_4BIT:-false}"

CMD="python train.py \
    --model_name $MODEL_LOCAL_PATH \
    --train_file $TRAIN_FILE \
    --val_file $VAL_FILE \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100"

if [ "$USE_4BIT" = "true" ]; then
    CMD="$CMD --use_4bit"
fi

echo "执行命令: $CMD"
echo ""
$CMD

# ========================================
# 步骤4：固化训练结果
# ========================================
echo ""
echo "[步骤4] 固化训练结果..."
echo "----------------------------------------"

# 创建最终存储目录
mkdir -p "$FINAL_MODEL_DIR"

# 复制模型文件
echo "复制模型到持久化存储: $FINAL_MODEL_DIR"
cp -r "$OUTPUT_DIR"/* "$FINAL_MODEL_DIR/"

# 保存训练元信息
cat > "${FINAL_MODEL_DIR}/training_info.json" << EOF
{
    "base_model": "$MODEL_NAME",
    "model_path": "$MODEL_LOCAL_PATH",
    "train_file": "$TRAIN_FILE",
    "val_file": "$VAL_FILE",
    "batch_size": $BATCH_SIZE,
    "num_epochs": $NUM_EPOCHS,
    "learning_rate": $LEARNING_RATE,
    "use_4bit": $USE_4BIT,
    "train_time": "$(date -Iseconds)",
    "output_dir": "$FINAL_MODEL_DIR"
}
EOF

echo "模型已保存到: $FINAL_MODEL_DIR"

# ========================================
# 步骤5：通过 Magnus 导出（可选）
# ========================================
echo ""
echo "[步骤5] Magnus 结果导出..."
echo "----------------------------------------"

if [ -n "$MAGNUS_ACTION" ] && [ -f "$MAGNUS_ACTION" ]; then
    echo "检测到 Magnus 环境，准备导出..."
    
    # 方法1：使用 Magnus CLI
    if command -v magnus &> /dev/null; then
        echo "使用 Magnus CLI 上传模型..."
        SECRET=$(magnus send "$FINAL_MODEL_DIR" 2>/dev/null)
        if [ -n "$SECRET" ]; then
            echo "$SECRET" > "$MAGNUS_ACTION"
            echo "模型已上传，Secret: $SECRET"
        fi
    else
        # 方法2：使用 Python SDK
        echo "使用 Magnus Python SDK 上传..."
        python3 << PYEOF
from magnus import send_file
import os

result = send_file("$FINAL_MODEL_DIR")
if result:
    with open(os.environ.get("MAGNUS_ACTION", "/magnus/workspace/.magnus_action"), "w") as f:
        f.write(result)
    print(f"模型已上传，Secret: {result}")
PYEOF
    fi
else
    echo "非 Magnus 环境或无需导出，跳过此步骤"
fi

# ========================================
# 完成
# ========================================
echo ""
echo "========================================="
echo "训练完成!"
echo "========================================="
echo ""
echo "模型位置:"
echo "  - 本地输出: $OUTPUT_DIR"
echo "  - 持久存储: $FINAL_MODEL_DIR"
echo ""
echo "测试模型:"
echo "  python inference.py --model_path $FINAL_MODEL_DIR --mode interactive"
echo ""