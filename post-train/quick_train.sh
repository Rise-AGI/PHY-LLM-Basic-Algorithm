#!/bin/bash

# ========================================
# 简化版：普通服务器快速启动脚本
# 不依赖 Magnus，适用于裸机/普通服务器
# ========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 配置
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/.cache/huggingface/models}"
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_LOCAL_PATH="$MODEL_CACHE_DIR/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"
OUTPUT_DIR="./output"
FINAL_DIR="${FINAL_DIR:-$HOME/trained_models/qwen_integral_$(date +%Y%m%d_%H%M%S)}"

echo "========================================="
echo "Qwen 1.5B 微调 - 普通服务器版"
echo "========================================="

# 步骤1：下载模型
echo ""
echo "[1/4] 检查模型..."
if [ ! -d "$MODEL_LOCAL_PATH" ]; then
    echo "下载模型 $MODEL_NAME..."
    python3 -c "
from huggingface_hub import snapshot_download
import os
cache_dir = '$MODEL_CACHE_DIR'
os.makedirs(cache_dir, exist_ok=True)
snapshot_download(
    repo_id='$MODEL_NAME',
    local_dir=os.path.join(cache_dir, 'Qwen2.5-1.5B-Instruct'),
    local_dir_use_symlinks=False
)
print('下载完成!')
"
    MODEL_LOCAL_PATH="$MODEL_CACHE_DIR/Qwen2.5-1.5B-Instruct"
else
    MODEL_LOCAL_PATH=$(ls -d $MODEL_LOCAL_PATH/*/ 2>/dev/null | head -1)
    echo "模型已存在: $MODEL_LOCAL_PATH"
fi

# 步骤2：准备数据
echo ""
echo "[2/4] 准备数据..."
if [ ! -f "train.json" ]; then
    python generate_data.py
fi

# 步骤3：训练
echo ""
echo "[3/4] 开始训练..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen_integral 2>/dev/null || true

python train.py \
    --model_name "$MODEL_LOCAL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4

# 步骤4：保存
echo ""
echo "[4/4] 保存模型..."
mkdir -p "$FINAL_DIR"
cp -r "$OUTPUT_DIR"/* "$FINAL_DIR/"
echo "模型已保存: $FINAL_DIR"

echo ""
echo "完成! 测试命令:"
echo "  python inference.py --model_path $FINAL_DIR"