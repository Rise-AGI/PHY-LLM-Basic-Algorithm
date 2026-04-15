#!/bin/bash

# 服务器训练启动脚本

# 切换到脚本所在目录（post-train）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "Qwen 1.5B 积分计算微调训练"
echo "工作目录: $SCRIPT_DIR"
echo "========================================="

# 检查数据是否存在
if [ ! -f "train.json" ] || [ ! -f "val.json" ]; then
    echo "未找到训练数据，开始生成..."
    python generate_data.py
fi

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen_integral

echo ""
echo "训练配置:"
echo "  模型: Qwen/Qwen2.5-1.5B-Instruct"
echo "  输出目录: ./output"
echo "  训练轮数: 3"
echo "  批次大小: 4"
echo ""

read -p "是否使用4bit量化? 节省显存但速度稍慢 (y/n): " use_4bit
read -p "是否使用Flash Attention? 加速训练 (y/n): " use_flash

CMD="python train.py --model_name Qwen/Qwen2.5-1.5B-Instruct --output_dir ./output --num_epochs 3 --batch_size 4 --learning_rate 2e-4 --logging_steps 10 --save_steps 100"

if [ "$use_4bit" = "y" ] || [ "$use_4bit" = "Y" ]; then
    CMD="$CMD --use_4bit"
fi

echo ""
echo "开始训练..."
echo "命令: $CMD"
echo ""

# 开始训练
$CMD

echo ""
echo "训练完成!"
echo "模型保存在: ./output"
echo ""
echo "测试模型:"
echo "  python inference.py --model_path ./output --mode interactive"