#!/bin/bash
# 一键环境配置脚本 - 推荐使用

# 切换到脚本所在目录（post-train）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="qwen_integral"

echo "========================================="
echo "一键环境配置"
echo "工作目录: $SCRIPT_DIR"
echo "========================================="

# 创建环境
conda create -n ${ENV_NAME} python=3.12 -y

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# 安装核心依赖（自动匹配CUDA版本）
pip install torch==2.6.0
pip install transformers==4.49.0
pip install datasets==3.3.2
pip install accelerate==1.4.0
pip install peft
pip install sympy==1.13.1
pip install tqdm==4.67.1
pip install numpy==2.2.3
pip install scipy

# 安装Flash Attention（可选，加速训练）
echo "是否安装Flash Attention? (y/n)"
read answer
if [ "$answer" = "y" ]; then
    pip install flash-attn==2.7.4.post1 --no-build-isolation
fi

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "========================================="
echo "环境配置完成!"
echo "========================================="
echo ""
echo "工作目录: $SCRIPT_DIR"
echo "环境名称: ${ENV_NAME}"
echo ""
echo "后续步骤:"
echo "  1. cd $SCRIPT_DIR"
echo "  2. conda activate ${ENV_NAME}"
echo "  3. python generate_data.py"
echo "  4. ./run_train.sh"