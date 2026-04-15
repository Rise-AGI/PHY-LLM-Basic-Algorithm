#!/bin/bash

# 切换到脚本所在目录（post-train）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "Qwen 1.5B 积分计算微调环境配置脚本"
echo "工作目录: $SCRIPT_DIR"
echo "========================================="

# 检查conda是否安装
if ! command -v conda &> /dev/null
then
    echo "错误: 未检测到conda，请先安装Anaconda或Miniconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 设置环境名称
ENV_NAME="qwen_integral"
PYTHON_VERSION="3.12"

echo ""
echo "步骤1: 创建conda环境 (Python ${PYTHON_VERSION})"
echo "----------------------------------------"
read -p "是否创建新环境? 环境名称: ${ENV_NAME} (y/n): " create_env
if [ "$create_env" = "y" ] || [ "$create_env" = "Y" ]; then
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    echo "环境创建完成: ${ENV_NAME}"
else
    read -p "请输入已存在的环境名称: " ENV_NAME
fi

echo ""
echo "步骤2: 激活环境"
echo "----------------------------------------"
echo "执行: conda activate ${ENV_NAME}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo ""
echo "步骤3: 检查CUDA版本"
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
    read -p "请确认GPU可用，按回车继续..."
else
    echo "警告: 未检测到NVIDIA GPU，将使用CPU训练（速度较慢）"
fi

echo ""
echo "步骤4: 安装PyTorch (CUDA版本)"
echo "----------------------------------------"
read -p "请选择CUDA版本 (1: CUDA 11.8, 2: CUDA 12.1, 3: CUDA 12.4): " cuda_choice
case $cuda_choice in
    1)
        CUDA_VERSION="cu118"
        ;;
    2)
        CUDA_VERSION="cu121"
        ;;
    3)
        CUDA_VERSION="cu124"
        ;;
    *)
        CUDA_VERSION="cu124"
        ;;
esac

echo "安装PyTorch with CUDA ${CUDA_VERSION}..."
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

echo ""
echo "步骤5: 安装核心依赖"
echo "----------------------------------------"
pip install transformers==4.49.0
pip install datasets==3.3.2
pip install accelerate==1.4.0
pip install sympy==1.13.1
pip install tqdm==4.67.1
pip install numpy==2.2.3
pip install huggingface-hub==0.29.1

echo ""
echo "步骤6: 安装PEFT (LoRA微调)"
echo "----------------------------------------"
pip install peft

echo ""
echo "步骤7: 安装Flash Attention (可选，加速训练)"
echo "----------------------------------------"
read -p "是否安装Flash Attention? 需要编译，耗时较长 (y/n): " install_flash
if [ "$install_flash" = "y" ] || [ "$install_flash" = "Y" ]; then
    pip install flash-attn==2.7.4.post1 --no-build-isolation
else
    echo "跳过Flash Attention安装"
fi

echo ""
echo "步骤8: 安装量化库 (可选，节省显存)"
echo "----------------------------------------"
read -p "是否安装bitsandbytes (4bit量化)? (y/n): " install_bnb
if [ "$install_bnb" = "y" ] || [ "$install_bnb" = "Y" ]; then
    pip install bitsandbytes
    pip install scipy
fi

echo ""
echo "步骤9: 验证安装"
echo "----------------------------------------"
echo "Python版本:"
python --version

echo ""
echo "PyTorch版本:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "关键包版本:"
python -c "
import transformers, datasets, accelerate
print(f'Transformers: {transformers.__version__}')
print(f'Datasets: {datasets.__version__}')
print(f'Accelerate: {accelerate.__version__}')
"

echo ""
echo "========================================="
echo "环境配置完成!"
echo "========================================="
echo ""
echo "工作目录: $SCRIPT_DIR"
echo ""
echo "使用方法:"
echo "  1. 进入项目目录: cd $SCRIPT_DIR"
echo "  2. 激活环境: conda activate ${ENV_NAME}"
echo "  3. 生成数据: python generate_data.py"
echo "  4. 开始训练: ./run_train.sh 或 python train.py"
echo ""
echo "如果遇到网络问题，可以设置镜像源:"
echo "  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
echo ""