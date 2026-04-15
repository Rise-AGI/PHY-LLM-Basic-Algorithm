#!/bin/bash

# 主启动脚本 - 自动定位post-train目录并执行

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo "Qwen 1.5B 积分计算微调 - 主启动脚本"
echo "========================================="
echo ""
echo "脚本目录: $SCRIPT_DIR"
echo ""

# 检查是否在post-train目录
if [ ! -f "$SCRIPT_DIR/train.py" ]; then
    echo "错误: 未在post-train目录下找到train.py"
    echo "请确保此脚本位于post-train文件夹内"
    exit 1
fi

cd "$SCRIPT_DIR"

echo "请选择操作:"
echo "  1) 配置环境 (首次运行)"
echo "  2) 生成训练数据"
echo "  3) 开始训练"
echo "  4) 测试模型"
echo "  5) 完整流程 (配置+生成+训练)"
echo ""
read -p "输入选项 (1-5): " choice

case $choice in
    1)
        echo ""
        echo "开始配置环境..."
        ./setup_quick.sh
        ;;
    2)
        echo ""
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate qwen_integral
        echo "生成训练数据..."
        python generate_data.py
        ;;
    3)
        echo ""
        ./run_train.sh
        ;;
    4)
        echo ""
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate qwen_integral
        echo "启动测试模式..."
        python inference.py --model_path ./output --mode interactive
        ;;
    5)
        echo ""
        echo "执行完整流程..."
        ./setup_quick.sh
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate qwen_integral
        echo ""
        echo "生成数据..."
        python generate_data.py
        echo ""
        ./run_train.sh
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "完成!"