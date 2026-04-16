#!/bin/bash

# ========================================
# Magnus 模型推理脚本
# 自动安装依赖后运行 inference.py
# ========================================

echo "========================================="
echo "模型推理测试"
echo "========================================="

# 自动定位项目目录
WORK_DIR="${MAGNUS_WORKSPACE:-/magnus/workspace}"
SCRIPT_DIR=""
SEARCH_DIRS=(
    "$WORK_DIR/repository/post-train"
    "$WORK_DIR/post-train"
    "$WORK_DIR/repository"
    "$WORK_DIR"
)

for dir in "${SEARCH_DIRS[@]}"; do
    if [ -d "$dir" ] && [ -f "$dir/inference.py" ]; then
        SCRIPT_DIR="$dir"
        break
    fi
done

if [ -z "$SCRIPT_DIR" ]; then
    SCRIPT_DIR="$(pwd)"
fi

cd "$SCRIPT_DIR"
echo "项目目录: $SCRIPT_DIR"
echo ""

# ========================================
# 安装依赖
# ========================================
echo "检查并安装依赖..."

PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MISSING_PKGS=""
for pkg in "torch" "transformers" "peft" "accelerate"; do
    python3 -c "import ${pkg}" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS $pkg"
done

if [ -n "$MISSING_PKGS" ]; then
    echo "安装缺失的依赖: $MISSING_PKGS"
    pip install $MISSING_PKGS --index-url "$PIP_INDEX" --quiet
fi

echo "✓ 依赖就绪"
echo ""

# 设置镜像
export HF_ENDPOINT="$HF_ENDPOINT"

# ========================================
# 运行推理
# ========================================
MODEL_PATH="${1:-$SCRIPT_DIR/output}"

# 自动搜索模型
if [ ! -d "$MODEL_PATH" ] || [ ! "$(ls -A $MODEL_PATH/*.safetensors 2>/dev/null)" ]; then
    echo "搜索模型..."
    
    SEARCH_PATHS=(
        "$SCRIPT_DIR/output"
        "$WORK_DIR/output"
        "/shared/trained_models"
    )
    
    for path in "${SEARCH_PATHS[@]}"; do
        if [ -d "$path" ]; then
            if ls "$path"/*.safetensors 1> /dev/null 2>&1; then
                MODEL_PATH="$path"
                echo "找到模型: $MODEL_PATH"
                break
            fi
            for subdir in "$path"/*; do
                if [ -d "$subdir" ] && ls "$subdir"/*.safetensors 1> /dev/null 2>&1; then
                    MODEL_PATH="$subdir"
                    echo "找到模型: $MODEL_PATH"
                    break 2
                fi
            done
        fi
    done
fi

echo "模型路径: $MODEL_PATH"
echo ""

# 测试问题
TEST_QUESTION="∫ 1/(x^2+1) dx"

echo "测试问题: $TEST_QUESTION"
echo ""

python3 inference.py --model_path "$MODEL_PATH" --question "$TEST_QUESTION"

echo ""
echo "========================================="
echo "完成"
echo "========================================="