#!/bin/bash

# ========================================
# Magnus 模型推理脚本
# 自动安装依赖后运行推理
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
for pkg in "torch" "transformers" "peft" "accelerate" "huggingface_hub"; do
    python3 -c "import ${pkg//-/_}" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS $pkg"
done

if [ -n "$MISSING_PKGS" ]; then
    echo "安装缺失的依赖: $MISSING_PKGS"
    pip install $MISSING_PKGS --index-url "$PIP_INDEX"
fi

echo "✓ 依赖就绪"
echo ""

# 设置镜像
export HF_ENDPOINT="$HF_ENDPOINT"

# ========================================
# 搜索模型
# ========================================
MODEL_PATH="${1:-$SCRIPT_DIR/output}"

echo "搜索模型..."
echo "当前路径: $MODEL_PATH"

if [ -d "$MODEL_PATH" ]; then
    echo "模型目录内容:"
    ls -la "$MODEL_PATH"
    echo ""
fi

# ========================================
# 运行推理（内置 Python，正确处理 LoRA）
# ========================================
TEST_QUESTION="∫ 1/(x^2+1) dx"
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

echo "基础模型: $BASE_MODEL"
echo "LoRA 路径: $MODEL_PATH"
echo "测试问题: $TEST_QUESTION"
echo ""

echo "========================================="
echo "开始推理"
echo "========================================="
echo ""

export MODEL_PATH="$MODEL_PATH"

python3 << 'PYEOF'
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

model_path = os.environ.get("MODEL_PATH", "./output")
base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
test_question = "∫ 1/(x^2+1) dx"

print(f"使用镜像: {HF_ENDPOINT}")
print()

try:
    # 检查是否是 LoRA adapter
    adapter_file = os.path.join(model_path, "adapter_model.safetensors")
    adapter_config = os.path.join(model_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_file) and os.path.exists(adapter_config)
    
    print(f"模型路径: {model_path}")
    print(f"是否 LoRA: {is_lora}")
    print()
    
    # 从基础模型加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        use_fast=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ tokenizer 加载完成")
    
    # 加载基础模型
    print("加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✓ 基础模型加载完成")
    
    # 如果是 LoRA，加载 adapter
    if is_lora:
        print("加载 LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        print("✓ LoRA 已合并")
    else:
        model = base_model
    
    model.eval()
    print("✓ 模型准备完成")
    print()
    
    # 生成回答
    prompt = f"<|im_start|>system\n计算以下不定积分，直接给出结果<|im_end|>\n<|im_start|>user\n{test_question}<|im_end|>\n<|im_start|>assistant\n"
    
    print("测试问题:", test_question)
    print()
    print("生成回答...")
    print("-" * 50)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.1,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 assistant 部分
    if "assistant" in full_response:
        answer = full_response.split("assistant")[-1].strip()
    else:
        answer = full_response
    
    print(answer)
    print("-" * 50)
    print()
    
    # 验证答案
    expected_keywords = ["arctan", "tan", "atan", "tan^-1", "tan⁻¹"]
    is_correct = any(kw.lower() in answer.lower() for kw in expected_keywords)
    
    print("验证结果:")
    if is_correct:
        print("✓ 回答包含正确关键词")
        print("  正确答案: arctan(x) + C (或 tan⁻¹(x) + C)")
    else:
        print("✗ 回答未包含预期关键词")
        print("  期望: arctan(x) + C")
    
    print()
    print("=========================================")
    print("推理完成")
    print("=========================================")

except Exception as e:
    print(f"推理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF