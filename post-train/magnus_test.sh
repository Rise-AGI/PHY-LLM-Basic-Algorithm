#!/bin/bash

# ========================================
# Magnus 模型验证脚本
# 测试训练后的模型是否能正确计算积分
# ========================================

set -e

echo "========================================="
echo "模型验证测试"
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
# 配置
# ========================================
# 模型路径（可通过参数指定）
MODEL_PATH="${1:-/shared/trained_models/qwen_integral_20260416_143532}"

# 测试问题
TEST_QUESTION="∫ 1/(x^2+1) dx"

echo "测试配置:"
echo "  模型路径: $MODEL_PATH"
echo "  测试问题: $TEST_QUESTION"
echo ""

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    echo ""
    echo "可用模型列表:"
    ls -la /shared/trained_models/ 2>/dev/null || echo "无已训练模型"
    exit 1
fi

echo "模型目录内容:"
ls -la "$MODEL_PATH"
echo ""

# ========================================
# 检查依赖
# ========================================
echo "检查依赖..."

PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

for pkg in "torch" "transformers" "peft"; do
    python3 -c "import ${pkg}" 2>/dev/null || {
        echo "安装 $pkg..."
        pip install "$pkg" --index-url "$PIP_INDEX" --quiet
    }
done

echo "✓ 依赖就绪"
echo ""

# ========================================
# 运行测试
# ========================================
echo "========================================="
echo "开始测试"
echo "========================================="
echo ""

export HF_ENDPOINT="$HF_ENDPOINT"

python3 << 'TEST_SCRIPT'
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 配置
model_path = os.environ.get("MODEL_PATH", "/shared/trained_models/qwen_integral_20260416_143532")
base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
test_question = "∫ 1/(x^2+1) dx"

print(f"加载基础模型: {base_model_name}")
print(f"加载 LoRA 权重: {model_path}")
print()

try:
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    # 检查是否有 LoRA adapter
    adapter_file = os.path.join(model_path, "adapter_model.safetensors")
    adapter_config = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_file) and os.path.exists(adapter_config):
        print("检测到 LoRA adapter，合并加载...")
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载 LoRA 权重
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        print("✓ LoRA 权重已合并")
    else:
        print("直接加载完整模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    print("✓ 模型加载完成")
    print()
    
    # 构建测试 prompt
    prompt = f"<|im_start|>system\n计算以下不定积分，直接给出结果<|im_end|>\n<|im_start|>user\n{test_question}<|im_end|>\n<|im_start|>assistant\n"
    
    print("测试问题:", test_question)
    print()
    print("生成回答...")
    print("-" * 50)
    
    # 生成回答
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    # 解码输出
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 assistant 部分
    if "assistant" in full_response:
        answer = full_response.split("assistant")[-1].strip()
    else:
        answer = full_response
    
    print(answer)
    print("-" * 50)
    print()
    
    # 验证答案（期望结果: arctan(x) + C 或 tan^-1(x) + C）
    expected_keywords = ["arctan", "tan", "atan", "tan^-1", "tan⁻¹"]
    is_correct = any(kw.lower() in answer.lower() for kw in expected_keywords)
    
    print("验证结果:")
    if is_correct:
        print("✓ 回答包含正确答案关键词")
        print("  正确答案: arctan(x) + C (或 tan⁻¹(x) + C)")
    else:
        print("✗ 回答未包含预期关键词")
        print("  期望: arctan(x) + C")
    
    print()
    print("========================================="
    print("测试完成"
    print("========================================="

except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
TEST_SCRIPT

# 传递环境变量
export MODEL_PATH="$MODEL_PATH"

# 重新运行（因为 heredoc 内无法访问外部变量）
python3 -c "
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_path = os.environ['MODEL_PATH']
base_model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
test_question = '∫ 1/(x^2+1) dx'

print(f'加载基础模型: {base_model_name}')
print(f'加载 LoRA 权重: {model_path}')
print()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

adapter_file = os.path.join(model_path, 'adapter_model.safetensors')
adapter_config = os.path.join(model_path, 'adapter_config.json')

if os.path.exists(adapter_file) and os.path.exists(adapter_config):
    print('检测到 LoRA adapter，合并加载...')
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()
    print('✓ LoRA 权重已合并')
else:
    print('直接加载完整模型...')
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)

print('✓ 模型加载完成')
print()

prompt = f'<|im_start|>system\n计算以下不定积分，直接给出结果<|im_end|>\n<|im_start|>user\n{test_question}<|im_end|>\n<|im_start|>assistant\n'

print('测试问题:', test_question)
print()
print('生成回答...')
print('-' * 50)

inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False, temperature=0.1, top_p=0.9, repetition_penalty=1.1)

full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

if 'assistant' in full_response:
    answer = full_response.split('assistant')[-1].strip()
else:
    answer = full_response

print(answer)
print('-' * 50)
print()

expected_keywords = ['arctan', 'tan', 'atan', 'tan^-1', 'tan⁻¹']
is_correct = any(kw.lower() in answer.lower() for kw in expected_keywords)

print('验证结果:')
if is_correct:
    print('✓ 回答包含正确答案关键词')
    print('  正确答案: arctan(x) + C (或 tan⁻¹(x) + C)')
else:
    print('✗ 回答未包含预期关键词')
    print('  期望: arctan(x) + C')

print()
print('=========================================')
print('测试完成')
print('=========================================')
"