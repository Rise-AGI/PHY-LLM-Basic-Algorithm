import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os

def is_lora_model(model_path):
    adapter_config = os.path.join(model_path, "adapter_config.json")
    adapter_model = os.path.join(model_path, "adapter_model.safetensors")
    return os.path.exists(adapter_config) or os.path.exists(adapter_model)

def load_model(model_path, base_model_name=None):
    print(f"加载模型: {model_path}")
    
    use_lora = is_lora_model(model_path)
    print(f"检测到 LoRA 模型: {use_lora}")
    
    if use_lora and not base_model_name:
        base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"自动检测基础模型: {base_model_name}")
    
    if use_lora and base_model_name:
        print(f"加载基础模型: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            use_fast=False
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"加载 LoRA 适配器: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    model.eval()
    
    return model, tokenizer

def solve_integral(model, tokenizer, question, max_new_tokens=512):
    prompt = f"<|im_start|>system\n你是一个数学专家,请计算以下积分。<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    assistant_start = response.find("assistant\n")
    if assistant_start != -1:
        response = response[assistant_start + len("assistant\n"):]
    
    return response.strip()

def interactive_mode(model, tokenizer):
    print("\n积分计算助手 (输入 'quit' 退出)")
    print("=" * 50)
    
    while True:
        question = input("\n请输入积分问题: ").strip()
        
        if question.lower() == 'quit':
            print("再见!")
            break
        
        if not question:
            continue
        
        print("\n正在计算...")
        answer = solve_integral(model, tokenizer, question)
        print(f"\n答案:\n{answer}")

def test_mode(model, tokenizer):
    test_questions = [
        "计算不定积分: ∫x³dx",
        "计算不定积分: ∫sin(x)dx",
        "计算不定积分: ∫e^x dx",
        "计算不定积分: ∫(x² + 2x + 1)dx",
        "计算定积分: ∫[0,1] x²dx",
    ]
    
    print("\n测试模式")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n测试 {i}: {question}")
        print("-" * 50)
        answer = solve_integral(model, tokenizer, question)
        print(f"答案: {answer}")
        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="使用微调后的模型进行积分计算")
    parser.add_argument("--model_path", type=str, required=True, help="微调后的模型路径")
    parser.add_argument("--base_model", type=str, default=None, help="基础模型路径(如果使用LoRA)")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "test"], help="运行模式")
    parser.add_argument("--question", type=str, default=None, help="单个问题(可选)")
    
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    if args.question:
        print(f"\n问题: {args.question}")
        answer = solve_integral(model, tokenizer, args.question)
        print(f"\n答案:\n{answer}")
    elif args.mode == "interactive":
        interactive_mode(model, tokenizer)
    else:
        test_mode(model, tokenizer)

if __name__ == "__main__":
    main()