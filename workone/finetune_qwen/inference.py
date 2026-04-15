import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen1.5-1.8B', help='Base model path')
    parser.add_argument('--lora_path', type=str, default='output/qwen_calculus', help='LoRA weights path')
    parser.add_argument('--prompt', type=str, default=None, help='Input prompt')
    parser.add_argument('--max_length', type=int, default=512, help='Max generation length')
    args = parser.parse_args()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if args.lora_path:
        print(f"Loading LoRA weights from {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
    
    model.eval()

    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "求函数 f(x) = x^3 + 2x 的导数",
            "计算不定积分 ∫x^3 dx",
            "求函数 f(x) = sin(3x) 的导数",
        ]

    print("\n" + "="*50)
    for prompt in prompts:
        full_prompt = f"### 问题:\n{prompt}\n\n### 回答:\n"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_length,
                do_sample=True,
                top_p=0.8,
                temperature=0.7,
                repetition_penalty=1.1,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"问题: {prompt}")
        print(f"回答: {response.split('### 回答:')[-1].strip()}")
        print("="*50)


if __name__ == "__main__":
    main()