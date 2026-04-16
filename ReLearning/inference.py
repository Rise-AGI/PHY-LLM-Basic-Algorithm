import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sympy import sympify, simplify, symbols, integrate, latex
from sympy.abc import x
import re

class IntegralInference:
    def __init__(self, model_path, base_model=None, use_lora=False):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path if base_model is None else base_model,
            trust_remote_code=True,
            use_fast=False
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        if use_lora and base_model:
            print(f"加载基础模型: {base_model}")
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print(f"加载LoRA适配器: {model_path}")
            self.model = PeftModel.from_pretrained(base, model_path)
        else:
            print(f"加载模型: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        self.model.eval()
        self.device = device
    
    def generate(self, question, max_new_tokens=256, temperature=0.7, top_p=0.9):
        prompt = f"<|im_start|>system\n你是一个数学专家,请计算以下积分。<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_start = response.find("assistant\n") + len("assistant\n")
        response = response[assistant_start:].strip()
        
        return response
    
    def interactive_mode(self):
        print("\n=== 积分计算器 (RL优化版) ===")
        print("输入积分问题进行计算,输入 'quit' 退出\n")
        
        while True:
            try:
                question = input("问题: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("再见!")
                    break
                
                if not question:
                    continue
                
                print("\n计算中...\n")
                response = self.generate(question)
                print(f"答案: {response}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as e:
                print(f"错误: {e}\n")
    
    def test_mode(self, test_file=None):
        test_questions = [
            "计算不定积分: ∫x²dx",
            "计算不定积分: ∫sin(x)dx",
            "计算不定积分: ∫e^x dx",
            "计算不定积分: ∫(1/x)dx",
            "计算不定积分: ∫x*cos(x)dx",
            "计算定积分: ∫[0,1] x²dx",
        ]
        
        print("\n=== 测试模式 ===\n")
        
        for i, question in enumerate(test_questions, 1):
            print(f"问题 {i}: {question}")
            response = self.generate(question)
            print(f"答案: {response}")
            print("-" * 50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="积分计算模型推理")
    
    parser.add_argument("--model_path", type=str, default="./ppo_model", help="模型路径")
    parser.add_argument("--base_model", type=str, default=None, help="基础模型路径(用于LoRA)")
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "test"], help="运行模式")
    parser.add_argument("--question", type=str, default=None, help="单个问题")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    
    args = parser.parse_args()
    
    inferencer = IntegralInference(
        args.model_path,
        args.base_model,
        args.use_lora
    )
    
    if args.question:
        print(f"问题: {args.question}")
        response = inferencer.generate(args.question, args.max_new_tokens, args.temperature)
        print(f"答案: {response}")
    elif args.mode == "interactive":
        inferencer.interactive_mode()
    else:
        inferencer.test_mode()

if __name__ == "__main__":
    main()