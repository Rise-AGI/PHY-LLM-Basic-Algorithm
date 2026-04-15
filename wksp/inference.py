import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_model_path, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    return model, tokenizer

def predict(model, tokenizer, instruction, input_text="", max_new_tokens=512):
    if input_text:
        prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_start = response.find("assistant\n")
    if assistant_start != -1:
        response = response[assistant_start + len("assistant\n"):]
    
    return response.strip()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen1.5-1.8B')
    parser.add_argument('--adapter', type=str, default='./output/qwen-integration')
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args()
    
    print("Loading model...")
    model, tokenizer = load_model(args.base_model, args.adapter)
    
    if args.interactive:
        print("\nEnter your integration problems (type 'quit' to exit):")
        while True:
            problem = input("\nProblem: ").strip()
            if problem.lower() == 'quit':
                break
            
            response = predict(model, tokenizer, "请计算以下不定积分:", problem)
            print(f"\nSolution:\n{response}")
    else:
        test_cases = [
            "计算不定积分: ∫x^2 dx",
            "计算不定积分: ∫sin(x) dx",
            "计算不定积分: ∫e^x dx",
        ]
        
        for problem in test_cases:
            response = predict(model, tokenizer, "请计算以下不定积分:", problem)
            print(f"Problem: {problem}")
            print(f"Solution: {response}\n")

if __name__ == '__main__':
    main()