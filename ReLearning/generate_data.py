import json
import random
from sympy import symbols, integrate, sin, cos, tan, exp, log, sqrt, Rational, pi, E, simplify
from sympy import sympify, latex, diff
from sympy.abc import x, t, u, theta
from tqdm import tqdm
import os

def generate_integral_with_answer():
    generators = [
        generate_basic_integral,
        generate_trig_integral,
        generate_exp_log_integral,
        generate_rational_integral,
        generate_complex_integral,
        generate_definite_integral,
    ]
    generator = random.choice(generators)
    return generator()

def generate_basic_integral():
    a = random.randint(1, 10)
    b = random.randint(-5, 5)
    n = random.randint(0, 5)
    
    if n == 0:
        expr = a
    else:
        expr = a * x**n + b
    
    integral = integrate(expr, x)
    
    prompt = f"计算不定积分: ∫({latex(expr)})dx"
    correct_answer = f"∫({latex(expr)})dx = {latex(integral)} + C"
    
    return {
        "prompt": prompt,
        "correct_answer": correct_answer,
        "type": "basic"
    }

def generate_trig_integral():
    choice = random.randint(0, 5)
    a = random.randint(1, 5)
    
    if choice == 0:
        expr = sin(a * x)
    elif choice == 1:
        expr = cos(a * x)
    elif choice == 2:
        expr = sin(x) * cos(x)
    elif choice == 3:
        expr = sin(x)**2
    elif choice == 4:
        expr = cos(x)**2
    else:
        expr = tan(x)
    
    integral = integrate(expr, x)
    prompt = f"计算不定积分: ∫{latex(expr)}dx"
    correct_answer = f"∫{latex(expr)}dx = {latex(integral)} + C"
    
    return {
        "prompt": prompt,
        "correct_answer": correct_answer,
        "type": "trigonometric"
    }

def generate_exp_log_integral():
    choice = random.randint(0, 3)
    a = random.randint(1, 5)
    
    if choice == 0:
        expr = exp(a * x)
    elif choice == 1:
        expr = exp(x) * x
    elif choice == 2:
        expr = 1 / x
    else:
        expr = log(a * x)
    
    integral = integrate(expr, x)
    prompt = f"计算不定积分: ∫{latex(expr)}dx"
    correct_answer = f"∫{latex(expr)}dx = {latex(integral)} + C"
    
    return {
        "prompt": prompt,
        "correct_answer": correct_answer,
        "type": "exp_log"
    }

def generate_rational_integral():
    a = random.randint(1, 5)
    b = random.randint(1, 5)
    
    choice = random.randint(0, 2)
    
    if choice == 0:
        expr = 1 / (a * x + b)
    elif choice == 1:
        expr = 1 / (x**2 + a**2)
    else:
        expr = x / (x**2 + a)
    
    integral = integrate(expr, x)
    prompt = f"计算不定积分: ∫{latex(expr)}dx"
    correct_answer = f"∫{latex(expr)}dx = {latex(integral)} + C"
    
    return {
        "prompt": prompt,
        "correct_answer": correct_answer,
        "type": "rational"
    }

def generate_complex_integral():
    a = symbols('a')
    examples = [
        (x * exp(x), "分部积分法"),
        (x**2 * exp(x), "分部积分法"),
        (x * sin(x), "分部积分法"),
        (x * cos(x), "分部积分法"),
        (log(x), "分部积分法"),
    ]
    
    idx = random.randint(0, len(examples) - 1)
    expr_template, method = examples[idx]
    expr = expr_template
    
    integral = integrate(expr, x)
    prompt = f"计算不定积分: ∫{latex(expr)}dx"
    correct_answer = f"使用{method}。∫{latex(expr)}dx = {latex(integral)} + C"
    
    return {
        "prompt": prompt,
        "correct_answer": correct_answer,
        "type": "complex"
    }

def generate_definite_integral():
    examples = [
        (x**2, 0, 1),
        (sin(x), 0, pi),
        (cos(x), 0, pi/2),
        (exp(x), 0, 1),
        (1/x, 1, E),
        (sqrt(x), 0, 4),
        (x**3, 0, 2),
    ]
    
    idx = random.randint(0, len(examples) - 1)
    expr_template, a, b = examples[idx]
    
    if random.random() > 0.5:
        factor = random.randint(1, 5)
        expr = factor * expr_template
    else:
        expr = expr_template
    
    integral = integrate(expr, (x, a, b))
    prompt = f"计算定积分: ∫[{latex(a)},{latex(b)}] {latex(expr)}dx"
    correct_answer = f"∫[{latex(a)},{latex(b)}] {latex(expr)}dx = {latex(integral)}"
    
    return {
        "prompt": prompt,
        "correct_answer": correct_answer,
        "type": "definite"
    }

def generate_wrong_answer(correct_answer, error_type="random"):
    wrong_answers = []
    
    if error_type == "sign":
        wrong_answers.append(correct_answer.replace("+ C", "- C"))
        wrong_answers.append(correct_answer.replace(" + ", " - "))
    elif error_type == "coefficient":
        import re
        nums = re.findall(r'\d+', correct_answer)
        if nums:
            num = random.choice(nums)
            wrong_num = str(int(num) + random.randint(1, 3))
            wrong = correct_answer.replace(num, wrong_num, 1)
            wrong_answers.append(wrong)
    
    wrong_answers.append(correct_answer + " (错误)")
    wrong_answers.append("无法计算此积分")
    wrong_answers.append("0 + C")
    
    return random.choice(wrong_answers)

def generate_preference_data(n_samples=1000):
    data = []
    print("生成偏好数据...")
    
    for _ in tqdm(range(n_samples)):
        item = generate_integral_with_answer()
        
        prompt = item["prompt"]
        correct = item["correct_answer"]
        
        chosen = f"解: {correct}"
        rejected = f"解: {generate_wrong_answer(correct)}"
        
        data.append({
            "prompt": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "chosen": chosen,
            "rejected": rejected,
            "type": item["type"]
        })
    
    return data

def generate_sft_data(n_samples=1000):
    data = []
    print("生成SFT数据...")
    
    for _ in tqdm(range(n_samples)):
        item = generate_integral_with_answer()
        
        data.append({
            "instruction": "你是一个数学专家,请计算以下积分。",
            "input": item["prompt"],
            "output": f"解: {item['correct_answer']}"
        })
    
    return data

def generate_prompt_dataset(n_samples=500):
    data = []
    print("生成Prompt数据集(用于PPO训练)...")
    
    for _ in tqdm(range(n_samples)):
        item = generate_integral_with_answer()
        data.append({
            "prompt": f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n",
            "correct_answer": item["correct_answer"],
            "type": item["type"]
        })
    
    return data

def main():
    os.makedirs("data", exist_ok=True)
    
    preference_data = generate_preference_data(n_samples=2000)
    train_size = int(len(preference_data) * 0.9)
    train_pref = preference_data[:train_size]
    val_pref = preference_data[train_size:]
    
    with open("data/preference_train.json", "w", encoding="utf-8") as f:
        for item in train_pref:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open("data/preference_val.json", "w", encoding="utf-8") as f:
        for item in val_pref:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"偏好数据训练集: {len(train_pref)} 条")
    print(f"偏好数据验证集: {len(val_pref)} 条")
    
    sft_data = generate_sft_data(n_samples=500)
    with open("data/sft_train.json", "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"SFT数据: {len(sft_data)} 条")
    
    prompt_data = generate_prompt_dataset(n_samples=500)
    with open("data/prompts.json", "w", encoding="utf-8") as f:
        for item in prompt_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Prompt数据集: {len(prompt_data)} 条")
    print("\n数据生成完成!")

if __name__ == "__main__":
    main()