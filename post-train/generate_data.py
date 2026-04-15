import json
import random
from sympy import symbols, integrate, sin, cos, tan, exp, log, sqrt, Rational, pi, E
from sympy import sympify, latex, simplify, diff
from sympy.abc import x, t, u, theta
import tqdm

def generate_basic_integrals(n_samples=100):
    data = []
    for _ in range(n_samples):
        a = random.randint(1, 10)
        b = random.randint(-5, 5)
        n = random.randint(0, 5)
        
        if n == 0:
            expr = a
        else:
            expr = a * x**n + b
        
        integral = integrate(expr, x)
        
        prompt = f"计算不定积分: ∫({latex(expr)})dx"
        response = f"解: ∫({latex(expr)})dx = {latex(integral)} + C"
        
        data.append({
            "instruction": "你是一个数学专家,请计算以下积分。",
            "input": prompt,
            "output": response
        })
    return data

def generate_trig_integrals(n_samples=100):
    data = []
    
    for _ in range(n_samples):
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
        
        try:
            integral = integrate(expr, x)
            prompt = f"计算不定积分: ∫{latex(expr)}dx"
            response = f"解: ∫{latex(expr)}dx = {latex(integral)} + C"
            
            data.append({
                "instruction": "你是一个数学专家,请计算以下积分。",
                "input": prompt,
                "output": response
            })
        except:
            continue
    return data

def generate_exp_log_integrals(n_samples=100):
    data = []
    
    for _ in range(n_samples):
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
        
        try:
            integral = integrate(expr, x)
            prompt = f"计算不定积分: ∫{latex(expr)}dx"
            response = f"解: ∫{latex(expr)}dx = {latex(integral)} + C"
            
            data.append({
                "instruction": "你是一个数学专家,请计算以下积分。",
                "input": prompt,
                "output": response
            })
        except:
            continue
    return data

def generate_rational_integrals(n_samples=80):
    data = []
    
    for _ in range(n_samples):
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        n = random.randint(1, 3)
        
        choice = random.randint(0, 2)
        
        if choice == 0:
            expr = 1 / (a * x + b)
        elif choice == 1:
            expr = 1 / (x**2 + a**2)
        else:
            expr = x / (x**2 + a)
        
        try:
            integral = integrate(expr, x)
            prompt = f"计算不定积分: ∫{latex(expr)}dx"
            response = f"解: ∫{latex(expr)}dx = {latex(integral)} + C"
            
            data.append({
                "instruction": "你是一个数学专家,请计算以下积分。",
                "input": prompt,
                "output": response
            })
        except:
            continue
    return data

def generate_complex_integrals(n_samples=120):
    data = []
    
    examples = [
        (x * exp(x), "分部积分法"),
        (x**2 * exp(x), "分部积分法"),
        (x * sin(x), "分部积分法"),
        (x * cos(x), "分部积分法"),
        (log(x), "分部积分法"),
        (sqrt(a - x**2), "三角换元法", {"a": random.randint(1, 5)}),
        (1 / sqrt(x**2 + a**2), "三角换元法", {"a": random.randint(1, 5)}),
    ]
    
    for _ in range(n_samples):
        idx = random.randint(0, len(examples) - 1)
        example = examples[idx]
        
        if len(example) == 3:
            expr_template, method, params = example
            expr = expr_template.subs(params)
        else:
            expr_template, method = example
            expr = expr_template
        
        try:
            integral = integrate(expr, x)
            prompt = f"计算不定积分: ∫{latex(expr)}dx"
            response = f"解: 使用{method}。\n∫{latex(expr)}dx = {latex(integral)} + C"
            
            data.append({
                "instruction": "你是一个数学专家,请计算以下积分。",
                "input": prompt,
                "output": response
            })
        except:
            continue
    return data

def generate_definite_integrals(n_samples=100):
    data = []
    
    examples = [
        (x**2, 0, 1),
        (sin(x), 0, pi),
        (cos(x), 0, pi/2),
        (exp(x), 0, 1),
        (1/x, 1, E),
        (x * exp(-x), 0, 10),
        (sqrt(x), 0, 4),
        (x**3, 0, 2),
    ]
    
    for _ in range(n_samples):
        idx = random.randint(0, len(examples) - 1)
        expr_template, a, b = examples[idx]
        
        if random.random() > 0.5:
            factor = random.randint(1, 5)
            expr = factor * expr_template
        else:
            expr = expr_template
        
        try:
            integral = integrate(expr, (x, a, b))
            prompt = f"计算定积分: ∫[{latex(a)},{latex(b)}] {latex(expr)}dx"
            response = f"解: ∫[{latex(a)},{latex(b)}] {latex(expr)}dx = {latex(integral)}"
            
            data.append({
                "instruction": "你是一个数学专家,请计算以下定积分。",
                "input": prompt,
                "output": response
            })
        except:
            continue
    return data

def generate_step_by_step_integrals(n_samples=100):
    data = []
    
    examples = [
        {
            "expr": x**2 * sin(x),
            "steps": [
                "使用分部积分法",
                "设 u = x², dv = sin(x)dx",
                "则 du = 2x dx, v = -cos(x)",
                "∫x²sin(x)dx = -x²cos(x) + ∫2x cos(x)dx",
                "对∫x cos(x)dx再次使用分部积分",
                "设 u = x, dv = cos(x)dx",
                "则 du = dx, v = sin(x)",
                "∫x cos(x)dx = x sin(x) - ∫sin(x)dx = x sin(x) + cos(x)",
                "因此∫x²sin(x)dx = -x²cos(x) + 2x sin(x) + 2cos(x) + C"
            ]
        },
        {
            "expr": exp(x) * sin(x),
            "steps": [
                "使用分部积分法两次",
                "设 I = ∫e^x sin(x)dx",
                "设 u = sin(x), dv = e^x dx",
                "则 du = cos(x)dx, v = e^x",
                "I = e^x sin(x) - ∫e^x cos(x)dx",
                "对∫e^x cos(x)dx再次使用分部积分",
                "设 u = cos(x), dv = e^x dx",
                "∫e^x cos(x)dx = e^x cos(x) + I",
                "因此 I = e^x sin(x) - e^x cos(x) - I",
                "2I = e^x(sin(x) - cos(x))",
                "I = e^x(sin(x) - cos(x))/2 + C"
            ]
        }
    ]
    
    for example in examples:
        expr = example["expr"]
        steps = example["steps"]
        
        integral = integrate(expr, x)
        
        prompt = f"计算不定积分: ∫{latex(expr)}dx (请给出详细步骤)"
        response = "解:\n" + "\n".join(steps)
        response += f"\n最终结果: {latex(integral)} + C"
        
        data.append({
            "instruction": "你是一个数学专家,请详细计算以下积分并给出步骤。",
            "input": prompt,
            "output": response
        })
    
    for _ in range(n_samples - len(examples)):
        a = random.randint(1, 5)
        n = random.randint(0, 3)
        
        if n == 0:
            expr = a * x**2
        else:
            expr = a * x**n
        
        integral = integrate(expr, x)
        
        prompt = f"计算不定积分: ∫{latex(expr)}dx"
        response = f"解: 使用幂函数积分公式 ∫x^n dx = x^(n+1)/(n+1) + C\n"
        response += f"∫{latex(expr)}dx = {latex(integral)} + C"
        
        data.append({
            "instruction": "你是一个数学专家,请计算以下积分。",
            "input": prompt,
            "output": response
        })
    
    return data

def main():
    print("正在生成积分数据集...")
    
    all_data = []
    
    print("生成基本积分...")
    all_data.extend(generate_basic_integrals(200))
    
    print("生成三角函数积分...")
    all_data.extend(generate_trig_integrals(200))
    
    print("生成指数对数积分...")
    all_data.extend(generate_exp_log_integrals(150))
    
    print("生成有理函数积分...")
    all_data.extend(generate_rational_integrals(150))
    
    print("生成复杂积分...")
    all_data.extend(generate_complex_integrals(200))
    
    print("生成定积分...")
    all_data.extend(generate_definite_integrals(150))
    
    print("生成详细步骤积分...")
    all_data.extend(generate_step_by_step_integrals(150))
    
    random.shuffle(all_data)
    
    train_size = int(len(all_data) * 0.9)
    train_data = all_data[:train_size]
    val_data = all_data[train_size:]
    
    with open("train.json", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open("val.json", "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"数据集生成完成!")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print(f"总计: {len(all_data)} 条")

if __name__ == "__main__":
    main()