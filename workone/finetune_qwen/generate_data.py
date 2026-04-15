import json
import random

def generate_derivative_problems():
    problems = []
    
    basic_rules = [
        ("x^{}", "kx^{{k-1}}", lambda k: f"{k}x^{{{k-1}}}"),
        ("sin(x)", "cos(x)", None),
        ("cos(x)", "-sin(x)", None),
        ("e^x", "e^x", None),
        ("ln(x)", "1/x", None),
        ("tan(x)", "sec^2(x)", None),
        ("arcsin(x)", "1/sqrt(1-x^2)", None),
        ("arctan(x)", "1/(1+x^2)", None),
    ]
    
    for _ in range(50):
        power = random.randint(2, 10)
        coef = random.randint(2, 9)
        
        problems.append({
            "instruction": f"求函数 f(x) = {coef}x^{power} 的导数",
            "output": f"对函数 f(x) = {coef}x^{power} 求导：\n\n根据幂函数求导公式，(ax^n)' = anx^(n-1)\n\nf'(x) = {coef * power}x^{power-1}\n\n因此，导数为 f'(x) = {coef * power}x^{power-1}"
        })
    
    composite_funcs = [
        ("sin({}x)", "-{}cos({}x)", lambda m: f"-{m}cos({m}x)"),
        ("cos({}x)", "-{}sin({}x)", lambda m: f"-{m}sin({m}x)"),
        ("e^{{{}x}}", "{}e^{{{}x}}", lambda m: f"{m}e^{{{m}x}}"),
        ("ln({}x)", "1/x", lambda m: "1/x"),
    ]
    
    for _ in range(30):
        m = random.randint(2, 5)
        func_template, _, result_func = random.choice(composite_funcs)
        func = func_template.format(m)
        
        problems.append({
            "instruction": f"求函数 f(x) = {func} 的导数",
            "output": f"对函数 f(x) = {func} 求导：\n\n使用链式法则求导。\n\nf'(x) = {result_func(m)}\n\n因此，导数为 f'(x) = {result_func(m)}"
        })
    
    return problems


def generate_integral_problems():
    problems = []
    
    for _ in range(50):
        power = random.randint(2, 8)
        coef = random.randint(2, 5)
        
        new_power = power + 1
        new_coef = coef / new_power if coef % new_power != 0 else coef // new_power
        
        if coef % new_power == 0:
            result = f"{int(new_coef)}x^{new_power}/{new_power}"
        else:
            result = f"{coef}x^{new_power}/{new_power}"
        
        problems.append({
            "instruction": f"计算不定积分 ∫{coef}x^{power} dx",
            "output": f"计算不定积分 ∫{coef}x^{power} dx：\n\n根据幂函数积分公式 ∫x^n dx = x^(n+1)/(n+1) + C\n\n∫{coef}x^{power} dx = {coef} · x^{new_power}/{new_power} + C = {result} + C\n\n因此，∫{coef}x^{power} dx = {result} + C"
        })
    
    basic_integrals = [
        ("cos(x)", "sin(x)"),
        ("sin(x)", "-cos(x)"),
        ("e^x", "e^x"),
        ("1/x", "ln|x|"),
        ("sec^2(x)", "tan(x)"),
        ("1/(1+x^2)", "arctan(x)"),
    ]
    
    for func, antideriv in basic_integrals:
        problems.append({
            "instruction": f"计算不定积分 ∫{func} dx",
            "output": f"计算不定积分 ∫{func} dx：\n\n由于 d({antideriv})/dx = {func}\n\n因此，∫{func} dx = {antideriv} + C"
        })
    
    return problems


def main():
    derivative_problems = generate_derivative_problems()
    integral_problems = generate_integral_problems()
    
    all_problems = derivative_problems + integral_problems
    random.shuffle(all_problems)
    
    with open('data/train_generated.jsonl', 'w', encoding='utf-8') as f:
        for problem in all_problems:
            f.write(json.dumps(problem, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(all_problems)} problems")
    print(f"  - Derivative problems: {len(derivative_problems)}")
    print(f"  - Integral problems: {len(integral_problems)}")


if __name__ == "__main__":
    main()