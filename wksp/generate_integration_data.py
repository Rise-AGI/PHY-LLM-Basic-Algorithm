import json
import random
import argparse
from sympy import *
from sympy.integrals.manualintegrate import manualintegrate

x, y, z, t = symbols('x y z t')

INTEGRATION_TEMPLATES = [
    ('polynomial', [
        (x**2, 'x^2'),
        (x**3, 'x^3'),
        (3*x**2 + 2*x + 1, '3x^2 + 2x + 1'),
        (x**4 - x**2, 'x^4 - x^2'),
        (5*x + 3, '5x + 3'),
    ]),
    ('rational', [
        (1/x, '1/x'),
        (1/(x**2), '1/x^2'),
        (x/(x**2 + 1), 'x/(x^2 + 1)'),
    ]),
    ('trigonometric', [
        (sin(x), 'sin(x)'),
        (cos(x), 'cos(x)'),
        (sin(2*x), 'sin(2x)'),
        (cos(3*x), 'cos(3x)'),
        (tan(x), 'tan(x)'),
        (sin(x)**2, 'sin^2(x)'),
        (sin(x)*cos(x), 'sin(x)cos(x)'),
    ]),
    ('exponential', [
        (exp(x), 'e^x'),
        (exp(2*x), 'e^(2x)'),
        (x*exp(x), 'xe^x'),
        (exp(-x), 'e^(-x)'),
    ]),
    ('logarithmic', [
        (1/x, '1/x (ln form)'),
        (log(x), 'ln(x)'),
        (log(x)/x, 'ln(x)/x'),
    ]),
    ('inverse_trig', [
        (1/sqrt(1 - x**2), '1/sqrt(1-x^2)'),
        (1/(1 + x**2), '1/(1+x^2)'),
    ]),
    ('combined', [
        (x*exp(x**2), 'xe^(x^2)'),
        (x*sin(x**2), 'x*sin(x^2)'),
        (exp(x)*sin(x), 'e^x * sin(x)'),
        (x**2 * log(x), 'x^2 * ln(x)'),
    ]),
]

def format_integral(expr_str):
    return f"计算不定积分: ∫{expr_str} dx"

def format_solution(expr, result):
    try:
        result_str = str(result).replace('**', '^')
        expr_str = str(expr).replace('**', '^')
        return f"解:\n∫{expr_str} dx = {result_str} + C"
    except:
        return f"解:\n∫{str(expr)} dx = {str(result)} + C"

def generate_sample():
    category, templates = random.choice(INTEGRATION_TEMPLATES)
    expr, expr_str = random.choice(templates)
    
    try:
        result = integrate(expr, x)
        if result is None or result.has(Integral):
            return None
        
        question = format_integral(expr_str)
        answer = format_solution(expr, result)
        
        return {
            "instruction": "请计算以下不定积分:",
            "input": question,
            "output": answer,
            "category": category
        }
    except:
        return None

def generate_advanced_sample():
    templates = [
        (x**2 + 3*x - 5, 'x^2 + 3x - 5'),
        (x**3 - 2*x**2 + x, 'x^3 - 2x^2 + x'),
        (sin(x)**2, 'sin^2(x)'),
        (cos(x)**2, 'cos^2(x)'),
        (1/(x**2 + 4), '1/(x^2 + 4)'),
        (x/(x**2 + 1), 'x/(x^2 + 1)'),
        (exp(x)*cos(x), 'e^x * cos(x)'),
        (x*exp(-x), 'x * e^(-x)'),
        (sqrt(x), 'sqrt(x)'),
        (1/sqrt(x), '1/sqrt(x)'),
        (x*sin(x), 'x * sin(x)'),
        (x**2 * cos(x), 'x^2 * cos(x)'),
        (sec(x)**2, 'sec^2(x)'),
        (1/(sqrt(1 - x**2)), '1/sqrt(1 - x^2)'),
    ]
    
    expr, expr_str = random.choice(templates)
    
    try:
        result = integrate(expr, x)
        if result is None or result.has(Integral):
            return None
        
        question = format_integral(expr_str)
        answer = format_solution(expr, result)
        
        return {
            "instruction": "请计算以下不定积分，并给出详细的解题步骤:",
            "input": question,
            "output": answer,
            "category": "advanced"
        }
    except:
        return None

def generate_dataset(size, output_file, seed=42):
    random.seed(seed)
    samples = []
    
    while len(samples) < size:
        if random.random() < 0.7:
            sample = generate_sample()
        else:
            sample = generate_advanced_sample()
        
        if sample:
            samples.append(sample)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len(samples)} samples to {output_file}")
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=int, default=10000)
    parser.add_argument('--eval_size', type=int, default=500)
    parser.add_argument('--output_dir', type=str, default='./data')
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_file = os.path.join(args.output_dir, 'integration_train.json')
    eval_file = os.path.join(args.output_dir, 'integration_eval.json')
    
    generate_dataset(args.train_size, train_file, seed=42)
    generate_dataset(args.eval_size, eval_file, seed=123)

if __name__ == '__main__':
    main()