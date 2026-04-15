#!/bin/bash
# 推理测试
python inference.py --base_model Qwen/Qwen1.5-1.8B --lora_path output/qwen_calculus

# 自定义问题测试
# python inference.py --base_model Qwen/Qwen1.5-1.8B --lora_path output/qwen_calculus --prompt "求函数 f(x) = x^4 的导数"