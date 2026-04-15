# Qwen 1.5B 积分计算微调项目

本项目使用LoRA技术对Qwen 1.5B模型进行微调，使其能够胜任复杂的积分计算任务。

## 项目结构

```
post-train/
├── generate_data.py      # 数据集生成脚本
├── train.py             # 模型训练脚本
├── inference.py         # 模型推理脚本
├── requirements.txt     # 依赖包
└── README.md           # 项目说明
```

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 硬件要求

- GPU显存: 建议16GB以上 (使用4bit量化可在8GB显存上运行)
- 内存: 建议16GB以上

## 使用步骤

### 1. 生成训练数据

运行数据生成脚本，生成积分相关的训练数据：

```bash
python generate_data.py
```

这将生成：
- `train.json`: 训练集 (约1000条数据)
- `val.json`: 验证集 (约100条数据)

数据类型包括：
- 基本积分 (幂函数、常数等)
- 三角函数积分
- 指数对数积分
- 有理函数积分
- 复杂积分 (分部积分、换元积分)
- 定积分
- 带详细步骤的积分

### 2. 训练模型

#### 基本训练命令

```bash
python train.py --model_name Qwen/Qwen2.5-1.5B-Instruct --output_dir ./output
```

#### 使用4bit量化训练 (节省显存)

```bash
python train.py --model_name Qwen/Qwen2.5-1.5B-Instruct --output_dir ./output --use_4bit
```

#### 自定义参数训练

```bash
python train.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --train_file train.json \
    --val_file val.json \
    --output_dir ./output \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --max_length 512 \
    --lora_r 16 \
    --lora_alpha 32
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name` | Qwen/Qwen2.5-1.5B-Instruct | 基础模型名称 |
| `--output_dir` | ./output | 输出目录 |
| `--num_epochs` | 3 | 训练轮数 |
| `--batch_size` | 4 | 批次大小 |
| `--learning_rate` | 2e-4 | 学习率 |
| `--max_length` | 512 | 最大序列长度 |
| `--lora_r` | 16 | LoRA秩 |
| `--lora_alpha` | 32 | LoRA alpha参数 |
| `--use_4bit` | False | 使用4bit量化 |

### 3. 模型推理

#### 交互模式

```bash
python inference.py --model_path ./output --mode interactive
```

#### 测试模式

```bash
python inference.py --model_path ./output --mode test
```

#### 单个问题推理

```bash
python inference.py --model_path ./output --question "计算不定积分: ∫x²dx"
```

#### 使用LoRA模型推理

```bash
python inference.py \
    --model_path ./output \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --mode interactive
```

## 示例对话

```
用户: 计算不定积分: ∫x³dx

助手: 解: 使用幂函数积分公式 ∫x^n dx = x^(n+1)/(n+1) + C
∫x³dx = x⁴/4 + C

用户: 计算不定积分: ∫sin(x)dx

助手: 解: ∫sin(x)dx = -cos(x) + C

用户: 计算定积分: ∫[0,1] x²dx

助手: 解: ∫[0,1] x²dx = [x³/3]₀¹ = 1³/3 - 0³/3 = 1/3
```

## 注意事项

1. 首次运行会自动从HuggingFace下载Qwen模型，请确保网络连接正常
2. 建议使用GPU训练，CPU训练速度较慢
3. 如果显存不足，可以：
   - 减小 `--batch_size`
   - 增加 `--gradient_accumulation_steps`
   - 使用 `--use_4bit` 参数
4. 训练过程中会自动保存checkpoint，可在 `--output_dir` 中查看
5. LoRA微调只会训练少量参数，训练速度快，显存占用少

## 数据集说明

数据集使用SymPy库自动生成，确保答案的正确性。数据格式：

```json
{
  "instruction": "你是一个数学专家,请计算以下积分。",
  "input": "计算不定积分: ∫x²dx",
  "output": "解: ∫x²dx = x³/3 + C"
}
```

## 扩展建议

1. 增加更多积分类型的数据
2. 添加数学证明题、极限计算等相关任务
3. 使用更大的模型 (如Qwen 7B) 获得更好效果
4. 调整LoRA参数以优化性能