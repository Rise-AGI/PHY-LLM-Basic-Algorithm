# Qwen 1.5B 积分计算 - 强化学习(RL)训练项目

本项目使用强化学习(RL)方法对Qwen模型进行训练,使其能够更好地完成积分计算任务。

## 项目结构

```
ReLearning/
├── generate_data.py       # 数据生成脚本
├── train_reward_model.py  # 奖励模型训练脚本
├── train_ppo.py          # PPO强化学习训练脚本
├── inference.py          # 模型推理脚本
├── requirements.txt      # 依赖包列表
└── README.md            # 项目说明
```

## RL训练流程

### 整体架构

本项目实现了一个完整的RLHF (Reinforcement Learning from Human Feedback)流程:

```
1. 数据生成 → 2. SFT预训练(可选) → 3. 奖励模型训练 → 4. PPO强化学习训练
```

### 核心组件

#### 1. 符号奖励函数 (Symbolic Reward)
使用SymPy数学库自动验证积分答案的正确性:
- 正确答案 → reward = 1.0
- 错误答案 → 根据格式给部分奖励 (-1.0 ~ 0.5)

#### 2. 学习奖励模型 (Learned Reward Model)
从偏好数据中学习,判断答案质量

#### 3. PPO算法
使用Proximal Policy Optimization优化策略模型:
- 策略模型 (Policy Model): 待优化的模型
- 参考模型 (Reference Model): 防止模型偏离太远
- KL散度约束: 保持模型在合理范围内

## 快速开始

### 1. 环境配置

```bash
# 创建conda环境
conda create -n qwen_rl python=3.12 -y
conda activate qwen_rl

# 安装依赖
pip install -r requirements.txt

# 安装PyTorch (根据CUDA版本选择)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 2. 生成数据

```bash
python generate_data.py
```

生成的数据包括:
- `data/preference_train.json` - 偏好训练数据 (1800条)
- `data/preference_val.json` - 偏好验证数据 (200条)
- `data/sft_train.json` - SFT训练数据 (500条)
- `data/prompts.json` - PPO训练prompts (500条)

### 3. 训练奖励模型 (可选但推荐)

```bash
python train_reward_model.py \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --train_file data/preference_train.json \
    --val_file data/preference_val.json \
    --output_dir ./reward_model \
    --num_epochs 1 \
    --batch_size 2 \
    --learning_rate 1e-5
```

### 4. PPO强化学习训练

#### 基础训练
```bash
python train_ppo.py \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --reward_model_path ./reward_model \
    --prompt_file data/prompts.json \
    --output_dir ./ppo_model \
    --num_epochs 1 \
    --batch_size 4
```

#### 使用SFT模型作为起点
```bash
python train_ppo.py \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --sft_model_path ../post-train/output \
    --reward_model_path ./reward_model \
    --prompt_file data/prompts.json \
    --output_dir ./ppo_model \
    --num_epochs 1
```

### 5. 模型推理

```bash
# 交互模式
python inference.py --model_path ./ppo_model --mode interactive

# 测试模式
python inference.py --model_path ./ppo_model --mode test

# 单个问题
python inference.py --model_path ./ppo_model --question "计算不定积分: ∫x²dx"
```

## 训练参数说明

### 奖励模型训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name` | Qwen/Qwen2.5-0.5B-Instruct | 基础模型 |
| `--output_dir` | ./reward_model | 输出目录 |
| `--num_epochs` | 1 | 训练轮数 |
| `--batch_size` | 2 | 批次大小 |
| `--learning_rate` | 1e-5 | 学习率 |

### PPO训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name` | Qwen/Qwen2.5-0.5B-Instruct | 基础模型 |
| `--sft_model_path` | None | SFT模型路径(可选) |
| `--reward_model_path` | ./reward_model | 奖励模型路径 |
| `--output_dir` | ./ppo_model | 输出目录 |
| `--num_epochs` | 1 | 训练轮数 |
| `--batch_size` | 4 | 批次大小 |
| `--learning_rate` | 1e-5 | 学习率 |
| `--kl_coef` | 0.1 | KL散度系数 |
| `--clip_range` | 0.2 | PPO裁剪范围 |
| `--max_new_tokens` | 128 | 最大生成token数 |

## 核心算法说明

### 1. 符号奖励计算

```python
def is_correct(model_answer, correct_answer):
    # 1. 提取积分表达式
    # 2. 使用SymPy规范化表达式
    # 3. 计算两个表达式的差
    # 4. 如果差为0,则答案正确
```

### 2. PPO损失函数

```python
# 策略比率
ratio = exp(log_prob_new - log_prob_old)

# PPO裁剪目标
surr1 = ratio * advantage
surr2 = clip(ratio, 1-ε, 1+ε) * advantage
policy_loss = -min(surr1, surr2)

# KL散度约束
kl_loss = KL(π_new || π_ref)

# 总损失
loss = policy_loss + β * kl_loss
```

### 3. 奖励计算

```python
# 组合奖励
reward = 0.3 * learned_reward + 0.7 * symbolic_reward

# learned_reward: 从奖励模型学习
# symbolic_reward: 符号验证
```

## 与SFT的对比

| 方面 | SFT | RL (本项目) |
|------|-----|------------|
| 训练数据 | 标注的问答对 | 偏好数据 + 交互反馈 |
| 训练目标 | 最大化似然 | 最大化期望奖励 |
| 奖励来源 | 无 | 符号验证 + 学习模型 |
| 优势 | 简单直接 | 可以超越训练数据质量 |
| 劣势 | 受限于训练数据 | 训练复杂,不稳定 |

## 建议的训练流程

1. **先进行SFT训练** (可选但推荐)
   ```bash
   cd ../post-train
   python train.py
   ```

2. **使用SFT模型作为RL起点**
   - 更稳定的初始策略
   - 更快的收敛速度

3. **训练奖励模型**
   - 从偏好数据学习质量判断

4. **PPO训练**
   - 使用组合奖励
   - 小学习率,多轮次

## 注意事项

1. **显存需求**
   - 奖励模型训练: ~8GB
   - PPO训练: ~16GB (需要加载3个模型)
   - 使用小模型或量化可降低需求

2. **训练稳定性**
   - PPO训练需要仔细调参
   - 建议从小学习率开始
   - 监控KL散度避免过大

3. **奖励函数设计**
   - 符号奖励是最可靠的
   - 学习奖励可能存在偏差
   - 组合使用效果更好

4. **训练时间**
   - 奖励模型: ~1-2小时 (1000条数据)
   - PPO: ~2-4小时 (500条prompts, 1 epoch)

## 扩展建议

1. 增加更多积分类型的偏好数据
2. 使用DPO替代PPO (更简单稳定)
3. 加入人工评估反馈
4. 使用更大的模型获得更好效果
5. 实现在线学习 (Online RL)

## 参考文献

- [PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [RLHF: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)