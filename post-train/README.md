# Qwen 1.5B 积分计算微调项目

本项目使用LoRA技术对Qwen 1.5B模型进行微调，使其能够胜任复杂的积分计算任务。

## 项目结构

```
post-train/
├── generate_data.py      # 数据集生成脚本
├── train.py             # 模型训练脚本
├── inference.py         # 模型推理脚本
├── start.sh             # 主启动脚本（推荐使用）
├── setup_env.sh         # 详细环境配置脚本
├── setup_quick.sh       # 一键环境配置脚本
├── run_train.sh         # 服务器训练启动脚本
├── requirements.txt     # 依赖包列表
└── README.md           # 项目说明
```

**重要说明：** 所有脚本文件均在 `post-train` 文件夹下运行，脚本会自动切换到正确目录。

## 快速开始

### 使用主启动脚本（最简单）

```bash
# 进入post-train目录
cd post-train

# 给脚本执行权限
chmod +x *.sh

# 运行主启动脚本（交互式菜单）
./start.sh

# 选项:
#   1) 配置环境 (首次运行)
#   2) 生成训练数据
#   3) 开始训练
#   4) 测试模型
#   5) 完整流程 (配置+生成+训练)
```

### 服务器快速运行流程

```bash
# 确保在post-train目录下
cd post-train

# 1. 配置环境（首次运行）
chmod +x setup_quick.sh
./setup_quick.sh

# 2. 激活环境
conda activate qwen_integral

# 3. 生成数据
python generate_data.py

# 4. 开始训练（使用启动脚本）
chmod +x run_train.sh
./run_train.sh

# 或者直接运行训练
python train.py --model_name Qwen/Qwen2.5-1.5B-Instruct --output_dir ./output

# 5. 测试模型
python inference.py --model_path ./output --mode interactive
```

## 环境配置

### 服务器环境配置 (推荐)

#### 方法1: 一键配置脚本（最简单）

```bash
# 给脚本执行权限
chmod +x setup_quick.sh

# 运行一键配置脚本
./setup_quick.sh
```

#### 方法2: 详细配置脚本（推荐新手）

```bash
# 给脚本执行权限
chmod +x setup_env.sh

# 运行详细配置脚本（交互式选择CUDA版本等）
./setup_env.sh
```

#### 方法3: 手动配置（适合有经验用户）

```bash
# 1. 创建conda环境
conda create -n qwen_integral python=3.12 -y

# 2. 激活环境
conda activate qwen_integral

# 3. 安装PyTorch（自动匹配CUDA）
pip install torch==2.6.0

# 4. 安装核心依赖
pip install transformers==4.49.0
pip install datasets==3.3.2
pip install accelerate==1.4.0
pip install peft
pip install sympy==1.13.1
pip install tqdm==4.67.1
pip install numpy==2.2.3
pip install scipy

# 5. 安装Flash Attention（可选，加速训练）
pip install flash-attn==2.7.4.post1 --no-build-isolation

# 6. 安装bitsandbytes（可选，4bit量化）
pip install bitsandbytes
```

#### 方法4: 指定CUDA版本安装PyTorch

```bash
# CUDA 11.8
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4 (推荐)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 验证安装

```bash
# 激活环境
conda activate qwen_integral

# 验证PyTorch和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# 验证关键包
python -c "import transformers, datasets, accelerate; print(f'Transformers: {transformers.__version__}'); print(f'Datasets: {datasets.__version__}'); print(f'Accelerate: {accelerate.__version__}')"
```

### 国内镜像源配置（加速下载）

```bash
# 设置清华镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 或者临时使用
pip install torch==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 硬件要求

- GPU显存: 建议16GB以上 (使用4bit量化可在8GB显存上运行)
- 内存: 建议16GB以上
- Python版本: 3.12 (推荐)
- CUDA版本: 11.8/12.1/12.4

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
   - 国内用户可能需要设置代理或使用镜像源
   - HuggingFace镜像: `export HF_ENDPOINT=https://hf-mirror.com`

2. 建议使用GPU训练，CPU训练速度较慢

3. 如果显存不足，可以：
   - 减小 `--batch_size`
   - 增加 `--gradient_accumulation_steps`
   - 使用 `--use_4bit` 参数

4. 训练过程中会自动保存checkpoint，可在 `--output_dir` 中查看

5. LoRA微调只会训练少量参数，训练速度快，显存占用少

6. Flash Attention安装可能需要较长时间编译，如遇问题可跳过

7. 在服务器后台训练建议使用nohup:
   ```bash
   nohup python train.py --model_name Qwen/Qwen2.5-1.5B-Instruct --output_dir ./output > train.log 2>&1 &
   ```

## 常见问题解决

### 问题1: CUDA版本不匹配

```bash
# 查看CUDA版本
nvidia-smi

# 根据CUDA版本安装对应的PyTorch
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 问题2: HuggingFace下载慢

```bash
# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 或者在Python脚本开头添加
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### 问题3: Flash Attention安装失败

```bash
# Flash Attention需要GPU支持，如果安装失败可以跳过
# 训练时仍可正常运行，只是速度稍慢
pip install transformers==4.49.0 datasets==3.3.2 accelerate==1.4.0 peft
```

### 问题4: 显存不足

```bash
# 使用4bit量化
python train.py --use_4bit --batch_size 2 --gradient_accumulation_steps 8

# 或减小模型加载时的显存占用
# 在train.py中添加: model = model.to("cuda:0")
```

### 问题5: 权限问题

```bash
# 给脚本执行权限
chmod +x setup_env.sh setup_quick.sh run_train.sh

# 或使用bash直接运行
bash setup_quick.sh
```

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