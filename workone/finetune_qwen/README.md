# Qwen1.5-1.8B 微调项目 - 求导与积分运算

## 项目结构
```
finetune_qwen/
├── train.py              # 训练脚本
├── inference.py          # 推理脚本
├── generate_data.py      # 数据生成脚本
├── requirements.txt      # 依赖包
├── config/
│   └── train_config.yaml # 训练配置
├── data/
│   └── train.jsonl       # 训练数据
└── scripts/
    ├── train.sh              # Linux单卡训练
    ├── train.bat             # Windows单卡训练
    ├── train_multigpu.sh     # Linux多卡训练
    ├── inference.sh          # Linux推理
    ├── inference.bat         # Windows推理
    └── check_cuda.py         # CUDA环境检查
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## CUDA环境检查

```bash
python scripts/check_cuda.py
```

## 训练

### 单卡训练
```bash
# Linux
bash scripts/train.sh

# Windows
scripts\train.bat

# 或直接运行
python train.py --config config/train_config.yaml
```

### 多卡训练 (Linux)
```bash
bash scripts/train_multigpu.sh

# 或使用 torchrun
torchrun --nproc_per_node=2 train.py --config config/train_config.yaml

# 或使用 accelerate
accelerate launch train.py --config config/train_config.yaml
```

## 推理测试

```bash
# Linux
bash scripts/inference.sh

# Windows
scripts\inference.bat

# 或直接运行
python inference.py --base_model Qwen/Qwen1.5-1.8B --lora_path output/qwen_calculus

# 自定义问题
python inference.py --base_model Qwen/Qwen1.5-1.8B --lora_path output/qwen_calculus --prompt "求函数 f(x) = x^4 的导数"
```

## 常见问题解决

### 1. CUDA版本不匹配
```bash
# 检查CUDA版本
nvcc --version
nvidia-smi

# 重新安装匹配的PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. 显存不足
修改 `config/train_config.yaml`:
```yaml
training:
  batch_size: 1              # 减小batch size
  gradient_accumulation_steps: 8  # 增加梯度累积
  max_length: 256            # 减小序列长度
```

### 3. 设置CUDA环境变量
```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0  # 指定GPU
```

### 4. 清理GPU缓存
```bash
python -c "import torch; torch.cuda.empty_cache()"
```

## 扩展训练数据

```bash
python generate_data.py
```

生成的数据保存在 `data/train_generated.jsonl`

## 配置说明

编辑 `config/train_config.yaml` 修改：
- `model.name`: 模型路径或HuggingFace名称
- `model.flash_attn`: 是否使用Flash Attention
- `training.batch_size`: 批次大小
- `training.learning_rate`: 学习率
- `lora.r`: LoRA秩
- `training.bf16`: 是否使用BF16 (需要Ampere架构GPU)

## 数据格式

```json
{
  "instruction": "求函数 f(x) = x^3 + 2x^2 的导数",
  "output": "对函数 f(x) = x^3 + 2x^2 求导：\n\nf'(x) = 3x^2 + 4x\n\n因此，导数为 f'(x) = 3x^2 + 4x"
}
```

## 硬件要求

- 单卡训练：至少 8GB 显存（使用 LoRA + gradient checkpointing）
- 推荐：RTX 3090/4090 或 A100
- BF16支持：需要Ampere架构(A100, RTX 30/40系列)