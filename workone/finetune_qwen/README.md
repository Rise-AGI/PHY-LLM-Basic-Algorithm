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
    ├── train.sh          # Linux训练启动脚本
    ├── train.bat         # Windows训练启动脚本
    ├── inference.sh      # Linux推理脚本
    └── inference.bat     # Windows推理脚本
```

## 安装依赖

```bash
pip install -r requirements.txt
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

### 多卡训练
```bash
# 使用 accelerate
accelerate launch --num_processes 2 train.py --config config/train_config.yaml
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

## 扩展训练数据

```bash
python generate_data.py
```

生成的数据保存在 `data/train_generated.jsonl`，可在配置文件中修改 `train_path` 使用。

## 配置说明

编辑 `config/train_config.yaml` 修改：
- 模型路径
- 训练轮数、批次大小、学习率
- LoRA 参数（rank, alpha, dropout）
- 输出路径

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