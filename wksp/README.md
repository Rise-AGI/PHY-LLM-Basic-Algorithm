# Qwen1.5B Integration Fine-tuning

微调Qwen1.5B模型使其能够高效计算积分。

## 项目结构

```
.
├── finetune_config.yaml      # 训练配置文件
├── generate_integration_data.py  # 数据生成脚本
├── train.py                  # 训练脚本
├── inference.py              # 推理脚本
├── requirements.txt          # 依赖包
├── run.sh                    # Linux启动脚本
└── run.ps1                   # Windows启动脚本
```

## 快速开始

### Linux服务器
```bash
chmod +x run.sh
./run.sh
```

### Windows服务器
```powershell
.\run.ps1
```

### 手动执行
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成数据
python generate_integration_data.py --train_size 10000 --eval_size 500 --output_dir ./data

# 3. 开始训练
python train.py --config finetune_config.yaml
```

## 单卡/多卡训练

### 单卡训练
```bash
python train.py --config finetune_config.yaml
```

### 多卡训练 (DDP)
```bash
torchrun --nproc_per_node=4 train.py --config finetune_config.yaml
```

### 多卡训练 (DeepSpeed)
```bash
deepspeed train.py --config finetune_config.yaml
```

## 推理测试

### 交互式测试
```bash
python inference.py --base_model Qwen/Qwen1.5-1.8B --adapter ./output/qwen-integration --interactive
```

### 批量测试
```bash
python inference.py --base_model Qwen/Qwen1.5-1.8B --adapter ./output/qwen-integration
```

## 配置说明

主要参数在 `finetune_config.yaml` 中配置：

- `model.name`: 模型名称或路径
- `lora.r`: LoRA秩，影响模型容量
- `training.learning_rate`: 学习率
- `training.num_train_epochs`: 训练轮数
- `training.per_device_train_batch_size`: 单卡批次大小

## 硬件要求

- 最低: 16GB GPU显存 (使用4bit量化)
- 推荐: 24GB+ GPU显存
- 多卡训练建议: 每卡至少16GB显存

## 数据格式

训练数据格式:
```json
{
  "instruction": "请计算以下不定积分:",
  "input": "计算不定积分: ∫x^2 dx",
  "output": "解:\n∫x^2 dx = x^3/3 + C",
  "category": "polynomial"
}
```