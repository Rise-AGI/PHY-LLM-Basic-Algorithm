#!/bin/bash
# 单卡训练
python train.py --config config/train_config.yaml

# 多卡训练 (使用2张GPU)
# accelerate launch --num_processes 2 train.py --config config/train_config.yaml