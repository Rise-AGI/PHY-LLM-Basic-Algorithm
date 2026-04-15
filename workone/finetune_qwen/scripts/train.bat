@echo off
REM 单卡训练
python train.py --config config/train_config.yaml

REM 多卡训练 (使用2张GPU)
REM accelerate launch --num_processes 2 train.py --config config/train_config.yaml