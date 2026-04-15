@echo off
REM 设置CUDA环境变量
set CUDA_VISIBLE_DEVICES=0

REM 设置PyTorch相关环境变量
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM 单卡训练
python train.py --config config/train_config.yaml

REM 多卡训练 (使用2张GPU) - 取消下面注释
REM torchrun --nproc_per_node=2 train.py --config config/train_config.yaml