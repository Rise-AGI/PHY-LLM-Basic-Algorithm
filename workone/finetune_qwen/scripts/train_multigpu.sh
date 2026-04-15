#!/bin/bash

# 设置CUDA环境变量
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 设置PyTorch相关环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

# 多卡训练使用 torchrun (推荐)
torchrun --nproc_per_node=2 --master_port=29500 train.py --config config/train_config.yaml

# 或者使用 accelerate (需要先配置: accelerate config)
# accelerate launch train.py --config config/train_config.yaml