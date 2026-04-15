#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

echo "Step 1: Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers>=4.36.0
pip install -q peft>=0.7.0
pip install -q datasets>=2.14.0
pip install -q accelerate>=0.24.0
pip install -q pyyaml sentencepiece tiktoken

echo ""
echo "Step 2: Checking environment..."
python scripts/check_env.py

echo ""
echo "Step 3: Starting training..."
python train.py --config config/train_config.yaml