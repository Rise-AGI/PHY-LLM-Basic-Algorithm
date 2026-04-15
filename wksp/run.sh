#!/bin/bash

set -e

echo "=== Qwen1.5B Integration Fine-tuning ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/4] Installing dependencies..."
pip install -r requirements.txt

echo "[2/4] Generating training data..."
python generate_integration_data.py \
    --train_size 10000 \
    --eval_size 500 \
    --output_dir ./data

echo "[3/4] Starting training..."
python train.py --config finetune_config.yaml

echo "[4/4] Training complete!"
echo "Model saved to: ./output/qwen-integration"
echo ""
echo "To test the model, run:"
echo "  python inference.py --base_model Qwen/Qwen1.5-1.8B --adapter ./output/qwen-integration --interactive"