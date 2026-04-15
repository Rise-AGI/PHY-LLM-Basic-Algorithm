#!/usr/bin/env pwsh

Write-Host "=== Qwen1.5B Integration Fine-tuning ===" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "[1/4] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "[2/4] Generating training data..." -ForegroundColor Yellow
python generate_integration_data.py `
    --train_size 10000 `
    --eval_size 500 `
    --output_dir ./data

Write-Host "[3/4] Starting training..." -ForegroundColor Yellow
python train.py --config finetune_config.yaml

Write-Host "[4/4] Training complete!" -ForegroundColor Green
Write-Host "Model saved to: ./output/qwen-integration" -ForegroundColor Green
Write-Host ""
Write-Host "To test the model, run:" -ForegroundColor Cyan
Write-Host "  python inference.py --base_model Qwen/Qwen1.5-1.8B --adapter ./output/qwen-integration --interactive" -ForegroundColor White