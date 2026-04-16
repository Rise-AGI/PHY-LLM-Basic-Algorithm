#!/bin/bash

echo "========================================"
echo "    积分计算 - RL训练项目"
echo "========================================"
echo ""
echo "请选择操作:"
echo "  1) 生成训练数据"
echo "  2) 训练奖励模型"
echo "  3) PPO强化学习训练"
echo "  4) 测试模型"
echo "  5) 完整流程 (生成数据+训练奖励+PPO)"
echo "  6) 退出"
echo ""
read -p "请输入选项 [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "正在生成训练数据..."
        python generate_data.py
        ;;
    2)
        echo ""
        echo "正在训练奖励模型..."
        python train_reward_model.py \
            --model_name Qwen/Qwen2.5-0.5B-Instruct \
            --train_file data/preference_train.json \
            --val_file data/preference_val.json \
            --output_dir ./reward_model \
            --num_epochs 1 \
            --batch_size 2
        ;;
    3)
        echo ""
        echo "正在训练PPO模型..."
        python train_ppo.py \
            --model_name Qwen/Qwen2.5-0.5B-Instruct \
            --reward_model_path ./reward_model \
            --prompt_file data/prompts.json \
            --output_dir ./ppo_model \
            --num_epochs 1 \
            --batch_size 4
        ;;
    4)
        echo ""
        echo "启动测试模式..."
        python inference.py --model_path ./ppo_model --mode test
        ;;
    5)
        echo ""
        echo "===== 完整训练流程 ====="
        echo ""
        echo "步骤 1/3: 生成训练数据..."
        python generate_data.py
        echo ""
        echo "步骤 2/3: 训练奖励模型..."
        python train_reward_model.py \
            --model_name Qwen/Qwen2.5-0.5B-Instruct \
            --train_file data/preference_train.json \
            --val_file data/preference_val.json \
            --output_dir ./reward_model \
            --num_epochs 1 \
            --batch_size 2
        echo ""
        echo "步骤 3/3: PPO强化学习训练..."
        python train_ppo.py \
            --model_name Qwen/Qwen2.5-0.5B-Instruct \
            --reward_model_path ./reward_model \
            --prompt_file data/prompts.json \
            --output_dir ./ppo_model \
            --num_epochs 1 \
            --batch_size 4
        echo ""
        echo "===== 训练完成! ====="
        ;;
    6)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac