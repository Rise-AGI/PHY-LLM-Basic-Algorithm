import torch
import json
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from tqdm import tqdm
import argparse
from sympy import sympify, latex, simplify, symbols, integrate
from sympy.abc import x
import re

class RewardDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        chosen_text = item["prompt"] + item["chosen"] + "<|im_end|>"
        rejected_text = item["prompt"] + item["rejected"] + "<|im_end|>"
        
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_encoding['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(),
        }

class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen_rewards = model(
            input_ids=inputs['chosen_input_ids'],
            attention_mask=inputs['chosen_attention_mask']
        ).logits.squeeze(-1)
        
        rejected_rewards = model(
            input_ids=inputs['rejected_input_ids'],
            attention_mask=inputs['rejected_attention_mask']
        ).logits.squeeze(-1)
        
        loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        return (loss, {'chosen_reward': chosen_rewards.mean(), 'rejected_reward': rejected_rewards.mean()}) if return_outputs else loss

class SymbolicRewardModel:
    def __init__(self):
        pass
    
    def extract_expression(self, text):
        patterns = [
            r'=\s*(.+?)\s*\+?\s*C',
            r'=\s*(.+?)\s*<\|im_end\|>',
            r'=\s*(.+?)$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None
    
    def is_correct(self, model_answer, correct_answer):
        try:
            model_expr_str = self.extract_expression(model_answer)
            correct_expr_str = self.extract_expression(correct_answer)
            
            if not model_expr_str or not correct_expr_str:
                return False
            
            model_expr_str = model_expr_str.replace('∫', '').replace('dx', '').strip()
            correct_expr_str = correct_expr_str.replace('∫', '').replace('dx', '').strip()
            
            model_expr_str = model_expr_str.replace('π', 'pi').replace('∞', 'oo')
            correct_expr_str = correct_expr_str.replace('π', 'pi').replace('∞', 'oo')
            
            model_expr_str = re.sub(r'\\[a-zA-Z]+', '', model_expr_str)
            correct_expr_str = re.sub(r'\\[a-zA-Z]+', '', correct_expr_str)
            
            try:
                model_expr = sympify(model_expr_str)
                correct_expr = sympify(correct_expr_str)
                
                diff = simplify(model_expr - correct_expr)
                return diff == 0
            except:
                return False
        except:
            return False
    
    def get_reward(self, model_answer, correct_answer):
        if self.is_correct(model_answer, correct_answer):
            return 1.0
        else:
            keywords = ['解', '积分', '=', '+ C', 'C']
            reward = -0.5
            for kw in keywords:
                if kw in model_answer:
                    reward += 0.1
            return max(reward, -1.0)

def train_reward_model(args):
    print(f"加载模型: {args.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print("加载数据集...")
    train_dataset = RewardDataset(args.train_file, tokenizer, args.max_length)
    val_dataset = RewardDataset(args.val_file, tokenizer, args.max_length)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=True,
        optim="adamw_torch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("开始训练奖励模型...")
    trainer.train()
    
    print("保存奖励模型...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"奖励模型已保存到: {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="训练奖励模型")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="基础模型")
    parser.add_argument("--train_file", type=str, default="data/preference_train.json", help="训练数据")
    parser.add_argument("--val_file", type=str, default="data/preference_val.json", help="验证数据")
    parser.add_argument("--output_dir", type=str, default="./reward_model", help="输出目录")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志步数")
    parser.add_argument("--save_steps", type=int, default=100, help="保存步数")
    parser.add_argument("--eval_steps", type=int, default=100, help="评估步数")
    
    args = parser.parse_args()
    
    train_reward_model(args)

if __name__ == "__main__":
    main()