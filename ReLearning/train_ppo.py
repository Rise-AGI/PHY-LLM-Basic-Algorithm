import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from sympy import sympify, simplify, symbols, integrate
from sympy.abc import x
import re
import numpy as np

class PromptDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
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
        
        encoding = self.tokenizer(
            item["prompt"],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'correct_answer': item.get("correct_answer", ""),
        }

class SymbolicRewardFunction:
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
    
    def normalize_latex(self, text):
        text = text.replace('∫', '').replace('dx', '').strip()
        text = text.replace('π', 'pi').replace('∞', 'oo')
        text = text.replace('\\frac', '/')
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = text.replace('{', '(').replace('}', ')')
        return text
    
    def is_correct(self, model_answer, correct_answer):
        try:
            model_expr_str = self.extract_expression(model_answer)
            correct_expr_str = self.extract_expression(correct_answer)
            
            if not model_expr_str or not correct_expr_str:
                return False
            
            model_expr_str = self.normalize_latex(model_expr_str)
            correct_expr_str = self.normalize_latex(correct_expr_str)
            
            try:
                model_expr = sympify(model_expr_str)
                correct_expr = sympify(correct_expr_str)
                
                diff = simplify(model_expr - correct_expr)
                return diff == 0
            except:
                return False
        except:
            return False
    
    def __call__(self, model_answer, correct_answer):
        if self.is_correct(model_answer, correct_answer):
            return 1.0
        else:
            reward = -0.5
            keywords = ['解', '积分', '=', '+ C', 'C', '步骤']
            for kw in keywords:
                if kw in model_answer:
                    reward += 0.15
            if '∫' in model_answer:
                reward += 0.1
            return max(min(reward, 1.0), -1.0)

class PPOTrainer:
    def __init__(
        self,
        policy_model,
        ref_model,
        reward_model,
        tokenizer,
        symbolic_reward,
        kl_coef=0.1,
        clip_range=0.2,
        gamma=1.0,
        gae_lambda=0.95,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.symbolic_reward = symbolic_reward
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.gamma = gamma
        self.gae_lambda = gae_lambda
    
    def generate_response(self, input_ids, attention_mask, max_new_tokens=128):
        self.policy_model.eval()
        with torch.no_grad():
            outputs = self.policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return outputs
    
    def compute_log_probs(self, model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        
        mask = attention_mask[:, 1:]
        log_probs_masked = selected_log_probs * mask
        
        return log_probs_masked.sum(dim=1)
    
    def compute_kl_divergence(self, policy_log_probs, ref_log_probs):
        return policy_log_probs - ref_log_probs
    
    def compute_rewards(self, responses, correct_answers, input_ids, device):
        rewards = []
        
        self.reward_model.eval()
        with torch.no_grad():
            for i, (response, correct) in enumerate(zip(responses, correct_answers)):
                response_text = self.tokenizer.decode(response, skip_special_tokens=True)
                symbolic_reward = self.symbolic_reward(response_text, correct)
                
                reward_input = self.tokenizer(
                    response_text,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).to(device)
                
                learned_reward = self.reward_model(
                    input_ids=reward_input['input_ids'],
                    attention_mask=reward_input['attention_mask']
                ).logits.item()
                
                combined_reward = 0.3 * learned_reward + 0.7 * symbolic_reward
                rewards.append(combined_reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=device)
    
    def train_step(self, batch, optimizer, scheduler, device):
        self.policy_model.train()
        self.ref_model.eval()
        self.reward_model.eval()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        correct_answers = batch['correct_answer']
        
        responses = self.generate_response(input_ids, attention_mask, max_new_tokens=128)
        
        policy_log_probs = self.compute_log_probs(self.policy_model, responses, 
                                                   torch.ones_like(responses, device=device))
        
        with torch.no_grad():
            ref_log_probs = self.compute_log_probs(self.ref_model, responses,
                                                    torch.ones_like(responses, device=device))
        
        rewards = self.compute_rewards(responses, correct_answers, input_ids, device)
        
        kl_div = self.compute_kl_divergence(policy_log_probs, ref_log_probs)
        
        advantages = rewards - kl_div * self.kl_coef
        
        with torch.no_grad():
            old_log_probs = policy_log_probs.clone()
        
        new_log_probs = self.compute_log_probs(self.policy_model, responses,
                                                torch.ones_like(responses, device=device))
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        kl_loss = kl_div.mean()
        
        loss = policy_loss + self.kl_coef * kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'mean_reward': rewards.mean().item(),
        }

def main():
    parser = argparse.ArgumentParser(description="PPO强化学习训练")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="基础模型")
    parser.add_argument("--reward_model_path", type=str, default="./reward_model", help="奖励模型路径")
    parser.add_argument("--sft_model_path", type=str, default=None, help="SFT模型路径(可选)")
    parser.add_argument("--prompt_file", type=str, default="data/prompts.json", help="Prompt数据")
    parser.add_argument("--output_dir", type=str, default="./ppo_model", help="输出目录")
    parser.add_argument("--max_length", type=int, default=256, help="最大输入长度")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--kl_coef", type=float, default=0.1, help="KL散度系数")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO裁剪范围")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="最大生成token数")
    parser.add_argument("--save_steps", type=int, default=50, help="保存步数")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    print(f"加载策略模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_path = args.sft_model_path if args.sft_model_path else args.model_name
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    
    print("加载参考模型...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    print(f"加载奖励模型: {args.reward_model_path}")
    try:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(device)
        reward_model.eval()
        for param in reward_model.parameters():
            param.requires_grad = False
        use_learned_reward = True
    except:
        print("未找到奖励模型,仅使用符号奖励")
        reward_model = None
        use_learned_reward = False
    
    symbolic_reward = SymbolicRewardFunction()
    
    print("加载数据集...")
    dataset = PromptDataset(args.prompt_file, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"数据集大小: {len(dataset)}")
    
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    num_training_steps = len(dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
    )
    
    ppo_trainer = PPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        symbolic_reward=symbolic_reward,
        kl_coef=args.kl_coef,
        clip_range=args.clip_range,
    )
    
    print("开始PPO训练...")
    global_step = 0
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        epoch_reward = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            metrics = ppo_trainer.train_step(batch, optimizer, scheduler, device)
            
            epoch_loss += metrics['loss']
            epoch_reward += metrics['mean_reward']
            global_step += 1
            
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'reward': f"{metrics['mean_reward']:.4f}",
                'kl': f"{metrics['kl_loss']:.4f}"
            })
            
            if global_step % args.save_steps == 0:
                save_path = f"{args.output_dir}/checkpoint-{global_step}"
                policy_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"\n模型已保存到: {save_path}")
        
        avg_loss = epoch_loss / len(dataloader)
        avg_reward = epoch_reward / len(dataloader)
        print(f"\nEpoch {epoch+1}: 平均Loss={avg_loss:.4f}, 平均Reward={avg_reward:.4f}")
    
    print(f"保存最终模型到: {args.output_dir}")
    policy_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("PPO训练完成!")

if __name__ == "__main__":
    main()