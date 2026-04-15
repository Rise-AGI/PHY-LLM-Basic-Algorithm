import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = None
class IntegralDataset(Dataset):
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
        
        prompt = f"<|im_start|>system\n{item['instruction']}<|im_end|>\n<|im_start|>user\n{item['input']}<|im_end|>\n<|im_start|>assistant\n"
        response = f"{item['output']}<|im_end|>"
        
        full_text = prompt + response
        
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            # padding='max_length',
            padding=False,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        # prompt_encoding = self.tokenizer(
        #     prompt,
        #     max_length=self.max_length,
        #     padding=False,
        #     truncation=True,
        #     return_tensors='pt'
        # )
        prompt_encoding = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        prompt_ids = prompt_encoding['input_ids'].squeeze()
        
        prompt_length = len(prompt_ids)
        if prompt_length > self.max_length:
            prompt_length = self.max_length
        labels[:prompt_length] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def load_model_and_tokenizer(model_name, use_4bit=False):
    print(f"加载模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": "auto"
    }
    
    if use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    return model, tokenizer

def setup_lora(model, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    print("配置LoRA...")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

def train(args):
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.use_4bit)
    
    model = setup_lora(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    print("加载数据集...")
    train_dataset = IntegralDataset(args.train_file, tokenizer, args.max_length)
    val_dataset = IntegralDataset(args.val_file, tokenizer, args.max_length)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        # evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not args.use_4bit,
        optim="adamw_torch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        )
    )
    
    print("开始训练...")
    trainer.train()
    
    print("保存模型...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"模型已保存到: {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="微调Qwen模型用于积分计算")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="模型名称或路径")
    parser.add_argument("--train_file", type=str, default="train.json", help="训练数据文件")
    parser.add_argument("--val_file", type=str, default="val.json", help="验证数据文件")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志步数")
    parser.add_argument("--save_steps", type=int, default=100, help="保存步数")
    parser.add_argument("--eval_steps", type=int, default=100, help="评估步数")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use_4bit", action="store_true", help="使用4bit量化")
    
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()