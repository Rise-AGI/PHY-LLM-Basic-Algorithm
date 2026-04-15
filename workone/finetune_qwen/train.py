import os
import sys
import json
import torch
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import argparse


def check_cuda():
    print("=" * 50)
    print("CUDA Environment Check")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        print("Please check your CUDA installation.")
        sys.exit(1)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    print("=" * 50)


def load_data(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_prompt(example, tokenizer):
    prompt = f"### 问题:\n{example['instruction']}\n\n### 回答:\n{example['output']}"
    return prompt


def preprocess_function(examples, tokenizer, max_length=512):
    prompts = []
    for inst, out in zip(examples['instruction'], examples['output']):
        prompt = f"### 问题:\n{inst}\n\n### 回答:\n{out}"
        prompts.append(prompt)
    
    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_config.yaml', help='Path to config file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    args = parser.parse_args()
    
    check_cuda()
    
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_name = config['model']['name']
    data_path = config['data']['train_path']
    output_dir = config['training']['output_dir']
    max_length = config['data'].get('max_length', 512)
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if config['training'].get('bf16', True) else torch.float16,
    }
    
    use_flash_attn = config['model'].get('flash_attn', False)
    if use_flash_attn:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")
        except Exception as e:
            print(f"Flash Attention not available: {e}")
    
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    if config['lora']['enabled']:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['lora']['r'],
            lora_alpha=config['lora']['lora_alpha'],
            lora_dropout=config['lora']['lora_dropout'],
            target_modules=config['lora']['target_modules'],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    print("Loading data...")
    raw_data = load_data(data_path)
    dataset = Dataset.from_list(raw_data)
    
    def preprocess(examples):
        return preprocess_function(examples, tokenizer, max_length)
    
    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        bf16=config['training']['bf16'],
        fp16=False,
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        optim=config['training']['optim'],
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=config['training'].get('num_workers', 4),
        dataloader_pin_memory=True,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training completed!")


if __name__ == "__main__":
    main()