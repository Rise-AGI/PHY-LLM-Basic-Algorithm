import os
import yaml
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_and_tokenizer(config):
    model_config = config['model']
    
    bnb_config = None
    if model_config.get('use_4bit', False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_compute_dtype=getattr(torch, model_config.get('bnb_4bit_compute_dtype', 'float16')),
            bnb_4bit_use_double_quant=model_config.get('use_nested_quant', False),
        )
    
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    if model_config.get('use_flash_attention', False):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        **model_kwargs
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['name'],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if model_config.get('use_4bit', False):
        model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def create_lora_config(config):
    lora_config = config['lora']
    return LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )

def format_data(sample):
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    output = sample.get('output', '')
    
    if input_text:
        prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    else:
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='finetune_config.yaml')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    print("Creating LoRA config...")
    peft_config = create_lora_config(config)
    
    data_config = config['data']
    
    print("Loading datasets...")
    train_dataset = load_dataset('json', data_files=data_config['train_file'], split='train')
    eval_dataset = load_dataset('json', data_files=data_config['eval_file'], split='train')
    
    training_config = config['training']
    
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        warmup_ratio=training_config['warmup_ratio'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        logging_steps=training_config['logging_steps'],
        save_steps=training_config['save_steps'],
        eval_steps=training_config['eval_steps'],
        save_total_limit=training_config['save_total_limit'],
        fp16=training_config.get('fp16', False),
        bf16=training_config.get('bf16', False),
        gradient_checkpointing=training_config.get('gradient_checkpointing', False),
        optim=training_config.get('optim', 'adamw_torch'),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
    )
    
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        formatting_func=format_data,
        max_seq_length=data_config['max_seq_length'],
        tokenizer=tokenizer,
        args=training_args,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_config['output_dir'])
    
    print(f"Training complete. Model saved to {training_config['output_dir']}")

if __name__ == '__main__':
    main()