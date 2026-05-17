#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) 训练脚本 — 纯 RL，无需奖励模型

对标 DeepSeek-R1-Zero：用规则奖励 + 组内归一化优势直接优化策略模型。
适合数学/物理推理场景（答案可自动验证）。

用法:
    # 内置规则奖励
    python grpo_train.py --model_path /path/to/sft_model --train_data /path/to/prompts.json --output_dir /tmp/grpo_out

    # 外部奖励 API
    python grpo_train.py --model_path /path/to/sft_model --train_data /path/to/prompts.json --reward_api_url http://reward-server:8080/score

    # 从 checkpoint 恢复
    python grpo_train.py --model_path /path/to/sft_model --train_data /path/to/prompts.json --output_dir /tmp/grpo_out --resume_from_checkpoint /tmp/grpo_out/checkpoint-100
"""

import argparse
import json
import math
import os
import re
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


MAGNUS_METRICS_DIR = os.environ.get("MAGNUS_METRICS_DIR", "/magnus/workspace/metrics")


def _write_metric(name: str, value: float, step: int, step_domain: str = "optimizer", unit: str = "gauge"):
    if not math.isfinite(value):
        return
    rec = {
        "name": name, "kind": "gauge", "value": value,
        "time_unix_ms": int(time.time() * 1000),
        "step": step, "step_domain": step_domain,
    }
    job_id = os.environ.get("MAGNUS_JOB_ID", "")
    if job_id:
        rec.setdefault("labels", {})["job"] = job_id[:8]
    os.makedirs(MAGNUS_METRICS_DIR, exist_ok=True)
    with open(os.path.join(MAGNUS_METRICS_DIR, "rank0.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ── 命令行参数 ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GRPO training")
    p.add_argument("--model_path",          type=str,   required=True)
    p.add_argument("--train_data",          type=str,   required=True)
    p.add_argument("--output_dir",          type=str,   default="/tmp/grpo_output")
    p.add_argument("--epochs",              type=int,   default=1)
    p.add_argument("--batch_size",          type=int,   default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate",       type=float, default=5e-7)
    p.add_argument("--warmup_ratio",        type=float, default=0.05)
    p.add_argument("--weight_decay",        type=float, default=0.01)
    p.add_argument("--max_prompt_length",   type=int,   default=512)
    p.add_argument("--max_response_length", type=int,   default=512)
    p.add_argument("--group_size",          type=int,   default=8,   help="G: 每条 prompt 采样几个 completion")
    p.add_argument("--kl_coef",             type=float, default=0.04, help="KL 惩罚系数 beta")
    p.add_argument("--clip_range",          type=float, default=0.2,  help="PPO clip epsilon")
    p.add_argument("--save_steps",          type=int,   default=100)
    p.add_argument("--logging_steps",       type=int,   default=10)
    p.add_argument("--num_workers",         type=int,   default=2)
    p.add_argument("--retry_seed",          type=int,   default=0)
    p.add_argument("--cpu_offload",         action="store_true", help="CPU offload optimizer states")
    p.add_argument("--use_8bit_adam",       action="store_true", help="Use 8-bit AdamW")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--temperature",         type=float, default=0.8,  help="采样温度")
    p.add_argument("--top_p",               type=float, default=0.95, help="Nucleus sampling")
    p.add_argument("--reward_api_url",      type=str,   default=None, help="外部奖励 API 地址（可选，留空=内置规则奖励）")
    return p.parse_args()


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_prompts(path: str) -> list:
    if path.endswith(".parquet"):
        import pandas as pd
        rows = pd.read_parquet(path).to_dict(orient="records")
    else:
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()
        rows = json.loads(raw) if raw.startswith("[") else [
            json.loads(line) for line in raw.splitlines() if line.strip()
        ]

    prompts = []
    for row in rows:
        if "messages" in row:
            prompts.append([m for m in row["messages"] if m["role"] in ("system", "user")])
        elif "instruction" in row:
            prompts.append([
                {"role": "system", "content": "你是一位数学解题专家。请逐步推理并解答以下问题。"},
                {"role": "user",   "content": row["instruction"]},
            ])
        elif "prompt" in row:
            prompts.append([
                {"role": "user", "content": row["prompt"]},
            ])
    assert prompts, f"数据集为空: {path}"
    log(f"[数据] 从 {path} 加载 {len(prompts)} 条 prompts")
    return prompts


class PromptDataset(Dataset):
    def __init__(self, prompts, tok, max_len):
        self.prompts = prompts
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        text = self.tok.apply_chat_template(
            self.prompts[i], tokenize=False, add_generation_prompt=True
        )
        enc = self.tok(text, return_tensors="pt", max_length=self.max_len, truncation=True)
        return {"input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0]}


def collate_prompts(batch, pad_id):
    max_len = max(b["input_ids"].size(0) for b in batch)
    ids, attn = [], []
    for b in batch:
        n = b["input_ids"].size(0)
        pad = max_len - n
        ids.append(F.pad(b["input_ids"],      (0, pad), value=pad_id))
        attn.append(F.pad(b["attention_mask"], (0, pad), value=0))
    return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(attn)}


# ── 奖励函数 ──────────────────────────────────────────────────────────────────

def math_physics_reward(response: str) -> float:
    """数学/物理推理结构化奖励 (0.0-1.0)。可替换为自定义 reward_fn。"""
    score = 0.0
    if any(tag in response for tag in ["<think>", "解答", "推导", "步骤"]):
        score += 0.2
    if re.search(r"[A-Za-z]\s*=\s*[-+]?\d", response):
        score += 0.2
    if re.search(r"\b(m/s|m/s\^2|kg|N|J|Pa|K|eV|Hz|W|C|V|T|mol|rad|cm|mm|km|g|s)\b", response):
        score += 0.2
    if re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", response):
        score += 0.2
    if re.search(r"\$.*?\$|\\frac|\\sqrt|\\int|\\sum|\\prod", response):
        score += 0.2
    return score


def call_reward_api(api_url: str, prompt: str, response: str, timeout: float = 30.0) -> float:
    """调用外部奖励 API 打分。约定: POST JSON {prompt, response} → {score: 0-1}"""
    import requests
    try:
        resp = requests.post(
            api_url,
            json={"prompt": prompt, "response": response},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("score", 0.0))
    except Exception as e:
        log(f"[奖励API] 调用失败: {e}，回退到 0.0")
        return 0.0


# ── 模型工具 ──────────────────────────────────────────────────────────────────

def sequence_log_prob(model, inp, attn, resp):
    """计算 response 序列的 sum log-prob [B]."""
    prompt_len = inp.size(1)
    full_ids  = torch.cat([inp, resp], dim=1)
    full_attn = torch.cat([attn, torch.ones_like(resp)], dim=1)
    out       = model(full_ids, attention_mask=full_attn)
    logits    = out.logits[:, prompt_len - 1 : -1, :]
    lp        = F.log_softmax(logits.float(), dim=-1)
    return lp.gather(2, resp.unsqueeze(2)).squeeze(2).sum(1)


@torch.no_grad()
def generate_responses(model, tok, inp, attn, max_new_tokens, pad_id, temperature, top_p):
    """采样 responses [B, L_resp]."""
    out = model.generate(
        inp, attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature, top_p=top_p,
        pad_token_id=pad_id,
    )
    return out[:, inp.size(1):]


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_ckpt(policy, tokenizer, output_dir, step, meta):
    path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(path, exist_ok=True)
    if hasattr(policy, "module"):
        policy.module.save_pretrained(path)
    else:
        policy.save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(os.path.join(path, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"  [Checkpoint] step={step} -> {path}")


# ── 主训练循环 ────────────────────────────────────────────────────────────────

def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count() if torch.cuda.is_available() else 0
    local_rank = 0

    if n_gpu > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=600))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    os.makedirs(args.output_dir, exist_ok=True)

    reward_source = f"外部API({args.reward_api_url})" if args.reward_api_url else "内置规则函数"
    log(f"[环境] device={device}, n_gpu={n_gpu}, rank={local_rank}")
    log(f"[GRPO] G={args.group_size}, beta={args.kl_coef}, eps={args.clip_range}, "
        f"lr={args.learning_rate}, temp={args.temperature}, reward={reward_source}")

    # ── Tokenizer ──
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    # ── Policy model（训练）────────────────────────────────────────────────
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    if hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False

    # ── Reference model（冻结，用于 KL 约束）────────────────────────────────
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── FSDP (multi-GPU) ──────────────────────────────────────────────────
    if n_gpu > 1:
        try:
            from torch.distributed.fsdp import (
                FSDP, ShardingStrategy, MixedPrecision, BackwardPrefetch, CPUOffload,
            )
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from functools import partial as _partial

            _layer_cls = None
            for _name, _mod in policy.named_modules():
                _cn = type(_mod).__name__
                if 'Decoder' in _cn and 'Layer' in _cn:
                    _layer_cls = type(_mod)
                    log(f"[FSDP] 检测到 transformer layer: {_cn}")
                    break

            if _layer_cls:
                _policy = _partial(transformer_auto_wrap_policy, transformer_layer_cls={_layer_cls})
            else:
                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
                _policy = _partial(size_based_auto_wrap_policy, min_num_params=1e8)

            _cpu_offload = CPUOffload(offload_params=True) if args.cpu_offload else None
            policy = FSDP(
                policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=_policy,
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                ),
                device_id=local_rank,
                limit_all_gathers=True,
                forward_prefetch=False,
                backward_prefetch=BackwardPrefetch.BACKWARD_POST,
                cpu_offload=_cpu_offload,
            )
            log(f"[FSDP] FULL_SHARD 完成")
        except ImportError:
            log("[FSDP] torch.distributed.fsdp 不可用，回退到 DataParallel")
            policy = torch.nn.DataParallel(policy)
        policy = policy.to(device)
    else:
        policy = policy.to(device)

    total_params = sum(p.numel() for p in policy.parameters()) / 1e9
    log(f"[模型] {total_params:.2f}B params | {n_gpu} GPU")

    # ── 数据 ────────────────────────────────────────────────────────────────
    prompts = load_prompts(args.train_data)
    dataset = PromptDataset(prompts, tok, args.max_prompt_length)
    sampler = DistributedSampler(dataset) if n_gpu > 1 else None
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_prompts(b, pad_id),
    )

    # ── 优化器 ──────────────────────────────────────────────────────────────
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            log("[优化器] 8-bit AdamW")
        except ImportError:
            optimizer = AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            log("[优化器] AdamW (8-bit 不可用)")
    else:
        optimizer = AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = len(loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    log(f"[优化器] steps_per_epoch={len(loader)}, total_steps={total_steps}, warmup={warmup_steps}")

    # ── 训练循环 ────────────────────────────────────────────────────────────
    global_step = 0
    train_log   = []
    start_epoch = 1

    if args.resume_from_checkpoint:
        log(f"[恢复] 从 checkpoint 加载: {args.resume_from_checkpoint}")
        ckpt_step = 0
        meta_path = os.path.join(args.resume_from_checkpoint, "checkpoint_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                ckpt_step = json.load(f).get("step", 0)
        global_step = ckpt_step
        log(f"[恢复] global_step={global_step}")

    eff_batch = args.batch_size * args.gradient_accumulation_steps * max(n_gpu, 1)
    log(f"[训练] 等效全局 batch={eff_batch} | 开始训练...")
    log("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch + args.retry_seed)

        for step, batch in enumerate(loader, 1):
            step_start = time.time()
            inp  = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            # ── 阶段 1: Rollout ─────────────────────────────────────────
            resp_list, old_lp_list, rew_list = [], [], []

            with torch.no_grad():
                for g in range(args.group_size):
                    resp = generate_responses(
                        policy, tok, inp, attn,
                        args.max_response_length, pad_id, args.temperature, args.top_p,
                    )
                    texts = tok.batch_decode(resp, skip_special_tokens=True)

                    # 选择奖励来源：外部 API 优先，否则用内置规则函数
                    if args.reward_api_url:
                        prompt_texts = tok.batch_decode(inp, skip_special_tokens=True)
                        rewards = torch.tensor(
                            [call_reward_api(args.reward_api_url, p, t) for p, t in zip(prompt_texts, texts)],
                            dtype=torch.float32, device=device,
                        )
                    else:
                        rewards = torch.tensor(
                            [math_physics_reward(t) for t in texts],
                            dtype=torch.float32, device=device,
                        )

                    old_lp = sequence_log_prob(policy, inp, attn, resp).detach()
                    resp_list.append(resp)
                    old_lp_list.append(old_lp)
                    rew_list.append(rewards)

            rewards_g = torch.stack(rew_list)    # [G, B]
            old_lp_g  = torch.stack(old_lp_list) # [G, B]

            # ── 阶段 2: 组内归一化优势 ───────────────────────────────────
            r_mean = rewards_g.mean(0, keepdim=True)
            r_std  = rewards_g.std(0, keepdim=True).clamp(min=1e-8)
            adv_g  = (rewards_g - r_mean) / r_std

            # ── 阶段 3: 策略梯度 ─────────────────────────────────────────
            optimizer.zero_grad()
            acc_kl, acc_clip = 0.0, 0.0

            for g in range(args.group_size):
                resp   = resp_list[g]
                new_lp = sequence_log_prob(policy, inp, attn, resp)
                ref_lp = sequence_log_prob(ref_model, inp, attn, resp).detach()

                ratio  = (new_lp - old_lp_g[g]).exp()
                kl     = new_lp - ref_lp
                adv    = adv_g[g]

                surr1   = ratio * adv
                surr2   = ratio.clamp(1 - args.clip_range, 1 + args.clip_range) * adv
                pg_loss = -torch.min(surr1, surr2).mean()
                kl_loss = kl.mean()
                loss_g  = (pg_loss + args.kl_coef * kl_loss) / args.group_size
                loss_g.backward()

                acc_kl   += kl_loss.item()
                acc_clip += ((ratio - 1).abs() > args.clip_range).float().mean().item()

            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            # ── 指标 ────────────────────────────────────────────────────
            step_time = time.time() - step_start
            mean_rew  = rewards_g.mean().item()
            std_rew   = rewards_g.std().item()
            mean_kl   = acc_kl   / args.group_size
            clip_ratio = acc_clip / args.group_size
            mean_adv  = adv_g.mean().item()
            lr_now    = scheduler.get_last_lr()[0]

            if local_rank == 0:
                if global_step % args.logging_steps == 0 or global_step == 1:
                    log(
                        f"  Ep{epoch} Step{global_step}/{total_steps} | "
                        f"reward={mean_rew:.4f}+-{std_rew:.3f} | "
                        f"kl={mean_kl:.4f} | clip={clip_ratio:.3f} | "
                        f"gnorm={grad_norm.item():.2f} | time={step_time:.1f}s | "
                        f"lr={lr_now:.2e}"
                    )

                _write_metric("grpo.reward.mean",    mean_rew,             global_step)
                _write_metric("grpo.reward.std",     std_rew,              global_step)
                _write_metric("grpo.kl_divergence",  mean_kl,              global_step)
                _write_metric("grpo.clip_ratio",     clip_ratio,           global_step)
                _write_metric("grpo.advantage.mean", mean_adv,             global_step)
                _write_metric("train.lr",            lr_now,               global_step)
                _write_metric("train.grad_norm",     grad_norm.item(),     global_step)
                _write_metric("train.step_time_s",   step_time,            global_step)

                train_log.append({
                    "step": global_step, "epoch": epoch,
                    "grpo.reward.mean": round(mean_rew, 6),
                    "grpo.reward.std":  round(std_rew,  6),
                    "grpo.kl_divergence": round(mean_kl, 6),
                    "grpo.clip_ratio":    round(clip_ratio, 6),
                    "grpo.advantage.mean": round(mean_adv, 6),
                    "train.lr": round(lr_now, 8),
                    "train.grad_norm": round(grad_norm.item(), 6),
                })

            if global_step % args.save_steps == 0:
                save_ckpt(policy, tok, args.output_dir, global_step, {
                    "step": global_step, "epoch": epoch,
                    "reward_mean": round(mean_rew, 6),
                })

        log(f"[Epoch {epoch}/{args.epochs}] 完成, global_step={global_step}")

    # ── 最终保存 ────────────────────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    m_final = policy.module if hasattr(policy, "module") else policy
    m_final.save_pretrained(final_path)
    tok.save_pretrained(final_path)
    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(train_log, f, ensure_ascii=False, indent=2)
    log(f"[完成] 最终模型 -> {final_path}")

    last = train_log[-1] if train_log else {}
    result = {
        "status": "success",
        "final_reward_mean": last.get("grpo.reward.mean"),
        "final_kl":          last.get("grpo.kl_divergence"),
        "total_steps":       global_step,
        "output_dir":        args.output_dir,
    }
    log(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    train()
