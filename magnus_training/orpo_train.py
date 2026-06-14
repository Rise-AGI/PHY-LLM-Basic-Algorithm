#!/usr/bin/env python3
"""
ORPO (Odds Ratio Preference Optimization) 训练脚本

无需参考模型，联合优化 SFT loss + 偏好损失。
参考: Hong et al. "ORPO: Monolithic Preference Optimization without Reference Model" (2024)

用法:
    python orpo_train.py --model_path /path/to/sft_model --train_data /path/to/preference_pairs.json --output_dir /tmp/orpo_out
"""

import argparse
import json
import math
import os
import time
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


def _write_metric(name: str, value: float, step: int, step_domain: str = "optimizer"):
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
    p = argparse.ArgumentParser(description="ORPO training")
    p.add_argument("--model_path",          type=str,   required=True)
    p.add_argument("--train_data",          type=str,   required=True)
    p.add_argument("--output_dir",          type=str,   default="/tmp/orpo_output")
    p.add_argument("--epochs",              type=int,   default=1)
    p.add_argument("--batch_size",          type=int,   default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate",       type=float, default=5e-7)
    p.add_argument("--warmup_ratio",        type=float, default=0.05)
    p.add_argument("--weight_decay",        type=float, default=0.01)
    p.add_argument("--max_prompt_length",   type=int,   default=512)
    p.add_argument("--max_response_length", type=int,   default=512)
    p.add_argument("--orpo_lambda",         type=float, default=0.1,
                    help="ORPO preference loss weight (λ). Higher=stronger preference signal")
    p.add_argument("--save_steps",          type=int,   default=100)
    p.add_argument("--logging_steps",       type=int,   default=10)
    p.add_argument("--num_workers",         type=int,   default=2)
    p.add_argument("--retry_seed",          type=int,   default=0)
    p.add_argument("--cpu_offload",         action="store_true")
    p.add_argument("--use_8bit_adam",       action="store_true")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


# ── 数据加载（与 DPO 共享偏好对格式）──────────────────────────────────────────

def load_preference_pairs(path: str) -> list:
    """加载偏好对数据。格式: {prompt/messages, chosen, rejected}"""
    if path.endswith(".parquet"):
        import pandas as pd
        rows = pd.read_parquet(path).to_dict(orient="records")
    else:
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()
        rows = json.loads(raw) if raw.startswith("[") else [
            json.loads(line) for line in raw.splitlines() if line.strip()
        ]

    pairs = []
    for row in rows:
        if "messages" in row:
            prompt_msgs = [m for m in row["messages"] if m["role"] in ("system", "user")]
        elif "instruction" in row:
            prompt_msgs = [
                {"role": "system", "content": "你是一位数学解题专家。"},
                {"role": "user",   "content": row["instruction"]},
            ]
        elif "question" in row:
            prompt_msgs = [
                {"role": "system", "content": "你是一位数学解题专家。"},
                {"role": "user",   "content": row["question"]},
            ]
        elif "prompt" in row:
            prompt_msgs = [{"role": "user", "content": row["prompt"]}]
        else:
            continue
        pairs.append({
            "prompt": prompt_msgs,
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        })
    assert pairs, f"数据集为空: {path}"
    log(f"[数据] 从 {path} 加载 {len(pairs)} 条偏好对")
    return pairs


class PreferenceDataset(Dataset):
    def __init__(self, pairs, tok, max_prompt_len, max_resp_len):
        self.pairs = pairs
        self.tok = tok
        self.max_prompt_len = max_prompt_len
        self.max_resp_len = max_resp_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pair = self.pairs[i]
        prompt_text = self.tok.apply_chat_template(
            pair["prompt"], tokenize=False, add_generation_prompt=True
        )
        prompt_enc = self.tok(prompt_text, return_tensors="pt",
                              max_length=self.max_prompt_len, truncation=True)
        chosen_enc = self.tok(pair["chosen"], return_tensors="pt",
                              max_length=self.max_resp_len, truncation=True)
        rejected_enc = self.tok(pair["rejected"], return_tensors="pt",
                                max_length=self.max_resp_len, truncation=True)
        return {
            "prompt_ids": prompt_enc["input_ids"][0],
            "prompt_mask": prompt_enc["attention_mask"][0],
            "chosen_ids": chosen_enc["input_ids"][0],
            "chosen_mask": chosen_enc["attention_mask"][0],
            "rejected_ids": rejected_enc["input_ids"][0],
            "rejected_mask": rejected_enc["attention_mask"][0],
        }


def collate_preference_batch(batch, pad_id):
    def _pad(seqs, pad_val):
        max_len = max(s.size(0) for s in seqs)
        return torch.stack([F.pad(s, (0, max_len - s.size(0)), value=pad_val) for s in seqs])

    return {
        "prompt_ids":   _pad([b["prompt_ids"]   for b in batch], pad_id),
        "prompt_mask":  _pad([b["prompt_mask"]  for b in batch], 0),
        "chosen_ids":   _pad([b["chosen_ids"]   for b in batch], pad_id),
        "chosen_mask":  _pad([b["chosen_mask"]  for b in batch], 0),
        "rejected_ids": _pad([b["rejected_ids"]  for b in batch], pad_id),
        "rejected_mask": _pad([b["rejected_mask"] for b in batch], 0),
    }


# ── ORPO 损失 ─────────────────────────────────────────────────────────────────

def sequence_log_prob(model, prompt_ids, prompt_mask, resp_ids, resp_mask):
    """计算 response 序列的 sum log-prob [B]."""
    full_ids = torch.cat([prompt_ids, resp_ids], dim=1)
    full_mask = torch.cat([prompt_mask, resp_mask], dim=1)
    out = model(full_ids, attention_mask=full_mask)
    logits = out.logits[:, prompt_ids.size(1) - 1 : -1, :]
    lp = F.log_softmax(logits.float(), dim=-1)
    return lp.gather(2, resp_ids.unsqueeze(2)).squeeze(2).sum(1)


def sequence_sft_loss(model, prompt_ids, prompt_mask, resp_ids, resp_mask):
    """SFT cross-entropy loss on response tokens only. Returns per-token avg [B]."""
    full_ids = torch.cat([prompt_ids, resp_ids], dim=1)
    full_mask = torch.cat([prompt_mask, resp_mask], dim=1)
    out = model(full_ids, attention_mask=full_mask)
    logits = out.logits[:, prompt_ids.size(1) - 1 : -1, :]
    # CE loss per token, averaged over response length
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        resp_ids.reshape(-1),
        reduction="none",
    )
    ce_per_seq = ce.reshape(resp_ids.shape).sum(1) / resp_ids.size(1).clamp(min=1)
    return ce_per_seq


def orpo_loss_fn(policy_chosen_lp, policy_rejected_lp,
                 sft_loss_chosen, orpo_lambda):
    """
    ORPO loss = L_SFT + λ * L_OR

    L_OR = -log σ(log_odds(chosen) - log_odds(rejected))
    where log_odds ≈ avg_log_prob for numerical stability
    """
    # Odds ratio preference loss
    log_odds_ratio = policy_chosen_lp - policy_rejected_lp
    l_or = -F.logsigmoid(log_odds_ratio).mean()

    # Combined loss
    l_sft = sft_loss_chosen.mean()
    loss = l_sft + orpo_lambda * l_or

    with torch.no_grad():
        chosen_acc = (log_odds_ratio > 0).float().mean().item()

    return loss, l_sft.item(), l_or.item(), chosen_acc


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_ckpt(policy, tokenizer, output_dir, step, meta):
    path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(path, exist_ok=True)
    m = policy.module if hasattr(policy, "module") else policy
    m.save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(os.path.join(path, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"  [Checkpoint] step={step} -> {path}")


# ── 主训练循环 ────────────────────────────────────────────────────────────────

def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    local_rank = 0

    if n_gpu > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=600))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    os.makedirs(args.output_dir, exist_ok=True)

    log(f"[环境] device={device}, n_gpu={n_gpu}, rank={local_rank}")
    log(f"[ORPO] lambda={args.orpo_lambda}, lr={args.learning_rate}")

    # ── Tokenizer ──
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    # ── Policy model（无 ref model！ORPO 不需要）──
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    if hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False

    # ── FSDP ──
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

            _wrap_policy = _partial(transformer_auto_wrap_policy, transformer_layer_cls={_layer_cls}) if _layer_cls else None
            _cpu_offload = CPUOffload(offload_params=True) if args.cpu_offload else None
            policy = FSDP(
                policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=_wrap_policy,
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16,
                ),
                device_id=local_rank, limit_all_gathers=True,
                forward_prefetch=False, backward_prefetch=BackwardPrefetch.BACKWARD_POST,
                cpu_offload=_cpu_offload,
            )
            log("[FSDP] FULL_SHARD 完成")
        except ImportError:
            log("[FSDP] 不可用，回退 DataParallel")
            policy = torch.nn.DataParallel(policy)
        policy = policy.to(device)
    else:
        policy = policy.to(device)

    total_params = sum(p.numel() for p in policy.parameters()) / 1e9
    log(f"[模型] {total_params:.2f}B params | {n_gpu} GPU | 无 ref model (ORPO)")

    # ── 数据 ──
    pairs = load_preference_pairs(args.train_data)
    dataset = PreferenceDataset(pairs, tok, args.max_prompt_length, args.max_response_length)
    sampler = DistributedSampler(dataset) if n_gpu > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_preference_batch(b, pad_id),
    )

    # ── 优化器 ──
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            log("[优化器] 8-bit AdamW")
        except ImportError:
            optimizer = AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = len(loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    log(f"[优化器] total_steps={total_steps}, warmup={warmup_steps}")

    # ── 训练循环 ──
    global_step = 0
    train_log = []
    start_epoch = 1

    if args.resume_from_checkpoint:
        meta_path = os.path.join(args.resume_from_checkpoint, "checkpoint_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                global_step = json.load(f).get("step", 0)
        log(f"[恢复] global_step={global_step}")

    log(f"[训练] 等效全局 batch={args.batch_size * args.gradient_accumulation_steps * max(n_gpu,1)} | 开始...")
    log("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch + args.retry_seed)

        for step, batch in enumerate(loader, 1):
            t0 = time.time()
            p_ids = batch["prompt_ids"].to(device)
            p_mask = batch["prompt_mask"].to(device)
            c_ids = batch["chosen_ids"].to(device)
            c_mask = batch["chosen_mask"].to(device)
            r_ids = batch["rejected_ids"].to(device)
            r_mask = batch["rejected_mask"].to(device)

            # ── Log-probs + SFT loss ──
            policy_chosen_lp  = sequence_log_prob(policy, p_ids, p_mask, c_ids, c_mask)
            policy_rejected_lp = sequence_log_prob(policy, p_ids, p_mask, r_ids, r_mask)
            sft_loss_chosen = sequence_sft_loss(policy, p_ids, p_mask, c_ids, c_mask)

            # ── ORPO loss ──
            loss, l_sft, l_or, chosen_acc = orpo_loss_fn(
                policy_chosen_lp, policy_rejected_lp,
                sft_loss_chosen, args.orpo_lambda,
            )
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step % args.gradient_accumulation_steps == 0) or (step == len(loader)):
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                step_time = time.time() - t0
                combined_loss = loss.item() * args.gradient_accumulation_steps
                if local_rank == 0:
                    if global_step % args.logging_steps == 0 or global_step == 1:
                        log(
                            f"  Ep{epoch} Step{global_step}/{total_steps} | "
                            f"loss={combined_loss:.4f} | sft={l_sft:.4f} | "
                            f"or={l_or:.4f} | acc={chosen_acc:.3f} | "
                            f"gnorm={grad_norm.item():.2f} | time={step_time:.1f}s"
                        )

                    _write_metric("orpo.loss",        combined_loss,     global_step)
                    _write_metric("orpo.sft_loss",    l_sft,             global_step)
                    _write_metric("orpo.or_loss",     l_or,              global_step)
                    _write_metric("orpo.chosen_acc",  chosen_acc,        global_step)
                    _write_metric("train.lr",         scheduler.get_last_lr()[0], global_step)
                    _write_metric("train.grad_norm",  grad_norm.item(),  global_step)
                    _write_metric("train.step_time_s", step_time,        global_step)

                    train_log.append({
                        "step": global_step, "epoch": epoch,
                        "orpo.loss": round(combined_loss, 6),
                        "orpo.sft_loss": round(l_sft, 6),
                        "orpo.or_loss": round(l_or, 6),
                        "orpo.chosen_acc": round(chosen_acc, 6),
                    })

                if global_step % args.save_steps == 0:
                    save_ckpt(policy, tok, args.output_dir, global_step, {
                        "step": global_step, "epoch": epoch,
                        "orpo_loss": round(combined_loss, 6),
                    })

        log(f"[Epoch {epoch}/{args.epochs}] 完成, global_step={global_step}")

    # ── 最终保存 ──
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
        "final_orpo_loss": last.get("orpo.loss"),
        "final_sft_loss": last.get("orpo.sft_loss"),
        "final_chosen_acc": last.get("orpo.chosen_acc"),
        "total_steps": global_step,
        "output_dir": args.output_dir,
    }
    log(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    train()
