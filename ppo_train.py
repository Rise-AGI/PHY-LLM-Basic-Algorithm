#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) 训练脚本 — 可选奖励模型 + 规则奖励 + 外部奖励 API

标准 RLHF PPO 流程：策略模型 + 冻结参考模型 + 奖励模型（可选）+ 规则奖励。

用法:
    # 使用奖励模型
    python ppo_train.py --model_path /path/to/sft_model --reward_model_path /path/to/rm --train_data /path/to/prompts.json --output_dir /tmp/ppo_out

    # 仅规则奖励（无奖励模型）
    python ppo_train.py --model_path /path/to/sft_model --train_data /path/to/prompts.json --output_dir /tmp/ppo_out

    # 外部奖励 API
    python ppo_train.py --model_path /path/to/sft_model --train_data /path/to/prompts.json --reward_api_url http://reward-server:8080/score
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
    AutoModelForSequenceClassification,
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
    p = argparse.ArgumentParser(description="PPO training with optional reward model")
    p.add_argument("--model_path",          type=str,   required=True, help="SFT 基础模型路径")
    p.add_argument("--reward_model_path",   type=str,   default=None, help="奖励模型路径（可选，留空仅用规则奖励）")
    p.add_argument("--train_data",          type=str,   required=True)
    p.add_argument("--output_dir",          type=str,   default="/tmp/ppo_output")
    p.add_argument("--epochs",              type=int,   default=1)
    p.add_argument("--batch_size",          type=int,   default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate",       type=float, default=5e-7)
    p.add_argument("--warmup_ratio",        type=float, default=0.05)
    p.add_argument("--weight_decay",        type=float, default=0.01)
    p.add_argument("--max_prompt_length",   type=int,   default=512)
    p.add_argument("--max_response_length", type=int,   default=512)
    p.add_argument("--kl_coef",             type=float, default=0.1,  help="KL 惩罚系数 beta")
    p.add_argument("--clip_range",          type=float, default=0.2,  help="PPO clip epsilon")
    p.add_argument("--reward_ratio",        type=float, default=0.3,  help="learned reward 权重（剩余给 rule reward）")
    p.add_argument("--value_coef",          type=float, default=0.5,  help="Value loss 系数")
    p.add_argument("--save_steps",          type=int,   default=100)
    p.add_argument("--logging_steps",       type=int,   default=10)
    p.add_argument("--num_workers",         type=int,   default=2)
    p.add_argument("--retry_seed",          type=int,   default=0)
    p.add_argument("--cpu_offload",         action="store_true", help="CPU offload")
    p.add_argument("--use_8bit_adam",       action="store_true")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--temperature",         type=float, default=0.7)
    p.add_argument("--top_p",               type=float, default=0.9)
    p.add_argument("--reward_api_url",      type=str,   default=None, help="外部奖励 API 地址（可选）")
    return p.parse_args()


# ── 数据 ──────────────────────────────────────────────────────────────────────

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
        elif "question" in row:
            prompts.append([
                {"role": "system", "content": "你是一位数学解题专家。请逐步推理并解答以下问题。"},
                {"role": "user",   "content": row["question"]},
            ])
        elif "prompt" in row:
            prompts.append([{"role": "user", "content": row["prompt"]}])
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


# ── 奖励 ──────────────────────────────────────────────────────────────────────

def rule_reward(response: str) -> float:
    """规则奖励 (0.0-1.0)。可替换为自定义打分函数。"""
    score = 0.0
    if any(tag in response for tag in ["<think>", "解答", "推导", "步骤"]):
        score += 0.2
    if re.search(r"[A-Za-z]\s*=\s*[-+]?\d", response):
        score += 0.2
    if re.search(r"\b(m/s|m/s\^2|kg|N|J|Pa|K|eV|Hz|W|C|V|T|mol|rad|cm|mm|km|g|s)\b", response):
        score += 0.2
    if re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", response):
        score += 0.2
    if re.search(r"\$.*?\$|\\frac|\\sqrt|\\int|\\sum", response):
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
    prompt_len = inp.size(1)
    full_ids  = torch.cat([inp, resp], dim=1)
    full_attn = torch.cat([attn, torch.ones_like(resp)], dim=1)
    out       = model(full_ids, attention_mask=full_attn)
    logits    = out.logits[:, prompt_len - 1 : -1, :]
    lp        = F.log_softmax(logits.float(), dim=-1)
    return lp.gather(2, resp.unsqueeze(2)).squeeze(2).sum(1)


@torch.no_grad()
def generate_responses(model, tok, inp, attn, max_new_tokens, pad_id, temperature, top_p):
    out = model.generate(
        inp, attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature, top_p=top_p,
        pad_token_id=pad_id,
    )
    return out[:, inp.size(1):]


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
    # 抑制 transformers/tqdm 进度条，避免 4 rank × 851 chunks 日志洪流
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
    except Exception:
        pass

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

    def _mem_report(tag: str):
        """每个 rank 报告本卡显存使用"""
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated(device) / 1024**3
            rsvd = torch.cuda.memory_reserved(device) / 1024**3
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            log(f"[显存] {tag} | used={used:.1f}GiB reserved={rsvd:.1f}GiB total={total:.1f}GiB")

    # ── 奖励来源描述 ────────────────────────────────────────────
    reward_parts = []
    if args.reward_api_url:
        reward_parts.append(f"外部API({args.reward_api_url})")
    if args.reward_model_path and os.path.exists(args.reward_model_path):
        reward_parts.append(f"本地RM(ratio={args.reward_ratio})")
    if not reward_parts:
        reward_parts.append("纯规则奖励")

    log(f"[环境] device={device}, n_gpu={n_gpu}, rank={local_rank}")
    log(f"[PPO] beta={args.kl_coef}, eps={args.clip_range}, lr={args.learning_rate}, "
        f"reward_ratio={args.reward_ratio}, temp={args.temperature}, "
        f"reward={' + '.join(reward_parts)}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    # ── Policy model（先加载到 CPU，再根据 FSDP/DataParallel 策略放 GPU）───
    log("[模型] 加载 policy (CPU)...")
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    if hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False
    total_params = sum(p.numel() for p in policy.parameters()) / 1e9
    _mem_report("policy loaded (CPU)")

    # ── FSDP / DataParallel（在加载 ref_model 之前确定，避免显存竞争）─────
    fsdp_ok = False
    if n_gpu > 1:
        _fsdp_cpu_offload = args.cpu_offload
        if _fsdp_cpu_offload and args.use_8bit_adam:
            log("[FSDP] 8bit Adam + CPU offload(offload_params=True) 不兼容，禁用 CPU offload")
            _fsdp_cpu_offload = False

        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
            )
            from torch.distributed.fsdp import (
                ShardingStrategy, MixedPrecision, BackwardPrefetch, CPUOffload,
            )
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from functools import partial as _partial

            _layer_cls = None
            for _name, _mod in policy.named_modules():
                _cn = type(_mod).__name__
                if 'Decoder' in _cn and 'Layer' in _cn:
                    _layer_cls = type(_mod)
                    log(f"[FSDP] transformer layer: {_cn}")
                    break
            if _layer_cls is None:
                log("[FSDP] 未检测到 DecoderLayer，使用默认 wrap policy")

            _wrap_policy = _partial(transformer_auto_wrap_policy, transformer_layer_cls={_layer_cls}) if _layer_cls else None
            _cpu_offload = CPUOffload(offload_params=True) if _fsdp_cpu_offload else None
            policy = FSDP(
                policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=_wrap_policy,
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                ),
                device_id=local_rank,
                use_orig_params=True,
                forward_prefetch=False,
                backward_prefetch=BackwardPrefetch.BACKWARD_POST,
                cpu_offload=_cpu_offload,
            )
            fsdp_ok = True
            log(f"[FSDP] FULL_SHARD 完成 (params={total_params:.2f}B, offload={_fsdp_cpu_offload})")
            _mem_report("FSDP wrap done")
        except Exception as _fsdp_err:
            log(f"[FSDP] 不可用: {type(_fsdp_err).__name__}: {_fsdp_err}")
            log("[FSDP] 回退 DataParallel — 27B 模型需 ref_model 保持 CPU 以防 OOM")
            policy = torch.nn.DataParallel(policy)
            policy = policy.to(device)
            _mem_report("DataParallel .to(device) done")
    else:
        policy = policy.to(device)

    # ── Reference model（冻结）────────────────────────────────────────────
    # FSDP: ref_model 用 device_map="auto" 跨卡分片 (~13.5GB/卡)
    # DataParallel: ref_model 必须放 CPU，否则与复制的 policy 抢显存导致 OOM
    if fsdp_ok:
        log("[模型] 加载 ref_model (device_map='auto')...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map="auto",
        )
        _mem_report("ref_model loaded (device_map=auto)")
    else:
        log("[模型] 加载 ref_model (CPU, DataParallel 回退)...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        _mem_report("ref_model loaded (CPU)")

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── Reward model（可选）───────────────────────────────────────────────────
    reward_model = None
    use_learned_reward = False
    if args.reward_model_path and os.path.exists(args.reward_model_path):
        log(f"[奖励模型] 加载: {args.reward_model_path}")
        try:
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                args.reward_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)
            reward_model.eval()
            for p in reward_model.parameters():
                p.requires_grad_(False)
            use_learned_reward = True
            log(f"[奖励模型] 已就绪，combined reward = learned*{args.reward_ratio} + rule*{1-args.reward_ratio}")
        except Exception as e:
            log(f"[奖励模型] 加载失败: {e}，回退到纯规则奖励")
    else:
        if not args.reward_model_path:
            log("[奖励模型] 未提供，使用纯规则奖励")
        else:
            log(f"[奖励模型] 路径不存在: {args.reward_model_path}，使用纯规则奖励")

    log(f"[模型] 训练就绪 | {n_gpu} GPU | FSDP={fsdp_ok} | learned_reward={use_learned_reward} | ref_cpu={not fsdp_ok}")

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
    else:
        optimizer = AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = len(loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    log(f"[优化器] total_steps={total_steps}, warmup={warmup_steps}")

    # ── 训练循环 ────────────────────────────────────────────────────────────
    global_step = 0
    train_log   = []
    start_epoch = 1

    if args.resume_from_checkpoint:
        log(f"[恢复] from {args.resume_from_checkpoint}")
        meta_path = os.path.join(args.resume_from_checkpoint, "checkpoint_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                global_step = json.load(f).get("step", 0)

    eff_batch = args.batch_size * args.gradient_accumulation_steps * max(n_gpu, 1)
    log(f"[训练] 等效全局 batch={eff_batch} | 开始训练...")
    log("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch + args.retry_seed)

        for step, batch in enumerate(loader, 1):
            t0 = time.time()
            inp  = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            B    = inp.size(0)

            # ── 1. 生成 responses ──────────────────────────────────────────
            with torch.no_grad():
                resp = generate_responses(
                    policy, tok, inp, attn,
                    args.max_response_length, pad_id, args.temperature, args.top_p,
                )
                texts = tok.batch_decode(resp, skip_special_tokens=True)

                # 计算奖励（优先级：外部 API > 本地奖励模型 > 规则函数）
                rule_rewards = torch.tensor(
                    [rule_reward(t) for t in texts], dtype=torch.float32, device=device,
                )

                if args.reward_api_url:
                    prompt_texts = tok.batch_decode(inp, skip_special_tokens=True)
                    api_rewards = torch.tensor(
                        [call_reward_api(args.reward_api_url, p, t) for p, t in zip(prompt_texts, texts)],
                        dtype=torch.float32, device=device,
                    )
                    # 外部 API 可与规则奖励融合
                    if use_learned_reward and reward_model is not None:
                        rm_inputs = tok(
                            texts, return_tensors="pt", max_length=512,
                            padding=True, truncation=True,
                        ).to(device)
                        learned_rewards = reward_model(**rm_inputs).logits.squeeze(-1)
                        rewards = (args.reward_ratio * learned_rewards +
                                   (1 - args.reward_ratio) * api_rewards)
                    else:
                        rewards = api_rewards
                elif use_learned_reward and reward_model is not None:
                    rm_inputs = tok(
                        texts, return_tensors="pt", max_length=512,
                        padding=True, truncation=True,
                    ).to(device)
                    learned_rewards = reward_model(**rm_inputs).logits.squeeze(-1)
                    rewards = args.reward_ratio * learned_rewards + (1 - args.reward_ratio) * rule_rewards
                else:
                    rewards = rule_rewards

                # Old log-probs + ref log-probs
                old_lp = sequence_log_prob(policy, inp, attn, resp).detach()
                if fsdp_ok:
                    ref_lp = sequence_log_prob(ref_model, inp, attn, resp).detach()
                else:
                    # DataParallel 回退: ref_model 在 CPU，输入需移过去
                    ref_lp = sequence_log_prob(
                        ref_model, inp.cpu(), attn.cpu(), resp.cpu()
                    ).detach().to(device)

            # ── 2. PPO loss ─────────────────────────────────────────────────
            new_lp = sequence_log_prob(policy, inp, attn, resp)
            ratio  = (new_lp - old_lp).exp()
            kl     = new_lp - ref_lp

            advantages = rewards - args.kl_coef * kl.detach()

            surr1   = ratio * advantages
            surr2   = ratio.clamp(1 - args.clip_range, 1 + args.clip_range) * advantages
            pg_loss = -torch.min(surr1, surr2).mean()
            kl_loss = kl.mean()
            loss    = pg_loss + args.kl_coef * kl_loss

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step % args.gradient_accumulation_steps == 0) or (step == len(loader)):
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ── 指标 ────────────────────────────────────────────────────
                step_time = time.time() - t0
                mean_reward = rewards.mean().item()
                clip_ratio  = ((ratio - 1).abs() > args.clip_range).float().mean().item()

                if local_rank == 0:
                    if global_step % args.logging_steps == 0 or global_step == 1:
                        log(
                            f"  Ep{epoch} Step{global_step}/{total_steps} | "
                            f"reward={mean_reward:.4f} | kl={kl_loss.item():.4f} | "
                            f"pg_loss={pg_loss.item():.4f} | clip={clip_ratio:.3f} | "
                            f"gnorm={grad_norm.item():.2f} | time={step_time:.1f}s"
                        )

                    _write_metric("ppo.reward.mean",    mean_reward,        global_step)
                    _write_metric("ppo.kl_divergence",  kl_loss.item(),     global_step)
                    _write_metric("ppo.pg_loss",        pg_loss.item(),     global_step)
                    _write_metric("ppo.clip_ratio",     clip_ratio,         global_step)
                    _write_metric("train.lr",           scheduler.get_last_lr()[0], global_step)
                    _write_metric("train.grad_norm",    grad_norm.item(),   global_step)
                    _write_metric("train.step_time_s",  step_time,          global_step)

                    train_log.append({
                        "step": global_step, "epoch": epoch,
                        "ppo.reward.mean": round(mean_reward, 6),
                        "ppo.kl_divergence": round(kl_loss.item(), 6),
                        "ppo.pg_loss": round(pg_loss.item(), 6),
                        "ppo.clip_ratio": round(clip_ratio, 6),
                    })

                if global_step % args.save_steps == 0:
                    save_ckpt(policy, tok, args.output_dir, global_step, {
                        "step": global_step, "epoch": epoch,
                        "reward_mean": round(mean_reward, 6),
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
        "final_reward_mean": last.get("ppo.reward.mean"),
        "final_kl":          last.get("ppo.kl_divergence"),
        "total_steps":       global_step,
        "output_dir":        args.output_dir,
    }
    log(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    train()
