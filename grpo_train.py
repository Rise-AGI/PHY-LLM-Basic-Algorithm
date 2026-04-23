"""
GRPO（Group Relative Policy Optimization）训练脚本 —— Magnus 容器内运行版本
对标 DeepSeek-R1-Zero 路线：纯 RL，无需奖励模型，用物理结构奖励直接优化。

数据格式（train.json / JSONL）:
    messages 格式（推荐）:
        {"messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}]}
    instruction 格式（兼容）:
        {"instruction": "物理题目", "output": "参考答案（可选，不用于训练）"}

指标输出（写入 $MAGNUS_METRICS_DIR）:
    rl.reward.mean / rl.reward.std / rl.kl_divergence
    rl.entropy / rl.clip_ratio / rl.advantage.mean
    train.lr / train.grad_norm

Checkpoint 输出结构:
    output_dir/
    ├── checkpoint-{step}/   每隔 save_steps 保存
    ├── final/               训练完成后的最终模型
    └── training_log.json    完整训练日志

待接入 RL 组接口（forward_step / reward_fn）:
    当 RL 组提供 forward_step 时，替换 `_generate_responses()` 函数体。
    当 RL 组提供 reward_fn 时，替换 `physics_reward()` 函数体。
"""

import argparse
import json
import math
import os
import re
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


# ── 命令行参数 ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GRPO RL training for Phy-LLM")
    p.add_argument("--model_path",          type=str,   required=True)
    p.add_argument("--train_data",          type=str,   required=True)
    p.add_argument("--output_dir",          type=str,   default="/tmp/grpo_output")
    p.add_argument("--epochs",              type=int,   default=1)
    p.add_argument("--batch_size",          type=int,   default=2)
    p.add_argument("--learning_rate",       type=float, default=5e-7)
    p.add_argument("--max_prompt_length",   type=int,   default=512)
    p.add_argument("--max_response_length", type=int,   default=512)
    p.add_argument("--group_size",          type=int,   default=8,   help="G: 每条 prompt 采样几个 completion")
    p.add_argument("--kl_coef",             type=float, default=0.04, help="KL 惩罚系数 β")
    p.add_argument("--clip_range",          type=float, default=0.2,  help="PPO clip ε")
    p.add_argument("--save_steps",          type=int,   default=100)
    p.add_argument("--logging_steps",       type=int,   default=10)
    return p.parse_args()


# ── 数据加载 ───────────────────────────────────────────────────────────

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
            # 只保留 system + user，丢弃 assistant（RL 自己生成）
            prompts.append([m for m in row["messages"] if m["role"] in ("system", "user")])
        elif "instruction" in row:
            prompts.append([
                {"role": "system", "content": "你是物理推理助手，请展示详细的推导过程。"},
                {"role": "user",   "content": row["instruction"]},
            ])

    assert prompts, f"数据集为空：{path}"
    print(f"[数据] 从 {path} 加载 {len(prompts)} 条 prompts")
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
        ids.append(F.pad(b["input_ids"],      (pad, 0), value=pad_id))
        attn.append(F.pad(b["attention_mask"], (pad, 0), value=0))
    return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(attn)}


# ── 奖励函数（待 RL 组接入 reward_fn 后替换此函数体）──────────────────

def physics_reward(response: str) -> float:
    """
    物理推理结构化奖励（0.0–1.0）。
    待 RL 组提供 reward_fn 时，替换此函数体。
    """
    score = 0.0
    # 有推理链
    if "<think>" in response and "</think>" in response:
        score += 0.3
    # 有方程式（字母 = 数字）
    if re.search(r"[A-Za-z]\s*=\s*[-+]?\d", response):
        score += 0.2
    # 有物理单位
    if re.search(r"\b(m/s|m/s\^2|kg|N|J|Pa|K|eV|Hz|W|C|V|T|mol|rad)\b", response):
        score += 0.2
    # 有数值结果
    if re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", response):
        score += 0.3
    return score


# ── 模型工具函数 ────────────────────────────────────────────────────────

def sequence_log_prob(model, inp: torch.Tensor, attn: torch.Tensor, resp: torch.Tensor) -> torch.Tensor:
    """计算 response 序列的 sum log-prob（[B] 张量）。"""
    prompt_len = inp.size(1)
    full_ids   = torch.cat([inp, resp], dim=1)
    full_attn  = torch.cat([attn, torch.ones_like(resp)], dim=1)
    out        = model(full_ids, attention_mask=full_attn)
    # logits[t] 预测 token t+1；所以 response 部分对应 [prompt_len-1 : -1]
    logits = out.logits[:, prompt_len - 1 : -1, :]
    lp     = F.log_softmax(logits, dim=-1)
    return lp.gather(2, resp.unsqueeze(2)).squeeze(2).sum(1)


def _generate_responses(model, tok, inp, attn, max_new_tokens, pad_id):
    """
    采样一批 responses。
    待 RL 组提供 forward_step 时，替换此函数体。
    """
    with torch.no_grad():
        out = model.generate(
            inp, attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.8, top_p=0.95,
            pad_token_id=pad_id,
        )
    return out[:, inp.size(1):]


# ── Magnus 指标上报 ────────────────────────────────────────────────────

def emit(name: str, kind: str, value: float, step: int, mdir: str):
    if not mdir:
        return
    v = -999999.0 if (math.isnan(value) or math.isinf(value)) else value
    rec = {
        "name": name, "kind": kind, "value": v,
        "time_unix_ms": int(time.time() * 1000),
        "step": step, "step_domain": "optimizer",
    }
    if v == -999999.0:
        rec["labels"] = {"nan": "true"}
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "rank0.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def emit_step(step: int, mdir: str, **kwargs):
    for name, value in kwargs.items():
        metric_name = name.replace("_", ".", 1) if "_" not in name[2:] else name
        emit(metric_name, "gauge", float(value), step, mdir)


# ── Checkpoint ─────────────────────────────────────────────────────────

def save_ckpt(policy, tokenizer, output_dir: str, step: int, meta: dict):
    path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(path, exist_ok=True)
    m = policy.module if hasattr(policy, "module") else policy
    m.save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(os.path.join(path, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  [Checkpoint] 已保存 step={step} → {path}")


# ── 主训练循环 ─────────────────────────────────────────────────────────

def train():
    args  = parse_args()
    mdir  = os.environ.get("MAGNUS_METRICS_DIR", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[环境] device={device}, n_gpu={n_gpu}")
    print(f"[GRPO] G={args.group_size}, β={args.kl_coef}, ε={args.clip_range}, lr={args.learning_rate}")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    # Policy model（训练）
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    # Reference model（冻结，用于 KL 约束）
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    if n_gpu > 1:
        policy = torch.nn.DataParallel(policy)

    total_params = sum(p.numel() for p in policy.parameters()) / 1e9
    print(f"[模型] {total_params:.2f}B 参数，全量 GRPO 微调")

    prompts = load_prompts(args.train_data)
    dataset = PromptDataset(prompts, tok, args.max_prompt_length)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate_prompts(b, pad_id),
    )

    total_steps = len(loader) * args.epochs
    optimizer   = AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.05)),
        num_training_steps=total_steps,
    )

    global_step = 0
    train_log   = []
    policy.train()

    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            inp  = batch["input_ids"].to(device)       # [B, L_prompt]
            attn = batch["attention_mask"].to(device)
            m    = policy.module if hasattr(policy, "module") else policy

            # ── 阶段 1：Rollout G 条 completions（无梯度）──────────────
            resp_list, old_lp_list, rew_list = [], [], []

            with torch.no_grad():
                for _ in range(args.group_size):
                    resp    = _generate_responses(m, tok, inp, attn, args.max_response_length, pad_id)
                    texts   = tok.batch_decode(resp, skip_special_tokens=True)
                    rewards = torch.tensor(
                        [physics_reward(t) for t in texts],
                        dtype=torch.float32, device=device,
                    )
                    old_lp  = sequence_log_prob(m, inp, attn, resp).detach()
                    resp_list.append(resp)
                    old_lp_list.append(old_lp)
                    rew_list.append(rewards)

            rewards_g = torch.stack(rew_list)    # [G, B]
            old_lp_g  = torch.stack(old_lp_list) # [G, B]

            # ── 阶段 2：组内归一化优势（GRPO 核心）─────────────────────
            r_mean = rewards_g.mean(0, keepdim=True)                # [1, B]
            r_std  = rewards_g.std(0, keepdim=True).clamp(min=1e-8) # [1, B]
            adv_g  = (rewards_g - r_mean) / r_std                   # [G, B]

            # ── 阶段 3：策略梯度更新（对 G 条 response 累加梯度）────────
            optimizer.zero_grad()
            acc_kl, acc_clip = 0.0, 0.0

            for g in range(args.group_size):
                resp   = resp_list[g]
                new_lp = sequence_log_prob(m, inp, attn, resp)           # 有梯度
                ref_lp = sequence_log_prob(ref_model, inp, attn, resp).detach()

                ratio  = (new_lp - old_lp_g[g]).exp()  # [B]  ≈1 at init
                kl     = new_lp - ref_lp                # [B]  近似 per-sequence KL
                adv    = adv_g[g]                       # [B]

                surr1   = ratio * adv
                surr2   = ratio.clamp(1 - args.clip_range, 1 + args.clip_range) * adv
                pg_loss = -torch.min(surr1, surr2).mean()
                kl_loss = kl.mean()

                # 除以 G 相当于对 G 条 loss 取均值
                loss_g = (pg_loss + args.kl_coef * kl_loss) / args.group_size
                loss_g.backward()

                acc_kl   += kl_loss.item()
                acc_clip += ((ratio - 1).abs() > args.clip_range).float().mean().item()

            grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            # ── 统计指标 ─────────────────────────────────────────────
            mean_rew   = rewards_g.mean().item()
            std_rew    = rewards_g.std().item()
            mean_kl    = acc_kl   / args.group_size
            clip_ratio = acc_clip / args.group_size
            mean_adv   = adv_g.mean().item()
            lr_now     = scheduler.get_last_lr()[0]

            if global_step % args.logging_steps == 0:
                print(
                    f"  Ep{epoch} Step{global_step} | "
                    f"reward={mean_rew:.4f}±{std_rew:.3f} | "
                    f"kl={mean_kl:.4f} | clip={clip_ratio:.3f} | "
                    f"gnorm={grad_norm:.3f} | lr={lr_now:.2e}"
                )
                train_log.append({
                    "step":              global_step,
                    "epoch":             epoch,
                    "rl.reward.mean":    round(mean_rew,          6),
                    "rl.reward.std":     round(std_rew,           6),
                    "rl.kl_divergence":  round(mean_kl,           6),
                    "rl.clip_ratio":     round(clip_ratio,        6),
                    "rl.advantage.mean": round(mean_adv,          6),
                    "train.lr":          round(lr_now,            8),
                    "train.grad_norm":   round(grad_norm.item(),  6),
                })

            # ── 上报 Magnus 指标 ──────────────────────────────────────
            emit("rl.reward.mean",    "gauge", mean_rew,             global_step, mdir)
            emit("rl.reward.std",     "gauge", std_rew,              global_step, mdir)
            emit("rl.kl_divergence",  "gauge", mean_kl,              global_step, mdir)
            emit("rl.entropy",        "gauge", 0.0,                  global_step, mdir)
            emit("rl.clip_ratio",     "gauge", clip_ratio,           global_step, mdir)
            emit("rl.advantage.mean", "gauge", mean_adv,             global_step, mdir)
            emit("train.lr",          "gauge", lr_now,               global_step, mdir)
            emit("train.grad_norm",   "gauge", grad_norm.item(),     global_step, mdir)

            if global_step % args.save_steps == 0:
                save_ckpt(policy, tok, args.output_dir, global_step, {
                    "step": global_step, "epoch": epoch, "reward_mean": round(mean_rew, 6),
                })

        print(f"[Epoch {epoch}/{args.epochs}] done, global_step={global_step}")

    # ── 保存最终模型 ──────────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    m_final = policy.module if hasattr(policy, "module") else policy
    m_final.save_pretrained(final_path)
    tok.save_pretrained(final_path)
    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(train_log, f, ensure_ascii=False, indent=2)
    print(f"[完成] 最终模型 → {final_path}")

    last   = train_log[-1] if train_log else {}
    result = {
        "status":             "success",
        "final_reward_mean":  last.get("rl.reward.mean"),
        "final_kl":           last.get("rl.kl_divergence"),
        "total_steps":        global_step,
        "output_dir":         args.output_dir,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    train()
