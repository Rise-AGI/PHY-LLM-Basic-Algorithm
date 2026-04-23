"""
DPO（Direct Preference Optimization）训练脚本 —— Magnus 容器内运行版本
不需要奖励模型，直接从偏好数据优化 policy。

数据格式（train.json / JSONL）:
    偏好对格式（推荐）:
        {"chosen": [...messages...], "rejected": [...messages...]}
    或 Alpaca 偏好格式：
        {"instruction": "...", "chosen": "好回答", "rejected": "差回答"}

DPO 损失：
    L = -E[log σ(β * (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))]

指标输出（写入 $MAGNUS_METRICS_DIR）:
    train.loss / val.loss / train.lr / train.grad_norm
    rl.reward.mean（chosen reward 均值）/ rl.kl_divergence（chosen KL 均值）
"""

import argparse
import json
import math
import os
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
    p = argparse.ArgumentParser(description="DPO training for Phy-LLM")
    p.add_argument("--model_path",    type=str,   required=True)
    p.add_argument("--train_data",    type=str,   required=True)
    p.add_argument("--test_data",     type=str,   default=None)
    p.add_argument("--output_dir",    type=str,   default="/tmp/dpo_output")
    p.add_argument("--epochs",        type=int,   default=1)
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--learning_rate", type=float, default=5e-7)
    p.add_argument("--max_length",    type=int,   default=1024)
    p.add_argument("--beta",          type=float, default=0.1,  help="DPO temperature β")
    p.add_argument("--save_steps",    type=int,   default=100)
    p.add_argument("--logging_steps", type=int,   default=10)
    return p.parse_args()


# ── 数据加载 ───────────────────────────────────────────────────────────

def load_pairs(path: str) -> list:
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
        if "chosen" in row and "rejected" in row:
            if isinstance(row["chosen"], list):
                # messages 格式: {"chosen": [...], "rejected": [...]}
                chosen_msgs   = row["chosen"]
                rejected_msgs = row["rejected"]
            else:
                # Alpaca 格式: {"instruction": ..., "chosen": str, "rejected": str}
                system_msg = {"role": "system", "content": "你是物理推理助手，请展示详细的推导过程。"}
                user_msg   = {"role": "user",   "content": row.get("instruction", row.get("prompt", ""))}
                chosen_msgs   = [system_msg, user_msg, {"role": "assistant", "content": row["chosen"]}]
                rejected_msgs = [system_msg, user_msg, {"role": "assistant", "content": row["rejected"]}]
            pairs.append({"chosen": chosen_msgs, "rejected": rejected_msgs})

    assert pairs, f"数据集为空：{path}"
    print(f"[数据] 从 {path} 加载 {len(pairs)} 条偏好对")
    return pairs


class DPODataset(Dataset):
    def __init__(self, pairs, tok, max_len):
        self.pairs   = pairs
        self.tok     = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pair = self.pairs[i]
        chosen_text   = self.tok.apply_chat_template(pair["chosen"],   tokenize=False)
        rejected_text = self.tok.apply_chat_template(pair["rejected"], tokenize=False)

        c_enc = self.tok(chosen_text,   return_tensors="pt", max_length=self.max_len, truncation=True)
        r_enc = self.tok(rejected_text, return_tensors="pt", max_length=self.max_len, truncation=True)

        return {
            "chosen_ids":     c_enc["input_ids"][0],
            "chosen_attn":    c_enc["attention_mask"][0],
            "rejected_ids":   r_enc["input_ids"][0],
            "rejected_attn":  r_enc["attention_mask"][0],
        }


def collate_dpo(batch, pad_id):
    def pad_seq(tensors):
        max_len = max(t.size(0) for t in tensors)
        return torch.stack([F.pad(t, (0, max_len - t.size(0)), value=pad_id) for t in tensors])

    def pad_attn(tensors):
        max_len = max(t.size(0) for t in tensors)
        return torch.stack([F.pad(t, (0, max_len - t.size(0)), value=0) for t in tensors])

    return {
        "chosen_ids":    pad_seq([b["chosen_ids"]   for b in batch]),
        "chosen_attn":   pad_attn([b["chosen_attn"] for b in batch]),
        "rejected_ids":  pad_seq([b["rejected_ids"]   for b in batch]),
        "rejected_attn": pad_attn([b["rejected_attn"] for b in batch]),
    }


# ── 核心函数 ───────────────────────────────────────────────────────────

def sequence_log_prob(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """计算完整序列（含 prompt）的 average log-prob per token。"""
    out    = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]                                    # [B, L-1, V]
    labels = input_ids[:, 1:]                                         # [B, L-1]
    lp     = F.log_softmax(logits, dim=-1)
    token_lp = lp.gather(2, labels.unsqueeze(2)).squeeze(2)          # [B, L-1]
    # 只计算非 pad token
    mask    = attention_mask[:, 1:].float()
    return (token_lp * mask).sum(1) / mask.sum(1).clamp(min=1)       # [B]


def dpo_loss(policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta):
    """标准 DPO 损失（Bradley-Terry + 对数 sigmoid）。"""
    pi_delta  = policy_chosen_lp - policy_rejected_lp
    ref_delta = ref_chosen_lp    - ref_rejected_lp
    logits    = beta * (pi_delta - ref_delta)
    loss      = -F.logsigmoid(logits).mean()
    reward_chosen   = beta * (policy_chosen_lp   - ref_chosen_lp).detach().mean()
    reward_rejected = beta * (policy_rejected_lp - ref_rejected_lp).detach().mean()
    return loss, reward_chosen.item(), reward_rejected.item()


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


# ── Checkpoint ─────────────────────────────────────────────────────────

def save_ckpt(policy, tokenizer, output_dir, step, meta):
    path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(path, exist_ok=True)
    m = policy.module if hasattr(policy, "module") else policy
    m.save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(os.path.join(path, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  [Checkpoint] 已保存 step={step} → {path}")


@torch.no_grad()
def evaluate(policy, ref_model, loader, device, beta, mdir=""):
    policy.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        c_ids = batch["chosen_ids"].to(device);    c_attn = batch["chosen_attn"].to(device)
        r_ids = batch["rejected_ids"].to(device);  r_attn = batch["rejected_attn"].to(device)
        m = policy.module if hasattr(policy, "module") else policy
        pol_c = sequence_log_prob(m, c_ids, c_attn)
        pol_r = sequence_log_prob(m, r_ids, r_attn)
        ref_c = sequence_log_prob(ref_model, c_ids, c_attn)
        ref_r = sequence_log_prob(ref_model, r_ids, r_attn)
        loss, _, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
        total_loss += loss.item(); n += 1
    policy.train()
    return total_loss / max(n, 1)


# ── 主训练循环 ─────────────────────────────────────────────────────────

def train():
    args  = parse_args()
    mdir  = os.environ.get("MAGNUS_METRICS_DIR", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[环境] device={device}, n_gpu={n_gpu}")
    print(f"[DPO] β={args.beta}, lr={args.learning_rate}, max_len={args.max_length}")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    if n_gpu > 1:
        policy = torch.nn.DataParallel(policy)

    total_params = sum(p.numel() for p in policy.parameters()) / 1e9
    print(f"[模型] {total_params:.2f}B 参数，DPO 全量微调")

    pairs = load_pairs(args.train_data)
    dataset = DPODataset(pairs, tok, args.max_length)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate_dpo(b, pad_id),
    )

    eval_loader = None
    if args.test_data and os.path.exists(args.test_data):
        eval_pairs   = load_pairs(args.test_data)
        eval_dataset = DPODataset(eval_pairs, tok, args.max_length)
        eval_loader  = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
            collate_fn=lambda b: collate_dpo(b, pad_id),
        )
        print(f"[测试集] {len(eval_pairs)} 条偏好对")

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
        epoch_loss, epoch_steps = 0.0, 0

        for batch in loader:
            c_ids  = batch["chosen_ids"].to(device);    c_attn  = batch["chosen_attn"].to(device)
            r_ids  = batch["rejected_ids"].to(device);  r_attn  = batch["rejected_attn"].to(device)
            m = policy.module if hasattr(policy, "module") else policy

            pol_c = sequence_log_prob(m, c_ids, c_attn)
            pol_r = sequence_log_prob(m, r_ids, r_attn)

            with torch.no_grad():
                ref_c = sequence_log_prob(ref_model, c_ids, c_attn)
                ref_r = sequence_log_prob(ref_model, r_ids, r_attn)

            loss, reward_chosen, reward_rejected = dpo_loss(pol_c, pol_r, ref_c, ref_r, args.beta)
            kl_chosen = (pol_c - ref_c).detach().mean().item()

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step  += 1
            epoch_loss   += loss.item()
            epoch_steps  += 1
            lr_now        = scheduler.get_last_lr()[0]

            if global_step % args.logging_steps == 0:
                avg_loss = epoch_loss / epoch_steps
                print(
                    f"  Ep{epoch} Step{global_step} | "
                    f"loss={avg_loss:.4f} | "
                    f"r_chosen={reward_chosen:.4f} r_rejected={reward_rejected:.4f} | "
                    f"kl={kl_chosen:.4f} | gnorm={grad_norm:.3f} | lr={lr_now:.2e}"
                )
                train_log.append({
                    "step":              global_step,
                    "epoch":             epoch,
                    "train.loss":        round(avg_loss,         6),
                    "rl.reward.mean":    round(reward_chosen,    6),
                    "rl.kl_divergence":  round(kl_chosen,        6),
                    "train.lr":          round(lr_now,           8),
                    "train.grad_norm":   round(grad_norm.item(), 6),
                })

            emit("train.loss",       "gauge", loss.item(),          global_step, mdir)
            emit("rl.reward.mean",   "gauge", reward_chosen,        global_step, mdir)
            emit("rl.kl_divergence", "gauge", kl_chosen,            global_step, mdir)
            emit("train.lr",         "gauge", lr_now,               global_step, mdir)
            emit("train.grad_norm",  "gauge", grad_norm.item(),     global_step, mdir)

            if global_step % args.save_steps == 0:
                eval_loss = evaluate(policy, ref_model, eval_loader, device, args.beta) if eval_loader else None
                save_ckpt(policy, tok, args.output_dir, global_step, {
                    "step": global_step, "epoch": epoch,
                    "train_loss": round(loss.item(), 6),
                    "eval_loss":  round(eval_loss, 6) if eval_loss is not None else None,
                })

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        eval_loss      = evaluate(policy, ref_model, eval_loader, device, args.beta) if eval_loader else None
        eval_str       = f" | val_loss={eval_loss:.4f}" if eval_loss is not None else ""
        print(f"[Epoch {epoch}/{args.epochs}] train_loss={avg_epoch_loss:.4f}{eval_str}")
        if eval_loss is not None:
            emit("val.loss", "gauge", eval_loss, global_step, mdir)

    # ── 保存最终模型 ───────────────────────────────────────────────
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
        "status":            "success",
        "final_train_loss":  last.get("train.loss"),
        "final_kl":          last.get("rl.kl_divergence"),
        "total_steps":       global_step,
        "output_dir":        args.output_dir,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    train()
