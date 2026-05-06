#!/usr/bin/env python3
"""
SFT 全参微调 — 独立训练/评估脚本

从 Magnus 蓝图（OpenFundus_SFT_zyz.magnus）提取的独立模块。
蓝图通过 wget 从 github.com/Rise-AGI/ 拉取此文件并执行。

用法:
    # 训练模式
    python sft_train.py --model_path /path/to/model --train_data /path/to/data --output_dir /tmp/out

    # 评估模式（训练完成后单独推理）
    python sft_train.py --eval-only --model_dir /tmp/out/final --test_path /path/to/test --output_dir /tmp/out
"""

import argparse
import json
import os
import time
import traceback
import warnings
from datetime import timedelta


def log(msg: str) -> None:
    """带时间戳打印，与 shell _log() 格式一致。flush=True 确保长时间操作中日志立即可见。"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


import re as _re

def parse_answer_solution(text: str):
    """
    将 "答案：...\\n\\n解答：..." 格式拆分为 (answer, solution)。
    对模型输出和标准答案通用。无法解析时 answer 返回全文。
    """
    if not text:
        return "", ""
    # 按 "解答" 分割（兼容中英文冒号）
    parts = _re.split(r'\n?\s*解答\s*[：:]\s*', text, maxsplit=1)
    if len(parts) >= 2:
        ans_part = parts[0]
        sol = parts[1].strip()
        m = _re.search(r'答案\s*[：:]\s*(.*?)$', ans_part, _re.DOTALL)
        ans = m.group(1).strip() if m else ans_part.strip()
        return ans, sol
    # fallback: 在单段文本中分别匹配
    m_ans = _re.search(r'答案\s*[：:]\s*(.*?)(?:\n\n|\n解答|$)', text, _re.DOTALL)
    m_sol = _re.search(r'解答\s*[：:]\s*(.*?)$', text, _re.DOTALL)
    ans = m_ans.group(1).strip() if m_ans else ""
    sol = m_sol.group(1).strip() if m_sol else ""
    if not ans and not sol:
        return text.strip(), ""
    return ans, sol


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.utils.data.distributed import DistributedSampler

# ═══════════════════════════════════════════════════════════════
# A100 硬件加速 + NCCL 优化（在 NCCL 初始化前设置）
# ═══════════════════════════════════════════════════════════════
torch.set_float32_matmul_precision('high')   # A100 TF32 tensor core 加速
torch.backends.cudnn.benchmark = True        # 固定 shape 输入下卷积/矩阵乘自寻最优算法

# NCCL: 单节点双卡走 NVLink（撤销蓝图可能设置的 P2P_DISABLE=1）
os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
os.environ["NCCL_P2P_DISABLE"] = "0"
# IB 在单节点场景不需要，保留 DISABLE 以消除 IB timeout 风险
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "^docker,lo,virbr")

# FSDP.state_dict_type() 已弃用但新 DCP get_state_dict 不支持 rank0_only，
# rank0_only 对 72B 模型至关重要（避免每 rank 各持一份完整 CPU state dict）
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="torch.distributed.fsdp")

# ── 兼容性补丁 ──
# pytorch/pytorch:2.5.1 镜像 + transformers>=5.7 的已知问题：
# transformers 5.7 因 CVE-2025-32434 禁止 torch<2.6 加载 .bin 权重文件。
# 模型为 .bin 格式时需绕过此检查。此补丁不影响 safetensors 加载。
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None


def parse_args():
    p = argparse.ArgumentParser(description="通用大模型 SFT on Magnus")
    # 模式
    p.add_argument("--eval-only", action="store_true", help="仅运行推理评估（不训练）")

    # 模型与数据
    p.add_argument("--model_path",    type=str, default=None,
                    help="训练模式：模型路径")
    p.add_argument("--train_data",    type=str, default=None,
                    help="训练模式：训练集路径")
    p.add_argument("--test_data",     type=str, default=None,
                    help="训练或评估模式：测试集路径（评估/验证）")
    p.add_argument("--output_dir",    type=str, default="/tmp/sft_output",
                    help="输出目录")

    # 评估模式专属
    p.add_argument("--model_dir",     type=str, default=None,
                    help="评估模式：已保存模型目录路径")

    # 超参数
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--max_length",    type=int,   default=1024)
    p.add_argument("--warmup_ratio",  type=float, default=0.05)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--logging_steps", type=int,   default=10)
    p.add_argument("--num_workers",  type=int,   default=2,
                    help="DataLoader worker 进程数")
    p.add_argument("--save_steps",    type=int,   default=100,
                    help="checkpoint 保存间隔（global_step 倍数）")
    p.add_argument("--retry_seed",    type=int,   default=0,
                    help="DistributedSampler retry seed（shell retry 递增）")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--prompt_prefix",     type=str, default=None,
                   help="统一添加到每条样本 instruction 前面的提示词。支持 {instruction} 占位符。")
    p.add_argument("--prompt_prefix_b64", type=str, default=None,
                   help="prompt_prefix 的 base64 编码（shell 安全传递）")

    args = p.parse_args()

    # base64 解码 prompt_prefix（shell 安全传递）
    if args.prompt_prefix_b64:
        import base64
        args.prompt_prefix = base64.b64decode(args.prompt_prefix_b64).decode("utf-8")

    return args


def load_json_dataset(path: str) -> list:
    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        data = df.to_dict(orient="records")
    elif os.path.isdir(path):
        try:
            from datasets import load_from_disk
            ds = load_from_disk(path)
            data = [row for row in ds]
        except Exception:
            import glob
            files = sorted(
                glob.glob(os.path.join(path, "**", "*.json"),  recursive=True) +
                glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True)
            )
            if not files:
                raise FileNotFoundError(
                    f"目录 {path} 既不是 HuggingFace Dataset，也没有找到任何 .json/.jsonl 文件"
                )
            data = []
            for fp in files:
                log(f"[数据] 读取文件：{fp}")
                with open(fp, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                if raw.startswith("["):
                    data.extend(json.loads(raw))
                else:
                    data.extend([json.loads(line) for line in raw.splitlines() if line.strip()])
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if raw.startswith("["):
            data = json.loads(raw)
        else:
            data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(data) > 0, f"数据集为空：{path}"
    log(f"[数据] 从 {path} 加载 {len(data)} 条样本")
    return data


class SFTDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length, prompt_prefix=None):
        self.samples      = samples
        self.tokenizer    = tokenizer
        self.max_length   = max_length
        self.prompt_prefix = prompt_prefix

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        instruction = item.get("instruction", "")
        extra_input = item.get("input", "")
        output      = item.get("output", "")
        # 统一提示词前缀：支持 {instruction} 占位符
        if self.prompt_prefix:
            instruction = self.prompt_prefix.replace("{instruction}", instruction)
        user_content = instruction
        if extra_input:
            user_content += f"\n{extra_input}"

        # 通用：使用 tokenizer.apply_chat_template
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]

        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        user_messages = [{"role": "user", "content": user_content}]
        user_prompt = self.tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        user_ids = self.tokenizer.encode(user_prompt, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False)

        input_ids = full_ids[:self.max_length]
        labels = ([-100] * len(user_ids) + full_ids[len(user_ids):])[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }


def collate_fn(batch, pad_id):
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids_list, labels_list, attn_list = [], [], []
    for b in batch:
        n   = b["input_ids"].size(0)
        pad = max_len - n
        input_ids_list.append(F.pad(b["input_ids"], (0, pad), value=pad_id))
        labels_list.append(F.pad(b["labels"],    (0, pad), value=-100))
        attn_list.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids":      torch.stack(input_ids_list),
        "labels":         torch.stack(labels_list),
        "attention_mask": torch.stack(attn_list),
    }


def unwrap_model(model):
    """解包 DataParallel / FSDP 包装，返回原始模型。"""
    if isinstance(model, FSDP):
        return model.module
    if hasattr(model, "module"):
        return model.module
    return model


def save_checkpoint(model, tokenizer, output_dir, step, meta, local_rank=0):
    """FSDP-safe checkpoint：原子保存防止中断损坏。"""
    ckpt_path = os.path.join(output_dir, "checkpoint-latest")
    if isinstance(model, FSDP):
        from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig
        full_config = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_config):
            state = model.state_dict()
            if local_rank == 0:
                import shutil
                tmp_path = os.path.join(output_dir, f".ckpt-tmp-{step}")
                os.makedirs(tmp_path, exist_ok=True)
                m = unwrap_model(model)
                m.save_pretrained(tmp_path, state_dict=state)
                tokenizer.save_pretrained(tmp_path)
                with open(os.path.join(tmp_path, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                if os.path.exists(ckpt_path):
                    shutil.rmtree(ckpt_path, ignore_errors=True)
                os.rename(tmp_path, ckpt_path)
                log(f"  [Checkpoint] 已保存 step={step} -> {ckpt_path} (原子覆盖)")
    else:
        if local_rank != 0:
            return
        import shutil
        tmp_path = os.path.join(output_dir, f".ckpt-tmp-{step}")
        os.makedirs(tmp_path, exist_ok=True)
        m = unwrap_model(model)
        m.save_pretrained(tmp_path)
        tokenizer.save_pretrained(tmp_path)
        with open(os.path.join(tmp_path, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path, ignore_errors=True)
        os.rename(tmp_path, ckpt_path)
        log(f"  [Checkpoint] 已保存 step={step} -> {ckpt_path} (原子覆盖)")


def save_final(model, tokenizer, output_dir, train_log, local_rank=0):
    """FSDP-safe：所有 rank 参与 state_dict gather，仅 rank 0 写盘。"""
    final_path = os.path.join(output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    if isinstance(model, FSDP):
        from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig
        full_config = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_config):
            state = model.state_dict()
            if local_rank == 0:
                m = unwrap_model(model)
                m.save_pretrained(final_path, max_shard_size="1800MB", state_dict=state)
                tokenizer.save_pretrained(final_path)
                log_path = os.path.join(output_dir, "training_log.json")
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(train_log, f, ensure_ascii=False, indent=2)
                log(f"[最终模型] 已保存 -> {final_path}")
                log(f"[训练日志] 已保存 -> {log_path}")
    else:
        if local_rank != 0:
            return
        m = unwrap_model(model)
        m.save_pretrained(final_path, max_shard_size="1800MB")
        tokenizer.save_pretrained(final_path)
        log_path = os.path.join(output_dir, "training_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(train_log, f, ensure_ascii=False, indent=2)
        log(f"[最终模型] 已保存 -> {final_path}")
        log(f"[训练日志] 已保存 -> {log_path}")


def _load_safetensors_state(ckpt_path):
    """从 safetensors 加载 state dict（处理分片 + 损坏检测）。"""
    try:
        import json as _json
        index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
        single_path = os.path.join(ckpt_path, "model.safetensors")
        if os.path.exists(index_path):
            from safetensors.torch import load_file
            with open(index_path) as f:
                _idx = _json.load(f)
            sd = {}
            for _sf in set(_idx["weight_map"].values()):
                sp = os.path.join(ckpt_path, _sf)
                if os.path.exists(sp):
                    sd.update(load_file(sp))
            return sd
        elif os.path.exists(single_path):
            from safetensors.torch import load_file
            return load_file(single_path)
    except Exception as e:
        log(f"  [警告] checkpoint safetensors 损坏: {e}")
        return None
    return None


@torch.no_grad()
def evaluate(model, dataloader, device, n_gpu=1, local_rank=0):
    model.eval()
    total_loss, total_steps = 0.0, 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        outputs   = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = outputs.loss
        total_loss  += loss.item()
        total_steps += 1
    # 跨卡汇总（FSDP 下每卡只算了部分数据）
    if n_gpu > 1:
        loss_t = torch.tensor([total_loss], device=device)
        cnt_t  = torch.tensor([total_steps], device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(cnt_t,  op=dist.ReduceOp.SUM)
        total_loss = loss_t.item()
        total_steps = int(cnt_t.item())
    model.train()
    return total_loss / max(total_steps, 1)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    local_rank = 0
    if n_gpu > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=600))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    os.makedirs(args.output_dir, exist_ok=True)

    log(f"[环境] 设备={device}, GPU 数量={n_gpu}, rank={local_rank}")
    log(f"[配置] epochs={args.epochs}, batch_size={args.batch_size}, grad_accum={args.gradient_accumulation_steps}, lr={args.learning_rate}, max_length={args.max_length}")

    # ── Step 1: Tokenizer ──
    t0 = time.time()
    log(f"[1/8] 加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\nAssistant: ' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}"
    log(f"[1/8] tokenizer 加载完成 ({time.time()-t0:.1f}s) | vocab_size={tokenizer.vocab_size} | pad_token_id={tokenizer.pad_token_id}")

    # ── Step 2: Config ──
    t0 = time.time()
    log(f"[2/8] 加载 config: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(config, 'rope_scaling') and isinstance(config.rope_scaling, dict):
        if "type" in config.rope_scaling and "rope_type" not in config.rope_scaling:
            config.rope_scaling["rope_type"] = config.rope_scaling["type"]
            log(f"[兼容] rope_scaling: 添加 rope_type={config.rope_scaling['type']} (新旧格式共存)")
    if hasattr(config, "attn_implementation"):
        try:
            import flash_attn  # noqa: F401
            config.attn_implementation = "flash_attention_2"
            log(f"[2/8] FlashAttention2 已启用")
        except ImportError:
            log(f"[2/8] flash_attn 未安装，使用默认 attention (sdpa)")
    log(f"[2/8] config 加载完成 ({time.time()-t0:.1f}s) | model_type={config.model_type}")

    # ── Step 3: 模型权重（72B 模型约 144GB，NFS 读取需 30-60 分钟）──
    load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    log(f"[3/8] 开始加载模型权重到内存 (dtype={load_dtype})...")
    log(f"[3/8] 提示: 72B 模型 ~144GB，从 /data NFS 读取预计 30-60 分钟，请耐心等待")
    log(f"[3/8] 模型路径: {args.model_path}")
    import glob as _glob
    _model_files = sorted(_glob.glob(os.path.join(args.model_path, "*.safetensors")) +
                          _glob.glob(os.path.join(args.model_path, "*.bin")))
    if _model_files:
        _total_size = sum(os.path.getsize(f) for f in _model_files) / 1e9
        log(f"[3/8] 模型文件: {len(_model_files)} 个, 总计 ~{_total_size:.1f}GB")
        for _f in _model_files:
            log(f"  {os.path.basename(_f):>40s}  {os.path.getsize(_f)/1e9:.2f}GB")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=load_dtype,
        trust_remote_code=True,
    )
    _elapsed = time.time() - t0
    log(f"[3/8] 模型权重加载完成 ({_elapsed:.1f}s = {_elapsed/60:.1f}min)")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            log(f"  GPU {i}: 空闲 {free/1e9:.1f}GB / 总计 {total/1e9:.1f}GB")

    # ── Step 4: Resume / Gradient Checkpointing / 参数量 ──
    t0 = time.time()
    log(f"[4/8] 配置训练优化...")
    start_step = 0
    if args.resume_from_checkpoint:
        ckpt = args.resume_from_checkpoint
        meta_path = os.path.join(ckpt, "checkpoint_meta.json")
        meta_step = 0
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                meta_step = meta.get("step", 0)
            except Exception as e:
                log(f"[4/8] checkpoint_meta.json 读取失败: {e}，忽略元数据")
        try:
            state = _load_safetensors_state(ckpt)
            if state is not None and len(state) > 0:
                model.load_state_dict(state, strict=False)
                start_step = meta_step
                log(f"[4/8] 从 checkpoint 恢复权重, start_step={start_step}")
            else:
                log(f"[4/8] checkpoint 无有效权重，从头开始训练")
        except Exception as e:
            log(f"[4/8] 加载 checkpoint 失败: {e}，从头开始训练")

    _raw_params = sum(p.numel() for p in model.parameters())
    total_params = _raw_params / 1e9
    log(f"[4/8] 原始参数量: {total_params:.2f}B ({_raw_params} params)")

    # 按参数量自动区分 7B / 72B 配置
    is_large_model = total_params > 30  # >30B → 大模型模式（显存安全优先）
    if hasattr(model, "gradient_checkpointing_enable"):
        if is_large_model:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            log("[4/8] gradient_checkpointing 已开启 (大模型模式)")
        else:
            log("[4/8] gradient_checkpointing 已跳过 (小模型模式，显存充足)")

    # ── Step 5: FSDP/分布式包装 ──
    log(f"[5/8] 分布式包装 ({n_gpu} GPU)...")
    t0 = time.time()
    if n_gpu > 1:
        estimated_gb_per_gpu = (total_params * 2) / n_gpu
        log(f"[5/8] 预计每卡 ~{estimated_gb_per_gpu:.1f}GB (模型+优化器分片)")
        # 检测 transformer decoder layer 类（兼容 Qwen2 / LLaMA / InternLM 等）
        _layer_cls = None
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from functools import partial as _partial
        for _name, _mod in model.named_modules():
            _cn = type(_mod).__name__
            if 'Decoder' in _cn and 'Layer' in _cn:
                _layer_cls = type(_mod)
                log(f"[5/8] 检测到 transformer layer: {_cn} (来自 {_name})")
                break
        if _layer_cls is None:
            log("[5/8] 未检测到标准 layer，使用 size_based 策略")
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
            _policy = _partial(size_based_auto_wrap_policy, min_num_params=1e8)
        else:
            _policy = _partial(transformer_auto_wrap_policy, transformer_layer_cls={_layer_cls})
        # 使用 auto_wrap_policy 逐层分片，避免 FSDP init 时将完整模型移至单卡
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=_policy,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            device_id=local_rank,
            limit_all_gathers=True,
            forward_prefetch=True,
        )
        log(f"[5/8] FSDP FULL_SHARD 完成 ({time.time()-t0:.1f}s)")
    else:
        model = model.to(device)
        log(f"[5/8] 单卡模式，模型已移至 {device} ({time.time()-t0:.1f}s)")

    eff_batch = args.batch_size * args.gradient_accumulation_steps * max(n_gpu, 1)
    log(f"[配置] 等效全局batch={eff_batch} ({args.batch_size}/卡 × {args.gradient_accumulation_steps}累积 × {max(n_gpu,1)}GPU)")

    # ── Step 6: 数据加载 ──
    t0 = time.time()
    log(f"[6/8] 加载训练数据: {args.train_data}")
    pad_id = tokenizer.pad_token_id

    def train_collate(b): return collate_fn(b, pad_id)
    def eval_collate(b):  return collate_fn(b, pad_id)

    train_samples = load_json_dataset(args.train_data)
    log(f"[6/8] 训练样本: {len(train_samples)} 条")
    train_dataset = SFTDataset(train_samples, tokenizer, args.max_length, args.prompt_prefix)
    train_sampler = DistributedSampler(train_dataset, rank=local_rank, shuffle=True) if n_gpu > 1 else None
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, prefetch_factor=2, collate_fn=train_collate)
    log(f"[6/8] DataLoader: {len(train_loader)} batches/epoch (batch_size={args.batch_size})")

    eval_loader = None
    eval_samples_raw = None  # 用于生成式评估
    if args.test_data and os.path.exists(args.test_data):
        log(f"[6/8] 加载测试数据: {args.test_data}")
        eval_samples_raw = load_json_dataset(args.test_data)
        eval_dataset = SFTDataset(eval_samples_raw, tokenizer, args.max_length, args.prompt_prefix)
        eval_loader  = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=2, collate_fn=eval_collate)
        log(f"[6/8] 测试集: {len(eval_samples_raw)} 条, {len(eval_loader)} batches")
    else:
        log("[6/8] 测试集: 未提供，跳过 eval loss")
    log(f"[6/8] 数据准备完成 ({time.time()-t0:.1f}s)")

    # ── Step 7: Optimizer & Scheduler ──
    t0 = time.time()
    log(f"[7/8] 创建 optimizer (AdamW, lr={args.learning_rate})...")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    accum         = args.gradient_accumulation_steps
    steps_per_epoch = (len(train_loader) + accum - 1) // accum
    total_steps   = steps_per_epoch * args.epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    log(f"[7/8] 优化器就绪 ({time.time()-t0:.1f}s) | steps_per_epoch={steps_per_epoch} | total_steps={total_steps} | warmup={warmup_steps}")

    # ── Step 8: 开始训练循环 ──
    log(f"[8/8] 开始训练循环")
    log(f"{'='*60}")
    log(f"  模型: {total_params:.2f}B params | FSDP: {'FULL_SHARD' if n_gpu > 1 else 'OFF'}")
    log(f"  数据: {len(train_samples)} 训练样本 | {len(train_loader)} batches/epoch")
    log(f"  训练: {args.epochs} epochs × {steps_per_epoch} steps = {total_steps} total steps")
    log(f"  保存: 每 {args.save_steps} steps（自动覆盖） | 日志: 每 {args.logging_steps} steps")
    log(f"  GPU: {n_gpu} × {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log(f"{'='*60}")

    train_log   = []
    global_step = 0
    optimizer.zero_grad()
    model.train()
    retry_seed = args.retry_seed

    # ── 诊断: 数据质量预检（仅 rank 0）──
    if local_rank == 0:
        bad_samples = 0
        for di, s in enumerate(train_dataset.samples):
            out = s.get("output", "")
            if not out or not out.strip():
                bad_samples += 1
                if bad_samples <= 3:
                    log(f"  [数据诊断] 空 output 样本 #{di}: instruction={s.get('instruction','')[:80]}")
        if bad_samples:
            log(f"  [数据诊断] 共 {bad_samples} 条空 output 样本（已统计）")
        else:
            log(f"  [数据诊断] 所有 {len(train_dataset.samples)} 条样本 output 均非空 ✓")

    # ── Initial loss @ step=0 ──
    init_eval_loss = None
    init_train_loss = None
    if local_rank == 0:
        log("  [初始化] 计算 step=0 初始 loss...")
    if eval_loader:
        init_eval_loss = evaluate(model, eval_loader, device, n_gpu, local_rank)
    if len(train_loader) > 0:
        try:
            first_batch = next(iter(train_loader))
            input_ids = first_batch["input_ids"].to(device)
            labels    = first_batch["labels"].to(device)
            attn_mask = first_batch["attention_mask"].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous().float()
                shift_labels = labels[..., 1:].contiguous()
                batch_loss = F.cross_entropy(
                    shift_logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                ).item()
            if n_gpu > 1:
                loss_t = torch.tensor([batch_loss], device=device)
                dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                batch_loss = loss_t.item() / n_gpu
            init_train_loss = batch_loss
        except Exception as e:
            if local_rank == 0:
                log(f"  [警告] step=0 train_loss 计算失败: {e}")
    if local_rank == 0:
        if init_train_loss is not None:
            log(f"  [初始化] step=0: train_loss={init_train_loss:.4f}")
        else:
            log("  [初始化] step=0: train_loss=None")
        if init_eval_loss is not None:
            log(f"  [初始化] step=0: eval_loss={init_eval_loss:.4f}")
        else:
            log("  [初始化] step=0: eval_loss=None")
    train_log.append({"global_step": 0, "epoch": 0.0, "train_loss": round(init_train_loss, 6) if init_train_loss is not None else None, "eval_loss": round(init_eval_loss, 6) if init_eval_loss is not None else None, "lr": round(scheduler.get_last_lr()[0], 8), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})

    # 训练前生成式评估已独立为 eval_baseline.py（按需手动执行）

    for epoch in range(1, args.epochs + 1):
        if n_gpu > 1:
            train_sampler.set_epoch(epoch + retry_seed)
        epoch_loss  = 0.0
        nan_count   = 0
        _skip_count = 0
        epoch_start = time.time()
        slow_count  = 0
        torch.cuda.reset_peak_memory_stats()
        log(f"[Epoch {epoch}/{args.epochs}] 开始...")

        for step, batch in enumerate(train_loader, 1):
            step_start = time.time()

            # ── 诊断: 每 10 步打印一次步进信息 ──
            if local_rank == 0 and step % 10 == 1:
                log(f"  [诊断] >>> Step {step}/{len(train_loader)} (global {global_step+1}) 开始")

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            seq_len   = input_ids.shape[-1]

            # ── 诊断: 定期记录序列长度和显存 ──
            if local_rank == 0 and step % 40 == 1:
                cur_mem = torch.cuda.memory_allocated() / 1024**3
                peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                log(f"  [诊断]   seq_len={seq_len}, GPU mem={cur_mem:.1f}GB (peak={peak_mem:.1f}GB)")

            # ── 前向传播（含异常恢复）──
            _skip_batch = False
            try:
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                # float32 计算 loss，避免 bfloat16 × 15 万词表数值下溢出
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous().float()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            except Exception as e:
                _skip_batch = True
                _skip_count += 1
                if local_rank == 0:
                    log(f"  [错误] Step {step} 前向异常: {e}")
                    for _tb_line in traceback.format_exc().rstrip().split("\n"):
                        log(f"  [错误]   {_tb_line}")

            if _skip_batch:
                # dummy 前向保持计算图连接（FSDP 依赖 backward hook 触发 NCCL sync）
                dummy = model(input_ids=input_ids, attention_mask=attn_mask)
                loss = dummy.logits[0, 0, 0] * 0.0

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if local_rank == 0 and nan_count <= 5:
                    log(f"  [警告] NaN/Inf loss @ step {step}, 跳过此 batch (#{nan_count})")
                # torch.where 保留计算图，确保 FSDP backward hook 能正常触发 NCCL sync
                loss = torch.where(
                    torch.isnan(loss) | torch.isinf(loss),
                    torch.zeros_like(loss),
                    loss,
                )

            (loss / accum).backward()
            epoch_loss += loss.item()

            # ── 诊断: 检测慢步 ──
            step_elapsed = time.time() - step_start
            if step_elapsed > 30:
                slow_count += 1
                log(f"  [SLOW] Step {step} 耗时 {step_elapsed:.1f}s | seq_len={seq_len} | "
                    f"GPU={torch.cuda.memory_allocated()/1024**3:.1f}GB "
                    f"(此 epoch 第 {slow_count} 次慢步)")
            elif step_elapsed > 10 and local_rank == 0:
                log(f"  [诊断] Step {step} 略慢: {step_elapsed:.1f}s | seq_len={seq_len}")

            is_update = (step % accum == 0) or (step == len(train_loader))
            if is_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) if n_gpu <= 1 else model.clip_grad_norm_(1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = epoch_loss / step
                    lr_now   = scheduler.get_last_lr()[0]
                    log(f"  Epoch {epoch}/{args.epochs} | Step {step}/{len(train_loader)} (global {global_step}) | Loss {avg_loss:.4f} | LR: {lr_now:.2e}")
                    train_log.append({"global_step": global_step, "epoch": round(epoch - 1 + step / len(train_loader), 3), "train_loss": round(avg_loss, 6), "lr": round(lr_now, 8), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})

                # ── 自动存档（仅 loss 评估 + 保存权重，不做生成式推理）──
                if global_step % args.save_steps == 0:
                    eval_loss = evaluate(model, eval_loader, device, n_gpu, local_rank) if eval_loader else None
                    save_checkpoint(model, tokenizer, args.output_dir, global_step, meta={"step": global_step, "epoch": round(epoch - 1 + step / len(train_loader), 3), "train_loss": round(epoch_loss / step, 6), "eval_loss": round(eval_loss, 6) if eval_loss is not None else None, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, local_rank=local_rank)
                    if local_rank == 0:
                        _e_str = f" | eval_loss={eval_loss:.4f}" if eval_loss is not None else ""
                        log(f"  [存档] step={global_step} train_loss={epoch_loss/step:.4f}{_e_str}")
                    train_log.append({"global_step": global_step, "epoch": round(epoch - 1 + step / len(train_loader), 3), "train_loss": round(epoch_loss / step, 6), "eval_loss": round(eval_loss, 6) if eval_loss is not None else None, "lr": round(scheduler.get_last_lr()[0], 8), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})

        avg_epoch_loss = epoch_loss / len(train_loader)
        elapsed        = time.time() - epoch_start
        eval_loss      = evaluate(model, eval_loader, device, n_gpu, local_rank) if eval_loader else None
        eval_str = f" | Eval Loss: {eval_loss:.4f}" if eval_loss is not None else ""
        log(f"[Epoch {epoch}/{args.epochs}] Train Loss: {avg_epoch_loss:.4f}{eval_str} | 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
        if _skip_count > 0:
            log(f"  [诊断] Epoch {epoch} 中共 {_skip_count} 个 batch 因错误跳过")
        train_log.append({"global_step": global_step, "epoch": epoch, "train_loss": round(avg_epoch_loss, 6), "eval_loss": round(eval_loss, 6) if eval_loss is not None else None, "elapsed_sec": round(elapsed, 1), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})

    save_final(model, tokenizer, args.output_dir, train_log, local_rank=local_rank)

    if eval_samples_raw is not None:
        run_generation_eval(model, tokenizer, eval_samples_raw, args,
                            tag="final", model_path=args.model_path,
                            output_dir=args.output_dir, device=device,
                            local_rank=local_rank, n_gpu=n_gpu)

    final_eval_loss = evaluate(model, eval_loader, device, n_gpu, local_rank) if eval_loader else None
    result = {"status": "success", "final_train_loss": round(train_log[-1]["train_loss"], 6), "final_eval_loss": round(final_eval_loss, 6) if final_eval_loss is not None else None, "total_steps": global_step, "output_dir": args.output_dir}
    log(f"[结果] {json.dumps(result, ensure_ascii=False)}")
    return result


@torch.no_grad()
def run_eval(args):
    """评估模式：基于已保存模型对测试集做推理并保存结果。"""
    model_dir = args.model_dir
    test_path = args.test_path or args.test_data
    out_dir   = args.output_dir

    log("[推理] 加载模型: " + model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n\\nAssistant: ' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + '\\n\\n' }}{% endif %}{% endfor %}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    log("[推理] 模型加载完成，设备: " + device)

    with open(test_path, "r", encoding="utf-8") as fh:
        raw = fh.read().strip()
    samples = json.loads(raw) if raw.startswith("[") else [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    log("[推理] 测试集共 " + str(len(samples)) + " 条")

    results = []

    for i, sample in enumerate(samples):
        instruction = sample.get("instruction", "")
        extra       = sample.get("input", "")
        gt_out      = sample.get("output", "")

        if args.prompt_prefix:
            instruction = args.prompt_prefix.replace("{instruction}", instruction)
        user_content = instruction + ("\n" + extra if extra else "")

        # 通用：使用 apply_chat_template 自动适配所有模型
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids  = out_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        gt_ans, gt_sol = parse_answer_solution(gt_out)
        md_ans, md_sol = parse_answer_solution(response)
        results.append({
            "id": sample.get("id", i),
            "question": user_content,
            "gt_full": gt_out,
            "gt_answer": gt_ans,
            "gt_solution": gt_sol,
            "model_full": response,
            "model_answer": md_ans,
            "model_solution": md_sol,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            log("  进度 %d/%d" % (i + 1, len(samples)))

    eval_dir = os.path.join(out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    eval_path = os.path.join(eval_dir, "eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    _fn = os.path.basename(eval_path)
    log(f"[推理] 已保存: {_fn} ({len(results)} 条)")
    log(f"[推理] 文件路径: {eval_path}")
    log(f"[推理] 下载: 任务结束后查看日志末尾的 magnus receive 命令")


@torch.no_grad()
def run_generation_eval(model, tokenizer, test_samples, args, tag,
                         model_path, output_dir, device, local_rank=0, n_gpu=1):
    """
    对测试集做生成式推理（逐样本生成 response），保存 JSON 结果。
    FSDP 兼容：所有 rank 参与 state_dict 收集（NCCL collective），
    rank 0 单独推理，其他 rank 在 barrier 等待，完成后同步返回训练循环。
    """
    # ── 1. 收集完整 state dict（ALL ranks 必须参与 NCCL collective）──
    _state = None
    if isinstance(model, FSDP):
        from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig
        _cfg = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, _cfg):
            _state = model.state_dict()
        if local_rank == 0:
            log(f"[推理-{tag}] FSDP state dict 收集完成")

    # barrier: 确保 state dict 的所有 NCCL 操作完成后 rank 0 再单独推理
    if n_gpu > 1:
        dist.barrier()

    # 非 rank 0 在此等待 rank 0 完成推理，防止其他 rank 进入训练循环
    # 发送需要 rank 0 参与的 NCCL collective（FSDP forward allgather）
    if local_rank != 0:
        if n_gpu > 1:
            dist.barrier()
        return

    log(f"[推理-{tag}] 开始生成式评估 ({len(test_samples)} 条)...")

    # ── 2. 创建临时推理模型（rank 0 only）──
    if isinstance(model, FSDP):
        _config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        _temp = AutoModelForCausalLM.from_config(_config, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16)
        _temp.load_state_dict(_state, strict=False)
        _temp = _temp.to(device)
        _temp.eval()
        _inf_model = _temp
        log(f"[推理-{tag}] 临时推理模型已创建")
    else:
        _inf_model = model
        _inf_model.eval()

    # ── 3. 逐样本推理 ──
    _results = []
    for _i, _sample in enumerate(test_samples):
        _inst = _sample.get("instruction", "")
        _extra = _sample.get("input", "")
        _gt = _sample.get("output", "")
        if args.prompt_prefix:
            _inst = args.prompt_prefix.replace("{instruction}", _inst)
        _content = _inst + ("\n" + _extra if _extra else "")

        _messages = [{"role": "user", "content": _content}]
        _prompt = tokenizer.apply_chat_template(_messages, tokenize=False,
                                                 add_generation_prompt=True)
        _inputs = tokenizer(_prompt, return_tensors="pt", truncation=True,
                            max_length=args.max_length).to(device)
        _out_ids = _inf_model.generate(
            **_inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        _new_ids = _out_ids[0][_inputs["input_ids"].shape[1]:]
        _response = tokenizer.decode(_new_ids, skip_special_tokens=True).strip()

        _gt_ans, _gt_sol = parse_answer_solution(_gt)
        _md_ans, _md_sol = parse_answer_solution(_response)
        _results.append({
            "id": _sample.get("id", _i),
            "question": _content,
            "gt_full": _gt,
            "gt_answer": _gt_ans,
            "gt_solution": _gt_sol,
            "model_full": _response,
            "model_answer": _md_ans,
            "model_solution": _md_sol,
        })

        if (_i + 1) % 10 == 0 or (_i + 1) == len(test_samples):
            log(f"  [推理-{tag}] {_i+1}/{len(test_samples)}")

    # ── 4. 保存 JSON ──
    _eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(_eval_dir, exist_ok=True)
    _path = os.path.join(_eval_dir, f"eval_results_{tag}.json")
    with open(_path, "w", encoding="utf-8") as _f:
        json.dump(_results, _f, ensure_ascii=False, indent=2)
    _fn = os.path.basename(_path)
    log(f"[推理-{tag}] 已保存: {_fn} ({len(_results)} 条)")
    log(f"[推理-{tag}] 文件路径: {_path}")
    log(f"[推理-{tag}] 下载: 任务结束后查看日志末尾的 magnus receive 命令")

    # ── 5. 清理 ──
    if isinstance(model, FSDP):
        del _temp, _state
        torch.cuda.empty_cache()
    else:
        _inf_model.train()

    # barrier: 通知其他 rank 推理完成，可以继续训练循环
    if n_gpu > 1:
        dist.barrier()


if __name__ == "__main__":
    _args = parse_args()

    if _args.eval_only:
        run_eval(_args)
    else:
        train(_args)
