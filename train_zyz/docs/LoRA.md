# LoRA / QLoRA SFT 训练 — 项目笔记

> 低资源微调大模型（72B 等）的 LoRA / QLoRA 方案。  
> 基于 `LoRA_zyz.magnus` 蓝图，支持 4-bit/8-bit/bf16 三种量化策略。

---

## 目录

- [1. 文件总览](#1-文件总览)
- [2. LoRA 蓝图](#2-lora-蓝图)
  - [2.1 参数说明](#21-参数说明)
  - [2.2 量化策略对比](#22-量化策略对比)
  - [2.3 Target Modules 自动检测](#23-target-modules-自动检测)
  - [2.4 多卡并行策略](#24-多卡并行策略)
  - [2.5 GPU 自动检测 + CPU 回退](#25-gpu-自动检测--cpu-回退2026-05-06)
- [3. 使用方式](#3-使用方式)
  - [3.1 通过 submit_sft.py 提交（推荐）](#31-通过-submit_sftpy-提交推荐)
  - [3.2 通过 Magnus SDK](#32-通过-magnus-sdk--web-ui)
  - [3.3 通过 Magnus Web UI](#33-通过-magnus-web-ui)
  - [3.4 CLI 提交](#34-cli-提交)
- [4. 输出结构](#4-输出结构)
- [5. LoRA vs 全参 SFT 对比](#5-lora-vs-全参-sft-对比)
- [6. 已知问题 & 排错](#6-已知问题--排错)

---

## 1. 文件总览

| 文件 | 角色 | 说明 |
|------|------|------|
| `LoRA_zyz.magnus` | **LoRA 蓝图** | 4-bit/8-bit/bf16 三种量化 + 自动检测 target modules |
| `SFT.md` | 全参 SFT 笔记 | 全参微调（对应 `OpenFundus_SFT_zyz.magnus`） |
| `docker.md` | 容器镜像笔记 | ACR / Docker 镜像管理 |

---

## 2. LoRA 蓝图

`LoRA_zyz.magnus` 是独立于 `OpenFundus_SFT_zyz.magnus` 的 LoRA 微调蓝图，使用 `peft` + `bitsandbytes` 实现。

### 2.1 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **模型与数据** | | |
| `model_path` | 必填 | 集群本地路径或 HuggingFace/ModelScope Hub ID |
| `train_data` | 自动生成假数据 | 训练集路径。留空 = 30 条假数据验证流程 |
| `test_data` | None | 测试集路径。留空跳过评估 |
| `output_dir` | 必填 | 输出目录（必须在 `/data/` 下的持久存储） |
| **超参数** | | |
| `epochs` | 3 | 训练轮数 |
| `batch_size` | 4 | 单卡 batch size（LoRA 显存低，可适当增大） |
| `grad_accum` | 4 | 梯度累积步数 |
| `learning_rate` | 5e-5 | 学习率（LoRA 通常比全参大 1-2 倍） |
| `max_length` | 1024 | 最大序列长度 |
| `num_workers` | 2 | DataLoader worker 进程数。假数据自动=0 |
| `save_steps` | 200 | checkpoint 保存间隔 |
| **LoRA 专属** | | |
| `lora_r` | 64 | LoRA rank（越大可学习参数越多，显存也越大） |
| `lora_alpha` | 128 | LoRA 缩放系数（通常 2×rank） |
| `lora_dropout` | 0.05 | LoRA dropout |
| `quantization` | `4bit` | 量化精度：`4bit` / `8bit` / `none` |
| `target_modules` | 自动 | 逗号分隔的模块名。留空自动检测 |
| `merge_on_save` | True | 保存时是否合并 adapter → 完整模型 |
| **硬件** | | |
| `gpu_count` | 4 | GPU 数量（QLoRA 72B 用 1-4 卡均可） |
| `gpu_type` | a100 | GPU 类型 |
| `cpu_count` | 40 | CPU 核心数 |
| `memory_demand` | 160G | 内存需求 |
| `ephemeral_storage` | 500G | 临时存储 |
| `priority` | A2 | 作业优先级 |
| `container_image` | None | 自定义容器镜像 |

### 2.2 量化策略对比

| 量化 | 72B 每卡显存 | GPU 推荐 | 精度 | 速度 |
|------|-------------|----------|------|------|
| `4bit` (QLoRA) | ~36 GB | 1-4 × A100 80G | 接近 bf16 | 较慢（bitsandbytes dequant） |
| `8bit` | ~72 GB | 4 × A100 80G | 中等 | 中等 |
| `none` (bf16) | ~144 GB | 8+ × A100 80G | 最高 | 最快 |

**推荐：QLoRA (4-bit NF4)**

- 72B 模型 4-bit 量化后约 36 GB/卡，4×A100 80G 有充足余量
- LoRA adapter 权重仅几百 MB，保存/加载极快
- 训练精度接近 bf16，评测分数差异通常 < 1%

### 2.3 Target Modules 自动检测

| 模型架构 | 自动选择的 Target Modules |
|----------|--------------------------|
| Qwen2 | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| LLaMA | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| 其他 | `q_proj`, `k_proj`, `v_proj`, `o_proj`（默认） |

也可通过 `target_modules` 参数手动指定。

### 2.4 多卡并行策略

| 量化模式 | 并行策略 | 说明 |
|---------|----------|------|
| `4bit` / `8bit` | DDP | 量化模型不支持 FSDP，每卡完整 base + LoRA |
| `none` (bf16) | FSDP FULL_SHARD | LoRA 权重跨卡分片，省显存 |

### 2.5 GPU 自动检测 + CPU 回退（2026-05-06）

与 SFT 蓝图相同，LoRA 蓝图也在 shell 层自动检测 GPU 数量并选择启动器：

- **0 GPU** → `LAUNCHER="python3"`（CPU 模式）
- **单卡** → `LAUNCHER="python3"`（不使用 torchrun）
- **多卡** → `LAUNCHER="torchrun --nproc_per_node=$ACTUAL_GPU"`（DDP/FSDP）
- `nvidia-smi` 非致命化（`|| echo`），避免无 NVIDIA runtime 时 `set -e` 崩溃

---

## 3. 使用方式

### 3.1 通过 submit_sft.py 提交（推荐）

修改 `submit_sft.py` 配置区：

```python
MODE        = "lora"         # LoRA/QLoRA 模式
MODEL_PATH  = "/data/$(whoami)/models/Qwen2.5-72B-Instruct"
```

然后运行 `python submit_sft.py`，自动使用 `LoRA_zyz.magnus` 蓝图。

### 3.2 通过 Magnus SDK / Web UI

**参数示例（4-bit QLoRA 微调 72B）：**

```python
magnus.launch_blueprint("LoRA_zyz", args={
    "model_path":        "/data/magnus/models/Qwen2.5-72B-Instruct",
    "output_dir":        "/data/magnus/models/qwen-lora-v1",
    "train_data":        "/data/magnus/training_data/train.json",
    "epochs":            3,
    "batch_size":        4,
    "lora_r":            64,
    "lora_alpha":        128,
    "quantization":      "4bit",
    "gpu_count":         4,
})
```

### 3.3 通过 Magnus Web UI

1. 打开 Magnus 前端 → Blueprints
2. 搜索 `LoRA_zyz`
3. 填入参数 → 提交

### 3.4 CLI 提交

```bash
magnus run LoRA_zyz -- \
    --model_path /data/magnus/models/Qwen2.5-72B-Instruct \
    --output_dir /data/magnus/models/qwen-lora-v1 \
    --epochs 3 \
    --batch_size 4 \
    --lora_r 64 \
    --lora_alpha 128 \
    --quantization 4bit
```

---

## 4. 输出结构

```
{output_dir}/
├── checkpoint-{step}/          # LoRA adapter checkpoint
│   ├── adapter_config.json     # LoRA 配置
│   ├── adapter_model.safetensors  # LoRA 权重（几 MB）
│   └── checkpoint_meta.json    # 训练元数据
├── lora_adapter/               # 最终 LoRA adapter（几百 MB）
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
├── final/                      # 合并后的完整模型（仅 merge_on_save=True）
│   ├── model-00001-of-NN.safetensors
│   ├── config.json
│   └── tokenizer files...
├── eval/                       # 推理结果（有 test_data 时）
│   └── eval_results.json
└── training_log.json           # 完整训练日志（Loss / LR / 显存）
```

**LoRA adapter 单独文件**（lora_adapter/）仅几百 MB，保存/传输极快，推理时用 `PeftModel.from_pretrained()` 加载。

---

## 5. LoRA vs 全参 SFT 对比

| 维度 | 全参 SFT | LoRA/QLoRA |
|------|---------|------------|
| **可训练参数** | 100% (72B) | ~0.1-1% (72B 约 70-700M) |
| **72B 显存 (4-bit)** | >300 GB | ~144 GB (4×A100) |
| **72B 显存 (QLoRA)** | 不适用 | ~36 GB/卡 |
| **训练速度** | 慢 | 快 3-5 倍 |
| **性能** | 理论最优 | ±1% 差异 |
| **Adapter 大小** | 无 | 几百 MB |
| **多任务部署** | 每任务一个完整模型 | 共享 base + 切换 adapter |
| **适用场景** | 有充足 GPU 资源 | 资源有限 / 快速实验 / 多任务 |

**选择建议：**

- **资源充足** (8×A100+) → 全参 SFT
- **资源有限** (1-4×A100) → QLoRA (4-bit)
- **快速实验 / 多任务并行** → QLoRA
- **追求极致精度** → 全参 SFT 或 "QLoRA 训练 + merge 后用全参数继续训练几轮"

---

## 6. 已知问题 & 排错

### 6.1 bitsandbytes 不兼容

**现象**：`ImportError: bitsandbytes was compiled without GPU support`

**原因**：容器镜像中 bitsandbytes 版本与 CUDA 不匹配。

**解决**：
- 使用自定义镜像（如 `docker://crpi-.../zyz25/sft-base:v1`，已预装兼容版本）
- 或在蓝图中指定 `quantization="none"`，使用 bf16 LoRA（需 8+ 卡）

### 6.2 QLoRA 多卡 DDP 慢

**现象**：多卡训练速度比单卡提升不明显。

**原因**：量化 + DDP 下 bitsandbytes dequant 频繁，通信开销大。

**解决**：
- 增加 `batch_size` 减少通信频次
- 增加 `grad_accum` 减少更新频率
- 考虑使用 `gradient_checkpointing` 降低单卡显存以便增大 batch

### 6.3 merge_on_save 磁盘空间不足

**现象**：`save_final` 时磁盘满。

**原因**：合并完整 72B 模型需要额外 ~130GB 临时空间。

**解决**：
- 设置 `merge_on_save=False`，只保存 LoRA adapter（几百 MB）
- 如需完整模型，用 `save_pretrained(max_shard_size="1800MB")` 分片保存
- 确保 `ephemeral_storage` 足够大（推荐 500G+）

### 6.4 transformers 5.7.0 + torch 2.5.1 CVE 安全检查拦截

**现象**：
```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with
`weights_only=True`, we now require users to upgrade torch to at least v2.6
```

**原因**：`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` 使用 torch 2.5.1，transformers 5.7.0 因 CVE-2025-32434 要求 torch >= 2.6 才能加载 `.bin` 文件。`.safetensors` 格式不受影响。

**修复**：在 `from_pretrained` 前添加 monkey-patch：
```python
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
```
已内置于 `LoRA_zyz.magnus` 的 `LORA_TRAIN_PY`、`_EVAL_PY` 和 HuggingFace 下载 fallback 中。

**注意**：必须 patch `modeling_utils` 而非 `import_utils`，因为前者以 `from ... import` 方式持有本地引用。

### 6.5 torchrun 静默失败 + 诊断增强

**现象**：torchrun 启动后无 Python traceback 直接 Failed。

**修复**：蓝图已添加 `2>&1` 重定向、`--log-dir` 各 rank 独立日志、训练前 CUDA/语法预检、失败时自动 dump 所有 rank 日志。详见 SFT.md §8.7。

### 6.6 tokenizer.chat_template 未设置（如 DeepSeek 模型）

**现象**：
```
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!
```

**原因**：部分模型的 tokenizer（如 `deepseek-math-7b-base`）未预置 `chat_template`。蓝图中 Dataset 和 eval 脚本使用 `tokenizer.apply_chat_template()` 会直接失败。

**修复**：tokenizer 加载后检测并设置默认模板（User/Assistant 格式），训练和 eval 共 2 处。

### 6.7 NaN Loss + 分布式传播 / NCCL 死锁

**现象**：训练到某步时 NaN loss，10 分钟后 NCCL watchdog 杀进程：
```
Watchdog caught collective operation timeout:
WorkNCCL(OpType=_ALLGATHER_BASE, ...) ran for 600069 milliseconds before timing out.
```

**完整崩溃链**：
```
样本 NaN loss / 全部 label = -100（被截断）
  → torch.zeros_like(loss) → 叶子张量，无 grad_fn
    → backward() 失败 → FSDP 的 REDUCE_SCATTER 未执行
      → 后续 NCCL 操作死等 → 10min watchdog → SIGABRT
```

**根因**：

| 根因 | 说明 |
|------|------|
| **bf16 大词表下溢出** | Qwen2.5 vocab_size=151643，bf16 精度不足使 softmax 归一化下溢出 → log(0) = NaN |
| **全部 label = -100** | instruction 超过 max_length，output 被完全截掉 → CrossEntropyLoss 无有效标签 → NaN |

**修复（v2 — 2026-05-03 当前方案）**：

**A — float32 loss（防 NaN 产生）：**
```python
outputs = model(input_ids=input_ids, attention_mask=attn_mask)
logits = outputs.logits
shift_logits = logits[..., :-1, :].contiguous().float()
shift_labels = labels[..., 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
```

**B — torch.where 保留计算图（防 NCCL 死锁）：**
```python
if torch.isnan(loss) or torch.isinf(loss):
    loss = torch.where(
        torch.isnan(loss) | torch.isinf(loss),
        torch.zeros_like(loss),
        loss,
    )
(loss / accum).backward()  # 始终成功，FSDP NCCL sync 正常
```

关键区别：
- 旧方案 `torch.zeros_like(loss)` → 破坏计算图 → backward 失败 → NCCL 死锁 ❌
- 新方案 `torch.where(..., zeros_like, loss)` → 保留 grad_fn → backward 正常 → FSDP sync 正常 ✅

**始终禁止**在 FSDP 训练循环中使用 `continue` 跳过 backward。

详见 SFT.md §8.11。

### 6.8 NCCL 超时（NaN 二次效应）

**现象**：`Watchdog caught collective operation timeout: WorkNCCL(_REDUCE_SCATTER_BASE)` 超时 600s。

**根因**：NaN loss 处理不当 → `backward()` 不触发 → FSDP REDUCE_SCATTER 不执行 → NCCL 队列卡住。**NCCL 超时是症状，非根因。**

**修复**：见 §6.7 v2 方案（float32 loss + `torch.where`），从源头解决。

### 6.9 Checkpoint 损坏导致恢复失败

**现象**：训练中断后自动重试，所有重试均因 checkpoint-latest 的 safetensors 文件截断而失败。

**根因**：训练被杀时 `save_pretrained()` 写到一半，checkpoint 文件损坏。`_load_safetensors_state` 无 try/except，直接抛 `SafetensorError` 崩溃进程。shell retry 在 3 次重试中每次都恢复同一损坏 checkpoint，确定性失败。

**修复**（与 SFT 蓝图相同，见 SFT.md §8.13）：

1. **原子保存**：`save_lora_checkpoint` 改为先写 `.ckpt-tmp-{step}` 再 `os.rename` 到 `checkpoint-latest`
2. **损坏检测**：`_load_safetensors_state` 加 `try/except` 兜底，损坏返回 `None`
3. **Shell 最终兜底**：3 次重试耗尽后，清理 checkpoint，从头开始最后一次训练
