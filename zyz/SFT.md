# SFT 训练工作流 — 项目笔记

> 基于 OpenFundus 仓库，在 Magnus 集群上运行。  
> 核心流程：**预热包 → 下载模型 → 提交 SFT 蓝图 → 绘图分析**

---

## 目录

- [1. 文件总览](#1-文件总览)
- [2. 两个 Magnus 蓝图](#2-两个-magnus-蓝图)
  - [2.1 旧版 `OpenFundSFT.magnus`（已弃用）](#21-旧版-openfundsftmagnus已弃用)
  - [2.2 新版 `OpenFundus_SFT_zyz.magnus`（唯一使用）](#22-新版-openfundus_sft_zyzmagnus唯一使用)
  - [2.3 蓝图对模型文件夹的文件要求](#23-蓝图对模型文件夹的文件要求)
- [3. 辅助 Python 脚本](#3-辅助-python-脚本)
- [4. 持久存储记录](#4-持久存储记录)
- [5. model-version 版本管理](#5-model-version-版本管理)
- [6. Magnus Monitor GUI — 独立 EXE 监控程序](#6-magnus-monitor-gui--独立-exe-监控程序)
- [7. 集群硬件与配置速查](#7-集群硬件与配置速查)
- [8. 已知问题 & 排错](#8-已知问题--排错)
- [9. 推荐工作流](#9-推荐工作流)
- [10. 关键参数与默认值速查](#10-关键参数与默认值速查)

---

## 1. 文件总览

| 文件 | 角色 | 说明 |
|------|------|------|
| `LoRA_zyz.magnus` | **LoRA/QLoRA 蓝图** | 低资源微调 72B，支持 4-bit/8-bit/bf16 |
| `LoRA.md` | **LoRA 笔记** | LoRA 参数说明、量化策略对比、排错 |
| `OpenFundus_SFT.magnus` | 旧版蓝图 | 仅支持 Qwen 模型，硬编码 ChatML 模板 |
| `OpenFundus_SFT_zyz.magnus` | **新版蓝图** | 通用版，兼容任何 ModelScope/HF 模型 |
| `monitor.py` | 监控模块 | 轮询任务状态 + 日志 + `notify_exe()` + 自动保存到 `data1/` |
| `monitor_gui.py` | **Web 监控程序** | 内嵌 HTTP 服务器 + 浏览器界面 + 三层轮询 + `@{}` 元数据 |
| `warmup_packages.py` | 包预热 | pip 依赖下载到 `/data/$USER/pip-cache/wheels/` + 自动排除 torch/nvidia |
| `warmup_test.py` | 持久存储测试 | 提交两个 B2 作业验证 `/data/` NFS 跨容器持久 |
| `download_model_auto.py` | 模型下载 | 从 ModelScope 下载模型到集群持久存储 |
| `magnus_sft.py` | SFT 提交（蓝图版） | 读取 .magnus 蓝图 → 保存 → 启动 → 监控 |
| `run_sft_blueprint.py` | SFT 提交（蓝图版 v2） | 直接注册蓝图 + 一键提交（更简洁） |
| `submit_sft.py` | **SFT 提交（蓝图版，推荐）** | 读取 .magnus → 保存蓝图到服务器 → 等 30s → 启动蓝图 |
| `plot_training.py` | 可视化 | 读取 `training_log.json` 绘制 Loss + LR 曲线 |
| `inspect_storage.py` | 存储检查 | 提交作业扫描 `/data/` 目录结构 + 校验 .safetensors 完整性 |
| `remove_storage.py` | 存储删除 | 提交作业 `rm -rf` 删除指定路径（交互确认 + `-y` 跳过） |
| `hooks/hook-magnus.py` | PyInstaller 钩子 | 打包 magnus-sdk 元数据到 EXE |

### 自动生成目录

| 路径 | 用途 | 后缀 |
|------|------|------|
| `data1/` | 完整日志文件（`monitor.py` + `monitor_gui.py` 快照） | `.data1` |
| `data2/` | 时间线文件 + `storage_record.json` + EXE 配置 `config.json` / `jobs.json` / `incoming/` | `.data2` |
| `SFT_data/` | 存放 submit_sft 下载的训练报告 | — |

**日志文件命名约定**：`{提交时间}-{状态}-{来源缩写}-{任务名}-{序号}`，按提交时间字母序排列（与 GUI 排序一致）。状态缩写：`s`=成功, `f`=失败, `t`=终止, `u`=未知。

**文件元数据头**：每个 `.data1` / `.data2` 文件第一行包含 BibTeX 风格的 `@{}` 头，记录任务全部元数据：

```
@{20260427-102800-s-wp-Warmup-SFT-Packages-001.data1,
  time = 2026-04-27 10:28:00,
  name = Warmup-SFT-Packages,
  submitter = warmup_packages.py,
  job_id = abc123def456,
  type = 日志,
  status = Success,
  source_abbr = wp,
  seq = 001,
  created_at = 2026-04-27T10:28:00,
  updated_at = 2026-04-27T11:00:00,
  gpu_count = N/A,
  gpu_type = N/A,
}
```

- `type` = `日志`（完整日志）或 `时间线`（时间线摘要）
- `data1/` 与 `data2/` 同基本文件名对应同一任务
- 文件命名异常时 EXE 自动读取 `@{}` 头纠正文件名

---

## 2. 两个 Magnus 蓝图

旧版 `OpenFundus_SFT.magnus` 已**完全被** `OpenFundus_SFT_zyz.magnus` 替代，后者参数更全、兼容性更好。

### 2.1 旧版 `OpenFundSFT.magnus`（已弃用）

- **专用型**：仅 Qwen 模型，硬编码 `<|im_start|>` ChatML 模板
- 训练脚本中 prompt 格式固定，无法适配其他模型
- 缺少清华镜像源、Warmup、ContainerImage 等参数
- 输出格式、评估脚本结构与新版一致

### 2.2 新版 `OpenFundus_SFT_zyz.magnus`（唯一使用）

#### zyz 相对旧版的改进

| 维度 | 旧版 | zyz |
|------|------|-----|
| 对话模板 | 硬编码 `<\|im_start\|>` ChatML | `apply_chat_template()` 自动适配所有模型 |
| 镜像源 | 无（默认 PyPI） | 清华源 + PIP_FIND_LINKS + Warmup |
| 并行策略 | 无（单卡） | FSDP SHARD_GRAD_OP（取代 DataParallel） |
| 参数 | 9 个 | 20 个（GPU 类型、CPU/内存/存储、ContainerImage、Resume 等） |
| 日志 | `echo` | `_log()` 带时间戳 |
| 兼容性 | 无 | InternLM2 rope_scaling 修补 |

#### 通用模型兼容
- **apply_chat_template**：使用 `tokenizer.apply_chat_template()` 自动适配各模型对话格式（Qwen、InternLM、DeepSeek、LLaMA 等均支持）
- **rope_scaling 兼容**：修补 transformers>=4.45 的 rope_scaling 格式变化（InternLM2 需要）
- **本地路径 vs Hub ID 智能判断**：以 `/` 开头视为本地路径，否则尝试 ModelScope → HF 自动下载

#### 包安装加速
- **PIP_FIND_LINKS**：`/data/$USERNAME/pip-cache/wheels`，预热作业下载 wheel 到持久目录，蓝图直接使用
- **清华镜像源**：`-i https://pypi.tuna.tsinghua.edu.cn/simple` 作为 fallback
- **单包依赖检查**：逐个检查，仅安装缺失的

#### 多卡并行：FSDP 替代 DataParallel

**背景**：`DataParallel` + `AdamW(fp32)` + 7B 模型 + A100-80GB 必然 OOM：
- 模型权重 (bf16): 14 GB
- 梯度 (bf16): 14 GB
- AdamW exp_avg + exp_avg_sq (fp32): 56 GB
- 合计 ~84.7 GB / 85.1 GB，反向传播无余量

**解决**：改用 FSDP（Fully Sharded Data Parallelism），7B~14B 模型用 `SHARD_GRAD_OP`（梯度分片，省通信），72B+ 用 `FULL_SHARD`（全分片）。

| 组件 | DataParallel | FSDP SHARD_GRAD_OP(7B,2卡) | FSDP FULL_SHARD(72B,3卡) |
|------|-------------|---------------------------|-------------------------|
| 模型权重 | 14 GB（每卡完整） | 14 GB（每卡完整） | **~48 GB**（分片到 3 卡） |
| 梯度 | 14 GB（每卡完整） | **7 GB**（分片） | **~48 GB**（分片） |
| AdamW 状态 | 56 GB（每卡完整） | **28 GB**（分片） | **~192 GB**（分片） |
| **合计** | **~84 GB** | **~49 GB** ✓  | **~288 GB** ✓ |
| 通信量/步 | 无（环等到） | **1 次 reduce-scatter** | 3 次 all-gather + reduce-scatter |
| 适用 | 小模型 | **≤14B 模型（推荐）** | 72B+ 大模型 |

> **通信量权衡**：SHARD_GRAD_OP 仅 backward 时做一次 reduce-scatter（梯度），forward 不需要通信。FULL_SHARD 每步 forward 前 all-gather 权重、backward 后 reduce-scatter 梯度+再次 all-gather，通信量 ~3×。对 A100 PCIe（无 NVLink，走 PCIe switch），减少通信能显著降低 NCCL 超时概率。

**关键改动**（SFT_TRAIN_PY 中）：

```python
# 之前
model = torch.nn.DataParallel(model)
model.to(device)

# 之后（7B~14B）
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,       # 72B+ 换 FULL_SHARD
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    device_id=local_rank,
)
```

选择依据：
- **7B~14B 模型**用 `SHARD_GRAD_OP`：仅梯度分片，forward 无需通信，显存足够（~49GB/2卡）
- **72B+ 大模型**改用 `FULL_SHARD`：权重/梯度/优化器全分片（~288GB/3卡）

配套改动：
- 启动命令从 `python3` 改为 `torchrun --nproc_per_node=N`（多卡时自动切换）
- `DataLoader` 增加 `DistributedSampler`，每卡只处理自己的数据子集
- `evaluate()` 用 `all_reduce` 汇总各 rank 的 loss
- 模型保存（`save_checkpoint` / `save_final`）仅 rank 0 执行，并解包 FSDP
- 梯度裁剪用 FSDP 内置 `model.clip_grad_norm_()`

#### 输出格式一致
- 训练日志自动通过 `magnus custody` 上传
- 最终结果写 `$MAGNUS_RESULT`

### 2.3 蓝图对模型文件夹的文件要求

`OpenFundus_SFT_zyz.magnus` 通过以下方式识别和加载模型：

```python
# 检查文件（第 634 行）
if [ -f "{model_path}/config.json" ]; then
    ACTUAL_MODEL_PATH="{model_path}"

# 加载方式
AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
AutoTokenizer.from_pretrained(path, trust_remote_code=True)
```

模型文件夹必须包含以下**必需文件**：

| 文件 | 用途 | 说明 |
|------|------|------|
| `config.json` | **入口检查** | 蓝图以此判断模型是否存在。**缺失则走 ModelScope 下载** |
| `model-*-of-*.safetensors` | 权重（推荐） | HuggingFace 标准分片格式，推荐 safetensors |
| 或 `pytorch_model.bin` | 权重（兼容） | PyTorch 旧格式 |
| `model.safetensors.index.json` | 分片索引 | 多分片 safetensors 必需 |
| `tokenizer_config.json` | Tokenizer 配置 | 必需 |
| `tokenizer.json` | Tokenizer 数据 | 推荐（tokenizers>=3.0 格式） |
| 或 `tokenizer.model` | SentencePiece | 部分模型（如 LLaMA） |
| 或 `vocab.json` + `merges.txt` | BPE | 部分模型（如 GPT-2） |

**特殊模型额外文件**（`trust_remote_code=True` 时由 transformers 自动加载/下载）：

| 模型 | 额外文件 | 说明 |
|------|----------|------|
| Qwen2.5 | `qwen2.5_tokenization.json` | Qwen 自定义 tokenizer |
| InternLM2 | `modeling_internlm2.py` | 自定义 modeling 代码 |
| DeepSeek-V2 | `modeling_deepseek.py` | MoE 架构 |
| 其他 trust_remote_code 模型 | 对应的 `modeling_*.py` | 使用远程代码的模型 |

> **注意**：`trust_remote_code=True` 的模型，transformers 会在首次加载时自动从 Hub 下载对应的 modeling 文件（如 `modeling_qwen2.py`）到缓存目录，不需要手动准备。但如果集群容器无网络，或模型 Hub 路径变动，则需要手动将这些文件放入模型文件夹。

---

## 3. 辅助 Python 脚本

### 3.1 `monitor.py` — 任务监控模块

**功能**：轮询 Magnus 任务状态和日志，支持成功/失败时自动保存全部输出。

**关键接口**：
```python
Monitor(
    poll_interval=60,         # 轮询间隔（秒）
    source="wp",              # 来源 py 文件缩写，用于日志文件命名
)
monitor.add(job_id)           # 添加要监控的任务
monitor.add_many(*job_ids)    # 添加多个任务
monitor.run()                 # 阻塞直到所有任务完成
```

**状态格式**：`[2026-04-26 19:50:00] [task_name] [Preparing] → [Running]`

**日志保存**（`source` 非 None 时）：
- 所有 print() 输出自动捕获到缓冲区
- 任务结束（正常或 Ctrl+C）自动写入 `data1/{提交时间}-{状态缩写}-{source缩写}-{任务名}-{序列号}.data1`
- 例：`20260427-102800-s-wp-Warmup-SFT-Packages-001.data1`
- 命名与 GUI 排序一致：字母序按提交时间排列（与 GUI 的时间分组一致）
- **序列号**确保同一时刻多个任务的日志不冲突

**日志文件命名格式**：

```
{YYYYMMDD}-{HHMMSS}-{s/f/t/u}-{source缩写}-{SafeTaskName}-{seq:03d}
```

| 段 | 来源 | 说明 |
|----|------|------|
| `YYYYMMDD-HHMMSS` | `get_job().created_at` | 任务提交时间（非保存时间） |
| `s/f/t/u` | 终态 | s=成功, f=失败, t=终止, u=未知 |
| `source缩写` | `auto_source()` | 自动根据调用者文件名映射 |
| `SafeTaskName` | `get_job().task_name` | 任务名，特殊字符替换为 `_` |
| `seq` | 全局递增计数器 | 跨实例递增，确保唯一性 |

**模块级函数**：
```python
record_storage(category, entry)              # 追加持久存储记录
check_model_version_exists(model_version)    # 检查 model-version 是否已存在
```

### 3.2 `warmup_packages.py` — 包预热

**目的**：预下载 20+ 个 SFT 依赖到 `/data/$USERNAME/pip-cache/wheels/`。

**source 缩写**：`wp`

**工作原理**：
1. 逐包执行 `pip download`，下载所有 wheel（含传递依赖）到 `/tmp/pip-wheels/`
2. **`accelerate` 单独处理**：使用 `--no-deps` 下载，避免拖入 `torch` + 20+ 个 nvidia/CUDA 包（~3GB）— 容器镜像 `pytorch:2.5.1-cuda12.4` 已自带
3. 安全网清理：`rm -f` 容器镜像已自带的包（`torch-*` / `nvidia_*` / `nvidia-*` / `triton-*` / `cuda_*` / `cuda-*`）
4. 用 `find -exec cp {}` 拷贝到 `/data/$USERNAME/pip-cache/wheels/`（避免 glob 参数列表溢出）
5. 写入 `.warmup_complete` 标记文件

**幂等机制**：仅当 `$SAVE_DIR/.warmup_complete` 存在时跳过；不存在或不完整则删除旧目录重建。上次 `cp` 失败留下的不完整缓存会被自动清理。

**成功时自动记录**：`data2/storage_record.json` → `pip` 分类

**用法**：
```bash
python train/warmup_packages.py
python train/warmup_packages.py --address http://xxx:3011/ --token sk-xxx
```

### 3.2b `warmup_test.py` — 持久存储连通性测试

**目的**：提交 Write + Read 两个 B2 作业，验证容器退出后 `/data/` NFS 写入是否持久。

**source 缩写**：`wt`

**用法**：
```bash
python train/warmup_test.py
```

### 3.3 `download_model_auto.py` — 模型下载

**目的**：从 ModelScope 下载模型到集群持久存储 `/data/<user>/models/<model_name>/`。

**下载策略**：直接用 `/data/` 做临时目录（避开容器 ephemeral storage 限制），下载完成后扁平化拷贝到目标目录。

**扁平化处理**：ModelScope 下载结构为 `{publisher}/{model_name}/`（如 `Qwen/Qwen2.5-72B-Instruct/`），脚本自动扁平化，把模型文件直接放入 `SAVE_DIR/`。

**source 缩写**：`dma`

**成功时自动记录**：`data2/storage_record.json` → `modelscope` 分类

**用法**：
```bash
python train/download_model_auto.py --model Qwen/Qwen2.5-7B
python train/download_model_auto.py --model Qwen/Qwen2.5-72B-Instruct
```

### 3.4 `magnus_sft.py` — SFT 训练提交（蓝图版）

**目的**：读取 .magnus 蓝图文件 → 保存到 Magnus → `launch_blueprint()` → 监控 → 后处理。

**source 缩写**：`ms`

**功能流程**（与旧版 `submit_sft.py` 相同）：
1. **版本检查**：自动生成 model-version，检查是否已存在
2. **提交**：`save_blueprint()` → `launch_blueprint()`
3. **监控**：`Monitor(source="ms")`
4. **后处理**：成功后下载报告到 `SFT_data/`，记录 model-version

**用法**：
```bash
python train/magnus_sft.py --model /data/magnus/models/Qwen2.5-1.5B
python train/magnus_sft.py --model /data/magnus/models/Qwen2.5-1.5B --model-version Qwen2.5-1.5B-v3
```

### 3.4b `run_sft_blueprint.py` — SFT 训练提交（蓝图版 v2，更简洁）

**目的**：直接注册蓝图 + 一键提交，比 `magnus_sft.py` 参数更简洁。

**source 缩写**：`rsb`（本地文件，不自动加入 SOURCE_ABBR）

**用法**：
```bash
python train/run_sft_blueprint.py --model Qwen/Qwen2.5-7B-Instruct --gpus 3
```

### 3.5 `submit_sft.py` — SFT 训练提交（蓝图版，推荐）

**目的**：读取本地 `.magnus` 蓝图 → 保存到 Magnus 服务器（长期/公共）→ 等 30s → 启动蓝图任务 → 监控。

**工作流**：
```
[0/5] model-version 去重检查
[1/5] 配置 Magnus 连接
[2/5] save_blueprint() → 保存/更新服务器蓝图
[3/5] time.sleep(30) → 等待服务器同步
[4/5] launch_blueprint() → 提交任务，获取 job_id
[5/5] Monitor → 后处理（下载报告 + 记录版本）
```

**配置驱动**：所有参数写在文件顶部配置区，修改后直接运行。

```python
# ═══ 配置区（修改此处后直接运行 python submit_sft.py）═══
BLUEPRINT_FILE  = "OpenFundus_SFT_zyz.magnus"
MODEL_PATH      = "/data/magnus/models/Qwen2.5-72B-Instruct"
MODEL_VERSION   = None        # None = 自动递增
TRAIN_DATA      = None        # None = 假数据
GPU_COUNT       = 2
PRIORITY        = "A2"
...
```

**source 缩写**：`ss`

**参数映射**：配置区变量自动映射为蓝图参数名（`MODEL_PATH` → `model_path`，`EPOCHS` → `epochs` 等）。

**与旧版区别**：

| 维度 | 旧版 | 新版 |
|------|------|------|
| 提交方式 | `submit_job()` + 内嵌脚本 | `save_blueprint()` + `launch_blueprint()` |
| 训练脚本位置 | 内嵌在 Python 中 (700+ 行) | 在 `.magnus` 服务器蓝图里 |
| 蓝图可见性 | 无 | 服务器长期保存，其他用户可调用 |
| 参数方式 | 文件顶部配置区 | 配置区 → 自动映射蓝图参数 |

**用法**：
```bash
python train/submit_sft.py                          # 使用配置区参数
python train/submit_sft.py --address http://...     # 仅覆盖连接参数
```

### 3.6 `inspect_storage.py` — 存储目录检查

**目的**：提交一个轻量检查作业，扫描 `/data/$USER/pip-cache/` 和 `/data/$USER/models/` 的目录结构。

**source 缩写**：`is`

**检查内容**：
- pip 缓存：wheel 数量、总大小、`.warmup_complete` 标记状态
- 模型目录：每个模型的 `.safetensors` 文件列表、文件数、大小
- **safetensors header 校验**：逐个字节检查确保未损坏

**用法**：
```bash
python train/inspect_storage.py
```

### 3.7 `remove_storage.py` — 删除长期存储

**目的**：删除 `/data/` 下的文件或目录。

**原理**：提交一个 B2 作业，通过 `system_entry_command` 挂载 `/data/`，在容器内执行 `rm -rf`。

**source 缩写**：`rs`

**防护**：默认交互确认 + `-y` 跳过确认 + 仅允许 `/data/` 前缀路径。

**用法**：
```bash
python train/remove_storage.py /data/magnus/models/old-model
python train/remove_storage.py /data/magnus/models/Qwen2.5-72B-Instruct -y
```

### 3.8 `plot_training.py` — 训练曲线可视化

**目的**：读取 `training_log.json`，绘制三张图（Loss 曲线、Eval 对比、LR 调度）。

**用法**：
```bash
python train/plot_training.py                    # 默认路径
python train/plot_training.py /path/to/training_log.json
```

---

## 4. 持久存储记录

文件：`data2/storage_record.json`

自动记录成功的服务器长期存储操作，按分类归档：

```json
{
    "pip": [
        {"time": "2026-04-26T19:50:00", "target": "/data/<用户名>/pip-cache/wheels", "packages": 42, "status": "success"}
    ],
    "modelscope": [
        {"time": "2026-04-26T19:52:00", "model": "deepseek-ai/deepseek-math-7b-base", "target": "/data/<user>/models/deepseek-math-7b-base", "status": "success"}
    ],
    "model-version": [
        {"time": "2026-04-26T20:00:00", "model": "Qwen2.5-7B-v1", "local_path": "/data/magnus/models/general-sft-v1", "status": "success"}
    ]
}
```

| 分类 | 记录者 | 触发条件 |
|------|--------|----------|
| `pip` | `warmup_packages.py` | 预热作业 Success |
| `modelscope` | `download_model_auto.py` | 下载作业 Success |
| `model-version` | `submit_sft.py` | 训练作业 Success |

---

## 5. model-version 版本管理

### 命名规则

格式：`{模型短名}-v{版本号}`

- `Qwen2.5-7B-v1`, `Qwen2.5-7B-v2`, ...
- 模型短名从 `--model` 路径自动提取（最后一个 `/` 后的部分）

### 自动递增

不指定 `--model-version` 时，自动查找 `storage_record.json` 中该模型已有的最高版本号，+1。

### 去重保护

`submit_sft.py` 提交前检查 `check_model_version_exists()`：
- 如果版本已存在 → 拒绝提交，提示用户指定新版本
- 通过 `--model-version` 可覆盖自动生成的值

### 报告下载

训练成功时：
1. 调用 `magnus.get_job_result(job_id)` 获取结果
2. 如果是 custody secret → `magnus.download_file()` 下载
3. 否者当作文本直接保存
4. 保存到 `SFT_data/{model-version}`

---

## 6. Magnus Monitor GUI — 独立 EXE 监控程序

### 6.1 概述

`monitor_gui.py` 是独立于 Python 解释器的 GUI 监控程序，可用 PyInstaller 打包为 `MagnusMonitor.exe`。
Python 脚本提交 job 后通过 HTTP 通知 EXE，EXE 独立轮询 Magnus API、保存日志、提供可视化界面。

**核心特性**：
- **三层轮询**：快速状态（15s）+ 全量日志（60s）+ 自动发现（60s）
- **双通道通知**：HTTP POST → `localhost:9876`，失败时写入 `data2/incoming/*.job.json`
- **自动发现**：每分钟调用 `magnus.list_jobs(limit=50)` 发现遗漏任务
- **去重保护**：按 `job_id` 唯一键，HTTP 通知 / 自动发现 / 文件导入共用同一空间
- **日志保证**：只要 EXE 在运行，每 60s 保存一次日志，与 Python 脚本生命周期无关
- **后台运行**：关闭窗口默认最小化到后台（可在设置中关闭），再次运行 EXE 自动恢复窗口
- **开机自启**：可在设置中勾选，写入 Windows 注册表 `Run` 键

### 6.2 架构

```
Python 脚本 → HTTP POST (localhost:9876) → Magnus Monitor EXE
  Fallback: data2/incoming/*.job.json        │
                                            ├── HTTP Server (thread, :9876)
                                            ├── Polling Engine
                                            │   ├── 快速状态 15s: get_job()
                                            │   ├── 全量日志 60s: get_job_logs()
                                            │   └── 自动发现 60s: list_jobs()
                                            └── GUI (tkinter)
                                                ├── 左栏: 任务列表 (状态颜色)
                                                ├── 上栏: 操作按钮
                                                ├── 主区: 日志显示
                                                └── 设置: 地址/Token
```

### 6.3 文件

| 文件 | 说明 |
|------|------|
| `train/monitor_gui.py` | 主程序（PyInstaller 入口） |
| `train/data2/config.json` | EXE 设置（地址/Token/轮询/最小化/开机自启） |
| `train/data2/jobs.json` | 任务注册表 |
| `train/data2/incoming/*.job.json` | HTTP 通知失败的 fallback |
| `train/data1/` | 完整日志（`.data1`）+ 终态快照 |
| `train/data2/` | 时间线（`.data2`）+ 快照同名文件 |

**日志快照命名**（任务进入终态时自动保存到 `data1/` + `data2/`）：
```
{YYYYMMDD-HHMMSS}-{s/f/t}-{submitter缩写}-{TaskName}-{seq:03d}.{data1|data2}
```
例：`20260427-102800-s-ss-FakeTest-Qwen2_5-72B-001.data1`

### 6.5 核心功能

- **关闭最小化**：关闭窗口默认最小化到后台（不退出）；可在设置中关闭此行为
- **开机自启**：勾选设置后自动写入 Windows 注册表 `Run` 键
- **单实例**：再次运行 EXE 自动恢复已隐藏的窗口
- **文件标准化**：启动时自动扫描 `data1/`/`data2/`，非标准命名文件通过 `@{}` 头纠正
- **快照保护**：每个任务仅保存一次终态快照，防止重复写入

### 6.6 用法

```bash
# 开发模式
python monitor_gui.py

# 构建 EXE
pip install pyinstaller
pyinstaller --noconsole --onefile --name MagnusMonitor monitor_gui.py
# 输出: dist/MagnusMonitor.exe，拷贝到 train/ 目录运行
```

### 6.7 界面

```
┌──────────────────────────────────────────────────────────────────┐
│  Magnus Job Monitor                                   ⚙ — □ ×   │
├──────────────────────────────────────────────────────────────────┤
│  [■ 终止任务]  [🌐 打开]  [⟳ 刷新]  [✕ 清除已完成]               │
├────────────────────────┬─────────────────────────────────────────┤
│  ● 10:00              │  [2026-04-27 10:00] [SFT-Qwen] [Run]   │
│    FakeTest-Qwen2.5.. │  --- 新增日志 ---                        │
│    submit_sft.py      │  [2026-04-27 10:00] 正在下载模型...      │
│    Running            │  [2026-04-27 10:01] 模型加载完成         │
│                       │                                         │
│  ● 09:30              │          选中任务日志显示区               │
│    Warmup-Pip-Cache   │                                         │
│    ⚡ magnus          │                                         │
│    Success            │                                         │
├────────────────────────┴─────────────────────────────────────────┤
│  共 5 个任务 | 2 个活跃 | 上次更新: 10:01:00                     │
└──────────────────────────────────────────────────────────────────┘
```

### 6.8 通知机制

所有提交脚本已在 `submit_job()` / `launch_blueprint()` 成功后自动调用 `notify_exe(job_id=job_id)`。

`notify_exe()` 位于 `monitor.py`，自动推断：
- **submitter**：通过 `inspect` 调用栈检测调用者文件名
- **address/token**：使用模块默认值
- **task_name**：可选，不传时 EXE 首次轮询自动从 `get_job()` 补全

EXE 不在线时自动降级：写入 `data2/incoming/*.job.json`，EXE 启动时扫描导入。

### 6.9 设置

点击工具栏 ⚙ 按钮 → 设置对话框：
- **Magnus 地址**：服务器 URL（默认 `http://162.105.151.134:3011/`）
- **API Token**：认证令牌
- **轮询间隔**：默认 60 秒
- **启用自动发现**：自动扫描 Magnus 服务器上最近的 50 个任务
- **关闭时最小化到后台**：点击 [X] 不退出，隐藏到系统托盘继续运行
- **开机自启**：写入 Windows 注册表 `HKCU\Software\Microsoft\Windows\CurrentVersion\Run\MagnusMonitor`

---

## 7. 集群硬件与配置速查

> 来源：`AI.md` + `magnus-main/configs/magnus_config.yaml.example`

### 7.1 硬件

| 资源 | 上限 | 默认值 |
|------|------|--------|
| CPU | 128 核 | 4 核 |
| 内存 | 256 GB | 1.6 GB |
| 临时磁盘 | — | 10 GB |
| GPU (RTX 5090 32GB) | 单任务最多 4 卡 | 0（纯 CPU） |
| 默认容器镜像 | — | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` |

### 7.2 优先级

| JobType | 数值 | 可抢占 | 适用 |
|---------|------|--------|------|
| A1 | 4 | 否 | 生产/紧急 |
| A2 | 3 | 否 | 常规训练（推荐） |
| B1 | 2 | 是 | 批量任务 |
| B2 | 1 | 是 | 下载/预热/测试 |

### 7.3 容器环境变量

| 变量 | 说明 |
|------|------|
| `MAGNUS_TOKEN` | SDK 自动认证 |
| `MAGNUS_ADDRESS` | 后端 API 地址 |
| `MAGNUS_JOB_ID` | 当前作业 ID |
| `MAGNUS_HOME` | 容器根路径 `/magnus` |
| `MAGNUS_RESULT` | 写此文件 → 返回结果给调度器 |
| `MAGNUS_ACTION` | 写此文件 → 客户端执行命令（如 custody 上传） |
| `MAGNUS_METRICS_DIR` | 指标文件目录 `metrics/` |

### 7.4 Job 状态机

| 状态 | 说明 | 终态 |
|------|------|------|
| `Pending` | 排队中 | 否 |
| `Preparing` | 拉镜像 + 克隆仓库 | 否 |
| `Running` | 运行中 | 否 |
| `Paused` | 被抢占暂停 | 否 |
| `Success` | 成功 | **是** |
| `Failed` | 失败（含 OOM） | **是** |
| `Terminated` | 用户取消 | **是** |

---

## 8. 已知问题 & 排错

### 8.1 Warmup 磁盘空间不足

**现象**：`cp: error writing '...whl': No space left on device`

**原因**：`accelerate` 的依赖 `torch` 引入 20+ 个 nvidia/CUDA 包（>3GB），撑满 `/tmp/` 或 `/data/`。

**修复**（`warmup_packages.py`）：
1. **核心修复**：`accelerate` 从主包列表移除，改为单独 `pip download --no-deps`（容器镜像已自带 torch/CUDA）
2. 安全网：拷贝前 `rm -f` 所有 `torch-*` / `nvidia_*` / `nvidia-*` / `triton-*` / `cuda_*` / `cuda-*`
3. 拷贝改用 `find -exec cp {}` 避免 glob 参数列表溢出
4. 跳过逻辑改用 `.warmup_complete` 标记文件（而非目录是否存在）
5. 无 `.warmup_complete` 则先 `rm -rf` 旧目录再重建

### 8.2 EXE 刷新空白 / 强制刷新卡死

**现象**：点击"刷新"后日志区空白，无 `data1/data2` 文件创建；"强制刷新"导致 GUI 卡死。

**原因**：
- `__file__` 在 PyInstaller onefile 中指向临时目录，`data1/data2` 创建在错误位置
- `poll_logs()` 在主线程同步 HTTP 调用，Magnus 响应慢时阻塞 GUI

**修复**：`monitor_gui.py` 已通过 `sys.executable` 定位正确路径 + 刷新操作移入后台线程。重新构建 EXE 即可。

### 8.3 时间线只有第一行

**现象**：时间线文件只有一行，丢失其余日志。

**原因**：`_save_timeline()` 只取了 `new_part.splitlines()[0]`。

**修复**：已改为遍历所有行，每行带 `[时间][状态]` 前缀。见 `monitor_gui.py:_save_timeline`。

### 8.4 完整日志文件只存了增量 diff

**现象**：`.data1` 文件只包含最近一次新增内容，不是完整日志。

**原因**：传给 `_save_full_log()` 的是 `new_part or text`，`new_part` 有值时传入的是增量。

**修复**：改为始终传入完整 `text`。见 `monitor_gui.py:poll_logs`。

---

### 8.5 容器内无法访问 `/data/`（平台更新）

**现象**：作业日志出现 `No such file or directory` 或 `cp: cannot create regular file '/data/...': No such file or directory`，但 `/data/` 在宿主机上存在。

**原因**：Magnus 平台更新后不再默认通过 Apptainer bind-mount 挂载 `/data/` 到容器内。容器的文件系统是隔离的，看不到宿主机 `/data/`。

**修复**：所有 `submit_job()` 调用添加 `system_entry_command` 参数，显式声明挂载：

```python
from monitor import SYSTEM_ENTRY_COMMAND

magnus.submit_job(
    ...
    system_entry_command = SYSTEM_ENTRY_COMMAND,
)
```

`SYSTEM_ENTRY_COMMAND`（定义在 `monitor.py`）内容：
```bash
mounts=(
    "/home:/home"
    "/data:/data"
)
export APPTAINER_BIND=$(IFS=,; echo "${mounts[*]}")
export MAGNUS_HOME=/magnus
unset -f nvidia-smi
unset VIRTUAL_ENV SSL_CERT_FILE
```

`APPTAINER_BIND` 是 Apptainer 的环境变量，格式为 `"宿主机路径:容器路径"`（逗号分隔）。此脚本在**宿主机上、容器启动前**执行。

`.magnus` 蓝图文件也已同步添加 `system_entry_command` 到 `submit_job()` 调用中。

### 8.6 大模型下载空间不足

**现象**：`[Errno 28] No space left on device: '/tmp/model_download/...'`，18 个文件下载失败。

**原因**：大模型（如 72B）超过 130GB，ModelScope 下载到 `/tmp/`（容器 ephemeral storage），120G 不够用。

**修复**（`download_model_auto.py`）：下载临时目录改为 `/data/$USERNAME/models/.dl_tmp`，直接走 NFS 持久存储，不受 ephemeral storage 限制。下载完成后 `rm -rf` 清理临时目录。

### 8.7 torchrun 训练静默失败（stderr 未捕获）

**现象**：日志显示 `torchrun` OMP 警告后直接 Failed，没有 Python traceback，无法定位崩溃点。

**原因**：torchrun 子进程的 Python 异常写入 stderr，而 Magnus 日志系统主要收集 stdout。`torchrun` 默认不保存各 rank 日志文件。

**修复**（`OpenFundus_SFT_zyz.magnus` + `LoRA_zyz.magnus`）：
- torchrun 命令行末尾添加 `2>&1`，将 stderr 合并到 stdout
- 添加 `--log-dir /tmp/torchrun_logs_$$`，每个 rank 保存独立日志
- 训练前添加诊断打印（`$CUDA_VISIBLE_DEVICES`、torch/cuda 版本、Python 脚本语法预检、import 预检）
- 失败时自动遍历并打印所有 rank 日志文件
- 打印 `$TRAIN_EXIT_CODE` 判断信号终止 vs Python 异常
- 增加 NCCL 调试日志（`NCCL_DEBUG=INFO`），通信 hang 时可直接定位卡住的 rank

### 8.8 transformers 5.7.0 + torch 2.5.1 CVE 安全检查拦截

**现象**：
```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with
`weights_only=True`, we now require users to upgrade torch to at least v2.6
in order to use the function. This version restriction does not apply when
loading files with safetensors.
```

**原因**：`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` 镜像含 torch 2.5.1，但 `transformers 5.7.0` 因 CVE-2025-32434 安全漏洞，禁止 torch < 2.6 加载 `.bin` 格式权重文件。`deepseek-math-7b-base` 模型恰好是 `.bin` 格式（不是 `.safetensors`），触发拦截。

**修复**（`SFT_TRAIN_PY`、`_EVAL_PY`、HuggingFace 下载 fallback）：
在 `from_pretrained` 之前 patch `modeling_utils` 中的安全函数：

```python
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
```

**关键细节**：不能 patch `import_utils.check_torch_load_is_safe`，因为 `modeling_utils.py` 以 `from import_utils import check_torch_load_is_safe` 方式导入，本地引用不受 `import_utils` 的 monkey-patch 影响。必须直接 patch `modeling_utils.check_torch_load_is_safe`。

**长期方案**：重建 Docker 镜像，使用 torch >= 2.6 的 base image，或将模型转为 `.safetensors` 格式。

### 8.9 bash 语法错误：`_log` 字符串缺少结尾引号

**现象**：
```
.magnus_user_script.sh: line 178: syntax error near unexpected token `('
```

**原因**：蓝图 entry_command 中 `_log` 命令的闭合双引号误删：
```python
# 错误：缺少结尾 "
f'\n_log "=== [4/5] 开始 SFT 训练 ===\n'
# 正确：
f'\n_log "=== [4/5] 开始 SFT 训练 ==="\n'
```

bash 将未闭合字符串后的 `(` 字符（例如诊断代码中的 `CUDA_VISIBLE_DEVICES=${...}`）解析为语法错误。

**修复**：确保所有 `_log` 字符串以 `"` 正确闭合。该 bug 已在两个蓝图中修复。

### 8.10 tokenizer.chat_template 未设置（如 DeepSeek 模型）

**现象**：
```
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set
and no template argument was passed! For information about writing templates and setting
the tokenizer.chat_template attribute, please see the documentation at
https://huggingface.co/docs/transformers/main/en/chat_templating
```

**原因**：部分模型的 tokenizer（如 `deepseek-math-7b-base`）未预置 `chat_template`。蓝图中的 `SFTDataset.__getitem__` 和 eval 脚本使用 `tokenizer.apply_chat_template()` 会直接失败。

**修复**：在 tokenizer 加载后检测并设置默认模板：
```python
if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ 'User: ' + message['content'] + '\n\nAssistant: ' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + '\n\n' }}"
        "{% endif %}"
        "{% endfor %}"
    )
```
已应用于 SFT 和 LoRA 蓝图的训练脚本及 eval 脚本（共 4 处）。

### 8.11 NaN Loss + FSDP 跨卡污染

**现象**（2 GPU FSDP 训练 deepseek-math-7b-base）：
- rank 0 从 step 20 开始持续 NaN loss，rank 1 初始正常
- step 140 后两个 rank 全部 NaN
- loss 曲线：`0.70 → 0.61 → 0.60 → NaN → NaN ...`

**原因**：
1. 训练数据中部分样本 `output` 为空 → `labels` 全部为 -100 → CrossEntropyLoss 返回 NaN
2. `batch_size=1` 时每张卡独立处理一个样本，坏样本只出现在一个 rank
3. `loss.backward()` 将 NaN 梯度写入模型参数 → FSDP 下一次 allreduce 时 NaN 传播到所有 rank
4. 一旦模型参数含 NaN，所有后续 batch 的 loss 全部 NaN

**修复**：在 `backward()` 前检测 NaN/Inf，跳过坏 batch：
```python
if torch.isnan(loss) or torch.isinf(loss):
    nan_count += 1
    if local_rank == 0 and nan_count <= 5:
        log(f"  [警告] NaN/Inf loss @ step {step}, 跳过此 batch (#{nan_count})")
    continue
```
已应用于 `OpenFundus_SFT_zyz.magnus` 和 `LoRA_zyz.magnus`。

### 8.12 NCCL 超时

存在两种场景：

**场景 A：NaN 导致 NCCL 超时**（NaN 时序图）

```
Loss: 0.70 → 0.61 → 0.60 → NaN → NaN → ...
[Watchdog] WorkNCCL OpType=_REDUCE_SCATTER_BASE 超时 (600s)
```

**原因**：NaN 梯度传播后 FSDP reduce_scatter 操作挂住，身份是 NaN 的"二次效应"。

**修复**：修复 NaN loss 传播（见 §8.11）即可消除此类 NCCL 超时。

---

**场景 B：纯通信挂起**（Loss 一直正常）

```
Loss: 1.37 → 1.08 → 0.98 → 0.88 → 0.82 → ...（全程正常）
[Rank 1] Watchdog caught collective operation timeout (600s)
```

**现象**：Qwen2.5-Math-7B-Instruct 训练，Loss 正常收敛，但 `_REDUCE_SCATTER_BASE` 在持续运行 25 分钟后突然超时（A100 PCIe，2卡，无 NVLink）。

**原因**：
1. A100 PCIe 版没有 NVLink，GPU 间通过 PCIe switch 通信
2. `FULL_SHARD` 策略每步通信量大（3 次 collective），在 PCIe 拓扑下容易 hang
3. CUDA 12.4 + PyTorch 2.5.1 + Driver 570 组合有已知 NCCL 兼容问题

**修复**（已应用于 `OpenFundus_SFT_zyz.magnus`）：

```bash
# 1. 增加 heartbeat 超时（默认 600s → 1800s），避免误杀
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

# 2. 禁止 P2P 直连，降级到共享内存/copy（A100 PCIe 兼容性）
export NCCL_P2P_DISABLE=1

# 3. 开启 NCCL 调试日志（下次出问题时可定位确切 rank/操作）
export NCCL_DEBUG=INFO

# 4. 对 ≤14B 模型换用 SHARD_GRAD_OP（通信量减少 ~66%）
#    SFT_TRAIN_PY 中: ShardingStrategy.SHARD_GRAD_OP 代替 FULL_SHARD
```

---

## 9. 推荐工作流

### 首次使用

```bash
# Step 1: 预热 pip 包（只需一次）
python train/warmup_packages.py

# Step 2: 验证存储
python train/warmup_test.py

# Step 3: 下载模型（每个模型只需一次）
python train/download_model_auto.py --model Qwen/Qwen2.5-72B-Instruct

# Step 4: 检查存储完整性
python train/inspect_storage.py

# Step 5: 提交 SFT 训练（修改 submit_sft.py 配置区后直接运行）
python train/submit_sft.py

# Step 5b: 或用 LoRA/QLoRA 低资源微调（详见 train/LoRA.md）
#          blueprint: LoRA_zyz.magnus，支持 4-bit/8-bit/bf16

# Step 6: 查看训练曲线
python train/plot_training.py ./SFT_data/Qwen2.5-72B-Instruct-v1
```

### 更新模型版本

```bash
# 修改 submit_sft.py 配置区 MODEL_PATH 和 OUTPUT_DIR 后
python train/submit_sft.py
```

---

## 10. 关键参数与默认值速查

### `OpenFundus_SFT_zyz.magnus` 蓝图参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_path` | 必填 | 本地路径或 Hub ID |
| ... | ... | （同之前版本） |

### `monitor.py` 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `poll_interval` | 60 | 轮询间隔（秒） |
| `source` | None | 文件缩写，用于日志命名 |

### `submit_sft.py` 参数（与 `magnus_sft.py` 一致）

| 参数 | 说明 |
|------|------|
| `--model-version` | 模型版本名（默认自动递增） |

### `warmup_packages.py` / `download_model_auto.py`

| 参数 | 默认值 |
|------|--------|
| `--address` | `http://162.105.151.134:3011/` |
| `--token` | 需手动修改 |

---

*基于 magnus-main 源码分析、OpenFundus 蓝图代码及集群运行经验整理。*
