# Magnus 训练工具 — 完整手册

> 适用于 Magnus GPU 集群的 SFT 训练工作流。
> 本文面向**所有部门同事**：前半部分为快速上手，后半部分为技术细节。

---

## 目录

- [快速开始（5 分钟上手）](#快速开始5-分钟上手)
- [文件总览](#文件总览)
  - [Markdown 笔记 (6个)](#markdown-笔记-6个)
  - [Python 程序 (12个)](#python-程序-12个)
  - [自动生成目录](#自动生成目录)
- [各文件详细说明](#各文件详细说明)
  - [笔记文件](#笔记文件)
  - [环境准备脚本](#环境准备脚本)
  - [存储管理脚本](#存储管理脚本)
  - [提交训练脚本](#提交训练脚本)
  - [监控与分析](#监控与分析)
- [SFT 蓝图对比](#sft-蓝图对比)
- [任务监控](#任务监控)
- [已知问题 & 排错](#已知问题--排错)
- [集群硬件与配置速查](#集群硬件与配置速查)
- [常用命令速查](#常用命令速查)

---

## 快速开始（5 分钟上手）

> 以下所有命令在 `train/` 目录下运行。

```bash
# 1. 编辑 submit_sft.py 顶部配置区（MODE, MODEL_PATH, GPU_COUNT, CPU_OFFLOAD 等）
# 2. 提交训练
python submit_sft.py
```

**关键配置项**：
- `MODE`: `"sft"` 全参微调 / `"lora"` LoRA
- `CPU_OFFLOAD = True` — 72B 模型 2 卡必备（优化器状态移至 CPU）
- `BWD_PREFETCH = "post"` — 进一步节省显存

---

## 文件总览

### Markdown 笔记 (9个)

| 文件 | 内容 |
|------|------|
| `blue.md` | Blueprint 引擎技术文档：沙箱执行、参数映射、类型转换、entry_command 构造、submit_job 参数速查 |
| `docker.md` | 容器镜像管理：URI 格式、HPC/Local 模式、Web UI/SDK/API 操作、自定义镜像 Dockerfile、关键陷阱 |
| `AI.md` | Magnus 平台深度分析（基于 magnus-main 源码）：完整交互链路、Metrics Protocol v1、CPU/GPU 指标语义变更 |
| `Magnus Platform 系统代码集成指南.md` | 非技术人员指南：API 调用示例（8 个场景）、完整 API 速查表 |
| `README.md` | 本文档 |
| `SFT.md` | SFT 工作流项目笔记：文件总览、蓝图详解、内存分析、已知问题、推荐工作流 |

### Python 程序 (8个)

#### 提交训练

| 脚本 | 功能 | 命令 |
|------|------|------|
| `submit_sft.py` | **推荐使用**。读取 .magnus 蓝图，提交 SFT 训练任务 | 编辑文件顶部配置区后 `python submit_sft.py` |
| `magnus_sft.py` | 同上，但通过命令行参数指定所有配置 | `python magnus_sft.py --model /data/.../Qwen2.5-7B --epochs 5` |
| `run_sft_blueprint.py` | 直接注册蓝图 + 一键提交（更简洁） | `python run_sft_blueprint.py --model Qwen/Qwen2.5-7B-Instruct --gpus 3` |

#### 评估与分析

| 脚本 | 功能 | 命令 |
|------|------|------|
| `auto_grade.py` | LLM 批改：用 QLoRA 72B 逐条批改 eval_results | `python auto_grade.py` |
| `eval_baseline.py` | **自动化评估启动器** | 扫描 Magnus 服务器模型 → 选择 → 提交评估 → 输出 file secret | `python eval_baseline.py` |
| `serve_model.py` | 在 Magnus 上启动 OpenAI 兼容 API 推理服务 | `python serve_model.py` |

#### 核心模块

| 脚本 | 功能 |
|------|------|
| `config.py` | **共享配置模块**：加载 secret.json、Magnus 连接、`wait_for_job()` 监控、`record_storage()`、`notify_exe()` |
| `sft_train.py` | **独立训练/评估脚本**：FSDP FULL_SHARD + CPU Offload + NCCL 优化 + CUDATimer 性能指标 |

### 自动生成目录

| 路径 | 用途 |
|------|------|
| `data2/` | `storage_record.json`（模型版本记录） |
| `SFT_data/` | 存放 submit_sft 下载的训练报告 |

---

## 各文件详细说明

### 笔记文件

#### `blue.md` — Blueprint 实现逻辑 & 入口指令笔记

**沙箱执行机制**：

```python
# 核心: 受限 builtins + 劫持 submit_job
def _hijacked_submit_job(**kwargs):
    raise _BlueprintCapture(kwargs)  # 不真提交，抛出异常捕获 payload

# 执行流程:
# 用户代码 → exec(blueprint_code, safe_builtins + execution_globals)
#   → blueprint(**validated_args) 被调用
#   → submit_job(...) 触发 _BlueprintCapture(kwargs)
#   → 捕获 payload → 构造 JobSubmission → 返回给调度器
```

**沙箱限制**：仅 `typing` 可 import，无 `open/read/write/sys/os/subprocess/eval/exec`

**参数类型 → 表单映射**：

| 类型 | 表单类型 | 支持选项 |
|------|----------|----------|
| `str` | text | allow_empty, placeholder, multi_line |
| `int` | number | min, max |
| `float` | float | min, max, placeholder |
| `bool` | boolean | — |
| `Literal["a","b"]` | select | options dict |
| `FileSecret` | file_secret | allow_empty 始终 False |
| `Optional[T]` | 可选 | 前端显示启用/禁用开关 |
| `List[T]` | 动态列表 | 前端添加/删除项 |

**entry_command 构造模式**：

```python
# 模式 A: 直接 CLI
entry_command = f"python train.py --lr {lr} --epochs {epochs}"

# 模式 B: 多行脚本
entry_command = f"""set -e
cd /project
uv sync --quiet
uv run python main.py --input {data_path}
"""

# 模式 C: 带条件参数
cmd = f"python train.py --data {data}"
if lr is not None:
    cmd += f" --lr {lr}"

# Shell 注入防御
def safe_quote(s: str) -> str:
    return f"'{str(s).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"
```

**容器内环境变量（自动注入）**：

| 变量 | 说明 |
|------|------|
| `MAGNUS_HOME` | 容器根路径 `/magnus` |
| `MAGNUS_TOKEN` | SDK 自动识别 |
| `MAGNUS_ADDRESS` | 后端 API 地址 |
| `MAGNUS_JOB_ID` | 当前 Job ID |
| `MAGNUS_RESULT` | 写此文件 = 返回结果给调度器 |
| `MAGNUS_ACTION` | 写此文件 = 客户端执行命令 |
| `MAGNUS_METRICS_DIR` | 指标文件目录 `metrics/` |

**submit_job 参数速查**：

```python
submit_job(
    task_name=...,          # str, 必填
    entry_command=...,      # str, 必填
    repo_name=...,          # str, 必填
    namespace="Rise-AGI",   # GitHub 组织
    branch=None,            # 自动检测默认分支
    gpu_type="cpu",         # GPU 型号, "cpu"=不用GPU
    gpu_count=0,            # GPU 数量
    cpu_count=None,         # 默认 4
    memory_demand=None,     # "32G", "1600M"
    ephemeral_storage=None, # 默认 10G
    job_type=JobType.A2,    # A1 > A2 > B1 > B2
    container_image=None,   # 默认 pytorch:2.5.1
    description=None,       # Markdown 格式
    system_entry_command=None,  # 宿主机侧预执行脚本
)
```

#### `docker.md` — Magnus 容器镜像管理 & 构建笔记

**镜像 URI 格式**：`docker://registry/name:tag`

**HPC 模式（Apptainer）**：将 docker:// 镜像拉取为只读 squashfs 文件 (.sif)，存放在 SIF 缓存目录，文件名规则 `registry_name_tag.sif`。容器缓存上限 80 GB，LRU 淘汰。原子写入 `.sif.tmp` → `chmod 644` → `rename` → `.sif`。

**镜像状态机**：
```
unregistered → pulling → cached → refreshing → failed → missing
```

| 状态 | 含义 | 颜色 |
|------|------|------|
| `cached` | .sif 文件就绪，可用 | 绿 |
| `pulling` | 正在拉取 | 蓝 |
| `refreshing` | 正在刷新(后台拉新文件替换) | 黄 |
| `failed` | 拉取失败 | 红 |
| `missing` | DB 有记录但文件不存在 | 红 |

**自定义镜像 Dockerfile 模板**：

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    curl ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_CACHE_DIR=/opt/uv/cache
ENV UV_LINK_MODE=copy
ENV PATH="/opt/uv/python/bin:/usr/local/bin:$PATH"

RUN uv python install 3.14

WORKDIR /opt/warmup
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-install-workspace
```

**关键陷阱**：

1. **`/root/` 空目录陷阱**：HPC 模式下 Apptainer 使用 rootlesskit，将 `/root/` 挂载为空 tmpfs。工具/二进制必须安装到 `/usr/local/bin/` 或 `/opt/`
2. **跨文件系统硬链接失败**：SIF (squashfs) 与可写层不是同一文件系统。解决：`ENV UV_LINK_MODE=copy`
3. **uv warmup 缓存不命中**：`uv pip install` 和 `uv sync` 使用不同缓存格式。解决：warmup 必须用 `uv sync --frozen`

#### `AI.md` — Magnus 平台深度分析 & 近期更新

> 基于 magnus-main 完整源码，分析日期 2026-04-25
> 近期更新 (2026-04): Metrics Protocol v1, CPU/cgroup 语义修正, GPU 泄露修复, magnus-sdk>=0.8.0

**Magnus 是什么**：PKU Plasma + Rise-AGI 开源的**科学计算基础设施平台**，将 HPC 集群转化为人类和 AI Agent 共享的执行后端。

**三层架构**：

| 层 | 职责 | 关键组件 |
|----|------|----------|
| **Execution** | 容器化任务在 SLURM 集群执行 | Apptainer 容器, 四级优先级调度, GPU/CPU/内存管理 |
| **Sedimentation** | 工作流固化与知识沉淀 | Blueprint (可执行函数), Skill (领域知识包) |
| **Collaboration** | 人类-Agent 协作 | 统一鉴权, Web UI, SDK/CLI |

**完整交互链路（7 步全景图）**：
```
客户端 SDK → FastAPI 后端 (POST /api/jobs/submit)
  → 资源准备 (镜像拉取 + 仓库克隆, 并行)
  → 调度决策 (四级优先级 + 抢占, 队头挂号)
  → SLURM 提交 (wrapper.py 生成, sbatch)
  → Apptainer 容器执行 (环境变量注入, 可写层管理)
  → 结果反馈 (wrapper epilogue, .magnus_success/.magnus_oom)
```

**Metrics Protocol v1 核心要点**：

- **双轴模型**：每个指标点必须带 `time_unix_ms`，可选带 `step`
- **指标命名前缀**：`system.` / `train.` / `eval.` / `inference.` / `app.`
- **JSONL 格式**写入 `$MAGNUS_METRICS_DIR/`
- **CPU 语义变更**：从 `/proc/stat` 节点级 → cgroup 用量/CPU quota（100%=任务用满分配的核）
- **GPU 泄露修复**：通过 `CUDA_VISIBLE_DEVICES` 过滤，CPU 任务跳过 GPU 采样
- **OOM 检测**：检查 cgroup `memory.events` / `memory.oom_control`

#### `Magnus Platform 系统代码集成指南.md` — 非技术人员集成指南

8 个场景完整代码示例：
1. 查看集群资源
2. 提交训练任务
3. 查看任务状态和日志
4. 列出所有任务
5. 终止任务
6. AI Explorer 对话
7. 运行蓝图
8. 批量自动化

**完整 API 速查表**：30+ 个端点的方法/路径/说明

**任务优先级**：A1（最高，不可抢占）> A2（高，不可抢占）> B1（普通，可被抢占）> B2（低，可被抢占）

#### `SFT.md` — SFT 训练工作流项目笔记

包含完整的技术细节：FSDP CPU Offload 显存分析、NCCL 通讯优化、性能指标系统、已知问题排错等。适合深入理解训练管线时查阅。

---

### 提交训练脚本

#### `submit_sft.py` — SFT 训练提交（配置版，推荐）

**工作流**：
```
[0/5] model-version 去重检查
[1/5] 配置 Magnus 连接
[2/5] save_blueprint() → 保存/更新服务器蓝图
[3/5] time.sleep(30) → 等待服务器同步
[4/5] launch_blueprint() → 提交任务，获取 job_id
[5/5] Monitor → 后处理（下载报告 + 记录版本）
```

**配置驱动**（文件顶部直接修改）：

```python
MODE        = "sft"          # "sft" = 全参微调, "lora" = LoRA/QLoRA
MODEL_PATH  = "/data/magnus/models/Qwen2.5-Math-7B-Instruct"
OUTPUT_DIR  = ""             # 留空自动推导
EPOCHS      = 3
BATCH_SIZE  = 2
GRAD_ACCUM  = 4
LEARNING_RATE = 2e-5
MAX_LENGTH  = 1024
GPU_COUNT   = 2
GPU_TYPE    = "a100"
CPU_OFFLOAD = False          # True = 优化器状态移至 CPU RAM（72B 2 卡必备）
BWD_PREFETCH = "pre"         # "pre"=速度优先, "post"=显存优先
```

**参数映射**：配置区变量自动映射为蓝图参数名（`MODEL_PATH` → `model_path`，`EPOCHS` → `epochs` 等）。

**source 缩写**：`ss`

```bash
python train/submit_sft.py
python train/submit_sft.py --address http://...  # 仅覆盖连接参数
```

#### `magnus_sft.py` — SFT 训练提交（CLI 版）

功能与 `submit_sft.py` 相同，但通过命令行参数指定所有配置。

完整流程：版本检查 → 保存蓝图 → 启动蓝图 → 监控 → 后处理（下载报告 + 记录 model-version）。

**source 缩写**：`ms`

```bash
python train/magnus_sft.py --model /data/magnus/models/Qwen2.5-1.5B
python train/magnus_sft.py --model /data/magnus/models/Qwen2.5-1.5B --model-version Qwen2.5-1.5B-v3
```

#### `run_sft_blueprint.py` — SFT 训练提交（精简版）

直接注册蓝图 + 一键提交，比 `magnus_sft.py` 参数更简洁。

**source 缩写**：`rsb`

```bash
python train/run_sft_blueprint.py --model Qwen/Qwen2.5-7B-Instruct --gpus 3
```

---

### 核心模块

#### `config.py` — 共享配置模块

所有其他脚本通过 `from config import ...` 引用此模块。主要组件：

**`wait_for_job()`** — 轮询任务状态直到终态：

```python
from config import wait_for_job

job = wait_for_job(job_id, poll_interval=60)  # 阻塞直到 Success/Failed/Terminated
```

- 自动打印状态变化（`Pending → Preparing → Running → Success/Failed`）
- 增量打印 Magnus 日志
- Ctrl+C 可中断

**`SYSTEM_ENTRY_COMMAND`**（容器挂载脚本）：

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

**`record_storage()`** / **`check_model_version_exists()`**：持久存储记录和版本去重。

---

## SFT 蓝图对比

### 旧版 `OpenFundus_SFT.magnus`（已弃用）

位置：`trian-share/OpenFundus_SFT.magnus`

**主要限制**：
- 对话模板硬编码为 Qwen 的 ChatML 格式（`<|im_start|>`），不支持其他模型
- 无多卡并行（`DataParallel` 导致大模型 OOM）
- 无 pip 镜像加速（依赖安装从 PyPI 直连，慢且不稳定）
- 参数较少（9 个），缺少 GPU 类型、容器镜像、内存/存储配置等

### 新版 `OpenFundus_SFT_zyz.magnus`（当前使用）

位置：`train/OpenFundus_SFT_zyz.magnus`

**核心改进**：

#### 1. 通用模型兼容（旧版仅支持 Qwen）
- 使用 `tokenizer.apply_chat_template()` 自动适配所有模型对话格式（Qwen / InternLM / DeepSeek / LLaMA 等）
- 自动修补 InternLM2 的 rope_scaling 兼容问题
- 本地路径 / Hub ID 智能判断（以 `/` 开头视为本地路径，否则自动从 ModelScope 下载）

#### 2. 多卡并行 FSDP（旧版 DataParallel 导致大模型 OOM）

| 组件 | DataParallel (单卡) | FSDP SHARD_GRAD_OP (3卡) | FSDP FULL_SHARD (3卡) |
|------|---------------------|--------------------------|------------------------|
| 模型权重 | 14 GB (完整) | 14 GB (完整) | **5 GB** (分片) |
| 梯度 | 14 GB (完整) | **5 GB** (分片) | **5 GB** (分片) |
| AdamW 状态 | 56 GB (完整) | **19 GB** (分片) | **19 GB** (分片) |
| **合计** | **~84 GB** | **~38 GB** ✓ | **~29 GB** ✓ |
| 适用 | 小模型 | ≤30B 模型 | **大模型（72B+）** |

72B 模型 bf16 ≈ 144GB，SHARD_GRAD_OP 每卡完整权重 > 80GB OOM → 已切换为 FULL_SHARD。

**关键实现**：

```python
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    device_id=local_rank,
)
```

配套改动：启动命令从 `python3` 改为 `torchrun --nproc_per_node=N`、`DistributedSampler`、`all_reduce` 汇总 loss、仅 rank 0 保存 checkpoint。

#### 3. pip 加速安装（旧版直连 PyPI）
- 优先使用本地缓存 `/data/$USERNAME/pip-cache/wheels/`（`--no-index --find-links`）
- 缓存未命中时回退到清华镜像源
- 不再从公网下载，安装从数十分钟缩短到数秒

#### 4. 更多可配置参数
旧版 9 个参数 → 新版 22+ 个参数：新增 `gpu_type`、`cpu_count`、`memory_demand`、`ephemeral_storage`、`container_image`、`cpu_offload`、`backward_prefetch` 等。

#### 5. 日志增强
- Python 脚本内每个步骤带时间戳和耗时（`[1/8]` ~ `[8/8]`）
- stdout 实时刷新（`flush=True`），不缓冲
- 模型加载时列出每个 safetensors 文件名和大小
- 训练失败时自动通过 `trap EXIT` 打包上传日志

### 改动速查表

| 维度 | 旧版 `OpenFundus_SFT` | 新版 `OpenFundus_SFT_zyz` |
|------|----------------------|--------------------------|
| 位置 | `trian-share/` | `train/` |
| 对话模板 | 硬编码 Qwen ChatML | `apply_chat_template()` 自动适配 |
| 并行策略 | DataParallel（大模型 OOM） | FSDP FULL_SHARD + 逐层包装 |
| 镜像加速 | 无 | 本地缓存 → 清华源 fallback |
| 参数数量 | 9 个 | 22+ 个 |
| 日志粒度 | `echo`（无时间戳） | `[时间] [步骤]` 含耗时统计 |
| 蓝图上传统计 | 提交时无等待 | save_blueprint() + 等 30s → launch |
| 提交方式 | `submit_sft.py`（内嵌训练脚本） | `submit_sft.py`（读取 .magnus 文件） |
| 断点续训 | 无 | 支持 `--resume_from_checkpoint` |

### 蓝图对模型文件夹的文件要求

模型文件夹必须包含以下**必需文件**：

| 文件 | 用途 | 说明 |
|------|------|------|
| `config.json` | **入口检查** | 蓝图以此判断模型是否存在。缺失则走 ModelScope 下载 |
| `model-*-of-*.safetensors` | 权重（推荐） | HuggingFace 标准分片格式 |
| `model.safetensors.index.json` | 分片索引 | 多分片 safetensors 必需 |
| `tokenizer_config.json` | Tokenizer 配置 | 必需 |
| `tokenizer.json` | Tokenizer 数据 | 推荐（tokenizers>=3.0 格式） |

---

## 任务监控

### 通知机制

提交脚本（`submit_sft.py`、`magnus_sft.py`）在 `launch_blueprint()` 后自动调用 `notify_exe(job_id=job_id)`（简化为终端 print 输出）。

### 轮询监控

`config.py` 中的 `wait_for_job(job_id, poll_interval=60)` 轮询 Magnus 任务状态直到终态，过程中实时打印状态变化和增量日志。

---

## 已知问题 & 排错

### 1. 容器内无法访问 `/data/`（平台更新）

**现象**：作业日志出现 `No such file or directory`

**原因**：Magnus 平台更新后不再默认通过 Apptainer bind-mount 挂载 `/data/` 到容器内。

**修复**：所有 `submit_job()` 调用添加 `system_entry_command = SYSTEM_ENTRY_COMMAND` 参数，显式声明挂载：

```python
from config import SYSTEM_ENTRY_COMMAND

magnus.submit_job(
    ...
    system_entry_command = SYSTEM_ENTRY_COMMAND,
)
```

### 2. FSDP Training OrderedDict KeyError

**现象**：`KeyError` from FSDP state dict `_flat_param` / `_fpw_module`.

**原因**：`resume_from_checkpoint` 时 FSDP 状态 dict 键名不匹配，或将完整 checkpoint 传入 rank≠0 的进程。

**修复**：参考 `OpenFundus_SFT_zyz.magnus` 中的 `_safe_fsdp_checkpoint_load()`，将模型的 single-GPU state dict 转换为 FSDP-compatible 格式。

---

## 集群硬件与配置速查

> 来源：`AI.md` + `magnus-main/configs/magnus_config.yaml.example`

### 硬件

| 资源 | 上限 | 默认值 |
|------|------|--------|
| CPU | 128 核 | 4 核 |
| 内存 | 256 GB | 1.6 GB |
| 临时磁盘 | — | 10 GB |
| GPU (RTX 5090 32GB) | 单任务最多 4 卡 | 0（纯 CPU） |
| 默认容器镜像 | — | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` |

### 优先级

| JobType | 数值 | 可被抢占 | 适用 |
|---------|------|----------|------|
| A1 | 4 | 否 | 生产/紧急 |
| A2 | 3 | 否 | 常规训练（推荐） |
| B1 | 2 | 是 | 批量任务 |
| B2 | 1 | 是 | 下载/预热/测试 |

### Job 状态机

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

## 推荐工作流

### 首次使用

```bash
# Step 1: 编辑 submit_sft.py 配置区（MODE, MODEL_PATH, GPU_COUNT, CPU_OFFLOAD 等）
# Step 2: 提交 SFT 训练
python train/submit_sft.py

# Step 3: 或用 LoRA/QLoRA 低资源微调（详见 train/docs/LoRA.md）
#          blueprint: LoRA_zyz.magnus，支持 4-bit/8-bit/bf16

# Step 4 (可选): 启动模型 API 推理服务
python train/serve_model.py
```

### 日常训练

```bash
# 编辑 submit_sft.py 配置区 MODE / MODEL_PATH / GPU_COUNT / CPU_OFFLOAD 等
python submit_sft.py
```

### 显存优化速查

| 场景 | 推荐配置 |
|------|----------|
| 72B + 3×80GB | `CPU_OFFLOAD = False`, `BWD_PREFETCH = "pre"` |
| 72B + 2×80GB | `CPU_OFFLOAD = True`, `BWD_PREFETCH = "post"` |
| 7B + 1×80GB | `CPU_OFFLOAD = False`, `BWD_PREFETCH = "pre"` |

---

## 常用命令速查

```bash
# ═══ 提交训练 ═══
python submit_sft.py                                   # 编辑配置区后运行（推荐）
python magnus_sft.py --model /data/.../Qwen2.5-7B --epochs 5  # CLI 版

# ═══ 评估与分析 ═══
python eval_baseline.py                                # 交互式选择模型+评估
python eval_baseline.py --scan                         # 仅扫描模型列表
python eval_baseline.py --model Qwen2.5-72B-Instruct   # 指定模型评估
python eval_baseline.py --all-versions                 # 对所有SFT版本评估
python auto_grade.py                                   # LLM 批改

# ═══ 推理服务 ═══
python serve_model.py                                  # 启动 API 服务
```

---

*基于 magnus-main 源码分析、OpenFundus 蓝图代码及集群运行经验整理。*
