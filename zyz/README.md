# Magnus 训练工具 — 完整手册

> 适用于 Magnus GPU 集群的 SFT 训练工作流。
> 本文面向**所有部门同事**：前半部分为快速上手，后半部分为技术细节。

---

## 目录

- [快速开始（5 分钟上手）](#快速开始5-分钟上手)
- [文件总览](#文件总览)
  - [Markdown 笔记](#markdown-笔记)
  - [Python 程序](#python-程序)
  - [自动生成目录 (已移除)](#自动生成目录-已移除)
- [各文件详细说明](#各文件详细说明)
  - [笔记文件](#笔记文件)
  - [环境准备脚本](#环境准备脚本)
  - [存储管理脚本](#存储管理脚本)
  - [提交训练脚本](#提交训练脚本)
  - [分析工具](#分析工具)
- [SFT 蓝图对比](#sft-蓝图对比)
- [已知问题 & 排错](#已知问题--排错)
- [集群硬件与配置速查](#集群硬件与配置速查)
- [常用命令速查](#常用命令速查)

---

## 快速开始（5 分钟上手）

> 以下所有命令在 `train/` 目录下运行。首次使用需按顺序执行。

```bash
# 1. 预热 pip 依赖包（只需一次，后续训练直接秒装）
python warmup_packages.py

# 2. 下载模型（每个模型只需一次）
python download_model_auto.py --model Qwen/Qwen2.5-72B-Instruct

# 3. 检查存储是否完整
python inspect_storage.py

# 4. 开始训练（编辑 submit_sft.py 顶部配置区，然后运行）
python submit_sft.py

# 5. 查看训练曲线
python plot_training.py
```

> **注意**：脚本提交作业后直接退出，状态可通过 Magnus Web UI 或 `magnus list-jobs` 查看。

---

## 文件总览

### Markdown 笔记

| 文件 | 内容 |
|------|------|
| `blue.md` | Blueprint 引擎技术文档：沙箱执行、参数映射、类型转换、entry_command 构造、submit_job 参数速查 |
| `docker.md` | 容器镜像管理：URI 格式、HPC/Local 模式、Web UI/SDK/API 操作、自定义镜像 Dockerfile、关键陷阱 |
| `AI.md` | Magnus 平台深度分析（基于 magnus-main 源码）：完整交互链路、Metrics Protocol v1、CPU/GPU 指标语义变更 |
| `Magnus Platform 系统代码集成指南.md` | 非技术人员指南：API 调用示例（8 个场景）、完整 API 速查表 |
| `README.md` | 本文档 |
| `SFT.md` | SFT 工作流项目笔记：文件总览、蓝图详解、内存分析、已知问题、推荐工作流 |

### Python 程序

| 脚本 | 功能 | 运行频率 | 分类 |
|------|------|----------|------|
| `warmup_packages.py` | 将 22 个训练依赖的 .whl 包下载到集群持久存储 | 首次使用一次 | 环境准备 |
| `download_model_auto.py` | 从 ModelScope 下载模型到集群持久存储 | 每个模型一次 | 环境准备 |
| `warmup_test.py` | 验证持久存储是否正常工作 | 怀疑存储故障时 | 环境准备 |
| `inspect_storage.py` | 查看集群上已缓存的 pip 包和模型列表 | 随时 | 存储管理 |
| `remove_storage.py` | 删除集群上的模型或文件（释放空间） | 需要清理时 | 存储管理 |
| `submit_sft.py` | **推荐**。读取 .magnus 蓝图，提交 SFT 训练任务 | 每次训练 | 提交训练 |
| `magnus_sft.py` | 通过命令行参数指定所有配置 | 每次训练 | 提交训练 |
| `run_sft_blueprint.py` | 注册蓝图 + 一键提交（精简版） | 测试快速验证 | 提交训练 |
| `plot_training.py` | 绘制训练 Loss 曲线和 LR 调度图 | 训练完成后 | 分析 |
| `plot_gpu_metrics.py` | 导出 GPU 指标到 CSV | 训练完成后 | 分析 |
| `push_to_acr.py` | 构建并推送 Docker 镜像到阿里云 ACR | 需要自定义镜像时 | 部署 |

### 自动生成目录 (已移除)

> 原 `data1/`（监控日志）、`data2/`（存储记录）、`SFT_data/`（训练报告）目录已随内置监视器移除。
> 作业状态请通过 Magnus Web UI 查看。

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

内容与本文档高度重叠，包含更多技术细节和问题排查记录。本文档整合了 SFT.md 的核心内容。

---

### 环境准备脚本

#### `warmup_packages.py` — pip 包预热

**目的**：将 22 个 SFT 训练依赖的 .whl 包下载到集群持久存储 `/data/<用户名>/pip-cache/wheels/`。

**预下载包列表**：
```
transformers, datasets, pandas, einops, sentencepiece, protobuf,
tokenizers, safetensors, numpy, scipy, pyarrow, jinja2,
huggingface-hub, tiktoken, modelscope, wandb, psutil, pyyaml,
click, scikit-learn, matplotlib, seaborn, requests
```

**关键设计**：
1. `accelerate` 单独用 `pip download --no-deps` 下载 — 避免拖入 `torch` + 20+ 个 nvidia/CUDA 包 (~3GB)。容器镜像 `pytorch:2.5.1-cuda12.4` 已自带 torch/CUDA
2. 安全网清理：`rm -f` 所有 `torch-*` / `nvidia_*` / `nvidia-*` / `triton-*` / `cuda_*` / `cuda-*`
3. 幂等机制：仅当 `.warmup_complete` 存在时跳过；不存在或不完整则删除旧目录重建
4. 拷贝用 `find -exec cp {}` 避免 glob 参数列表溢出
5. 幂等标记：`.warmup_complete` 标记文件

```bash
python train/warmup_packages.py
python train/warmup_packages.py --address http://xxx:3011/ --token sk-xxx
```

#### `download_model_auto.py` — 模型下载

**目的**：从 ModelScope 下载模型到集群持久存储。

**下载策略**：直接以 `/data/<user>/models/.dl_tmp` 做临时目录（避开容器 ephemeral storage 限制），下载完成后扁平化拷贝到目标目录，`rm -rf` 清理临时目录。

**扁平化处理**：ModelScope 下载结构为 `{publisher}/{model_name}/`（如 `Qwen/Qwen2.5-72B-Instruct/`），脚本自动扁平化，把模型文件直接放入 SAVE_DIR。


```bash
python train/download_model_auto.py --model Qwen/Qwen2.5-7B
python train/download_model_auto.py --model Qwen/Qwen2.5-72B-Instruct
```

#### `warmup_test.py` — 持久存储连通性测试

**目的**：提交 Write + Read 两个 B2 作业，验证容器退出后 `/data/` NFS 写入是否持久。

1. Write 作业：在 `/data/$USERNAME/persist-test/hello.txt` 写入标记文件
2. Read 作业：在新容器中检查该标记文件是否存在

```bash
python train/warmup_test.py
```

---

### 存储管理脚本

#### `inspect_storage.py` — 集群存储检查

**目的**：提交一个轻量检查作业，扫描 `/data/` 和 `/tmp/` 的目录结构。

**检查内容**：
- pip 缓存：wheel 数量、总大小、`.warmup_complete` 标记状态
- 模型目录：每个模型的 `.safetensors` 文件列表、文件数、大小
- safetensors header 校验：逐个字节检查确保未损坏

**原理**：内嵌 `TREE_PY` Python 脚本，通过 `os.walk` 遍历目录，以树形图输出。作业完成后从日志中提取结果。

```bash
python train/inspect_storage.py
```

#### `remove_storage.py` — 删除集群持久存储

**目的**：删除 `/data/` 下的文件或目录。

**安全防护**：
- 仅允许 `/data/` 前缀路径
- 默认交互确认
- `-y` 跳过确认
- 提交 B2 作业在容器内执行 `rm -rf`

```bash
python train/remove_storage.py /data/magnus/models/old-model
python train/remove_storage.py /data/magnus/models/Qwen2.5-72B-Instruct -y
```

---

### 提交训练脚本

#### `submit_sft.py` — SFT 训练提交（配置版，推荐）

**工作流**：
```
[0/5] 配置验证
[1/5] 配置 Magnus 连接
[2/5] save_blueprint() → 保存/更新服务器蓝图
[3/5] time.sleep(10) → 等待服务器同步
[4/5] launch_blueprint() → 提交任务，获取 job_id
[5/5] 提交完成（状态自行查看 Magnus Web UI）
```

**配置驱动**（文件顶部直接修改）：

```python
BLUEPRINT_FILE  = "OpenFundus_SFT_zyz.magnus"
MODEL_PATH      = "/data/magnus/models/Qwen2.5-72B-Instruct"
OUTPUT_DIR      = "/data/magnus/models/Qwen2.5-72B-Instruct-sft-v1"
EPOCHS          = 3
BATCH_SIZE      = 2
GRAD_ACCUM      = 4
LEARNING_RATE   = 2e-5
MAX_LENGTH      = 1024
SAVE_STEPS      = 200
GPU_COUNT       = 2
GPU_TYPE        = "a100"
CPU_COUNT       = 40
MEMORY          = "160G"
STORAGE         = "1024G"
PRIORITY        = "A2"
```

**参数映射**：配置区变量自动映射为蓝图参数名（`MODEL_PATH` → `model_path`，`EPOCHS` → `epochs` 等）。


```bash
python train/submit_sft.py
python train/submit_sft.py --address http://...  # 仅覆盖连接参数
```

#### `magnus_sft.py` — SFT 训练提交（CLI 版）

功能与 `submit_sft.py` 相同，但通过命令行参数指定所有配置。

```bash
python train/magnus_sft.py --model /data/magnus/models/Qwen2.5-1.5B
python train/magnus_sft.py --model /data/magnus/models/Qwen2.5-1.5B --model-version Qwen2.5-1.5B-v3
```

#### `run_sft_blueprint.py` — SFT 训练提交（精简版）

直接注册蓝图 + 一键提交，比 `magnus_sft.py` 参数更简洁。

```bash
python train/run_sft_blueprint.py --model Qwen/Qwen2.5-7B-Instruct --gpus 3
```

---

### 分析工具


#### `plot_training.py` — 训练曲线可视化

读取 `training_log.json`，绘制三张图：

1. **Train Loss 曲线**：逐步 loss + 滑动平均 + epoch 结束散点
2. **Train vs Eval Loss**：epoch 级别对比
3. **学习率曲线**：学习率调度

```bash
python train/plot_training.py                # 默认路径
python train/plot_training.py /path/to/training_log.json
```

#### `hooks/hook-magnus.py` — PyInstaller 钩子

```python
from PyInstaller.utils.hooks import copy_metadata
datas = copy_metadata("magnus-sdk")
```

用于 PyInstaller 打包 EXE 时复制 magnus-sdk 的元数据。

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
旧版 9 个参数 → 新版 20 个参数：新增 `gpu_type`、`cpu_count`、`memory_demand`、`ephemeral_storage`、`container_image`、`execute_action`、`resume_from` 等。

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
| 参数数量 | 9 个 | 20 个 |
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

## 已知问题 & 排错

### 1. Warmup 磁盘空间不足

**现象**：`cp: error writing '...whl': No space left on device`

**原因**：`accelerate` 的依赖 `torch` 引入 20+ 个 nvidia/CUDA 包（>3GB），撑满 `/tmp/` 或 `/data/`。

**修复**：
1. `accelerate` 从主包列表移除，改为单独 `pip download --no-deps`
2. 安全网：拷贝前 `rm -f` 所有 `torch-*` / `nvidia_*` / `triton-*` / `cuda_*`
3. 拷贝改用 `find -exec cp {}` 避免 glob 参数列表溢出
4. 幂等机制改用 `.warmup_complete` 标记文件

### 2. 大模型下载空间不足

**现象**：`[Errno 28] No space left on device: '/tmp/model_download/...'`

**原因**：大模型（如 72B）超过 130GB，ModelScope 下载到 `/tmp/`（容器 ephemeral storage），120G 不够用。

**修复**：下载临时目录改为 `/data/$USERNAME/models/.dl_tmp`，直接走 NFS 持久存储。

### 3. FSDP Training OrderedDict KeyError

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

# Step 6: 查看训练曲线
python train/plot_training.py /path/to/training_log.json
```

### 日常训练

```bash
# 编辑 submit_sft.py 配置区 MODEL_PATH / OUTPUT_DIR / GPU_COUNT 等
python submit_sft.py

# 通过 Magnus Web UI 查看进度
```

### 清理

```bash
# 删除旧模型
python remove_storage.py /data/magnus/models/old-model -y

# 检查存储使用情况
python inspect_storage.py
```

---

## 常用命令速查

```bash
# ═══ 一次性准备 ═══
python warmup_packages.py                              # 预热 pip 包
python download_model_auto.py --model Qwen/Qwen2.5-72B-Instruct  # 下载模型
python inspect_storage.py                              # 检查存储

# ═══ 每次训练 ═══
python submit_sft.py                                   # 提交训练（推荐）
python magnus_sft.py --model /data/.../Qwen2.5-7B --epochs 5  # CLI 版

# ═══ 分析 ═══
python plot_training.py                                # 绘制训练曲线

# ═══ 清理 ═══
python remove_storage.py /data/magnus/models/旧模型 -y  # 删除模型
```

---

*基于 magnus-main 源码分析、OpenFundus 蓝图代码及集群运行经验整理。*
