# Magnus 平台深度分析 & 近期更新

> 分析日期: 2026-04-25 | 基于 magnus-main 完整源码  
> 近期更新 (2026-04): Metrics Protocol v1, CPU/cgroup 语义修正, GPU 泄露修复, magnus-sdk>=0.8.0

---

## 目录

- [1. 项目概览与核心文件树](#1-项目概览与核心文件树)
- [2. 完整交互链路](#2-完整交互链路)
- [3. 任务提交规范与硬件管理](#3-任务提交规范与硬件管理)
- [4. download_model.py 深度分析](#4-download_modelpy-深度分析)
- [5. 具体修改方案](#5-具体修改方案)
- [6. Magnus Metrics Protocol v1](#6-magnus-metrics-protocol-v1)

---

## 1. 项目概览与核心文件树

### 1.1 Magnus 是什么

Magnus 是 PKU Plasma + Rise-AGI 开源的**科学计算基础设施平台**，将 HPC 集群转化为人类和 AI Agent 共享的执行后端。核心三层架构：

| 层 | 职责 | 关键组件 |
|----|------|----------|
| **Execution** | 容器化任务在 SLURM 集群执行 | Apptainer 容器, 四级优先级调度, GPU/CPU/内存管理 |
| **Sedimentation** | 工作流固化与知识沉淀 | Blueprint (可执行函数), Skill (领域知识包) |
| **Collaboration** | 人类-Agent 协作 | 统一鉴权, Web UI, SDK/CLI |

**技术栈**: FastAPI (Python >= 3.14) + Next.js 14 + SQLAlchemy + SLURM + Apptainer

### 1.2 核心目录树

```
magnus-main/
├── back_end/
│   ├── server/                          # Magnus 特定代码 ★核心
│   │   ├── main.py                      # FastAPI 入口 + lifespan 管理
│   │   ├── models.py                    # SQLAlchemy ORM (User, Job, ClusterSnapshot, ...)
│   │   ├── schemas.py                   # Pydantic 请求/响应模型
│   │   ├── database.py                  # SQLite 连接池
│   │   ├── _scheduler.py                # ★调度器核心 (心跳, wrapper.py 生成)
│   │   ├── _slurm_manager.py            # ★SLURM CLI (sbatch/squeue/scancel)
│   │   ├── _resource_manager.py         # ★镜像拉取 + 仓库克隆 (LRU 缓存)
│   │   ├── _metrics_collector.py        # ★Docker 模式系统指标采集 (cgroup v1/v2)
│   │   ├── _blueprint_manager.py        # 蓝图解析/沙箱执行
│   │   ├── _magnus_config.py            # 配置加载与校验
│   │   ├── _docker_manager.py           # local 模式 Docker 后端
│   │   ├── _service_manager.py          # 弹性服务管理
│   │   ├── _file_custody_manager.py     # 文件中转 (magnus-secret)
│   │   ├── _chat_manager.py             # Explorer AI 对话
│   │   └── routers/
│   │       ├── jobs.py                  # API /api/jobs/*
│   │       ├── auth.py                  # 飞书 OAuth + JWT + Trust Token
│   │       ├── blueprints.py            # Blueprint CRUD
│   │       ├── cluster.py               # 集群资源查询
│   │       ├── metrics.py               # ★Metrics Protocol v1 API (streams/query/render)
│   │       ├── files.py                 # 文件代管
│   │       ├── images.py                # 容器镜像缓存
│   │       └── ...
│   └── library/
│       ├── fundamental/                 # 基础工具 (JWT, YAML, JSON)
│       └── functional/                  # 功能模块 (飞书 SDK, OpenCode Agent)
├── front_end/                           # Next.js 14 Web UI
│   └── src/
│       ├── app/(main)/                  # 页面路由
│       ├── components/                  # UI 组件库 (含 MetricsChart)
│       ├── lib/api.ts                   # API 客户端
│       └── types/                       # TypeScript 类型定义
├── configs/
│   └── magnus_config.yaml.example       # ★配置模板
├── docs/
│   ├── Magnus_SDK_Guide.md              # ★SDK/CLI 文档 (含 metrics 感知)
│   ├── Magnus_metrics_guide.md          # ★Metrics Protocol v1 协议文档
│   ├── Magnus_job_runtime.md            # ★Job 运行时协议
│   ├── Blueprint_Crafting_Guide.md      # ★蓝图编写指南
│   ├── local_magnus.md                  # 本地模式文档
│   └── opencode_integration/            # Agent 集成
├── docker/
│   └── magnus-runtime/Dockerfile
└── scripts/
    └── deploy.py
```

---

## 2. 完整交互链路

### 2.1 全链路流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. 客户端 (本地 SDK/CLI)                                                  │
│                                                                          │
│   import magnus                                                          │
│   magnus.configure(token="sk-xxx", address="http://host:8017")           │
│                                                                          │
│   job_id = magnus.submit_job(                                            │
│       task_name="DownloadModel-Qwen2.5-1.5B",                           │
│       entry_command="pip install modelscope && python download.py",      │
│       repo_name="OpenFundus",                                            │
│       namespace="Rise-AGI",                                              │
│       gpu_type="a100", gpu_count=1,                                      │
│       job_type="A2",                                                     │
│   )                                                                      │
│                    │                                                     │
│                    ▼                                                     │
│   HTTP POST /api/jobs/submit                                             │
│   Header: Authorization: Bearer sk-xxxxxx                                │
│   Body: { task_name, entry_command, repo_name, ... }                    │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────────┐
│ 2. Magnus 后端 (FastAPI)                                                  │
│                                                                          │
│   routers/jobs.py: submit_job()                                          │
│     ├── Pydantic 校验 (JobSubmission schema)                             │
│     ├── create_job(): resources check                                    │
│     └── status=PREPARING                                                 │
│                                                                          │
│   调度器后台异步 (heartbeat=2s):                                          │
│     tick() → _sync_reality + _make_decisions + _record_snapshot          │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────────┐
│ 3. 资源准备 (PREPARING → PENDING)                                         │
│                                                                          │
│   asyncio.gather (并行):                                                 │
│   ├── 镜像拉取 (_ensure_image_decoupled, shield 保护)                     │
│   │   docker://image → .sif (Apptainer), LRU 缓存 80G                    │
│   │   3次重试 + 指数退避, 非瞬态错误直接失败                                │
│   └── 仓库克隆 (ensure_repo)                                              │
│       git clone → cache → copy → checkout → setfacl                      │
│                                                                          │
│   成功 → PENDING, 失败 → FAILED                                          │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────────┐
│ 4. 调度决策 (_make_decisions)                                             │
│                                                                          │
│   队头挂号模式:                                                           │
│   ├── PENDING jobs 按优先级排序 (A1=4 > A2=3 > B1=2 > B2=1)              │
│   ├── 同级按 created_at FIFO                                             │
│   ├── A 类可抢占 RUNNING B 类 (优先 B2, LIFO)                            │
│   ├── SLURM 队列中最多 1 个 QUEUED job                                    │
│   ├── 抢占恢复: PAUSED → PREPARING (重新准备资源)                         │
│   └── 资源充足 → _submit_to_slurm                                         │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────────┐
│ 5. SLURM 提交 (_submit_to_slurm)                                          │
│                                                                          │
│   _build_wrapper_content() 生成 wrapper.py:                               │
│   ┌──────────────────────────────────────────────────┐                   │
│   │ wrapper.py (Python, SLURM 直接运行)                │                   │
│   │   ├── _metrics_sidecar (daemon thread, 5s 轮询)   │                   │
│   │   │   ├── system.gpu.utilization (nvidia-smi)     │                   │
│   │   │   │   └── CUDA_VISIBLE_DEVICES 过滤 ★          │                   │
│   │   │   ├── system.cpu.utilization (cgroup, quota)  │                   │
│   │   │   └── system.memory.used_bytes (cgroup)       │                   │
│   │   ├── .magnus_user_script.sh 生成                  │                   │
│   │   ├── shell_cmd (Bash)                             │                   │
│   │   │   ├── APPTAINERENV_* 环境变量注入               │                   │
│   │   │   ├── system_entry_command 执行 (宿主机侧)      │                   │
│   │   │   ├── overlay 创建 (sparse ext3)              │                   │
│   │   │   ├── apptainer exec → 容器内执行用户脚本       │                   │
│   │   │   └── OOM 检测 (_check_oom + .magnus_oom)     │                   │
│   │   └── epilogue → 写 .magnus_success                │                   │
│   └──────────────────────────────────────────────────┘                   │
│                                                                          │
│   sbatch 提交:                                                            │
│     sbatch --parsable --job-name={task_name}                              │
│       --output={work}/slurm/output.txt                                    │
│       --gres=gpu:{gpu_type}:{gpu_count}                                   │
│       --mem={memory_demand}                                               │
│       --cpus-per-task={cpu_count}                                         │
│       python3 {work}/wrapper.py                                           │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────────┐
│ 6. 容器执行 (Apptainer)                                                   │
│                                                                          │
│   容器内注入的环境变量 (via APPTAINERENV_*):                               │
│   ┌──────────────────────────────────────────────┐                       │
│   │ MAGNUS_TOKEN      → SDK 自动识别              │                       │
│   │ MAGNUS_ADDRESS    → Magnus 后端地址            │                       │
│   │ MAGNUS_JOB_ID     → 当前 Job ID               │                       │
│   │ MAGNUS_HOME       → /magnus (容器内根路径)     │                       │
│   │ MAGNUS_RESULT     → .magnus_result 文件路径    │                       │
│   │ MAGNUS_ACTION     → .magnus_action 文件路径    │                       │
│   │ MAGNUS_METRICS_DIR → metrics/ 目录 ★新增       │                       │
│   │ MAGNUS_METRICS_PROTO → metrics.v1 ★新增        │                       │
│   └──────────────────────────────────────────────┘                       │
│                                                                          │
│   可写层策略:                                                              │
│   ┌────────────────┬───────────────────┬──────────────────┐              │
│   │ 模式            │ 可写层              │ 容量限制           │              │
│   ├────────────────┼───────────────────┼──────────────────┤              │
│   │ containall+     │ ephemeral overlay │ ephemeral_storage │              │
│   │ overlay         │ (sparse ext3)     │                   │              │
│   │ contain+        │ RAM tmpfs         │ 与 memory_demand  │              │
│   │ writable-tmpfs  │                   │ 共享              │              │
│   │ none (裸跑)     │ host 文件系统      │ 无限制             │              │
│   └────────────────┴───────────────────┴──────────────────┘              │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────────────┐
│ 7. 结果反馈                                                               │
│                                                                          │
│   wrapper.py epilogue:                                                    │
│   ├── 用户命令 exit 0 → 写 .magnus_success                               │
│   ├── 用户命令 exit != 0 → OOM 检测 (_check_oom)                         │
│   │   └── OOM 命中 → 写 .magnus_oom 标记                                  │
│   └── finally: 清理 overlay 镜像                                           │
│                                                                          │
│   调度器心跳:                                                              │
│   ├── _sync_reality: squeue 查询 SLURM 状态                               │
│   │   ├── COMPLETED + .magnus_success存在 → SUCCESS                       │
│   │   ├──  + .magnus_oom存在 → FAILED (OOM,格式化消息)                     │
│   │   ├── COMPLETED + 无标记 → FAILED                                     │
│   │   └── FAILED/CANCELLED/TIMEOUT → FAILED                              │
│   ├── _record_snapshot: ClusterSnapshot (每 300s)                        │
│   └── 惰性读取 .magnus_result / .magnus_action 到 DB                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 鉴权链路（不变）

```
认证方式: Trust Token (sk-*) / JWT (Web) / Local mode (免鉴权)
Trust Token: sk-{secrets.token_urlsafe(24)}
JWT: HS256, 7 天过期
鉴权缓存: TTLCache(60s, 1000条)
```

---

## 3. 任务提交规范与硬件管理

### 3.1 完整参数表 (JobSubmission)

| 参数 | 类型 | 必填 | 默认值 | 约束 |
|------|------|------|--------|------|
| `task_name` | str | **是** | — | 任务显示名称 |
| `entry_command` | str | **是** | — | shell 命令, 支持多行 |
| `repo_name` | str | **是** | — | GitHub 仓库名 |
| `namespace` | str | 否 | `"Rise-AGI"` | 组织名 |
| `branch` | str\|None | 否 | None | 自动检测默认分支 |
| `commit_sha` | str\|None | 否 | None | None=HEAD, 支持 `msg:regex` |
| `gpu_type` | str | 否 | `"cpu"` | GPU 型号, `"cpu"`=无 GPU |
| `gpu_count` | int | 否 | 0 | GPU 数量 |
| `job_type` | JobType | 否 | A2 | A1/A2/B1/B2 |
| `description` | str\|None | 否 | None | Markdown 描述 |
| `container_image` | str\|None | 否 | None | Docker URI |
| `cpu_count` | int\|None | 否 | None | 默认 4 |
| `memory_demand` | str\|None | 否 | None | 默认 1600M |
| `ephemeral_storage` | str\|None | 否 | None | 默认 10G |
| `runner` | str\|None | 否 | None | 默认集群 runner |
| `system_entry_command` | str\|None | 否 | None | 宿主机侧预执行 |

### 3.2 硬件规格与限制

```yaml
cluster:
  gpus:
    - value: rtx5090
      label: NVIDIA GeForce RTX 5090
      meta: 32GB • Blackwell
      limit: 4
  max_cpu_count: 128
  max_memory_demand: 256G
  default_cpu_count: 4
  default_memory_demand: 1600M
  default_ephemeral_storage: 10G
  default_container_image: docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
```

### 3.3 调度规则

| 优先级 | 数值 | 可抢占 | 说明 |
|--------|------|--------|------|
| A1 | 4 | 否 | 最高, 不可被抢占 |
| A2 | 3 | 否 | 高, 不可被抢占 |
| B1 | 2 | 是 | 低, 可被 A 类抢占 |
| B2 | 1 | 是 | 最低, 可被 A 类抢占 |

- 同优先级 FIFO, SLURM 队列最多 1 个 QUEUED job
- 抢占: 优先杀 B2 → B1, 同优先级 LIFO (后启动先杀)
- 被抢占 → PAUSED → 重新 PREPARING (重新拉镜像/仓库)

### 3.4 新增: ClusterSnapshot 集群快照

```python
class ClusterSnapshot(Base):
    total_gpus: int          # SLURM 报告的总 GPU
    slurm_used_gpus: int     # SLURM 使用的 GPU
    magnus_used_gpus: int    # Magnus RUNNING job 占用的 GPU
    timestamp: datetime
```

调度器每 `snapshot_interval` (默认 300s) 记录一次, 用于历史资源使用趋势分析。

---

## 4. download_model.py 深度分析

（内容不变, 仍为之前的分析）

---

## 5. 具体修改方案

（内容不变, 仍为之前的改进版 download_model.py 方案）

---

## 6. Magnus Metrics Protocol v1

### 6.1 概述

Magnus Metrics Protocol v1 是在 2026-04 版本中引入的全新指标协议, 目的是统一承载系统指标与训练指标，使 Agent/CLI/Web 都能感知作业运行状态。

**核心原则**: Job 以最小、稳定、跨语言的 JSONL 点流主动上报指标；每个点同时面向真实时间和逻辑 step 两个坐标系。

**SDK 支持**: `pip install magnus-sdk>=0.8.0` 后, Agent 可通过 CLI 像人看前端一样感知 metrics。

### 6.2 双轴模型

每个指标点必须带 `time_unix_ms` (物理时间), 可以选择额外带 `step` (逻辑推进坐标):

| 形态 | 适用场景 | 示例 |
|------|----------|------|
| 仅时间序列 | 纯系统指标 | `system.gpu.utilization` |
| 时间+step | 训练/模拟指标 | `train.loss`, `train.lr` |
| 时间+非训练step | 推理/求解 | `inference.tokens.generated` |

`step_domain` 说明 step 的语义 (如 `optimizer`, `eval`, `epoch` 等)。

### 6.3 指标名规范

推荐前缀:

| 前缀 | 用途 | 示例 |
|------|------|------|
| `system.` | 系统资源 | `system.cpu.utilization`, `system.gpu.utilization`, `system.memory.used_bytes` |
| `train.` | 训练 | `train.loss`, `train.lr`, `train.tokens.total` |
| `eval.` | 评估 | `eval.loss` |
| `inference.` | 推理 | `inference.tokens.total` |
| `app.` | 应用自定义 | 不限 |

### 6.4 数据格式 (JSONL)

```
$MAGNUS_METRICS_DIR/    (由 Magnus 运行时注入)
├── system.jsonl        ← metrics sidecar 写入
├── rank-0.jsonl        ← 用户代码写入
└── rank-1.jsonl
```

每行一个完整 JSON 对象:

```json
{
  "name": "system.cpu.utilization",
  "kind": "gauge",
  "value": 63.2,
  "time_unix_ms": 1770000123456,
  "unit": "percent",
  "labels": {"node": "node-01"}
}
```

系统指标由 wrapper.py 的 `_metrics_sidecar` (daemon thread, 5s 间隔) 自动采集写入, 用户任务无需处理。

### 6.5 **关键语义变更: CPU 指标**

| 版本 | CPU 计算方式 | 语义 |
|------|-------------|------|
| 旧版 | `/proc/stat` (节点级) | 百分比表示整节点 CPU 使用率 |
| **新版 (v1)** | **cgroup 用量 / CPU quota** | **100% = 任务用满分配的 CPU 核心** |

新语义使 CPU 指标直接反映任务对 allocated CPU 的使用程度, 而非干扰的节点级数值。关键代码:

```python
# wrapper.py _metrics_sidecar:
alloc_cpus = float(_allocated_cpus())  # 从 SLURM_CPUS_PER_TASK 或 os.cpu_count()
cpu_pct = (d_usage / (d_wall_ms * 1000.0) / alloc_cpus) * 100
```

### 6.6 **GPU 泄露修复**

旧版: 当 `CUDA_VISIBLE_DEVICES` 为空或未设置时, 指标 sidecar 采样**所有可见 GPU**, 导致 CPU-only 任务仍有 GPU 指标读取, 运维侧 GPU 驱动保持活跃, 造成泄露。

新版修复:

```python
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES") or ""
allowed_gpus = set(s.strip() for s in _cvd.split(",") if s.strip().isdigit())
# 空字符串 → 空集合 → 跳过 GPU 采样
```

| 场景 | 旧版 | 新版 |
|------|------|------|
| GPU 任务 (`CUDA_VISIBLE_DEVICES=0,1`) | 采样全部 GPU (泄露) | **只采样 GPU 0,1** |
| CPU 任务 (无 GPU 环境变量) | 采样全部 GPU (泄露) | **跳过 GPU 采样** |

### 6.7 API 端点

指标数据不入数据库, 直接从 JSONL 文件读取:

| 端点 | 说明 |
|------|------|
| `GET /api/jobs/{id}/metrics/streams` | 列出指标 stream (去重) |
| `GET /api/jobs/{id}/metrics/query?name=X&labels=Y&max_points=2000` | 查询指标数据点 (支持 time/step 过滤, 均匀降采样) |
| `GET /api/jobs/{id}/metrics/render?name=X&format=png` | 服务端渲染指标折线图 PNG |
| `GET /api/jobs/{id}/metrics/initial` | 一次获取最相关指标的 stream list + 默认数据 |

### 6.8 Docker 模式对等采集

SLURM 模式下系统指标由 wrapper.py 内嵌的 `_metrics_sidecar` 采集。Docker 模式无 wrapper, 因此新增 `_metrics_collector.py`:

- 独立 asyncio task, 在调度器启动时创建
- 每 5s 检查所有 RUNNING docker job
- 通过 docker inspect 获取容器 PID → `/proc/<pid>/cgroup` 解析 cgroup v1/v2
- cgroup 解析逻辑与 wrapper.py 保持行为一致 (源码有注释警告同步维护)
- 共享 nvidia-smi 采样, 按 `NVIDIA_VISIBLE_DEVICES` 过滤 GPU

### 6.9 OOM 检测

wrapper.py 新增 OOM 检测 (`_check_oom`), 在用户命令非零退出后检查 cgroup `memory.events` 或 `memory.oom_control`:

- cgroup v2: `/sys/fs/cgroup/<rel>/memory.events` → `oom_kill` 计数器
- cgroup v1: `/sys/fs/cgroup/memory/<rel>/memory.oom_control` → `oom_kill` 计数器
- OOM 命中 → 写 `.magnus_oom` 标记 → 调度器读取 → 格式化消息 `"Out of memory: memory_demand=X"`

### 6.10 文件系统变更

| 路径 | 生命周期 | 说明 |
|------|----------|------|
| `{work}/metrics/` | submit → **永久** | metrics JSONL 文件, 与 slurm/output.txt 同策略, 不随 job 清理 |
| `{work}/.magnus_oom` | OOM 发生时 → sync_reality | OOM 标记, 调度器读取后清理 |

`_clean_up_working_table` 不再清理 `metrics/` 目录。

---

## 附录: 关键源文件索引 (更新版)

| 文件 | 职责 | 关键函数 |
|------|------|----------|
| [server/main.py](magnus-main/back_end/server/main.py) | FastAPI 入口, lifespan | `run_scheduler_loop()`, `lifespan()` |
| [server/models.py](magnus-main/back_end/server/models.py) | ORM 模型 | `Job`, `JobType`, `JobStatus`, `ClusterSnapshot` |
| [server/schemas.py](magnus-main/back_end/server/schemas.py) | Pydantic Schema | `JobSubmission`, `ClusterResources` |
| [server/_scheduler.py](magnus-main/back_end/server/_scheduler.py) | 调度器核心 | `tick()`, `_build_wrapper_content()`, `_sync_reality()`, `_make_decisions()`, `_submit_to_slurm()` |
| [server/_metrics_collector.py](magnus-main/back_end/server/_metrics_collector.py) | Docker 模式指标采集 ★新增 | `DockerMetricsCollector`, cgroup v1/v2 双栈解析, `_sample_nvidia_smi()` |
| [server/_slurm_manager.py](magnus-main/back_end/server/_slurm_manager.py) | SLURM 操作 | `submit_job_simple()`, `check_job_status()`, `kill_job()` |
| [server/_resource_manager.py](magnus-main/back_end/server/_resource_manager.py) | 资源准备 | `ensure_image()`, `ensure_repo()`, LRU 淘汰 |
| [server/routers/jobs.py](magnus-main/back_end/server/routers/jobs.py) | Job API | `create_job()`, `submit_job()` |
| [server/routers/metrics.py](magnus-main/back_end/server/routers/metrics.py) | Metrics API ★新增 | `list_metric_streams()`, `query_metrics()`, `render_metric_chart()` |
| [server/routers/auth.py](magnus-main/back_end/server/routers/auth.py) | 鉴权 | `get_current_user()`, `feishu_login()` |
| [server/_blueprint_manager.py](magnus-main/back_end/server/_blueprint_manager.py) | 蓝图引擎 | 沙箱执行, submit_job 劫持 |
| [configs/magnus_config.yaml.example](magnus-main/configs/magnus_config.yaml.example) | 配置模板 | 所有可配置项 |
| [docs/Magnus_SDK_Guide.md](magnus-main/docs/Magnus_SDK_Guide.md) | SDK 文档 | `submit_job`, `execute_job`, `magnus job metrics` (≥0.8.0) |
| [docs/Magnus_metrics_guide.md](magnus-main/docs/Magnus_metrics_guide.md) | Metrics 协议文档 ★新增 | 双轴模型, JSONL 格式, 命名规范 |
| [docs/Magnus_job_runtime.md](magnus-main/docs/Magnus_job_runtime.md) | 运行时协议 | wrapper.py 结构, 文件系统, 环境变量 |
| [docs/Blueprint_Crafting_Guide.md](magnus-main/docs/Blueprint_Crafting_Guide.md) | 蓝图编写 | 类型系统, 沙箱, FileSecret |
