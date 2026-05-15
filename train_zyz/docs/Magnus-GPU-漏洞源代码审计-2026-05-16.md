# Magnus 平台 GPU 资源管理漏洞 — 源代码审计报告

**日期**: 2026-05-16
**审计范围**: `magnus-main/back_end/` 全部 Python 源代码
**审计方法**: 全量关键词搜索 + 关键函数逐行审读
**审计目标**: 验证 GPU 状态污染漏洞的存在性、Grace Period 修复方案的可行性、发现新漏洞

---

## P0-1: wrapper.py 信号处理逻辑审计

**文件**: `back_end/server/_scheduler/_wrapper_template.py`
**搜索**: `signal.signal`, `SIGTERM`, `SIGINT`, `handler`, `_signal_user_processes`

### 关键代码段

**信号处理器注册** (第 430-436 行):
```python
_signaled = [False]
def _on_sigterm(_signum, _frame):
    _signaled[0] = True
    try:
        _signal_user_processes(signal.SIGTERM)
    except Exception:
        pass
signal.signal(signal.SIGTERM, _on_sigterm)
```

**信号转发逻辑 `_signal_user_processes()`** (第 323-389 行):
```python
def _signal_user_processes(sig):
    """SIGTERM handler 内调用：向 user 容器内进程全员转发 sig。"""
    # 读取 cgroup.procs → 遍历所有 PID
    # 读 /proc/<pid>/status 的 NSpid 字段区分:
    #   NSpid 单列 → host-only 进程 (apptainer starter, FUSE helpers) → 跳过
    #   NSpid 双列 + inner==1 → 容器 init → 跳过
    #   NSpid 双列 + inner!=1 → user 进程 → os.kill(pid, sig)
```

**SIGKILL 路径** (第 425-427 行，注释):
```python
# 强终止走 kill_job 的 scancel --signal=KILL --full：SIGKILL 内核侧不可
# ignore，由 proctrack 广播 cgroup 全员瞬间清场
```

**Epilogue — marker 决策** (第 599-622 行):
```python
# Phase 3: ret==0 → 写 .magnus_success marker
# ret!=0 + _signaled[0] + .magnus_result 存在 → 仍写 marker (fallback)
# 其它 → 不写 marker, 透传 ret_code
```

**finally 块** (第 627-634 行):
```python
finally:
    for _candidate in (overlay_path, overlay_path + ".ext3"):
        try:
            if os.path.exists(_candidate):
                os.remove(_candidate)
        except Exception:
            pass
```

### 结论

| 问题 | 结论 |
|------|------|
| 是否注册了 SIGTERM 处理器？ | **是**。`_on_sigterm` 注册于 `main()` 第 436 行 |
| 处理器是否转发信号给用户进程？ | **是**。`_signal_user_processes(SIGTERM)` 通过 cgroup.procs + NSpid 筛选用户容器内进程并逐 `os.kill()` 转发 |
| 是否包含 CUDA/NCCL 清理逻辑？ | **否**。wrapper.py 仅负责信号转发，不执行任何 CUDA/NCCL 操作。无 `torch.cuda.empty_cache()`、无 `ncclCommDestroy()` |
| `finally` 在 SIGKILL 下是否执行？ | **否**。SIGKILL 直接杀死进程，`finally` 不执行。`finally` 仅清理 overlay 文件，不涉及 GPU |
| SIGTERM handler 本身是否 fail-safe？ | **是**。`_on_sigterm` 先置 `_signaled[0]=True` 再 forward；forward 内部的每个 `os.kill` 独立 try/except，单个失败不影响整体 |

**对 Grace Period 方案的影响**: **需重新评估**。Grace Period 方案假设"SIGTERM → wrapper fan-out → user handler 清理 CUDA"。但 wrapper.py 本身**不执行任何 CUDA 清理**，它只负责把信号转发到 user 进程。清理的实际执行者必须是 **user 代码的 SIGTERM handler**。这意味着：

1. 若 user 代码安装了 SIGTERM handler 并执行 `cuCtxDestroy()` + `ncclCommDestroy()`，Grace Period 方案有效
2. 若 user 代码**未装 handler**（继承 SIG_IGN），SIGTERM 被无视 → grace period 空转 → SIGKILL 兜底，与当前行为无差异
3. **不能指望** Grace Period 自动修复 GPU 清理问题 —— 需要 user 代码配合

---

## P0-2: 抢占机制强制终止分支审计

**文件**: `back_end/server/_scheduler/_decisions.py`
**函数**: `_kill_and_pause()` (第 229-251 行)

### 关键代码段

```python
def _kill_and_pause(self, db: Session, job: Job):
    """Kill SLURM job and mark as PAUSED for preemption"""
    if job.slurm_job_id:
        logger.info(f"Killing victim job {job.id} (SLURM: {job.slurm_job_id})")
        assert self.slurm_manager is not None
        self.slurm_manager.kill_job(
            job.slurm_job_id,
            runner = job.runner if job.runner is not None else "magnus",
            token = job.user.token if job.user.token is not None else "",
        )

    self._clean_up_working_table(job.id)
    job.status = JobStatus.PAUSED
    job.start_time = None
    db.commit()
```

### 结论

| 问题 | 结论 |
|------|------|
| 是否存在 `force_kill=True` 参数？ | **否**。代码中无此参数 |
| 是否存在 `skip_grace_period=True` 参数？ | **否**。代码中无此参数 |
| 是否有注释要求抢占在 X ms 内完成？ | **否**。代码注释提到"瞬时让出 GPU"在 `_control.py` 的 `kill_job()` 中，此处未提及时限 |
| 抢占与用户 terminate 是否走同一条 `kill_job()` 路径？ | **是**。两者完全相同，均直接调用 `slurm_manager.kill_job()` → `scancel --signal=KILL --full` |

**对 Grace Period 方案的影响**: 抢占场景需要独立设计。若 Grace Period 引入后抢占仍用同一路径，则每次抢占都需要等待 grace period（默认 10s），这使得"瞬时让出 GPU"的设计承诺不再成立。建议为抢占场景单独配置更短的 grace period（如 2-3s），或在 `_kill_and_pause` 中传递 `grace_period` 参数。

---

## P0-3: GPU 分配前置清理逻辑审计

**文件**: `back_end/server/_scheduler/_resources.py`, `back_end/server/_scheduler/_submit.py`, `back_end/server/_scheduler/_core.py`
**搜索**: `nvidia-smi`, `nvml`, `cuda`, `reset`, `cleanup`

### 关键发现

**`_resources.py`** (全文件 1-184 行): 仅处理镜像拉取和仓库 clone。**零 GPU 操作**。

**`_submit.py:_submit_to_slurm()`** (第 88-180 行):
```python
def _submit_to_slurm(self, db: Session, job: Job) -> bool:
    # ... 准备 wrapper_content, sif_path, system_entry_command ...
    slurm_id = self.slurm_manager.submit_job_simple(
        entry_command = f"python3 {wrapper_path}",
        gpus = job.gpu_count,
        job_name = job.task_name,
        # ... 无 GPU 健康检查参数 ...
    )
    job.status = JobStatus.QUEUED
```
**零 GPU 健康检查或清理操作**。提交前不验证 GPU 是否可用。

**`_control.py:submit_job_simple()`** (第 12-90 行): 构造 `sbatch` 命令并提交。`submit_job_simple` 的注释（第 25 行）明确说"简单提交：不做 sleep + 状态检查，让 SLURM 自己排队和调度"。**零 GPU 预检**。

### 结论

| 问题 | 结论 |
|------|------|
| 新 Job 启动前是否对即将分配的 GPU 执行清理？ | **否** |
| 新 Job 启动前是否对即将分配的 GPU 执行健康检查？ | **否** |
| SLURM 是否有机制确保分配的 GPU 处于可用状态？ | **否**。SLURM 的 GRES 只做记账（计数），不验证 GPU 实际健康状态 |
| GPU 污染是否会永久残留？ | **是**。一旦 nvidia.ko 清理失败，GPU 状态污染无限期残留，直到外部干预（`nvidia-smi -r` 或节点重启） |

---

## P1-4: SLURM 动态 epilog/prolog 生成逻辑审计

**文件**: `back_end/server/_slurm_manager/_control.py`
**搜索**: `--epilog`, `--prolog`, `epilog.sh`, `prolog.sh`

### 关键发现

`submit_job_simple()` 构造的 `sbatch` 命令（第 47-68 行）:
```python
command = [
    "sbatch",
    "--parsable",
    f"--job-name={job_name}",
    # ... gres, mem, cpus-per-task, output ...
]
```

**无任何 `--epilog`、`--prolog` 或动态 epilog/prolog 脚本参数**。

SLURM 的 `sbatch` 支持 `--epilog=<path>` 和 `--prolog=<path>` 作为 job 级别的覆盖，但 Magnus 不使用这些参数。

### 结论

| 问题 | 结论 |
|------|------|
| Magnus 是否为每个 Job 动态生成 epilog/prolog 脚本？ | **否** |
| Magnus 的 sbatch 命令是否传递 `--epilog`/`--prolog` 参数？ | **否** |
| Magnus 源代码中 epilog 的唯一引用是什么？ | 仅存在于代码注释中（`_core.py:109`, `_decisions.py:243`, `_resource_query.py:309`），作为对 SLURM 集群配置的**假设** |

**修正原有结论**: 之前分析报告中"Magnus 假设 SLURM epilog 会做 GPU reset"的结论**正确**，但程度被低估。Magnus 不仅不配置 epilog，而且没有任何机制验证 epilog 是否存在。这是一个**完全未被满足的隐性依赖**。

---

## P1-5: NCCL 共享内存启动前清理审计

**文件**: `back_end/server/_scheduler/_wrapper_template.py`
**搜索**: `/dev/shm`, `nccl-`, `shm_unlink`, `find`

### 关键发现

**全代码库搜索结果**: 零匹配。代码库中**不存在**任何对 `/dev/shm`、`shm_open`、`shm_unlink` 或 NCCL shm 文件的引用。

`wrapper.py` 的 `main()` 函数启动流程 (第 391-444 行)：信号处理器注册 → 工作目录初始化 → metrics sidecar 启动 → Phase 1 (user script 渲染) → Phase 2 (apptainer exec) → Phase 3 (epilogue marker)。**无任何 NCCL shm 清理步骤**。

### 结论

| 问题 | 结论 |
|------|------|
| 新 Job 启动时是否清理旧 NCCL 共享内存文件？ | **否** |
| 是否存在任何 NCCL `/dev/shm` 清理逻辑？ | **否**。代码库中完全没有此概念 |
| NCCL `/dev/shm` 泄漏漏洞是否真实存在？ | **是，确认为真实漏洞**。所有 SIGKILL/SIGABRT 异常退出的 NCCL communicator 的 shm 文件永久残留，直到手动清理或节点重启（tmpfs） |

**量化**：A100 6-GPU FSDP 训练的 NCCL shm 使用量约每 rank 几十 MB（ring buffer + peer mapping）。每次异常终止残留 ≈ 100-300MB。`/dev/shm` 默认大小 ≈ RAM*50%。若节点有 512GB RAM，则 `/dev/shm` ≈ 256GB，可容纳数百次异常终止的残留。**触发周期长但一旦触发（`ENOSPC`）即致命**。

---

## P1-6: 服务重启全局清理逻辑审计

**文件**: `back_end/server/main.py`
**函数**: `lifespan()` (第 288-350 行)

### 关键代码段

**启动阶段** (第 306 行):
```python
scheduler_task = asyncio.create_task(run_scheduler_loop())
```

**关闭阶段** (第 331-349 行):
```python
logger.info("Shutting down...")
scheduler_task.cancel()       # 取消调度循环
service_manager_task.cancel()
file_custody_task.cancel()
# ...
try:
    await scheduler_task       # 等待 CancelledError
    # ...
except asyncio.CancelledError:
    logger.info("Scheduler loop stopped.")

file_custody_manager.shutdown()
```

### 结论

| 问题 | 结论 |
|------|------|
| 服务重启时是否会遍历并终止所有运行中 Job？ | **否**。仅 `cancel()` 调度循环。运行中的 SLURM job 不受影响，它们独立于 Magnus 进程继续在 SLURM 中运行 |
| 是否执行任何全局 GPU 清理操作？ | **否** |
| `file_custody_manager.shutdown()` 执行什么？ | 文件托管清理（上传残留），与 GPU 无关 |

**重要发现 — 修正 §4.6 的错误结论**:

之前分析报告中 §4.6 写道"每次重启时，所有 Magnus 管理的 GPU job 均被终止（走 SIGKILL 路径）"。**此陈述经源代码审计确认为错误**。Magnus 服务重启时：
- 运行中的 SLURM job **不被终止**，它们在 SLURM 中独立运行
- Magnus 只是暂停了调度循环，重启后 `_sync_reality` 会重新发现这些 inflight job 并恢复追踪
- **Magnus 重启不会对 GPU 状态产生任何影响**（无论正面还是负面）

**重新评估 §4.6 的逻辑**: Magnus 重启不产生"被动清除污染"效应。如果节点确实观察到 GPU 状态在 Magnus 重启后恢复正常，其原因是：
1. 节点重启（非 Magnus 重启）→ tmpfs 清空 + GPU 硬件重置
2. 管理员在重启期间手动执行了 `nvidia-smi -r`
3. 巧合：运行中的 job 在 Magnus 重启期间自然完成了，GPU 正常释放

**但仍需保留 §4.6 作为分析维度**: 如果 Magnus 的运维流程中包含了"节点级重启"（而不仅仅是进程级重启），则我的原始分析成立。将此发现作为对 §4.6 的修正记录在案。

---

## P2-7: Apptainer GPU 挂载与信号配置审计

**文件**: `back_end/server/_scheduler/_wrapper_template.py` (第 539-554 行), `back_end/server/_scheduler.py` (第 1237-1252 行)

### 关键代码段

```bash
# containment 启用时:
APPTAINER_FLAGS="--nv --$APPTAINER_CONTAIN --no-mount tmp"

# containment 禁用时:
APPTAINER_FLAGS="--nv"
```

**环境变量注入** (第 494-495 行):
```bash
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi
```

### 结论

| 问题 | 结论 |
|------|------|
| Apptainer 启动参数是否影响信号传递？ | **否**。`--nv` 仅启用 NVIDIA GPU 支持（bind-mount GPU 设备），不影响信号传递。信号传递由 wrapper.py 的 `_signal_user_processes` + cgroup.procs + NSpid 控制 |
| Apptainer 启动参数是否影响 GPU 资源释放？ | **间接**。`--nv` bind-mount 了 `/dev/nvidia*` 设备，但这些设备的生命周期由 host nvidia.ko 管理。Apptainer 容器退出时不负责 GPU 资源清理 |
| CUDA_VISIBLE_DEVICES 是否正确传递？ | **是**。通过 `APPTAINERENV_CUDA_VISIBLE_DEVICES` 传递到容器内 |
| 是否存在信号相关的 Apptainer 配置缺失？ | **否**。信号分发不依赖 Apptainer 配置 |

---

## P2-8: NVML API 调用审计 + 全局关键词搜索

### NVML API

**全代码库搜索 `pynvml`, `nvmlInit`, `nvmlDeviceReset`, `nvmlDeviceGetComputeRunningProcesses`**:

**结果: 零匹配。**

Magnus 完全不使用 NVML (NVIDIA Management Library) API。GPU 状态的唯一交互是通过 `nvidia-smi` CLI 的**只读查询**（用于 metrics 采集）:

**`_metrics_collector.py`** (第 267 行):
```python
["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", ...]
```

**`_metrics_collector_nvidia_smi.py`** (全文件): 单次 `nvidia-smi` 全节点 GPU 采样，返回 `{gpu_idx: (util_pct, mem_used_bytes)}`。**只读操作，不修改 GPU 状态**。

### 全局关键词搜索 (TODO/FIXME/HACK/NOTE/BUG)

| 关键词 | 匹配数 | 位置 | 性质 |
|--------|--------|------|------|
| TODO | 1 | `back_end/python_scripts/tests/test_github_tools.py:7` | 测试文件，非生产代码 |
| FIXME | 0 | — | — |
| HACK | 0 | — | — |
| XXX | 0 | — | — |
| BUG | 0 | — | — |
| NOTE | 5 | `services.py:180`, `cluster.py:210`, `_scheduler.py:918`, `_wrapper_template.py:36`, `_resource_manager.py:335` | 均为代码说明注释，非缺陷标记 |

**结论**: Magnus 生产代码中不存在任何 TODO/FIXME/HACK/XXX/BUG 标记。代码注释风格严谨。

### nvidia-smi `-r` (GPU Reset)

**搜索**: `nvidia-smi.*-r`, `nvidia-smi.*reset`, `gpu.*reset`

**结果**: 代码库中唯一提及 GPU reset 的是注释中的假设（`_core.py:109`, `_decisions.py:243`）。**无任何代码实际执行 `nvidia-smi -r`**。

---

## 审计总结

### 新发现的漏洞

| # | 漏洞 | 严重程度 | 发现来源 |
|---|------|---------|---------|
| 1 | **Magnus 重启不终止运行中 Job — §4.6 分析基础错误** | 中 | P1-6 审计 |
| 2 | **Grace Period 方案依赖 User 代码协作，无自动 CUDA 清理** | 中 | P0-1 审计 |

### 修正的原有结论

| 原有结论 | 修正后 | 修正来源 |
|---------|--------|---------|
| §4.6 "Magnus 重启会 SIGKILL 所有 Job 从而清除污染证据" | **错误**。Magnus 重启仅停止调度循环，SLURM job 独立运行不受影响。GPU 污染不会因 Magnus 重启而被动清除 | P1-6 |
| Grace Period 能自动让 CUDA runtime 完成清理 | **部分正确**。仅当 user 代码安装了 SIGTERM handler 并执行 `cuCtxDestroy` 时有效。wrapper.py 本身不执行 CUDA 清理 | P0-1 |
| "SLURM epilog 不存在"的结论 | **确认且强化**。不仅 Magnus 不配置 epilog，且 sbatch 命令也不传递 `--epilog` 参数。这是一个完全未被满足的隐性依赖 | P1-4 |

### 确认的原有结论

| 原有结论 | 确认状态 | 确认来源 |
|---------|---------|---------|
| Magnus 所有异常终止走 SIGKILL | ✅ 确认 | P0-2, P0-3 |
| Magnus 无 GPU 状态清理 | ✅ 确认，且无 NVML API | P0-3, P2-8 |
| 抢占与 terminate 同一 SIGKILL 路径 | ✅ 确认 | P0-2 |
| NCCL `/dev/shm` 孤儿文件漏洞 | ✅ 确认为真实漏洞，代码库无任何清理 | P1-5 |
| wrapper.py `finally` 在 SIGKILL 下不执行 | ✅ 确认 | P0-1 |
| 无 GPU 健康检查 | ✅ 确认，nvidia-smi 仅用于 metrics 采集（只读） | P2-8 |

### 对修复方案的影响

**Grace Period 方案 (P1)**:

| 影响 | 说明 |
|------|------|
| ✅ 信号投射机制存在 | wrapper.py 已有完整的 SIGTERM → NSpid fan-out → user 进程机制 |
| ✅ `_signaled` 标记设计完善 | fail-open，forward 失败不影响 marker fallback |
| ⚠️ CUDA 清理不由 Magnus 控制 | Grace Period 方案的实际效果取决于 user 代码是否安装 SIGTERM handler |
| ⚠️ 抢占需要独立参数 | 不能与 terminate 共用 grace period 值，需要单独的 `preemption_grace_period` |
| ✅ 向后兼容 | `grace_period=0` → 行为完全不变，安全回退 |

**NCCL shm 清理 (P1)**:

| 影响 | 说明 |
|------|------|
| ✅ 漏洞真实存在 | 代码库无任何清理逻辑 |
| ✅ 修复简单 | 一行 `find /dev/shm -name 'nccl-*' -user $USER -delete` 即可 |
| ⚠️ 需在正确位置添加 | SLURM epilog（集群级）或 wrapper.py `main()` 开头（job 级） |

**GPU 健康检查 (P2)**:

| 影响 | 说明 |
|------|------|
| ✅ 当前完全缺失 | 无 NVML API，无 nvidia-smi 验证，无健康反馈回路 |
| ✅ 低风险添加 | `_finalize_completed_job` 中 fail-open 的健康检查不影响 job 终态 |

**§4.6 需更新**:

确认 Magnus 服务重启不影响 GPU job 运行状态。若用户运维流程中包含节点级重启（非纯 Magnus 进程重启），则 GPU 状态可能被被动清除。需在分析文档中修正此逻辑。
