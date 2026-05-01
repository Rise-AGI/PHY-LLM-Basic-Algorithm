# Blueprint 实现逻辑 & 入口指令笔记

---

## 一、Blueprint 引擎核心逻辑

### 1.1 沙箱执行机制

```python
# _blueprint_manager.py 核心：受限 builtins + 劫持 submit_job

def __init__(self):
    def _hijacked_submit_job(**kwargs):
        raise _BlueprintCapture(kwargs)  # ← 不真提交, 抛出异常捕获 payload

    self.execution_globals = {
        "submit_job": _hijacked_submit_job,  # 劫持
        "JobType": JobType, "FileSecret": FileSecret,
        "Annotated": Annotated, "Literal": Literal,
        "Optional": Optional, "List": List,
    }
```

**执行流程**:
```
用户代码 → exec(blueprint_code, safe_builtins + execution_globals)
  → blueprint(**validated_args) 被调用
  → submit_job(...) 触发 _BlueprintCapture(kwargs)
  → 捕获 payload → 构造 JobSubmission → 返回给调度器
```

**沙箱限制**:
- 仅 `typing` 可 import
- 无 `open/read/write/sys/os/subprocess/eval/exec`
- 可用: 基本类型, `len/range/enumerate/zip/map/filter/sorted/sum/min/max/print`

### 1.2 参数类型 → 表单映射

```python
# analyze_signature() 自动推导类型

str            → "text"         支持: allow_empty, placeholder, multi_line
int            → "number"       支持: min, max
float          → "float"        支持: min, max, placeholder
bool           → "boolean"
Literal["a","b"] → "select"    支持: options dict 自定义 label/description
FileSecret     → "file_secret"  allow_empty 始终 False

# 包装器
Optional[T]    → 可选 (前端显示启用/禁用开关)
List[T]        → 动态列表 (前端添加/删除项)
```

### 1.3 类型转换与校验

```python
# execute() 内部: 动态 Pydantic 模型转换
DynamicModel = create_model("DynamicBlueprintModel", **field_definitions)
validated_args = DynamicModel(**processed_inputs).model_dump()
# "10" → 10, "3.14" → 3.14, "true" → True

# 列表预处理: CLI 传单值自动转为列表
if _is_list_type(annotation) and not isinstance(value, list):
    processed_inputs[param_name] = [value]
```

### 1.4 蓝图中唯一可做的事

```python
def blueprint(参数):
    submit_job(                          # ← 唯一允许的副作用
        task_name=...,                   # 必填, 任务显示名
        entry_command=...,               # 必填, 容器内 shell 命令
        repo_name=...,                   # 必填, GitHub 仓库名
        ...
    )
```

**本质**: Blueprint = Pydantic Schema 生成器 + submit_job 参数捕获器。不真算、不真跑。

---

## 二、entry_command 构造模式

### 2.1 基础模式

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
entry_command = cmd

# 模式 D: 纯 CPU
entry_command = "python preprocess.py"  # gpu_type="cpu", gpu_count=0
```

### 2.2 Shell 注入防御

```python
# 用户传入的 str 拼进 shell 时必须转义引号
def safe_quote(s: str) -> str:
    return f"'{str(s).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"

entry_command = f"python main.py --path {safe_quote(user_path)}"
# 等价 bash: python main.py --path '/path/with'\"'\"'single/quote'
```

### 2.3 容器内环境变量 (自动注入)

```python
# wrapper.py 注入, 容器内可直接读取
MAGNUS_HOME      = "/magnus"          # 容器根路径
MAGNUS_TOKEN     = "sk-xxx"           # SDK 自动识别
MAGNUS_ADDRESS   = "http://host:port"  # 后端地址
MAGNUS_JOB_ID    = "abc123"           # 当前 Job ID
MAGNUS_RESULT    = "$MAGNUS_HOME/workspace/.magnus_result"   # 写此文件 = 返回结果
MAGNUS_ACTION    = "$MAGNUS_HOME/workspace/.magnus_action"   # 写此文件 = 客户端执行
MAGNUS_METRICS_DIR = "$MAGNUS_HOME/workspace/metrics"        # 指标文件目录

# 容器内工作目录 = $MAGNUS_HOME/workspace/repository (git checkout)
```

### 2.4 结果回传模式

```python
# 小结果 → MAGNUS_RESULT (后端惰性读取)
echo '{"accuracy": 0.95, "loss": 0.12}' > "$MAGNUS_RESULT"

# 大文件 → MAGNUS_ACTION (SDK 自动执行)
SECRET=$(magnus custody ./outputs/checkpoint-1200)
echo "magnus receive $SECRET --output ./checkpoint-1200" > "$MAGNUS_ACTION"
```

### 2.5 文件中转 (FileSecret)

```python
# 蓝图中声明
Model = Annotated[FileSecret, {"label": "Base Model"}]

# 容器内接收
from magnus import download_file
download_file(model_secret, "/tmp/model")   # SDK 自动处理解压
# 或 CLI
# magnus receive "$MODEL_SECRET" --output /tmp/model

# 自动上传: SDK 端传本地路径即可
job_id = magnus.launch_blueprint("my-bp", args={"model": "/data/models/qwen.pt"})
# SDK → 上传文件 → 得到 magnus-secret:xxxx → 传给蓝图
```

---

## 三、submit_job 参数速查

```python
submit_job(
    # ── 必填 ──
    task_name,          # str
    entry_command,      # str
    repo_name,          # str: GitHub 仓库名

    # ── Git ──
    namespace="Rise-AGI",    # GitHub 组织
    branch=None,             # None=自动检测默认分支
    commit_sha=None,         # None=HEAD, "msg:v2.1"=搜索commit message

    # ── 硬件 ──
    gpu_type="cpu",          # GPU 型号, "cpu"=不用GPU
    gpu_count=0,             # GPU 数量
    cpu_count=None,          # None=集群默认(4)
    memory_demand=None,      # "32G", "1600M", None=集群默认
    ephemeral_storage=None,  # 容器临时磁盘, None=集群默认(10G)

    # ── 调度 ──
    job_type=JobType.A2,     # A1 > A2 > B1 > B2
    runner=None,             # None=集群默认
    container_image=None,    # None=集群默认(pytorch:2.5.1)

    # ── 可选 ──
    description=None,        # Markdown 格式
    system_entry_command=None,  # 宿主机侧预执行脚本
)
```

---

## 四、资源上限 (集群配置)

```yaml
cluster:
  max_cpu_count: 128            # CPU 上限
  max_memory_demand: 256G       # 内存上限
  default_cpu_count: 4          # 默认 CPU
  default_memory_demand: 1600M  # 默认内存
  default_ephemeral_storage: 10G
  default_container_image: docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
  gpus:
    - value: rtx5090
      limit: 4                  # 单任务最大 GPU 数
```

---

## 五、完整 Blueprint 模板

```python
# ============ 本地 IDE 辅助导入 (web 端忽略) ============
from magnus import submit_job, JobType, FileSecret
from typing import Annotated, Optional, Literal, List
# =====================================================

# ── 参数定义 ──
GpuCount = Annotated[int, {
    "label": "GPU 数量",
    "min": 1, "max": 8,
}]

Lr = Annotated[Optional[float], {
    "label": "学习率",
    "scope": "超参数",
    "min": 0, "max": 1,
}]

# ── 蓝图函数 ──
def blueprint(
    gpu_count: GpuCount = 4,
    learning_rate: Lr = None,
):
    cmd = f"python train.py --gpus {gpu_count}"
    if learning_rate is not None:
        cmd += f" --lr {learning_rate}"

    submit_job(
        task_name = f"Train-{gpu_count}gpu",
        entry_command = cmd,
        repo_name = "my-project",
        gpu_count = gpu_count,
        gpu_type = "rtx5090",
        job_type = JobType.A2,
    )
```

---

## 六、优先级速查

| JobType | 数值 | 可被抢占 | 适用场景 |
|---------|------|----------|----------|
| A1 | 4 | 否 | 生产/紧急任务 |
| A2 | 3 | 否 | 常规训练 |
| B1 | 2 | 是 | 批量/低优 |
| B2 | 1 | 是 | 下载/预处理/调试 |

---

## 七、本地运行 & 部署

```bash
# 本地执行 (安装 SDK 后蓝图代码可直接跑)
python blueprint.py

# 提交到远程
magnus run my-blueprint -- --gpu-count 4 --learning-rate 0.001

# 查看
magnus logs -1         # 最新任务日志
magnus status -1       # 最新任务状态
magnus job result -1   # 最新任务结果
```
