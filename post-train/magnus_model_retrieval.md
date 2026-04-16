# Magnus 训练模型读取方法

> 基于 [Rise-AGI/magnus](https://github.com/Rise-AGI/magnus) 源代码分析

---

## 核心机制：File Custody + Apptainer Bind

Magnus 不直接挂载模型文件，而是通过 **文件托管 (File Custody)** 中转层，用 `magnus-secret:...` token 传递文件。

**Token 格式：**
```
magnus-secret:{prime}-{word}-{word}-{word}
# 示例：magnus-secret:7919-calm-boat-fire
```

---

## 方法一：在容器内用 CLI 读取（推荐）

```bash
# 在 Job 的入口脚本中
magnus receive "magnus-secret:7919-calm-boat-fire" --output /tmp/my_model

# 然后直接使用
python eval.py --model-path /tmp/my_model
```

---

## 方法二：在容器内用 Python SDK 读取

```python
from magnus import download_file
import os

model_secret = os.environ.get("MODEL_SECRET")  # 由 blueprint 传入的参数
local_path = download_file(model_secret, "/tmp/my_model")
# local_path 即为下载后的本地路径
```

---

## Apptainer Bind 配置

Magnus 通过 `APPTAINER_BIND` 环境变量控制挂载。

### 默认挂载路径

| 宿主机路径 | 容器内路径 | 说明 |
|---|---|---|
| `{server.root}/workspace/jobs/{job_id}/` | `/magnus/workspace/` | 工作目录（自动挂载） |
| `{server.root}/workspace/jobs/{job_id}/ephemeral_overlay.img` | writable layer | 容器可写层（rootless 模式） |

### 在 magnus_config.yaml 中配置额外挂载

如需挂载集群共享存储中的模型，修改配置文件：

```yaml
cluster:
  default_system_entry_command: |-
    mounts=(
      "/home:/home"
      "/opt/miniconda3:/opt/miniconda3"
      "/shared/models:/models"      # 挂载模型目录到容器内
    )
    export APPTAINER_BIND=$(IFS=,; echo "${mounts[*]}")
```

配置后，容器内可直接访问 `/models/your_checkpoint`。

### Apptainer 运行模式对比

| 模式 | 可写层类型 | 命令参数 |
|---|---|---|
| rootless apptainer | sparse ext3 overlay image | `--containall --overlay ephemeral_overlay.img` |
| setuid apptainer | RAM tmpfs | `--contain --writable-tmpfs` |
| 裸运行 | 宿主机文件系统 | `MAGNUS_CONTAIN_LEVEL=none` |

---

## 将训练完的模型导出

训练结束后，在 Job 入口脚本中执行：

```bash
# 1. 将模型上传至 Magnus 文件托管，获得 secret
magnus send /tmp/trained_model > /tmp/secret.txt

# 2. 将 secret 写入 MAGNUS_ACTION，平台将自动下载
cat /tmp/secret.txt > "$MAGNUS_ACTION"
```

平台会自动把模型下载到你指定的位置。

---

## 关键注意事项

1. **File Custody 有 TTL**：默认 1 小时，最长 1 天。不适合长期存储大模型。
   - 大模型建议放在集群 NFS / S3，通过 `system_entry_command` 的 bind mount 挂进容器。

2. **容器内文件位置**：下载到容器内的文件默认在 `/magnus/.tmp/`（writable overlay 层），不在宿主机 bind mount 目录内。

3. **环境变量参考**：

   | 变量名 | 说明 | 示例 |
   |---|---|---|
   | `MAGNUS_HOME` | 容器根目录 | `/magnus` |
   | `MAGNUS_TOKEN` | SDK 认证 token | `sk-...` |
   | `MAGNUS_ADDRESS` | 后端服务地址 | `http://server:3011` |
   | `MAGNUS_JOB_ID` | 当前 Job ID | `abc123` |
   | `MAGNUS_RESULT` | 结果文件路径 | `/magnus/workspace/.magnus_result` |
   | `MAGNUS_ACTION` | 动作文件路径 | `/magnus/workspace/.magnus_action` |
   | `MAGNUS_METRICS_DIR` | 指标输出目录 | `/magnus/workspace/metrics` |

---

## 相关源码位置

| 文件 | 说明 |
|---|---|
| `back_end/server/_scheduler.py` | Wrapper 生成、bind mount、apptainer 执行逻辑 |
| `back_end/server/_file_custody_manager.py` | 文件托管后端、secret token 生成 |
| `sdks/python/src/magnus/http_download.py` | `download_file()` 实现 |
| `sdks/python/src/magnus/file_transfer.py` | 文件中转目录检测 |
| `configs/magnus_config.yaml.example` | 完整配置示例 |
