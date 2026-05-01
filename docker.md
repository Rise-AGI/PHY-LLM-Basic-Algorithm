# Magnus 容器镜像管理 & 构建笔记

---

## 一、镜像 URI 与底层存储

### 1.1 URI 格式

```
docker://registry/name:tag
```

示例: `docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`

### 1.2 HPC 模式 (Apptainer)

Apptainer 将 `docker://` 镜像拉取为 **只读 squashfs** 文件 (.sif)，存放在 SIF 缓存目录:

```
{magnus_root}/container_cache/
├── pytorch_pytorch_2.5.1-cuda12.4-cudnn9-runtime.sif
├── nvidia_cuda_12.4.0-base-ubuntu22.04.sif
└── ...
```

- **文件名规则**: `registry_name_tag.sif` (斜杠 → 下划线, 冒号 → 下划线)
- **容器缓存上限**: 80 GB, LRU 淘汰
- **仓库缓存上限**: 20 GB (仓库存放下载中间层)
- **原子写入**: 写入 `.sif.tmp` → `chmod 644` → `rename` → `.sif` (崩溃安全)
- **安全刷新**: 拉取到新 `.sif.tmp` → 原子替换旧 `.sif` — 旧镜像在刷新期间仍可用

### 1.3 Local 模式 (Docker)

直接使用 Docker daemon，不产生 .sif 文件:

```
docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
docker run -d --name magnus-job-{job_id} ...
```

### 1.4 镜像状态机

```
  ┌──────────┐   submit pull    ┌──────────┐   pull完成   ┌─────────┐
  │ unregistered │ ──────────────→ │  pulling  │ ───────────→ │ cached  │
  └──────────┘                   └──────────┘              └─────────┘
                                    ↑                          │
                                    │                     refresh
                                    │                          ↓
                               ┌──────────┐              ┌──────────┐
                               │refreshing│              │ missing  │
                               └──────────┘              └──────────┘
                                                           (DB有记录但.sif文件丢失)
```

| 状态 | 含义 |
|------|------|
| `cached` | .sif 文件就绪，可用 |
| `pulling` | 正在拉取 |
| `refreshing` | 正在刷新(后台拉新文件替换) |
| `failed` | 拉取失败 |
| `missing` | DB 有记录但文件不存在 |
| `unregistered` | 文件存在但 DB 无记录 (异常路径) |

---

## 二、Web UI 操作镜像

### 2.1 页面入口

`{frontend}/images` — 镜像管理页面

### 2.2 功能总览

| 功能 | 操作入口 |
|------|----------|
| 查看列表 | 表格/卡片，显示 URI、状态、大小、所有者 |
| 搜索 | 按 URI 关键词过滤 |
| 按用户筛选 | 下拉选择 "全部/我的" |
| 预热(拉取) | Preheat 抽屉 → 输入 URI → 提交 |
| 查看详情 | 点击行/卡片 → 详情抽屉 |
| 刷新 | 详情抽屉 → Refresh 按钮 (仅 cached 状态) |
| 删除 | 详情抽屉 → Delete 按钮 (禁止 pulling/refreshing 时删除) |

### 2.3 预热 (Preheat) 流程

```
1. 点击 "预热镜像" 按钮 → 打开抽屉
2. 输入 URI (如 docker://pytorch/pytorch:2.5.1)
3. 点击提交 → POST /api/images
4. 状态变为 pulling → 前端每 5s 轮询直到 cached
5. 镜像就绪，可在蓝图中引用
```

Web UI 的预热适用于 **不紧急的预下载**，这样作业提交时镜像已缓存，避免拉取等待。

### 2.4 状态颜色标识

| 状态 | 颜色 | 行为 |
|------|------|------|
| cached | 绿 | 可刷新/可删除 |
| pulling | 蓝 | 只读(按钮禁用) |
| refreshing | 黄 | 只读(按钮禁用) |
| failed | 红 | 可删除 |
| missing | 红 | 可删除 |
| unregistered | 灰 | — |

### 2.5 刷新 (Refresh)

- 仅在 `cached` 状态可用
- 后台拉取新版本到临时文件 → 原子替换旧 .sif
- 刷新期间旧镜像继续可用
- 用于更新 tag 不固定的镜像 (如 `:latest`)

---

## 三、API / SDK 操作镜像

### 3.1 REST API

所有镜像操作定义在 `back_end/server/routers/images.py`:

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/images` | 列出镜像 (支持 ?search=&user_id=) |
| POST | `/api/images` | 拉取镜像 (body: {uri}) → 返回 202 |
| POST | `/api/images/{id}/refresh` | 刷新镜像 → 返回 202 |
| DELETE | `/api/images/{id}` | 删除镜像 |
| POST | `/api/images/{id}/transfer` | 转移所有者 (body: {user_id}) |

**异步模式说明**:
- POST 拉取/刷新立即返回 `202 Accepted`
- 后台 `_do_pull()` 任务在服务端执行
- 通过 GET `/api/images` 轮询状态直到 `cached` 或 `failed`

### 3.2 Python SDK

```python
import magnus

# 配置连接
magnus.configure(address="http://xxx:3011/", token="sk-xxx")

# 拉取镜像 (异步)
magnus.pull_image("docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
# → 返回 202, 后台任务执行

# 列出镜像
images = magnus.list_images(search="pytorch")
for img in images:
    print(img["uri"], img["status"], img["size_bytes"])

# 刷新镜像 (异步, 仅 cached 状态)
magnus.refresh_image(image_id="img_xxx")

# 删除镜像
magnus.remove_image(image_id="img_xxx")
```

### 3.3 API vs Web UI 选择

| 场景 | 推荐方式 | 原因 |
|------|----------|------|
| 作业脚本自动拉取 | SDK `pull_image()` | 可嵌入 pipeline |
| 管理员批量管理 | API 直接调用 | 支持脚本化 |
| 临时查看镜像列表 | Web UI | 可视化，操作便捷 |
| 调试/排查 | Web UI | 状态颜色直观，可快速删除异常镜像 |
| 蓝图中使用 | 参数 `container_image` | 直接引用，无需手动拉取 |

---

## 四、在蓝图中使用镜像

### 4.1 container_image 参数

```python
submit_job(
    task_name = "MyTask",
    entry_command = "python train.py",
    repo_name = "my-project",

    # ── 镜像 ──
    container_image = "docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
    # 或使用已有镜像名 (不含 docker:// 前缀的 SIF 文件名)
    # container_image = "pytorch_pytorch_2.5.1-cuda12.4-cudnn9-runtime.sif",
)
```

- **不传 `container_image`** → 使用集群默认镜像 (`configs/magnus_config.yaml` 中的 `default_container_image`)
- **传入 URI** → 调度器自动确保镜像已缓存 (不存在则自动拉取)
- **不同任务可以用不同镜像** — 不在一个蓝图里耦合

### 4.2 镜像选择策略

| 场景 | 推荐镜像 | GPU |
|------|----------|-----|
| PyTorch 训练 | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` | rtx5090/a100 |
| 纯 CPU 处理 | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` 或更小的 ubuntu | cpu |
| 下载/预处理 | 自定义轻量镜像 | cpu |
| 多 Python 包依赖 | 自定义镜像 (见下一章) | 按需 |

### 4.3 自动拉取时机

```
作业提交 → 调度器检查 container_image
  ├── 已有缓存 (SIF 存在) → 直接使用
  └── 无缓存 → _do_pull() 后台拉取
        ├── 小镜像 (几秒 ~ 几分钟)
        └── 大镜像 (数十分钟)
```

**重要**: 首次使用未缓存的镜像，作业会进入 PENDING 直到拉取完成。如果作业需要立即运行，应先用预热功能提前拉取。

---

## 五、配置多个 Python 宏包 — 自定义镜像

### 5.1 推荐方案: 基于 uv-runtime 构建

`docker/uv-runtime/Dockerfile` 是 Magnus 官方提供的最小运行时:

```dockerfile
FROM ubuntu:22.04

# 系统依赖
RUN apt-get update && apt-get install -y \
    curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# uv (从 context 复制，避免网络下载 install.sh)
COPY uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv

# Python (uv 托管)
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv python install 3.14
```

**扩展为项目镜像**:

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    curl ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── uv + Python ──
COPY uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv python install 3.14

# ── 项目依赖 (提前缓存) ──
WORKDIR /opt/build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-install-workspace

# ── 运行时环境变量 (关键！) ──
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_CACHE_DIR=/opt/uv/cache
ENV UV_LINK_MODE=copy
ENV PATH="/opt/uv/python/bin:/usr/local/bin:$PATH"
```

### 5.2 直接在容器内安装 (不构建镜像)

如果不想维护自定义镜像，可以在 `entry_command` 中安装:

```bash
set -e
uv sync --frozen
uv run python train.py --lr 0.001
```

缺点: 每次作业都重复安装，耗时较长。

### 5.3 Dockerfile 构建命令

```bash
# 构建
docker build --network=host \
    --build-arg HTTP_PROXY=http://your-proxy:port \
    --build-arg HTTPS_PROXY=http://your-proxy:port \
    -t your-image:tag .

# 推送到 registry
docker push your-image:tag
```

### 5.4 构建建议

| 建议 | 原因 |
|------|------|
| 锁定 uv 版本 | 避免构建环境不一致 |
| 使用 `uv sync --frozen` | 确保锁文件与依赖完全一致 |
| 将 uv 二进制 COPY 而非 curl 安装 | 避免构建时网络失败 |
| 构建时指定 platform | `--platform linux/amd64` 保证与集群一致 |

---

## 六、关键陷阱与解决

### 6.1 `/root/` 空目录陷阱 (rootlesskit + containall)

**现象**: 容器内 `/root/` 是空 tmpfs，`uv python install` 安装到 `/root/.local/bin/uv` 消失。

**原因**: HPC 模式下 Apptainer 使用 rootlesskit, containall 参数将 `/root/` 挂载为空 tmpfs。

**解决**: 所有工具/二进制安装到 `/usr/local/bin/` 或 `/opt/`:

```dockerfile
# 错误: uv 安装在 /root/.local/bin/
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 正确: uv 复制到系统路径
COPY uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv
```

### 6.2 跨文件系统硬链接失败

**现象**: `uv sync` 报 cross-device link 错误。

**原因**: uv 缓存位于 SIF (squashfs, 只读) 层，但 venv 在可写 bind mount 上。文件系统不同，硬链接不可用。

**解决**: 设置 `UV_LINK_MODE=copy`

```dockerfile
ENV UV_LINK_MODE=copy
```

### 6.3 uv warmup 缓存不命中

**现象**: warmup 阶段安装了依赖，但作业运行时重新下载。

**原因**: `uv pip install` 和 `uv sync` 使用不同的缓存格式。warmup 用 `uv pip install` 则作业时 `uv sync` 缓存命中率为 0%。

**解决**: warmup 必须用 `uv sync --frozen`:

```dockerfile
# ✓ 正确: warmup
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-install-workspace

# 作业 entry_command 中用同样的 uv sync
cd /workspace/repository
uv sync --frozen
uv run python train.py
```

### 6.4 镜像构建时的代理

```bash
# 构建时设置代理 (集群环境通常需要)
docker build --network=host \
    --build-arg HTTP_PROXY=http://proxy:port \
    --build-arg HTTPS_PROXY=http://proxy:port \
    -t my-image:tag .
```

### 6.5 torch 2.5.1 + transformers 5.7.0 不兼容（sft-base 已知问题）

**现象**: 加载 `.bin` 格式模型时：
```
ValueError: Due to a serious vulnerability issue in `torch.load`
...require users to upgrade torch to at least v2.6
```

**原因**: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` 基础镜像的 torch 是 2.5.1。`sft-base:v1`/`v2` 预装 `transformers==5.7.0`（最新版），该版本因 CVE-2025-32434 禁止 torch < 2.6 加载 `.bin` 权重。`.safetensors` 格式无此限制。

**短期解决**（已内置）: 蓝图中 monkey-patch `modeling_utils.check_torch_load_is_safe`。详见 SFT.md §8.8。

**长期方案**: 下次构建镜像时，改用 `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` 或更新版本作为基础镜像。或提前将模型转为 `.safetensors` 格式。

---

## 七、完整模板汇总

### 7.1 自定义镜像 Dockerfile

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

### 7.2 蓝图中引用镜像

```python
def blueprint(
    lr: Lr = 0.001,
    epochs: int = 10,
):
    submit_job(
        task_name = "train-with-custom-image",
        entry_command = f"""
        set -e
        cd /workspace/repository
        uv sync --frozen
        uv run python train.py --lr {lr} --epochs {epochs}
        """,
        repo_name = "my-project",
        container_image = "docker://your-registry/my-image:tag",
        gpu_type = "rtx5090",
        gpu_count = 1,
    )
```

### 7.3 预热 + 提交 (两步)

```bash
# 先预热
magnus pull "docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"

# 检查状态
magnus list-images

# 提交作业 (使用默认镜像，已被预热)
magnus run my-bp -- --lr 0.001
```

---

## 八、使用阿里云 ACR (Alibaba Container Registry)

### 8.1 登录

```bash
docker login --username=<your-username> <your-registry>.cn-beijing.personal.cr.aliyuncs.com
```

- 用户名为阿里云账号全名
- 密码为开通服务时设置的密码（可在访问凭证页面修改）
- RAM 用户（子账号）登录时，企业别名不能带英文句号（`.`）

### 8.2 拉取镜像

```bash
docker pull <your-registry>.cn-beijing.personal.cr.aliyuncs.com/<your-namespace>/sft-base:[镜像版本号]
```

### 8.3 推送镜像

```bash
# 登录
docker login --username=<your-username> <your-registry>.cn-beijing.personal.cr.aliyuncs.com

# 给本地镜像打 ACR tag
docker tag [ImageId] <your-registry>.cn-beijing.personal.cr.aliyuncs.com/<your-namespace>/sft-base:[镜像版本号]

# 推送
docker push <your-registry>.cn-beijing.personal.cr.aliyuncs.com/<your-namespace>/sft-base:[镜像版本号]
```

### 8.4 内网地址（VPC）

从阿里云 ECS 推送时使用内网地址，速度更快且不损耗公网流量：

```
<your-registry>.cn-beijing.personal.cr.aliyuncs.com
```

### 8.5 操作示例

```bash
# 查看本地镜像
docker images
# REPOSITORY                                      TAG       IMAGE ID
# crpi-.../<your-namespace>/sft-base                         v1        abc123def456

# 重新打 tag 并推送
docker tag abc123def456 <your-registry>.cn-beijing.personal.cr.aliyuncs.com/<your-namespace>/sft-base:v1
docker push <your-registry>.cn-beijing.personal.cr.aliyuncs.com/<your-namespace>/sft-base:v1
```

### 8.6 自动化脚本

配套脚本 `push_to_acr.py` 一键构建+推送：

```bash
python train/push_to_acr.py -u <username> -p <password>

# 仅构建不推送（调试用）
python train/push_to_acr.py -u <username> -p <password> --no-push

# 指定 tag
python train/push_to_acr.py -u <username> -p <password> --tag v2
```

Windows 用户也可使用 `push_to_acr.cmd`：

```cmd
push_to_acr.cmd -u <your-username> -p your_password
```

### 8.7 当前项目 ACR 配置

| 配置项 | 值 |
|--------|-----|
| Registry | `<your-registry>.cn-beijing.personal.cr.aliyuncs.com` |
| 命名空间 | `<your-namespace>` |
| 仓库 | `sft-base` |
| 当前 Tag | `v1` |
| 镜像全称 | `<your-registry>.cn-beijing.personal.cr.aliyuncs.com/<your-namespace>/sft-base:v1` |
| Magnus URI | `docker://<your-registry>.cn-beijing.personal.cr.aliyuncs.com/<your-namespace>/sft-base:v1` |

---

## 九、参考文件

| 文件 | 内容 |
|------|------|
| `back_end/server/routers/images.py` | 镜像 API 完整实现 |
| `back_end/server/_resource_manager.py` | SIF 缓存管理、LRU 淘汰、原子写入 |
| `front_end/src/app/(main)/images/page.tsx` | Web UI 镜像管理页面 |
| `front_end/src/components/images/image-table.tsx` | 镜像表格/卡片组件 |
| `docs/about_uv_image.md` | 容器镜像 uv 配置最佳实践 |
| `docker/uv-runtime/Dockerfile` | 最小 uv 运行时 |
| `docker/magnus-runtime/Dockerfile` | 完整 Magnus 运行时 |
