"""
在 Magnus 集群内下载模型到 /data/<用户名>/models/ 持久目录。

用法：
    python download_model.py
    python download_model.py --model Qwen/Qwen2.5-1.5B   # 默认值
    python download_model.py --model Qwen/Qwen2.5-7B      # 换成其他模型
"""

import sys
import argparse
from magnus import execute_job

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="deepseek-math-7b-base", help="ModelScope 模型 ID")
args = parser.parse_args()

model_id = args.model
# ModelScope 的 namespace 用小写，Qwen/Qwen2.5-1.5B → qwen/Qwen2.5-1.5B
ms_model_id = model_id[0].lower() + model_id[1:]

entry_command = f"""
set -e
pip install -q modelscope

echo "=== 当前用户 ==="
whoami
USERNAME=$(whoami)
SAVE_DIR="/data/$USERNAME/models/{model_id.split('/')[-1]}"
echo "模型将保存到: $SAVE_DIR"

# 如果已存在就跳过
if [ -f "$SAVE_DIR/config.json" ]; then
    echo "[跳过] 模型已存在: $SAVE_DIR"
    exit 0
fi

mkdir -p "$SAVE_DIR"
echo "=== 开始从 ModelScope 下载 {ms_model_id} ==="
python3 -c "
from modelscope import snapshot_download
path = snapshot_download('{ms_model_id}', local_dir='/tmp/model_download')
print('下载完成，临时路径:', path)
"

echo "=== 移动到持久目录 ==="
cp -r /tmp/model_download/* "$SAVE_DIR/"
echo "=== 完成！模型已保存到: $SAVE_DIR ==="
ls "$SAVE_DIR"
"""

print(f"[1/2] 提交下载作业，模型: {model_id}")
print(f"      ModelScope ID: {ms_model_id}")
print(f"      目标路径: /data/<用户名>/models/{model_id.split('/')[-1]}")
print()

execute_job(
    task_name     = f"DownloadModel-{model_id.split('/')[-1]}",
    description   = f"下载 {model_id} 到集群持久存储",
    entry_command = entry_command,
    namespace     = "Rise-AGI",
    repo_name     = "OpenFundus",
    cpu_count= 6,           # CPU 核心数
    memory_demand="16G",    # 内存，单位 M/G
    ephemeral_storage="50G",# 容器临时磁盘空间（overlay 大小）
    gpu_count     = 0,
    gpu_type      = "a100",
    job_type      = "A2",
    execute_action = False,
)

print()
print("[2/2] 作业已提交！")
print("请去 Magnus 网页查看作业日志，下载完成后日志末尾会显示模型路径。")
print("下载完成后把该路径填入蓝图的「模型路径」字段即可。")
