
import sys
import argparse
import magnus


DEFAULT_ADDRESS = "http://162.105.151.134:3011/"
DEFAULT_TOKEN   = "sk-V-XXX"  #!!!  需修改  !!!  



def main():
    parser = argparse.ArgumentParser(description="一键配置 Magnus 并提交模型下载作业")

    # !!! 需修改 !!!(必须:发布者/名称, 之后程序自动识别名称)

    parser.add_argument("--model",  default="deepseek-ai/deepseek-math-7b-base",  
                        help="ModelScope 模型 ID")
    parser.add_argument("--address", default=DEFAULT_ADDRESS,
                        help="Magnus 服务器地址 (默认: %(default)s)")
    parser.add_argument("--token",   default=DEFAULT_TOKEN,
                        help="Magnus Trust Token (默认: 已配置)")
    args = parser.parse_args()

    #自动匹配名称
    model_id    = args.model
    ms_model_id = model_id[0].lower() + model_id[1:]
    model_name  = model_id.split("/")[-1]

    # ── 第 1 步：配置 Magnus（等价于 magnus login） ────────────
    print(f"[1/3] 配置 Magnus 连接...")
    print(f"      地址  : {args.address}")
    print(f"      Token : {args.token[:8]}...{args.token[-4:]}")
    magnus.configure(address=args.address, token=args.token)

    # ── 第 2 步：提交下载作业 ────────────────────────────────
    entry_command = f"""
set -e
pip install -q modelscope

echo "=== 当前用户 ==="
whoami
USERNAME=$(whoami)
SAVE_DIR="/data/$USERNAME/models/{model_name}"
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

    print(f"[2/3] 提交下载作业，模型: {model_id}")
    print(f"      ModelScope ID: {ms_model_id}")
    print(f"      目标路径     : /data/<用户名>/models/{model_name}")
    print()

    #!!!    需修改  !!!

    magnus.execute_job(
        task_name      = f"DownloadModel-{model_name}",
        description    = f"下载 {model_id} 到集群持久存储",
        entry_command  = entry_command,
        namespace      = "Rise-AGI",
        repo_name      = "OpenFundus",
        gpu_count      = 0,
        gpu_type       = "cpu",
        cpu_count      = 4,
        memory_demand  = "8G",
        ephemeral_storage = "25G",
        job_type       = "B2",
        execute_action = False,
    )

    # ── 第 3 步：完成提示 ────────────────────────────────────
    print()
    print("[3/3] 作业已提交！")
    print("请去 Magnus 网页查看作业日志，下载完成后日志末尾会显示模型路径。")
    print("下载完成后把该路径填入蓝图的「模型路径」字段即可。")


if __name__ == "__main__":
    main()
