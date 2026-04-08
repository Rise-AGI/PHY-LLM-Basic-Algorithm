import time
import requests

# ========== 配置 ==========
MAGNUS_HOST = "http://162.105.151.134:3011"
TOKEN = "sk-V-v0UMARpccPA_QwXGIoRDc9G2K67XiA"
GIT_REPO = "https://github.com/Rise-AGI/PHY-LLM-Basic-Algorithm"

# 🔥 修复：彻底删除 magnus custody 命令（容器里没有这个命令！）
ENTRY_CMD = """
pip install plotly
GITHUB_TOKEN="ghp_XXX" python magnus_code/neuralnet3.py
"""
# ========================

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "task_name": "API上传",
    "repo_name": "PHY-LLM-Basic-Algorithm",
    "blueprint_id": "hello-world",
    "entry_command": ENTRY_CMD,
    "container_image": "docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
    "gpu_count": 0,
    "cpu_count": 4,
    "memory_demand": "16G",
    "git_repo": GIT_REPO,
    "git_commit": "main"
}
# 查询任务状态
def get_status(job_id):
    try:
        return requests.get(f"{MAGNUS_HOST}/api/jobs/{job_id}", headers=headers).json()
    except:
        return {}

# 主流程
if __name__ == "__main__":
    print("📤 提交任务至 Magnus...")
    resp = requests.post(f"{MAGNUS_HOST}/api/jobs/submit", json=payload, headers=headers)
    job_id = resp.json()["id"]
    print(f"✅ 任务提交成功 | ID: {job_id}")
    print("-" * 60)

    # 30秒轮询状态
    while True:
        status = get_status(job_id).get("status", "未知")
        print(f"[{time.strftime('%H:%M:%S')}] 任务状态: {status}")

        if status == "Success":
            print("-" * 60)
            break
        if status == "Failed":
            print("❌ 任务执行失败")
            break

        time.sleep(1)
