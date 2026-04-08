import requests

# ========== 配置 ==========
MAGNUS_HOST = "http://162.105.151.134:3011"
TOKEN = "sk-V-v0UMARpccPA_QwXGIoRDc9G2K67XiA"
GIT_REPO = "https://github.com/Rise-AGI/PHY-LLM-Basic-Algorithm"

# 🔥 修复：彻底删除 magnus custody 命令（容器里没有这个命令！）
ENTRY_CMD = """
pip install matplotlib numpy -i https://pypi.tuna.tsinghua.edu.cn/simple && 
python magnus_code/neuralnet1.py
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
    "gpu_count": 1,
    "cpu_count": 4,
    "memory_demand": "16G",
    "git_repo": GIT_REPO,
    "git_commit": "main"
}

try:
    resp = requests.post(f"{MAGNUS_HOST}/api/jobs/submit", json=payload, headers=headers, timeout=30)
    print("✅ 请求状态码:", resp.status_code)
    result = resp.json()
    job_id = result.get("id")
    print(f"\n最终结果：任务ID = {job_id}")
    print(f"任务状态 = {result['status']}")

except Exception as e:
    print(f"\n❌ 错误详情: {e}")
