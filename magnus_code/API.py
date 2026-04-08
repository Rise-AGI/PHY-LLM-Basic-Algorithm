import requests

# ========== 用户必须修改的部分 ==========
MAGNUS_HOST = "http://162.105.151.134:3011/"   # 你的 Magnus 后端地址
TOKEN = "sk-V-v0UMARpccPA_QwXGIoRDc9G2K67XiA"                         # 集群请替换为真实 token
GIT_REPO = "https://github.com/Rise-AGI/PHY-LLM-Basic-Algorithm"  # 你的代码仓库
ENTRY_CMD = """
pip install matplotlib numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
python magnus_code/neuralnet1.py
magnus custody magnus_code/neuralnet1.py"""          # 容器内要执行的命令
# ======================================

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "blueprint_id": "hello-world",
    "entry_command": ENTRY_CMD,
    "container_image": "docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
    "gpu_count": 1,
    "cpu_count": 4,
    "memory_demand": "16G",
    "ephemeral_storage": "10G",
    "priority": "B2",
    "git_repo": GIT_REPO,
    "git_commit": "main"  # 分支名或 commit hash
}

# 提交任务
resp = requests.post(f"{MAGNUS_HOST}/api/jobs/submit", json=payload, headers=headers)
resp.raise_for_status()
job_id = resp.json()["job_id"]
print(f"任务提交成功，Job ID = {job_id}")