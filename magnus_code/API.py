import requests

# ========== 修复后的配置（仅修改这里） ==========
# 1. 修复：后端端口 8017，删除末尾斜杠
MAGNUS_HOST = "http://162.105.151.134:8017"
# 你的认证Token
TOKEN = "sk-V-v0UMARpccPA_QwXGIoRDc9G2K67XiA"
# 代码仓库
GIT_REPO = "https://github.com/Rise-AGI/PHY-LLM-Basic-Algorithm"
# 2. 修复：单行命令，用 && 分隔，删除无效的 magnus custody
ENTRY_CMD = "pip install matplotlib numpy -i https://pypi.tuna.tsinghua.edu.cn/simple && python magnus_code/neuralnet1.py"
# ======================================

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# 3. 修复：精简请求体，保证后端兼容
payload = {
    "blueprint_id": "hello-world",
    "entry_command": ENTRY_CMD,
    "container_image": "docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
    "gpu_count": 1,
    "cpu_count": 4,
    "memory_demand": "16G",
    "git_repo": GIT_REPO,
    "git_commit": "main"
}

# 调试：打印请求信息
print("正在提交任务到:", f"{MAGNUS_HOST}/api/jobs/submit")

try:
    resp = requests.post(f"{MAGNUS_HOST}/api/jobs/submit", json=payload, headers=headers)
    # 打印详细错误信息（方便排查）
    if resp.status_code != 200:
        print("后端返回错误详情:", resp.text)
    resp.raise_for_status()
    
    job_id = resp.json()["job_id"]
    print(f"✅ 任务提交成功，Job ID = {job_id}")
    
except Exception as e:
    print(f"❌ 提交失败: {e}")
    