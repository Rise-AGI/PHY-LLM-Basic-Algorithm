import requests

# ========== 配置 ==========
MAGNUS_HOST = "http://162.105.151.134:3011"
TOKEN = "sk-V-v0UMARpccPA_QwXGIoRDc9G2K67XiA"
GIT_REPO = "https://github.com/Rise-AGI/PHY-LLM-Basic-Algorithm"
ENTRY_CMD = """
pip install matplotlib numpy -i https://pypi.tuna.tsinghua.edu.cn/simple && 
python magnus_code/neuralnet1.py && 
# 托管容器内生成的文件（按需修改文件名）
magnus custody magnus_code/result.png && 
magnus custody magnus_code/output.txt
"""
# ========================

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "task_name": "phy_llm_task",
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
    
    # 🔥 关键：打印完整响应，查看真实返回内容
    print("✅ 请求状态码:", resp.status_code)
    print("🔥 后端完整返回:", resp.json())  # 这行能看到所有字段
    resp.raise_for_status()

    # 安全获取 job_id（防止字段名不一致）
    result = resp.json()
    job_id = result.get("job_id") or result.get("id") or "未找到任务ID"
    print(f"\n最终结果：任务ID = {job_id}")

except Exception as e:
    print(f"\n❌ 错误详情: {e}")