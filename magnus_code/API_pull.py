import requests

TOKEN = "sk-V-v0UMARpccPA_QwXGIoRDc9G2K67XiA"
MAGNUS_HOST = "http://162.105.151.134:3011"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
JOB_ID = "你的任务ID"  # 替换成上面获取的ID

# 下载容器内生成的图片文件
file_path = "magnus_code/xor_loss_curve.png"
resp = requests.get(
    f"{MAGNUS_HOST}/api/jobs/{JOB_ID}/repository/{file_path}",
    headers=HEADERS
)

# 保存到本地
with open("xor_loss_curve.png", "wb") as f:
    f.write(resp.content)

print("✅ 图片下载成功！")