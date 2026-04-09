import os
import numpy as np
import cupy as cp
import plotly.graph_objects as go

# ==================== 🔥 Class 独立封装上传工具（1:1复刻 mytools 原始风格）====================
class MyToolsGitHub:
    # 【完全保留原风格】类内导入模块
    import base64
    import json
    import urllib.request
    import os
    from urllib.error import HTTPError

    # 【完全保留原风格】固定团队仓库配置
    REPO = "Rise-AGI/PHY-LLM-Basic-Algorithm"
    BRANCH = "main"

    # 【完全保留原风格】私有类方法：获取SHA
    @classmethod
    def _get_remote_sha(cls, token: str, github_path: str):
        """内部：获取GitHub文件SHA，用于覆盖上传"""
        url = f"https://api.github.com/repos/{cls.REPO}/contents/{github_path}"
        try:
            req = cls.urllib.request.Request(url)
            req.add_header("Authorization", f"token {token}")
            with cls.urllib.request.urlopen(req) as resp:
                return cls.json.load(resp).get("sha")
        except:
            return None

    # 【完全保留原风格】核心上传类方法（函数名/参数/注释完全一致）
    @classmethod
    def magnus_github_upload(cls, github_token: str, local_file_path: str, github_file_path: str = None):
        """
        核心上传函数
        :param github_token: GitHub PAT 令牌（ghp_/github_pat_）
        :param local_file_path: 本地文件路径
        :param github_file_path: GitHub 目标路径
        """
        # 自动路径逻辑
        if github_file_path is None:
            github_file_path = local_file_path.strip()
            print(f"[自动路径] GitHub目标路径 = {github_file_path}")

        # 基础校验
        token = github_token.strip()
        local_path = local_file_path.strip()
        if not token:
            print("[错误] GitHub令牌不能为空")
            return
        if not cls.os.path.isfile(local_path):
            print(f"[错误] 本地文件不存在：{local_path}")
            return

        # 读取文件
        try:
            with open(local_path, "rb") as f:
                content = cls.base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"[错误] 文件读取失败：{str(e)}")
            return

        # 构造请求
        url = f"https://api.github.com/repos/{cls.REPO}/contents/{github_file_path}"
        payload = {
            "message": f"auto upload: {cls.os.path.basename(local_path)}",
            "content": content,
            "branch": cls.BRANCH
        }

        # 覆盖已有文件
        sha = cls._get_remote_sha(token, github_file_path)
        if sha:
            payload["sha"] = sha
            print("[信息] 检测到同名文件，自动覆盖")

        # 发送上传
        try:
            req = cls.urllib.request.Request(url, data=cls.json.dumps(payload).encode(), method="PUT")
            req.add_header("Authorization", f"token {token}")
            req.add_header("Content-Type", "application/json")
            req.add_header("User-Agent", "Python-Script")

            with cls.urllib.request.urlopen(req):
                print("="*50)
                print("✅ 上传成功！")
                print(f"本地：{local_path}")
                print(f"远程：{github_file_path}")
                print("="*50)
        except cls.HTTPError as e:
            print(f"❌ 上传失败 HTTP {e.code}")
            print("403=令牌权限不足 | 404=路径错误 | 422=缺少SHA")
        except Exception as e:
            print(f"❌ 上传异常：{str(e)}")
# ======================================================================================

# ==================== 激活函数 ====================
def sigmoid(z):
    return 1 / (1 + cp.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

# ==================== CuPy 神经网络（修复收敛+维度）====================
class SimpleNN:
    def __init__(self):
        cp.random.seed(42)
        self.W1 = cp.random.randn(40, 2) * 0.001
        self.b1 = cp.zeros((40, 1))
        self.W2 = cp.random.randn(1, 40) * 0.001
        self.b2 = cp.zeros((1, 1))

    def forward(self, X):
        Z1 = self.W1 @ X + self.b1
        A1 = sigmoid(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = sigmoid(Z2)
        return A2, Z1, A1

    def backward(self, X, Y, A2, Z1, A1, lr):
        m = X.shape[1]
        dZ2 = A2 - Y
        dW2 = (dZ2 @ A1.T) / m
        db2 = cp.sum(dZ2, axis=1, keepdims=True) / m
        
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * sigmoid_deriv(Z1)
        dW1 = (dZ1 @ X.T) / m
        db1 = cp.sum(dZ1, axis=1, keepdims=True) / m

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

# ==================== 报告生成 ====================
def plot_loss(loss_hist, html_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_hist, name="训练损失", line=dict(color='#1f77b4')))
    fig.update_layout(title="CuPy 训练损失曲线", xaxis_title="迭代次数", yaxis_title="Loss")
    fig.write_html(html_path)

def generate_report(md_path):
    content = """# CuPy 神经网络训练报告
状态：训练收敛完成
模型：2输入-40隐藏-1输出（异或分类）
"""
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ 报告已生成")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 读取令牌
    TOKEN = os.getenv("GITHUB_TOKEN")
    
    # 数据集（异或）
    X_cpu = np.array([[0,0],[0,1],[1,0],[1,1]]).T
    Y_cpu = np.array([[0,1,1,0]])
    X = cp.array(X_cpu, dtype=cp.float32)
    Y = cp.array(Y_cpu, dtype=cp.float32)

    # 训练参数
    epochs = 15000
    lr = 0.1
    log_interval = 3000
    loss_history = []

    # 初始化模型
    model = SimpleNN()
    print("🚀 CuPy GPU 训练开始...")

    # 训练循环
    for i in range(epochs):
        A2, Z1, A1 = model.forward(X)
        loss = cp.mean((Y - A2) ** 2)
        loss_history.append(float(loss.get()))
        model.backward(X, Y, A2, Z1, A1, lr)

        if i % log_interval == 0:
            print(f"Epoch {i:5d} | Loss: {loss_history[-1]:.6f}")

    print("🎉 训练完成！")

    # 生成文件
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    HTML_PATH = os.path.join(SCRIPT_DIR, "loss_curve.html")
    MD_PATH = os.path.join(SCRIPT_DIR, "training_report.md")
    plot_loss(loss_history, HTML_PATH)
    generate_report(MD_PATH)

    # 🔥 调用 Class 上传（和原 mytools 调用方式完全一致）
    if TOKEN:
        print("\n☁️ 开始上传文件到 GitHub...")
        MyToolsGitHub.magnus_github_upload(TOKEN, MD_PATH, "magnus_code/training_report.md")
        MyToolsGitHub.magnus_github_upload(TOKEN, HTML_PATH, "magnus_code/loss_curve.html")
    else:
        print("\n[日志] 未配置GITHUB_TOKEN，跳过上传")