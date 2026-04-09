import os
import numpy as np
import cupy as cp
import plotly.graph_objects as go

# ==================== 🔥 独立类隔离环境：原封不动内嵌mytools1（无exec）====================
# 仅新增：class 定义 + 代码缩进，mytools1 内部代码一字不改！
class MyToolsGitHub:
    # 👇 直接粘贴你的 mytools1 全部代码（仅缩进，无任何修改）
    # -*- coding: utf-8 -*-
    """
    Magnus 平台 → GitHub 文件上传工具库
    ✅ 纯Python内置库 | ✅ 无依赖 | ✅ 自动同路径上传 | ✅ 支持覆盖
    调用规则：
    1. 自动模式：main(令牌, 本地路径) → GitHub路径 = 本地路径
    2. 自定义模式：main(令牌, 本地路径, GitHub自定义路径)
    """
    import base64
    import json
    import urllib.request
    import os
    from urllib.error import HTTPError

    # 固定团队仓库配置（无需修改）
    REPO = "Rise-AGI/PHY-LLM-Basic-Algorithm"
    BRANCH = "main"

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

    @classmethod
    def magnus_github_upload(cls, github_token: str, local_file_path: str, github_file_path: str = None):
        """
        核心上传函数
        :param github_token: GitHub PAT 令牌（ghp_/github_pat_）
        :param local_file_path: Magnus 本地文件相对路径
        :param github_file_path: GitHub 目标路径（默认=本地路径）
        """
        # ================= 自动路径逻辑 =================
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
            req.add_header("User-Agent", "Magnus-Upload")

            with cls.urllib.request.urlopen(req):
                print("="*50)
                print("上传成功！")
                print(f"本地：{local_path}")
                print(f"远程：{github_file_path}")
                print("="*50)
        except cls.HTTPError as e:
            print(f"上传失败 HTTP {e.code}")
            print("403=令牌权限不足 | 404=路径错误 | 422=缺少SHA")
        except Exception as e:
            print(f"上传异常：{str(e)}")
# ======================================================================================

# ==================== 激活函数 (CuPy GPU) ====================
def sigmoid(z):
    return 1 / (1 + cp.exp(-z))
def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return cp.maximum(0, z)
def relu_deriv(z):
    return (z > 0).astype(cp.float64)

def tanh(z):
    return cp.tanh(z)
def tanh_deriv(z):
    return 1 - cp.tanh(z) ** 2

def linear(z):
    return z
def linear_deriv(z):
    return z

# ==================== 神经网络 (CuPy GPU) ====================
class FlexibleNN:
    def __init__(self, layer_dims, activations):
        self.layer_dims = layer_dims
        self.activations = activations
        self.parameters = {}
        self.L = len(layer_dims) - 1
        self._init_params()

    def _init_params(self):
        cp.random.seed(42)
        for i in range(1, self.L+1):
            self.parameters[f'W{i}'] = cp.random.randn(self.layer_dims[i], self.layer_dims[i-1]) * 0.1
            self.parameters[f'b{i}'] = cp.zeros((self.layer_dims[i], 1))

    def forward(self, X):
        cache = {"A0": X}
        A = X
        for i in range(1, self.L+1):
            W = self.parameters[f'W{i}']
            b = self.parameters[f'b{i}']
            act, _ = self.activations[i-1]
            Z = W @ A + b
            A = act(Z)
            cache[f'Z{i}'], cache[f'A{i}'] = Z, A
        return A, cache

    def backward(self, Y, cache, m):
        grads = {}
        A = cache[f'A{self.L}']
        dA = 2 * (A - Y) / m

        for i in reversed(range(1, self.L+1)):
            Z = cache[f'Z{i}']
            A_prev = cache[f'A{i-1}']
            W = self.parameters[f'W{i}']
            _, deriv = self.activations[i-1]
            
            dZ = dA * deriv(Z)
            grads[f'dW{i}'] = dZ @ A_prev.T
            grads[f'db{i}'] = cp.sum(dZ, axis=1, keepdims=True)
            if i > 1:
                dA = W.T @ dZ
        return grads

    def update(self, grads, lr):
        for i in range(1, self.L+1):
            self.parameters[f'W{i}'] -= lr * grads[f'dW{i}']
            self.parameters[f'b{i}'] -= lr * grads[f'db{i}']

# ==================== 报告生成 ====================
def matrix_to_latex(mat_gpu, name):
    mat = cp.asnumpy(mat_gpu)
    if mat.size > 100:
        return f"${name} \\in \\mathbb{{R}}^{{{mat.shape[0]} \\times {mat.shape[1]}}}$"
    rows = " \\\\ ".join([" & ".join([f"{x:.4f}" for x in r]) for r in mat])
    return f"${name} = \\begin{{bmatrix}} {rows} \\end{bmatrix}$"

def plot_loss(loss_hist, html_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_hist, name="Training Loss", line=dict(color='#1f77b4')))
    fig.update_layout(title="GPU 训练损失曲线", xaxis_title="Epoch", yaxis_title="Loss")
    fig.write_html(html_path)

def generate_report(nn, X, Y, pred, loss_hist, interval, html_path, md_path):
    table = "| Epoch | Loss |\n|-------|------|\n"
    for i in range(0, len(loss_hist), interval):
        table += f"| {i} | {loss_hist[i]:.6f} |\n"
    
    content = f"""# 神经网络训练报告
## 网络结构 {nn.layer_dims}
## 损失变化
{table}
## 预测结果
真实: {matrix_to_latex(Y, 'Y')}
预测: {matrix_to_latex(pred, 'A')}
"""
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"报告已生成: {md_path}")
    print(f"曲线已生成: {html_path}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    TOKEN = os.getenv("GITHUB_TOKEN")
    
    # 数据集
    X_cpu = np.array([[0,0,1,1],[0,1,0,1]])
    Y_cpu = np.array([[0,1,1,0]])
    
    layer_dimensions = [2, 40, 1]
    activation_functions = [(sigmoid, sigmoid_deriv), (sigmoid, sigmoid_deriv)]
    learning_rate = 0.5
    epochs = 15000
    log_interval = 3000

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    HTML_PATH = os.path.join(SCRIPT_DIR, "loss_curve.html")
    MD_PATH = os.path.join(SCRIPT_DIR, "training_report.md")

    X = cp.array(X_cpu)
    Y = cp.array(Y_cpu)
    m = X.shape[1]

    nn = FlexibleNN(layer_dimensions, activation_functions)
    loss_history = []
    print("CuPy GPU 训练开始...")

    for i in range(epochs):
        pred, cache = nn.forward(X)
        loss = cp.mean((Y - pred)**2)
        loss_history.append(float(cp.asnumpy(loss)))
        grads = nn.backward(Y, cache, m)
        nn.update(grads, learning_rate)
        if i % log_interval == 0:
            print(f"Epoch {i:5d} | Loss: {loss_history[-1]:.6f}")

    print("训练完成！")

    # 主动显存释放
    final_pred = pred
    del pred, cache, grads, X, Y
    cp.get_default_memory_pool().free_all_blocks()

    # 生成文件
    plot_loss(loss_history, HTML_PATH)
    generate_report(nn, cp.array(X_cpu), cp.array(Y_cpu), final_pred, loss_history, log_interval, HTML_PATH, MD_PATH)

    # 🔥 调用独立类中的上传函数（无exec、无导入、完美隔离）
    if TOKEN:
        try:
            print("开始上传...")
            MyToolsGitHub.magnus_github_upload(TOKEN, MD_PATH, "magnus_code/training_report.md")
            MyToolsGitHub.magnus_github_upload(TOKEN, HTML_PATH, "magnus_code/loss_curve.html")
        except Exception as e:
            print(f"❌ 上传失败: {str(e)}")
    else:
        print("[日志] 未配置TOKEN，跳过上传")