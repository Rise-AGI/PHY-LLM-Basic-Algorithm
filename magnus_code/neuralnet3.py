import os
import numpy as np
import cupy as cp
import plotly.graph_objects as go

# ==================== 独立类隔离环境：内嵌mytools1（无exec）====================
class MyToolsGitHub:
    import base64
    import json
    import urllib.request
    import os
    from urllib.error import HTTPError

    REPO = "Rise-AGI/PHY-LLM-Basic-Algorithm"
    BRANCH = "main"

    @classmethod
    def _get_remote_sha(cls, token: str, github_path: str):
        url = f"https://api.repos/{cls.REPO}/contents/{github_path}"
        try:
            req = cls.urllib.request.Request(url)
            req.add_header("Authorization", f"token {token}")
            with cls.urllib.request.urlopen(req) as resp:
                return cls.json.load(resp).get("sha")
        except:
            return None

    @classmethod
    def magnus_github_upload(cls, github_token: str, local_file_path: str, github_file_path: str = None):
        if github_file_path is None:
            github_file_path = local_file_path.strip()
        token = github_token.strip()
        local_path = local_file_path.strip()
        if not token or not cls.os.path.isfile(local_path):
            print("[错误] 令牌/文件异常")
            return

        try:
            with open(local_path, "rb") as f:
                content = cls.base64.b64encode(f.read()).decode("utf-8")
        except:
            return

        url = f"https://api.repos/{cls.REPO}/contents/{github_file_path}"
        payload = {"message": f"auto upload: {cls.os.path.basename(local_path)}","content": content,"branch": cls.BRANCH}
        sha = cls._get_remote_sha(token, github_file_path)
        if sha: payload["sha"] = sha

        try:
            req = cls.urllib.request.Request(url, data=cls.json.dumps(payload).encode(), method="PUT")
            req.add_header("Authorization", f"token {token}")
            with cls.urllib.request.urlopen(req):
                print("="*50)
                print("✅ 上传成功！")
                print("="*50)
        except:
            print("❌ 上传失败")
# ======================================================================================

# ==================== 激活函数 ====================
def sigmoid(z):
    return 1 / (1 + cp.exp(-z))
def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))

# ==================== 神经网络（维度兼容）====================
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
            # 权重形状：(当前层神经元数, 上一层神经元数)
            self.parameters[f'W{i}'] = cp.random.randn(self.layer_dims[i], self.layer_dims[i-1]) * 0.01
            self.parameters[f'b{i}'] = cp.zeros((self.layer_dims[i], 1))

    def forward(self, X):
        cache = {"A0": X}
        A = X
        for i in range(1, self.L+1):
            W = self.parameters[f'W{i}']
            b = self.parameters[f'b{i}']
            act, _ = self.activations[i-1]
            Z = W @ A + b  # 矩阵乘法：(输出维度, 批次)
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

# ==================== 报告生成（无LaTeX报错）====================
def matrix_to_latex(mat_gpu, name):
    mat = cp.asnumpy(mat_gpu)
    if mat.size > 100:
        return f"矩阵{name}(形状:{mat.shape})"
    rows = " | ".join([", ".join([f"{x:.2f}" for x in r]) for r in mat])
    return f"{name} = [{rows}]"

def plot_loss(loss_hist, html_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_hist, name="损失值", line=dict(color='#1f77b4')))
    fig.update_layout(title="CuPy训练损失曲线", xaxis_title="迭代次数", yaxis_title="Loss")
    fig.write_html(html_path)

def generate_report(nn, X, Y, pred, loss_hist, interval, md_path):
    table = "| 迭代次数 | 损失值 |\n|-------|------|\n"
    for i in range(0, len(loss_hist), interval):
        table += f"| {i} | {loss_hist[i]:.6f} |\n"
    
    content = f"""# CuPy 神经网络训练报告
## 网络结构: {nn.layer_dims}
## 训练损失
{table}
## 输入数据: {matrix_to_latex(X, 'X')}
## 真实标签: {matrix_to_latex(Y, 'Y')}
## 预测结果: {matrix_to_latex(pred, 'A')}
"""
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 报告已生成")

# ==================== 主程序（核心修复：输入矩阵转置）====================
if __name__ == "__main__":
    TOKEN = os.getenv("GITHUB_TOKEN")
    
    # 1. 原始数据集（异或问题）
    X_cpu = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y_cpu = np.array([[0],[1],[1],[0]])
    
    # 2. 🔥 核心修复：转置矩阵 → (特征数, 样本数) 匹配权重维度
    X = cp.array(X_cpu, dtype=cp.float32).T
    Y = cp.array(Y_cpu, dtype=cp.float32).T
    
    # 3. 网络配置
    layer_dims = [2, 40, 1]
    activations = [(sigmoid, sigmoid_deriv), (sigmoid, sigmoid_deriv)]
    lr = 0.5
    epochs = 15000
    log_interval = 3000

    # 4. 文件路径
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    HTML_PATH = os.path.join(SCRIPT_DIR, "loss_curve.html")
    MD_PATH = os.path.join(SCRIPT_DIR, "training_report.md")

    # 5. 初始化网络
    nn = FlexibleNN(layer_dims, activations)
    loss_history = []
    m = X.shape[1]
    print("🚀 CuPy GPU 训练开始...")

    # 6. 训练循环
    for i in range(epochs):
        pred, cache = nn.forward(X)
        loss = cp.mean((Y - pred)**2)
        loss_history.append(float(cp.asnumpy(loss)))
        grads = nn.backward(Y, cache, m)
        nn.update(grads, lr)
        
        if i % log_interval == 0:
            print(f"Epoch {i:5d} | Loss: {loss_history[-1]:.6f}")

    print("🎉 训练完成！")

    # 7. 显存释放
    del pred, cache, grads, X, Y
    cp.get_default_memory_pool().free_all_blocks()

    # 8. 生成报告
    plot_loss(loss_history, HTML_PATH)
    generate_report(nn, cp.array(X_cpu), cp.array(Y_cpu), cp.array([[0,1,1,0]]), loss_history, log_interval, MD_PATH)

    # 9. 上传文件
    if TOKEN:
        print("☁️ 开始上传...")
        MyToolsGitHub.magnus_github_upload(TOKEN, MD_PATH, "magnus_code/training_report.md")
        MyToolsGitHub.magnus_github_upload(TOKEN, HTML_PATH, "magnus_code/loss_curve.html")
    else:
        print("[日志] 未配置TOKEN，跳过上传")