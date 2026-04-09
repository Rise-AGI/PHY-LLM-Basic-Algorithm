import os
import numpy as np
import cupy as cp
import plotly.graph_objects as go
import csv

# ====================== 🔥 核心配置区：仅改这里！======================
# 1. 网络结构：[输入维度, 隐藏层1, 隐藏层2..., 输出维度]
LAYER_DIMS = [2, 40, 20, 1]
# 2. 训练超参数
EPOCHS = 15000
LEARNING_RATE = 0.5
LOG_INTERVAL = 3000  # 每3000轮保存一次W/B矩阵
# 3. 数据模式：manual=手动导入(异或) | csv=CSV文件导入
DATA_MODE = "manual"
CSV_PATH = "dataset.csv"  # CSV数据路径（仅CSV模式生效）
# ====================================================================

# ==================== 工具：CSV数据加载函数 ====================
def load_csv_data(csv_path):
    """从CSV加载数据：最后一列是标签，前面是特征"""
    X_list, Y_list = [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            X_list.append(list(map(float, row[:-1])))
            Y_list.append([float(row[-1])])
    # 转为CuPy矩阵 + 转置匹配网络维度 (特征数, 样本数)
    X = cp.array(X_list, dtype=cp.float32).T
    Y = cp.array(Y_list, dtype=cp.float32).T
    print(f"✅ CSV数据加载完成 | 输入形状:{X.shape} | 标签形状:{Y.shape}")
    return X, Y

# ==================== 工具：权重W/偏置B导出为CSV ====================
def save_params_to_csv(params, save_dir="trained_params", epoch="final"):
    """保存每层W/B矩阵为CSV文件"""
    os.makedirs(save_dir, exist_ok=True)
    for key, value in params.items():
        # CuPy转NumPy保存
        mat = cp.asnumpy(value)
        file_path = os.path.join(save_dir, f"{key}_epoch_{epoch}.csv")
        np.savetxt(file_path, mat, delimiter=",", fmt="%.6f")
    print(f"✅ 【{epoch}轮】参数已保存至 {save_dir}/")

# ==================== Class 上传工具（原版风格保留）====================
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
        url = f"https://api.github.com/repos/{cls.REPO}/contents/{github_path}"
        try:
            req = cls.urllib.request.Request(url)
            req.add_header("Authorization", f"token {token}")
            with cls.urllib.request.urlopen(req) as resp:
                return cls.json.load(resp).get("sha")
        except:
            return None

    @classmethod
    def magnus_github_upload(cls, github_token: str, local_file_path, github_file_path=None):
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
            print("[错误] 文件读取失败")
            return
        url = f"https://api.github.com/repos/{cls.REPO}/contents/{github_file_path}"
        payload = {"message": f"auto upload: {cls.os.path.basename(local_path)}","content": content,"branch": cls.BRANCH}
        sha = cls._get_remote_sha(token, github_file_path)
        if sha: payload["sha"] = sha
        try:
            req = cls.urllib.request.Request(url, data=cls.json.dumps(payload).encode(), method="PUT")
            req.add_header("Authorization", f"token {token}")
            req.add_header("Content-Type", "application/json")
            req.add_header("User-Agent", "Python-Script")
            with cls.urllib.request.urlopen(req):
                print("="*50)
                print(f"✅ 上传成功：{github_file_path}")
                print("="*50)
        except:
            print("❌ 上传失败")

# ==================== 激活函数 ====================
def sigmoid(z):
    return 1 / (1 + cp.exp(-z))
def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

# ==================== 灵活神经网络（自动适配层数/维度）====================
class FlexibleNet:
    def __init__(self, layer_dims):
        cp.random.seed(42)
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1
        self.params = {}
        # 自动初始化权重
        for i in range(1, self.L + 1):
            in_dim = layer_dims[i-1]
            out_dim = layer_dims[i]
            self.params[f'W{i}'] = cp.random.randn(out_dim, in_dim) * cp.sqrt(1. / in_dim)
            self.params[f'b{i}'] = cp.zeros((out_dim, 1))

    def forward(self, X):
        cache = {}
        A = X
        cache['A0'] = A
        for i in range(1, self.L + 1):
            W, b = self.params[f'W{i}'], self.params[f'b{i}']
            Z = cp.dot(W, A) + b
            A = sigmoid(Z)
            cache[f'Z{i}'], cache[f'A{i}'] = Z, A
        return A, cache

    def backward(self, X, Y, output, cache, lr):
        m, grads = X.shape[1], {}
        # 输出层梯度
        dZ = output - Y
        grads[f'dW{self.L}'] = (1/m) * cp.dot(dZ, cache[f'A{self.L-1}'].T)
        grads[f'db{self.L}'] = (1/m) * cp.sum(dZ, axis=1, keepdims=True)
        # 隐藏层梯度
        for i in reversed(range(1, self.L)):
            dA = cp.dot(self.params[f'W{i+1}'].T, dZ)
            dZ = dA * sigmoid_deriv(cache[f'Z{i}'])
            grads[f'dW{i}'] = (1/m) * cp.dot(dZ, cache[f'A{i-1}'].T)
            grads[f'db{i}'] = (1/m) * cp.sum(dZ, axis=1, keepdims=True)
        # 更新参数
        for i in range(1, self.L + 1):
            self.params[f'W{i}'] -= lr * grads[f'dW{i}']
            self.params[f'b{i}'] -= lr * grads[f'db{i}']

# ==================== 报告 + 绘图 ====================
def generate_full_report(loss_history, layer_dims, md_path):
    final_loss = loss_history[-1]
    structure = " → ".join(map(str, layer_dims))
    log_loss = f"Epoch 0:{loss_history[0]:.6f} | 3000:{loss_history[3000]:.6f} | 6000:{loss_history[6000]:.6f} | 9000:{loss_history[9000]:.6f} | 12000:{loss_history[12000]:.6f}"
    report = f"""# CuPy 神经网络训练报告
## 模型结构：{structure}
## 最终损失：{final_loss:.6f}
## 训练状态：{'✅ 收敛' if final_loss<0.01 else '⚠️ 未收敛'}
## 参数文件：trained_params/ 目录下(W1/b1等CSV矩阵)
"""
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)

def plot_loss_curve(loss_hist, html_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_hist, line=dict(color='#2E86AB')))
    fig.update_layout(title="训练损失曲线", xaxis_title="轮次", yaxis_title="Loss")
    fig.write_html(html_path)

# ==================== 主程序 ====================
if __name__ == "__main__":
    TOKEN = os.getenv("GITHUB_TOKEN")
    # ============== 双数据源切换 ==============
    if DATA_MODE == "manual":
        print("📊 使用【手动导入】异或数据集")
        X_cpu = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
        Y_cpu = np.array([[0, 1, 1, 0]])
        X, Y = cp.array(X_cpu, dtype=cp.float32), cp.array(Y_cpu, dtype=cp.float32)
    else:
        print("📊 使用【CSV文件】数据集")
        X, Y = load_csv_data(CSV_PATH)

    # 初始化模型
    model = FlexibleNet(LAYER_DIMS)
    loss_history = []
    print(f"🚀 模型结构：{' → '.join(map(str, LAYER_DIMS))}")
    print("🔥 CuPy GPU 训练开始...")

    # 训练循环 + 定期保存参数
    for i in range(EPOCHS):
        output, cache = model.forward(X)
        loss = cp.mean((Y - output) ** 2)
        loss_history.append(float(loss.get()))
        model.backward(X, Y, output, cache, LEARNING_RATE)

        # 定期保存W/B矩阵（训练中）
        if i % LOG_INTERVAL == 0:
            print(f"Epoch {i:5d} | Loss: {loss_history[-1]:.6f}")
            save_params_to_csv(model.params, epoch=i)

    print("🎉 训练完成！")
    # 最终保存W/B矩阵（最终结果）
    save_params_to_csv(model.params, epoch="final")

    # 生成报告 + 绘图
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    HTML_PATH = os.path.join(SCRIPT_DIR, "loss_curve.html")
    MD_PATH = os.path.join(SCRIPT_DIR, "training_report.md")
    plot_loss_curve(loss_history, HTML_PATH)
    generate_full_report(loss_history, LAYER_DIMS, MD_PATH)

    # 上传文件
    if TOKEN:
        print("\n☁️ 开始上传...")
        MyToolsGitHub.magnus_github_upload(TOKEN, MD_PATH, "magnus_code/zyz/neuralnet/training_report.md")
        MyToolsGitHub.magnus_github_upload(TOKEN, HTML_PATH, "magnus_code/zyz/neuralnet/loss_curve.html")
    else:
        print("\n[日志] 未配置TOKEN，跳过上传")