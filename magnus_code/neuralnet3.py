import os
import numpy as np
import cupy as cp
import plotly.graph_objects as go
import importlib.util

# ==================== 100%成功动态导入（同级文件专用）====================
# 强制获取当前脚本所在的 magnus_code 文件夹
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 精准定位同级 mytools1.py
MYTOOLS_PATH = os.path.join(SCRIPT_DIR, "mytools1.py")

mytools1 = None
TOKEN = None

# 动态加载模块
try:
    spec = importlib.util.spec_from_file_location("mytools1", MYTOOLS_PATH)
    mytools1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mytools1)
    print("✅ mytools1 导入成功！")
except Exception as e:
    print(f"❌ mytools1 导入失败: {e}")

# 读取TOKEN
try:
    TOKEN = os.getenv("GITHUB_TOKEN")
    if TOKEN:
        print(f"✅ GITHUB_TOKEN 读取成功: {TOKEN[:5]}...")
except Exception as e:
    print(f"❌ TOKEN 读取失败: {e}")
# ======================================================================

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
    return f"${name} = \\begin{{bmatrix}} {rows} \\end{{bmatrix}}$"

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
    print(f"✅ 报告已生成: {md_path}")
    print(f"✅ 曲线已生成: {html_path}")

# ==================== 主程序 ====================
if __name__ == "__main__":
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
    print("🚀 CuPy GPU 训练开始...")

    for i in range(epochs):
        pred, cache = nn.forward(X)
        loss = cp.mean((Y - pred)**2)
        loss_history.append(float(cp.asnumpy(loss)))
        grads = nn.backward(Y, cache, m)
        nn.update(grads, learning_rate)
        if i % log_interval == 0:
            print(f"Epoch {i:5d} | Loss: {loss_history[-1]:.6f}")

    print("🎉 训练完成！")

    # 主动显存释放
    final_pred = pred
    del pred, cache, grads, X, Y
    cp.get_default_memory_pool().free_all_blocks()

    plot_loss(loss_history, HTML_PATH)
    generate_report(nn, cp.array(X_cpu), cp.array(Y_cpu), final_pred, loss_history, log_interval, HTML_PATH, MD_PATH)

    # 上传
    if mytools1 and TOKEN:
        try:
            print("☁️ 开始上传...")
            mytools1.magnus_github_upload(TOKEN, MD_PATH, "magnus_code/training_report.md")
            mytools1.magnus_github_upload(TOKEN, HTML_PATH, "magnus_code/loss_curve.html")
            print("✅ 上传成功！")
        except Exception as e:
            print(f"❌ 上传失败: {e}")
    else:
        print("[日志] 跳过上传")