import os
import numpy as np
import cupy as cp
import plotly.graph_objects as go

# 工具库兼容
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import mytools1
except ImportError:
    mytools1 = None
    print("mytools1 = None")
try:
    TOKEN = os.getenv("GITHUB_TOKEN")
except ImportError:
    TOKEN = None
    print("TOKEN = None")

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
    return cp.ones_like(z)

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
# ==================== 修复后的矩阵转LaTeX函数 ====================
def matrix_to_latex(mat_gpu, name):
    mat = cp.asnumpy(mat_gpu)
    if mat.size > 100:
        return f"${name} \\in \\mathbb{{R}}^{{{mat.shape[0]} \\times {mat.shape[1]}}}$ (内容过长省略)"
    rows = " \\\\ ".join([" & ".join([f"{x:.4f}" for x in r]) for r in mat])
    return f"${name} = \\begin{{bmatrix}} {rows} \\end{{bmatrix}}$"

def plot_loss(loss_hist, html_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_hist, name="Training Loss", line=dict(color='#1f77b4')))
    fig.update_layout(title="GPU 训练损失曲线", xaxis_title="Epoch", yaxis_title="Loss")
    fig.write_html(html_path)

def generate_report(nn, X, Y, pred, loss_hist, interval, html_path, md_path):
    # 损失表格
    table = "| Epoch | Loss |\n|-------|------|\n"
    for i in range(0, len(loss_hist), interval):
        table += f"| {i} | {loss_hist[i]:.6f} |\n"
    
    # 生成Markdown
    content = f"""# 神经网络训练报告 (CuPy GPU)
## 网络结构
{nn.layer_dims}
## 训练参数
- 学习率: 0.5
- 迭代次数: {len(loss_hist)}
## 损失变化
{table}
## 预测结果
真实标签: {matrix_to_latex(Y, 'Y')}
预测结果: {matrix_to_latex(pred, 'A')}
## 网络参数
"""
    for i in range(1, nn.L+1):
        content += f"### 层 {i}\n{matrix_to_latex(nn.parameters[f'W{i}'], f'W_{i}')}\n{matrix_to_latex(nn.parameters[f'b{i}'], f'b_{i}')}\n\n"
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 报告已生成: {md_path}")
    print(f"✅ 曲线已生成: {html_path}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # ============== 【仅修改这里的参数】 ==============
    # 数据集
    X_cpu = np.array([[0,0,1,1],[0,1,0,1]])
    Y_cpu = np.array([[0,1,1,0]])
    
    # 网络结构
    layer_dimensions = [2, 40, 1]
    activation_functions = [(sigmoid, sigmoid_deriv), (sigmoid, sigmoid_deriv)]
    
    # 训练参数
    learning_rate = 0.5
    epochs = 15000
    log_interval = 3000

    # 文件路径（脚本所在目录，自动生成）
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    HTML_PATH = os.path.join(SCRIPT_DIR, "loss_curve.html")
    MD_PATH = os.path.join(SCRIPT_DIR, "training_report.md")
    # ==================================================

    # 数据加载到GPU
    X = cp.array(X_cpu)
    Y = cp.array(Y_cpu)
    m = X.shape[1]

    # 训练
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

    # ==================== 核心修复：备份变量 + 释放显存 ====================
    # 1. 先把需要的结果备份到CPU（防止被删除后报错）
    final_pred = pred
    # 2. 主动释放GPU显存（你要求的唯一优化）
    del pred, cache, grads, X, Y
    cp.get_default_memory_pool().free_all_blocks()
    # ======================================================================
    
    # 生成输出文件（使用备份的变量，不报错）
    plot_loss(loss_history, HTML_PATH)
    generate_report(nn, cp.array(X_cpu), cp.array(Y_cpu), final_pred, loss_history, log_interval, HTML_PATH, MD_PATH)

    # GitHub 自动上传
    if mytools1 and TOKEN:
        try:
            print("☁️ 开始上传文件到 GitHub...")
            mytools1.magnus_github_upload(
                github_token=TOKEN,
                local_file_path=MD_PATH,
                github_file_path="magnus_code/training_report.md"
            )
            mytools1.magnus_github_upload(
                github_token=TOKEN,
                local_file_path=HTML_PATH,
                github_file_path="magnus_code/loss_curve.html"
            )
            print("✅ 所有文件上传 GitHub 成功！")
        except Exception as e:
            print(f"❌ 上传失败: {str(e)}")
    else:
        print("[日志] 未配置 TOKEN 或未找到 mytools1，跳过上传")