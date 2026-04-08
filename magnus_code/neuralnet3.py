import os
import numpy as np  # 保留 numpy 用于 CPU 端数据类型转换
import cupy as cp   # 引入 cupy
import plotly.graph_objects as go # 替代 matplotlib
import mytools1
from plotly.io import write_html

# ==========================================
# 1. 激活函数库 (使用 Cupy)
# ==========================================
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

# ==========================================
# 2. 灵活的神经网络类 (基于 Cupy)
# ==========================================
class FlexibleNN:
    def __init__(self, layer_dims, activations):
        self.layer_dims = layer_dims
        self.activations = activations
        self.parameters = {}
        self.L = len(layer_dims) - 1 
        self._initialize_parameters()

    def _initialize_parameters(self):
        cp.random.seed(42) 
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] = cp.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.1
            self.parameters[f'b{l}'] = cp.zeros((self.layer_dims[l], 1))

    def forward(self, X):
        caches = {'A0': X}
        A_prev = X
        for l in range(1, self.L + 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            activation_func, _ = self.activations[l-1]
            Z = cp.dot(W, A_prev) + b
            A = activation_func(Z)
            caches[f'Z{l}'] = Z
            caches[f'A{l}'] = A
            A_prev = A
        return A, caches

    def backward(self, Y, caches, m):
        grads = {}
        AL = caches[f'A{self.L}']
        dA = 2 * (AL - Y) / m
        
        for l in reversed(range(1, self.L + 1)):
            _, deriv_func = self.activations[l-1]
            Z = caches[f'Z{l}']
            A_prev = caches[f'A{l-1}']
            W = self.parameters[f'W{l}']
            
            dZ = dA * deriv_func(Z)
            dW = cp.dot(dZ, A_prev.T)
            db = cp.sum(dZ, axis=1, keepdims=True)
            
            if l > 1:
                dA = cp.dot(W.T, dZ)
            
            grads[f'dW{l}'] = dW
            grads[f'db{l}'] = db
        return grads

    def update(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']

# ==========================================
# 3. 辅助函数：生成 Markdown 文档 & Plotly 绘图
# ==========================================
def matrix_to_latex(mat_gpu, name):
    """将 Cupy 矩阵转回 CPU 并转换为 LaTeX 字符串"""
    mat = cp.asnumpy(mat_gpu) 
    if mat.size > 100: 
        return f"${name} \\in \\mathbb{{R}}^{{{mat.shape[0]} \\times {mat.shape[1]}}}$ (内容过长省略)"
    lines = []
    for row in mat:
        formatted_row = " & ".join([f"{x:.4f}" for x in row])
        lines.append(formatted_row)
    body = " \\\\ ".join(lines)
    return f"$$ {name} = \\begin{{bmatrix}} {body} \\end{{bmatrix}} $$"

def plot_loss_curve_plotly(loss_history, save_path_html):
    """使用 Plotly 绘制损失曲线并保存为 HTML"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=loss_history, 
        mode='lines', 
        name='Training Loss',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title='Neural Network Training Loss Curve',
        xaxis_title='Epochs',
        yaxis_title='Loss',
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig.write_html(save_path_html)
    print(f"[日志] 交互式图表已保存: {save_path_html}")

def generate_markdown_report(nn, X_gpu, Y_gpu, final_A_gpu, loss_history, log_interval, save_path_html, save_path_md):
    """生成完整的 Markdown 报告"""
    
    # 构建 Loss 表格
    table_rows = []
    for i in range(0, len(loss_history), log_interval):
        table_rows.append({"Epoch": i, "Loss": f"{loss_history[i]:.6f}"})
    if table_rows[-1]["Epoch"] != len(loss_history)-1:
        table_rows.append({"Epoch": len(loss_history)-1, "Loss": f"{loss_history[-1]:.6f}"})

    md_content = f"# 神经网络训练实验报告 (GPU Accelerated)\n\n"
    
    md_content += "## 1. 网络结构与配置\n\n"
    md_content += f"- **层结构**: {nn.layer_dims}\n"
    act_names = [func.__name__ for func, _ in nn.activations]
    md_content += f"- **激活函数**: {act_names}\n"
    md_content += f"- **样本数 (m)**: {X_gpu.shape[1]}\n"
    md_content += f"- **特征数**: {X_gpu.shape[0]}\n"
    md_content += f"- **计算后端**: CuPy (GPU)\n"
    md_content += f"- **可视化**: Plotly (Interactive HTML)\n\n"

    md_content += "## 2. 训练损失 (Loss) 变化\n\n"
    md_content += "| Epoch | Loss |\n"
    md_content += "|-------|------|\n"
    for row in table_rows:
        md_content += f"| {row['Epoch']} | {row['Loss']} |\n"
    md_content += "\n"

    md_content += "## 3. 损失曲线 (交互式)\n\n"
    md_content += f"请在浏览器中打开 [{os.path.basename(save_path_html)}]({os.path.basename(save_path_html)}) 查看动态损失曲线。\n\n"

    md_content += "## 4. 预测结果\n\n"
    md_content += "### 真实标签 Y:\n"
    md_content += matrix_to_latex(Y_gpu, "Y") + "\n\n"
    md_content += "### 预测结果 A (Final):\n"
    md_content += matrix_to_latex(final_A_gpu, "A") + "\n\n"
    md_content += "### 四舍五入后:\n"
    final_A_cpu = cp.asnumpy(final_A_gpu)
    md_content += f"```\n{np.round(final_A_cpu, 4)}\n```\n\n"

    md_content += "## 5. 最终网络参数\n\n"
    for l in range(1, nn.L + 1):
        md_content += f"### 第 {l} 层\n\n"
        md_content += matrix_to_latex(nn.parameters[f'W{l}'], f'W_{l}') + "\n\n"
        md_content += matrix_to_latex(nn.parameters[f'b{l}'], f'b_{l}') + "\n\n"

    with open(save_path_md, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"[日志] Markdown报告已生成: {save_path_md}")

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    # --- 配置区域 ---
    X_cpu = np.array([[0, 0, 1, 1],
                      [0, 1, 0, 1]])
    Y_cpu = np.array([[0, 1, 1, 0]])
    
    # 网络结构
    layer_dimensions = [2, 40, 1] 
    
    activation_functions = [
        (sigmoid, sigmoid_deriv),
        (sigmoid, sigmoid_deriv)
    ]
    
    learning_rate = 0.5
    epochs = 15000
    log_interval = 3000
    
    # 路径配置：强制保存在脚本所在的目录
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    HTML_PATH = os.path.join(SCRIPT_DIR, "xor_loss_curve.html")
    MD_PATH = os.path.join(SCRIPT_DIR, "training_report.md")
    
    # --- 数据迁移至 GPU ---
    print(f"[日志] 正在将数据迁移至 GPU...")
    X = cp.asarray(X_cpu)
    Y = cp.asarray(Y_cpu)
    
    # --- 训练 ---
    m = X.shape[1]
    nn = FlexibleNN(layer_dimensions, activation_functions)
    loss_history = []

    print("开始训练 (GPU)...")
    for i in range(epochs):
        AL, caches = nn.forward(X)
        loss = cp.mean((Y - AL) ** 2)
        loss_history.append(cp.asnumpy(loss).item())
        
        grads = nn.backward(Y, caches, m)
        nn.update(grads, learning_rate)
        
        if i % log_interval == 0:
            print(f"Epoch {i:5d} | Loss: {loss_history[-1]:.6f}")

    print(f"Epoch {epochs:5d} | Loss: {loss_history[-1]:.6f}")
    print("训练完成！")

    # --- 使用 Plotly 绘图 (生成 HTML) ---
    plot_loss_curve_plotly(loss_history, HTML_PATH)

    # --- 生成报告 ---
    final_A, _ = nn.forward(X)
    generate_markdown_report(nn, X, Y, final_A, loss_history, log_interval, HTML_PATH, MD_PATH)



# 从环境变量读取令牌（安全，无明文）
TOKEN = os.getenv("GITHUB_TOKEN")
# 自动上传：本地路径 = GitHub路径
mytools1.magnus_github_upload(
    github_token=TOKEN,
    local_file_path="python magnus_code/training_report.md"
)
mytools1.magnus_github_upload(
    github_token=TOKEN,
    local_file_path="python magnus_code/xor_loss_curve.html"
)

''' 入口指令cmd
pip install plotly
GITHUB_TOKEN=" " python magnus_code/neuralnet.py

'''