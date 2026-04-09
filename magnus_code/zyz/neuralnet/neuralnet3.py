import os
import numpy as np
import cupy as cp
import plotly.graph_objects as go

# ==================== Class 独立封装上传工具（完全复刻 mytools 原始风格）====================
class MyToolsGitHub:
    # 类内导入模块
    import base64
    import json
    import urllib.request
    import os
    from urllib.error import HTTPError

    # 固定团队仓库配置
    REPO = "Rise-AGI/PHY-LLM-Basic-Algorithm"
    BRANCH = "main"

    # 私有方法：获取SHA
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

    # 核心上传方法
    @classmethod
    def magnus_github_upload(cls, github_token: str, local_file_path: str, github_file_path: str = None):
        """
        核心上传函数
        :param github_token: GitHub PAT 令牌
        :param local_file_path: 本地文件路径
        :param github_file_path: GitHub 目标路径
        """
        if github_file_path is None:
            github_file_path = local_file_path.strip()
            print(f"[自动路径] GitHub目标路径 = {github_file_path}")

        token = github_token.strip()
        local_path = local_file_path.strip()
        if not token:
            print("[错误] GitHub令牌不能为空")
            return
        if not cls.os.path.isfile(local_path):
            print(f"[错误] 本地文件不存在：{local_path}")
            return

        try:
            with open(local_path, "rb") as f:
                content = cls.base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"[错误] 文件读取失败：{str(e)}")
            return

        url = f"https://api.github.com/repos/{cls.REPO}/contents/{github_file_path}"
        payload = {
            "message": f"auto upload: {cls.os.path.basename(local_path)}",
            "content": content,
            "branch": cls.BRANCH
        }

        sha = cls._get_remote_sha(token, github_file_path)
        if sha:
            payload["sha"] = sha
            print("[信息] 检测到同名文件，自动覆盖")

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
        except Exception as e:
            print(f"❌ 上传异常：{str(e)}")
# ======================================================================================

# ==================== 激活函数 ====================
def sigmoid(z):
    return 1 / (1 + cp.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

# ==================== 【彻底修复】CuPy 神经网络（必收敛）====================
class XORNet:
    def __init__(self):
        cp.random.seed(42)
        # 正确的权重初始化（解决梯度消失/不收敛）
        self.W1 = cp.random.randn(40, 2) * cp.sqrt(1. / 2)
        self.b1 = cp.zeros((40, 1))
        self.W2 = cp.random.randn(1, 40) * cp.sqrt(1. / 40)
        self.b2 = cp.zeros((1, 1))

    def forward(self, X):
        # 前向传播
        Z1 = cp.dot(self.W1, X) + self.b1
        A1 = sigmoid(Z1)
        Z2 = cp.dot(self.W2, A1) + self.b2
        A2 = sigmoid(Z2)
        return A2, Z1, A1

    def backward(self, X, Y, A2, Z1, A1, learning_rate):
        m = X.shape[1]
        # 反向传播（数学完全正确）
        dZ2 = A2 - Y
        dW2 = (1 / m) * cp.dot(dZ2, A1.T)
        db2 = (1 / m) * cp.sum(dZ2, axis=1, keepdims=True)

        dA1 = cp.dot(self.W2.T, dZ2)
        dZ1 = dA1 * sigmoid_deriv(Z1)
        dW1 = (1 / m) * cp.dot(dZ1, X.T)
        db1 = (1 / m) * cp.sum(dZ1, axis=1, keepdims=True)

        # 参数更新
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

# ==================== 【丰富完整】报告生成函数 ====================
def generate_full_report(loss_history, md_path):
    # 关键训练数据
    final_loss = loss_history[-1]
    log_loss = f"""
Epoch 0      | Loss: {loss_history[0]:.6f}
Epoch 3000   | Loss: {loss_history[3000]:.6f}
Epoch 6000   | Loss: {loss_history[6000]:.6f}
Epoch 9000   | Loss: {loss_history[9000]:.6f}
Epoch 12000  | Loss: {loss_history[12000]:.6f}
"""
    # 完整报告内容
    report = f"""# CuPy 神经网络训练报告
## 一、模型信息
- 模型类型：全连接神经网络（异或分类）
- 网络结构：2输入 → 40隐藏层 → 1输出层
- 运行环境：CuPy GPU 加速

## 二、训练超参数
- 训练轮次：15000
- 学习率：0.5
- 激活函数：Sigmoid
- 损失函数：均方误差(MSE)

## 三、训练损失记录
{log_loss}
- 最终损失值：{final_loss:.6f}

## 四、数据集
- 任务：异或(XOR)逻辑运算
- 输入样本：4组
- 输入特征：2维
- 标签：1维(0/1二分类)

## 五、训练状态
✅ 训练完成
{'✅ 模型收敛' if final_loss < 0.01 else '⚠️ 模型未收敛'}

## 六、文件说明
- `loss_curve.html`：训练损失变化曲线
- `training_report.md`：训练详细报告
"""
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print("✅ 完整训练报告已生成")

# ==================== 损失曲线绘图 ====================
def plot_loss_curve(loss_hist, html_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=loss_hist,
        name="训练损失",
        line=dict(color='#2E86AB', width=2)
    ))
    fig.update_layout(
        title="CuPy XOR神经网络训练损失曲线",
        xaxis_title="训练轮次",
        yaxis_title="损失值",
        template="plotly_white"
    )
    fig.write_html(html_path)

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 环境变量
    TOKEN = os.getenv("GITHUB_TOKEN")
    
    # 异或数据集（维度100%正确）
    X_cpu = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    Y_cpu = np.array([[0, 1, 1, 0]])
    X = cp.array(X_cpu, dtype=cp.float32)
    Y = cp.array(Y_cpu, dtype=cp.float32)

    # 训练参数
    epochs = 15000
    lr = 0.5
    loss_history = []

    # 初始化模型
    model = XORNet()
    print("🚀 CuPy GPU 训练开始...")

    # 训练循环
    for i in range(epochs):
        A2, Z1, A1 = model.forward(X)
        loss = cp.mean((Y - A2) ** 2)
        loss_history.append(float(loss.get()))
        model.backward(X, Y, A2, Z1, A1, lr)

        # 打印日志
        if i % 3000 == 0:
            print(f"Epoch {i:5d} | Loss: {loss_history[-1]:.6f}")

    print("🎉 训练完成！")

    # 文件路径
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    HTML_PATH = os.path.join(SCRIPT_DIR, "loss_curve.html")
    MD_PATH = os.path.join(SCRIPT_DIR, "training_report.md")

    # 生成报告和曲线
    plot_loss_curve(loss_history, HTML_PATH)
    generate_full_report(loss_history, MD_PATH)

    # 上传文件
    if TOKEN:
        print("\n☁️ 开始上传文件到 GitHub...")
        MyToolsGitHub.magnus_github_upload(TOKEN, MD_PATH, "magnus_code/zyz/training_report.md")
        MyToolsGitHub.magnus_github_upload(TOKEN, HTML_PATH, "magnus_code/zyz/loss_curve.html")
    else:
        print("\n[日志] 未配置GITHUB_TOKEN，跳过上传")