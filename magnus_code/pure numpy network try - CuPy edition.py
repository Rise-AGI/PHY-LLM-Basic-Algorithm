import numpy as np
import cupy as cp
# import matplotlib.pyplot as plt
import time

# ============ Adam 超参数 ============
lr = 0.001          # Adam 默认学习率
beta1 = 0.9         # 一阶矩衰减系数
beta2 = 0.999       # 二阶矩衰减系数
eps = 1e-8          # 数值稳定项
total_iters = 2000  # 总迭代次数
warmup_iters = 200  # 学习率 warmup 步数

# 全局时间步计数器
adam_t = 0

def get_lr(step):
    """Warmup + Cosine Annealing 学习率调度（纯数学实现）"""
    if step < warmup_iters:
        # 线性 warmup
        return lr * (step + 1) / warmup_iters
    else:
        # Cosine annealing 衰减到 lr 的 1%
        progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
        return lr * 0.01 + 0.5 * (lr - lr * 0.01) * (1.0 + float(cp.cos(cp.float64(progress * cp.pi))))

rng = cp.random.default_rng()

class Linlayer:
    def __init__(self, idim=128, odim=128):
        # He 初始化（适配 tanh 激活）
        self.W = rng.standard_normal(size=(odim, idim), dtype=cp.float32) * cp.sqrt(1 / idim, dtype=cp.float32)
        self.b = cp.zeros((odim, 1), dtype=cp.float32)
        # Adam 状态变量
        self.m_W = cp.zeros_like(self.W)
        self.v_W = cp.zeros_like(self.W)
        self.m_b = cp.zeros_like(self.b)
        self.v_b = cp.zeros_like(self.b)

    def fwdpp(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.x = x
        return cp.matmul(self.W, x) + self.b

    def bwdpp(self, input_delta):
        batch_size = input_delta.shape[1]
        self.dW = (input_delta @ self.x.T) / batch_size
        self.db = cp.mean(input_delta, axis=1, keepdims=True)
        return self.W.T @ input_delta

    def step(self):
        """Adam 优化器更新步"""
        global adam_t
        current_lr = get_lr(adam_t)

        # ---- 更新 W ----
        # 一阶矩估计
        self.m_W = beta1 * self.m_W + (1 - beta1) * self.dW
        # 二阶矩估计
        self.v_W = beta2 * self.v_W + (1 - beta2) * (self.dW ** 2)
        # 偏差修正
        m_W_hat = self.m_W / (1 - beta1 ** adam_t)
        v_W_hat = self.v_W / (1 - beta2 ** adam_t)
        # 参数更新
        self.W -= current_lr * m_W_hat / (cp.sqrt(v_W_hat) + eps)

        # ---- 更新 b ----
        self.m_b = beta1 * self.m_b + (1 - beta1) * self.db
        self.v_b = beta2 * self.v_b + (1 - beta2) * (self.db ** 2)
        m_b_hat = self.m_b / (1 - beta1 ** adam_t)
        v_b_hat = self.v_b / (1 - beta2 ** adam_t)
        self.b -= current_lr * m_b_hat / (cp.sqrt(v_b_hat) + eps)


class th_activate:
    def fwdpp(self, x):
        self.y = cp.tanh(x)
        return self.y

    def bwdpp(self, input_delta):
        self.delta = input_delta * (1 - self.y ** 2)
        return self.delta


L1 = Linlayer(idim=200)
L2 = Linlayer()
L3 = Linlayer()
L4 = Linlayer(odim=1)
th1 = th_activate()
th2 = th_activate()
th3 = th_activate()


def eval(x):
    x = L1.fwdpp(x)
    x = th1.fwdpp(x)
    x = L2.fwdpp(x)
    x = th2.fwdpp(x)
    x = L3.fwdpp(x)
    x = th3.fwdpp(x)
    x = L4.fwdpp(x)
    return x


def f(x, y):
    return cp.cos(23 * x) * cp.sin(17 * y) * x * y


def bwdp(yh, yr):
    delta = yh - yr
    delta = L4.bwdpp(delta)
    delta = th3.bwdpp(delta)
    delta = L3.bwdpp(delta)
    delta = th2.bwdpp(delta)
    delta = L2.bwdpp(delta)
    delta = th1.bwdpp(delta)
    L1.bwdpp(delta)
    L4.step()
    L3.step()
    L2.step()
    L1.step()


B = rng.random(size=(2, 100), dtype=cp.float32) * 30
t0 = time.time()

for i in range(total_iters):
    adam_t = i + 1  # Adam 时间步从 1 开始

    x_train = rng.random(size=(1024, 2), dtype=cp.float32)
    x0 = (x_train @ B).T
    x_input = cp.vstack((cp.cos(x0), cp.sin(x0)))
    yr = f(x_train[:, 0:1], x_train[:, 1:2]).T
    yh = eval(x_input)
    loss = cp.mean((yh - yr) ** 2) / 2
    bwdp(yh, yr)
    if i % 100 == 0:
        print(f"Iter {i}, Loss: {loss.get():.6f}, LR: {get_lr(i):.6f}")

t1 = time.time()
print(f"Training time: {t1 - t0:.2f}s")

# # ============ 可视化 ============
# x_steps = cp.linspace(0, 1, 300, dtype=cp.float64)
# y_steps = cp.linspace(0, 1, 300, dtype=cp.float64)
# grid_x, grid_y = cp.meshgrid(x_steps, y_steps)
# input_grid = cp.vstack([grid_x.flatten(), grid_y.flatten()]).T
# x0 = cp.matmul(input_grid, B).T
# x_input_plot = cp.vstack((cp.cos(x0), cp.sin(x0)))
# yh = eval(x_input_plot).reshape((300, 300)).get()

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# X_np = grid_x.get()
# Y_np = grid_y.get()
# ax.plot_surface(X_np, Y_np, yh, cmap='viridis')
# ax.set_title('Neural Network Approximation (Adam Optimizer)')
# plt.show()
