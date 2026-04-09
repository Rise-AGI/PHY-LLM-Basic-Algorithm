# magnus_code/zyz/neuralnet1.py 【强制生成图片版本】
import numpy as np
import matplotlib.pyplot as plt

# --------------- XOR 训练代码 ---------------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(0)
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

losses = []
epochs = 12000
lr = 0.1

for epoch in range(epochs):
    # 前向传播
    z1 = np.dot(X, W1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = 1 / (1 + np.exp(-z2))

    # 损失
    loss = np.mean((a2 - y) ** 2)
    losses.append(loss)

    # 反向传播
    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)
    dz1 = np.dot(dz2, W2.T) * (1 - a1 ** 2)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # 更新
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 3000 == 0:
        print(f"训练次数 {epoch//1000}k, 当前损失: {loss:.4f}")

# --------------- 强制保存图片（核心！）---------------
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Loss')
plt.title('XOR Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 🔥 绝对路径，100%生成图片
plt.savefig("/magnus/workspace/repository/xor_loss_curve.png", dpi=150)
plt.close()

# 打印确认（日志里会看到，证明图片生成了）
print("✅ 图片已成功保存到容器目录！")
print("预测结果（理想值：[[0, 1, 1, 0]]）:")
print(a2.T)