'''
-------------------------
Z = W@X +b
A = f(Z)

X是矩阵(特征数, 样本数)
# 2*2 ABCD 矩阵（直接用变量/字母表示）
mat = np.array([[A, B],   # 第一行
                [C, D]])  # 第二行

---------------------------

loss = np.mean((Y - A2) ** 2)
                
\ frac{\partial L}{\partial w} 
= \ frac{\partial L}{\partial z} \cdot \ frac{\partial z}{\partial w}

dL = (Y-A2) dA2 =(Y-A2) sigmoid_deriv(Z2)

'''


import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['axes.unicode_minus'] = False
# 1. 定义激活函数 ReLU
def relu(z):
    # np.maximum 是逐元素比较，取大的那个
    return np.maximum(0, z)

# 1. 定义激活函数和它的导数（反向传播用）
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    # Sigmoid 的导数（反向传播需要）
    return sigmoid(z) * (1 - sigmoid(z))

# 2. 准备一个超简单的数据集：异或问题 (XOR)
# 输入：[0,0] -> 输出 0
# 输入：[0,1] -> 输出 1
# 输入：[1,0] -> 输出 1
# 输入：[1,1] -> 输出 0
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])  # 形状 (2, 4)
Y = np.array([[0, 1, 1, 0]])  # 真实标签 (1, 4)
m = 4                         # 4个样本

# 3. 初始化参数 (权重和偏置随机开始)
input_size = 2    # 输入层 2 个神经元
hidden_size = 40    # 隐藏层 40 个神经元
output_size = 1   # 输出层 1 个神经元

# 随机初始化权重
W1 = np.random.randn(hidden_size, input_size)
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size)
b2 = np.zeros((output_size, 1))

learning_rate = 0.1
epochs = 1000*15 # 训练 k 次
loss_history = []  # 用来保存每一步的损失


# 4. 训练循环 (前向传播 + 反向传播)
for i in range(epochs):
    # --- 第一步：前向传播 (Forward Pass) ---
    # 第一层 (隐藏层)
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    
    # 第二层 (输出层)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    # 计算损失 (MSE)
    loss = np.mean((Y - A2) ** 2)
    loss_history.append(loss)
    # --- 第二步：反向传播 (Backward Pass) --- # 除以样本数 m
    # 输出层误差
    dZ2 = (A2 - Y) * sigmoid_deriv(Z2) #先除以m
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m

        # 隐藏层误差 (链式法则)
    dZ1 = np.dot(W2.T, dZ2) * sigmoid_deriv(Z1)
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    
    # --- 第三步：梯度下降 (更新权重) ---
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    if i % 3000 == 0:
        print(f"训练次数 {int(i/1000)}k, 当前损失: {loss:.4f}")

print("\n训练完成！")
print("预测结果（理想值：[[0, 1, 1, 0]]）:")
print(np.round(A2, 4))  # 四舍五入保留4位小数，方便看结果
print(f"Z1{Z1}, b1{b1}, Z2{Z2}, b2{b2}")

plt.figure(figsize=(10, 5))
plt.plot(loss_history, color='#1f77b4', linewidth=1.2, label='Training Loss')
plt.xlabel('Epochs ')
plt.ylabel('Loss ')
plt.title('XOR Neural Network Training Loss Curve')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("xor_loss_curve.png", dpi=300, bbox_inches='tight')
plt.close()