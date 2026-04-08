# 神经网络反向传播（BP）公式推导与应用

反向传播（Backpropagation, BP）是**神经网络训练的核心算法**，本质是**基于链式求导法则**，计算损失函数对各层权重、偏置的梯度，再通过梯度下降更新参数，让网络输出逼近真实标签。

下面以**三层全连接神经网络（输入层+1层隐藏层+输出层）** 为例，完成完整公式推导，再讲通用化与实际应用。

---

## 一、前置定义与前向传播

### 1. 符号约定（统一且通用）

|符号|含义|
|---|---|
| $a^0=x$ |输入层特征（输入向量）|
| $w^l$ |第 $l$ 层→第 $l+1$ 层的权重矩阵|
| $b^l$ |第 $l$ 层的偏置向量|
| $z^l$ |第 $l$ 层的**加权和**： $z^l = w^l a^{l-1} + b^l$ |
| $a^l=\sigma(z^l)$ |第 $l$ 层的**激活值**， $\sigma$ 为激活函数（常用Sigmoid/ReLU）|
| $y$ |真实标签向量|
| $L$ |损失函数（以**均方误差MSE**为例）|
| $\eta$ |学习率|
| $\odot$ |哈达玛积（矩阵/向量**元素对应相乘**）|
### 2. 前向传播流程

三层网络（ $l=0$ 输入层， $l=1$ 隐藏层， $l=2$ 输出层）：

1. 隐藏层加权和： $\boldsymbol{z^1 = w^1 a^0 + b^1}$ 

2. 隐藏层激活值： $\boldsymbol{a^1 = \sigma(z^1)}$ 

3. 输出层加权和： $\boldsymbol{z^2 = w^2 a^1 + b^2}$ 

4. 输出层激活值： $\boldsymbol{a^2 = \sigma(z^2)}$ 

### 3. 损失函数

均方误差（回归任务常用）：

 $\boldsymbol{L = \frac{1}{2}\|y - a^2\|^2 = \frac{1}{2}\sum_{i}(y_i - a_i^2)^2}$ 

乘 $\frac{1}{2}$ 是为了**求导后消去系数**，简化计算。

### 4. 核心工具：链式求导法则

复合函数求导：

 $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w}$ 

反向传播的本质就是**用链式法则，从输出层往输入层逐层递推梯度**。

---

## 二、反向传播公式核心推导

定义**层误差项**（核心中间量）：

 $\boldsymbol{\delta^l = \frac{\partial L}{\partial z^l}}$ 

 $\delta^l$ 表示损失对第 $l$ 层加权和的梯度，后续所有权重、偏置的梯度都由它推导。

### 1. 输出层误差  $\delta^2$  推导

对 $z^2$ 求导，链式法则展开：

 $\delta^2 = \frac{\partial L}{\partial z^2} = \frac{\partial L}{\partial a^2} \cdot \frac{\partial a^2}{\partial z^2}$ 

分步计算：

1. 损失对激活值的偏导：

 $\frac{\partial L}{\partial a^2} = a^2 - y$ 

1. 激活函数对加权和的偏导（以Sigmoid为例， $\sigma(z)=\frac{1}{1+e^{-z}}$ ）：

 $\sigma'(z) = \sigma(z)(1-\sigma(z)) \implies \frac{\partial a^2}{\partial z^2} = \sigma'(z^2)$ 

最终**输出层误差**：

 $\boldsymbol{\delta^2 = (a^2 - y) \odot \sigma'(z^2)}$ 

### 2. 隐藏层误差  $\delta^1$  推导

隐藏层误差需要**从输出层往回传递**，链式法则：

 $\delta^1 = \frac{\partial L}{\partial z^1} = \frac{\partial L}{\partial z^2} \cdot \frac{\partial z^2}{\partial a^1} \cdot \frac{\partial a^1}{\partial z^1}$ 

分步计算：

1.  $\frac{\partial L}{\partial z^2}=\delta^2$ 

2.  $z^2=w^2 a^1 + b^2 \implies \frac{\partial z^2}{\partial a^1}=(w^2)^T$ （权重矩阵转置）

3.  $\frac{\partial a^1}{\partial z^1}=\sigma'(z^1)$ 

最终**隐藏层误差**：

 $\boldsymbol{\delta^1 = \left((w^2)^T \delta^2\right) \odot \sigma'(z^1)}$ 

### 3. 权重与偏置的梯度

#### （1）权重梯度

 $z^l = w^l a^{l-1} + b^l \implies \frac{\partial z^l}{\partial w^l} = a^{l-1}$ 

结合误差项 $\delta^l$ ：

 $\boldsymbol{\frac{\partial L}{\partial w^l} = \delta^l \cdot (a^{l-1})^T}$ 

#### （2）偏置梯度

 $\frac{\partial z^l}{\partial b^l} = 1$ ，因此：

 $\boldsymbol{\frac{\partial L}{\partial b^l} = \delta^l}$ 

### 4. 梯度下降参数更新

得到梯度后，沿**梯度反方向**更新参数（减小损失）：

 $\boldsymbol{w^l = w^l - \eta \cdot \frac{\partial L}{\partial w^l}}$ 

 $\boldsymbol{b^l = b^l - \eta \cdot \frac{\partial L}{\partial b^l}}$ 

### 5. 反向传播代码实战详解（结合异或问题）

结合上述推导的反向传播理论公式，以下逐行拆解你提供的异或问题代码中，反向传播的每一步实现逻辑，明确代码与理论公式的对应关系，帮你彻底理解“理论如何落地为代码”。

先明确代码中核心变量与理论符号的对应关系（关键！）：

|代码变量|理论符号|含义说明|
|---|---|---|
|dZ2| $\delta^2$ |输出层误差项（损失对Z2的梯度）|
|dW2| $\frac{\partial L}{\partial w^2}$ |输出层权重W2的梯度|
|db2| $\frac{\partial L}{\partial b^2}$ |输出层偏置b2的梯度|
|dZ1| $\delta^1$ |隐藏层误差项（损失对Z1的梯度）|
|dW1| $\frac{\partial L}{\partial w^1}$ |隐藏层权重W1的梯度|
|db1| $\frac{\partial L}{\partial b^1}$ |隐藏层偏置b1的梯度|
|A2/Y| $a^2/y$ |输出层激活值/真实标签|
|A1| $a^1$ |隐藏层激活值|
接下来逐行拆解代码中反向传播核心代码（注释已补充理论对应关系）：

```python
# --- 第二步：反向传播 (Backward Pass) ---
# 1. 输出层误差 dZ2（对应理论公式：δ² = (a² - y) ⊙ σ’(z²)）
# 代码中 * 运算符 = 哈达玛积（对应理论中的 ⊙），sigmoid_deriv(Z2) = σ’(z²)
dZ2 = (A2 - Y) * sigmoid_deriv(Z2)

# 2. 输出层权重梯度 dW2（对应理论公式：∂L/∂w² = δ² · (a¹)ᵀ）
# np.dot(dZ2, A1.T) 对应 δ² 与 (a¹)ᵀ 的矩阵乘法；除以4是因为4个样本，求平均梯度（批量梯度下降）
dW2 = np.dot(dZ2, A1.T) / 4  

# 3. 输出层偏置梯度 db2（对应理论公式：∂L/∂b² = δ²）
# np.sum(dZ2, axis=1, keepdims=True) 对所有样本的误差求和，除以4求平均；keepdims=True保证维度与b2一致（避免广播错误）
db2 = np.sum(dZ2, axis=1, keepdims=True) / 4

# 4. 隐藏层误差 dZ1（对应理论公式：δ¹ = [(w²)ᵀ δ²] ⊙ σ’(z¹)）
# np.dot(W2.T, dZ2) 对应 (w²)ᵀ 与 δ² 的矩阵乘法；* sigmoid_deriv(Z1) 对应哈达玛积 ⊙ σ’(z¹)
dZ1 = np.dot(W2.T, dZ2) * sigmoid_deriv(Z1)

# 5. 隐藏层权重梯度 dW1（对应理论公式：∂L/∂w¹ = δ¹ · (a⁰)ᵀ）
# X是输入层a⁰，X.T是(a⁰)ᵀ；除以4同样是求4个样本的平均梯度
dW1 = np.dot(dZ1, X.T) / 4

# 6. 隐藏层偏置梯度 db1（对应理论公式：∂L/∂b¹ = δ¹）
# 与db2逻辑一致，求和后求平均，保证维度与b1一致
db1 = np.sum(dZ1, axis=1, keepdims=True) / 4
```

#### 关键补充说明（代码与理论的核心关联点）

1. **哈达玛积的代码实现**：理论中的  $\odot$ （哈达玛积），在NumPy中直接用 `*` 运算符实现，这也是代码中`(A2 - Y) * sigmoid_deriv(Z2)`、`np.dot(W2.T, dZ2) * sigmoid_deriv(Z1)` 的核心逻辑，与此前补充的哈达玛积实现完全一致。

2. **批量梯度下降的平均操作**：代码中所有梯度（dW1、dW2、db1、db2）都除以4（样本数），原因是采用“批量梯度下降”（一次用所有样本计算梯度），除以样本数是为了让梯度大小不受样本数量影响，保证训练稳定性（若不除以样本数，样本越多梯度越大，会导致参数更新震荡）。

3. **维度匹配的关键**：`keepdims=True` 是反向传播代码的“避坑点”。比如b2的形状是 (1, 1)，dZ2的形状是 (1, 4)，求和后若不保持维度，会变成 (1,)，与b2维度不匹配，无法进行后续参数更新；加上 `keepdims=True` 后，求和结果仍为 (1, 1)，确保维度一致。

4. **反向传播的流程闭环**：代码中反向传播的顺序（dZ2 → dW2、db2 → dZ1 → dW1、db1），完全遵循理论推导的“从输出层往输入层递推”逻辑，先计算输出层误差，再将误差回传至隐藏层，最终得到所有参数的梯度，为后续梯度下降更新参数做准备。

补充：代码中后续的参数更新 `W1 -= learning_rate * dW1` 等，对应理论公式  $w^l = w^l - \eta \cdot \frac{\partial L}{\partial w^l}$ ，即沿梯度反方向更新参数，减小损失，与此前推导完全一致。

---

## 三、多层网络通用化公式

对于**任意深度**的全连接神经网络（ $L$ 为总层数）：

1. **输出层误差**（ $l=L$ ）：

 $\delta^L = (a^L - y) \odot \sigma'(z^L)$ 

1. **隐藏层误差**（ $l=L-1,L-2,...,1$ ）：

 $\delta^l = \left((w^{l+1})^T \delta^{l+1}\right) \odot \sigma'(z^l)$ 

1. **参数梯度**：

 $\frac{\partial L}{\partial w^l} = \delta^l (a^{l-1})^T,\quad \frac{\partial L}{\partial b^l} = \delta^l$ 

1. **参数更新**：

 $w^l \leftarrow w^l - \eta \delta^l (a^{l-1})^T,\quad b^l \leftarrow b^l - \eta \delta^l$ 

---

## 四、关键补充：激活函数导数

反向传播的计算依赖激活函数导数，常用函数导数：

1. **Sigmoid**： $\sigma'(z)=\sigma(z)(1-\sigma(z))$ 

2. **ReLU**： $\sigma'(z)=\begin{cases}1, & z>0 \\ 0, & z\leq0\end{cases}$ 

3. **Tanh**： $\sigma'(z)=1-\tanh^2(z)$ 

---

## 五、反向传播的核心应用

### 1. 神经网络训练的核心引擎

反向传播是**监督学习训练神经网络的标准算法**：

- 前向传播：计算网络输出与损失

- 反向传播：计算所有参数梯度

- 迭代更新：重复上述步骤，直到损失收敛

### 2. 支撑所有深度学习模型

几乎所有深度模型都基于反向传播优化：

- 卷积神经网络（CNN）：图像分类/检测，仅将全连接权重替换为卷积核

- 循环神经网络（RNN/LSTM）：时序预测、NLP，加入时序梯度递推

- Transformer：大语言模型核心，基于自注意力+反向传播

### 3. 梯度优化与问题解决

反向传播直接暴露网络训练问题，推动优化方法发展：

- **梯度消失**：Sigmoid导数<1，深层网络梯度逐层衰减→改用ReLU、残差连接

- **梯度爆炸**：梯度过大导致参数震荡→梯度裁剪、权重归一化

- 优化器升级：从普通梯度下降→SGD、Adam、RMSprop（均基于反向传播梯度）

### 4. 实际任务落地

- 回归任务：房价预测、销量预测

- 分类任务：手写数字识别（MNIST）、图像分类

- 序列任务：文本生成、语音识别

- 其他：推荐系统、目标检测、自动驾驶感知

---

## 六、极简总结

1. **原理**：链式求导+梯度下降，从输出层回传误差

2. **核心公式**：

输出层误差 $\delta^2=(a^2-y)\odot\sigma'(z^2)$ 

隐藏层误差 $\delta^1=(w^2)^T\delta^2\odot\sigma'(z^1)$ 

权重更新 $w\leftarrow w-\eta\delta a^{l-1}$ 

1. **价值**：让神经网络可训练，是深度学习的**底层基石**。
> （注：文档部分内容可能由 AI 生成）