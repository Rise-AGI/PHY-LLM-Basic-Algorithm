# 从基础到实践：2 层神经网络反向传播公式推导

我会**从零开始，用链式法则一步步推导**这个 2层神经网络（输入层→隐藏层→输出层）的反向传播公式，**完全对应你代码中的每一行反向传播代码**，让你清晰理解公式来源。

# 一、先明确：网络结构 + 前向传播公式（推导基础）

你的网络是**最简单的全连接神经网络**：

输入层(2神经元) → 隐藏层(4神经元，Sigmoid) → 输出层(1神经元，Sigmoid)

损失函数：**均方误差 MSE**

### 统一符号定义（和代码严格对应）

|符号|含义|形状|代码变量|
|---|---|---|---|
| $X$ |输入数据|(2, 4)| $X$ |
| $Y$ |真实标签|(1, 4)| $Y$ |
| $W_1,b_1$ |隐藏层权重、偏置|(4,2),(4,1)| $W1,b1$ |
| $Z_1$ |隐藏层加权和 ( $W_1X+b_1$ )|(4, 4)| $Z1$ |
| $A_1$ |隐藏层激活值 ( $\sigma(Z_1)$ )|(4, 4)| $A1$ |
| $W_2,b_2$ |输出层权重、偏置|(1,4),(1,1)| $W2,b2$ |
| $Z_2$ |输出层加权和 ( $W_2A_1+b_2$ )|(1, 4)| $Z2$ |
| $A_2$ |输出层激活值 ( $\sigma(Z_2)$ )|(1, 4)| $A2$ |
| $\sigma(z)$ |Sigmoid激活函数|-|sigmoid|
| $L$ |损失函数(MSE)|标量|loss|
| $m=4$ |样本数量|标量|4|
---

### 1. 激活函数与导数

Sigmoid函数：

 $\sigma(z) = \frac{1}{1+e^{-z}}$ 

Sigmoid导数（核心公式）：

 $\sigma'(z) = \sigma(z) \cdot (1-\sigma(z))$ 

→ 对应代码：`sigmoid_deriv(z)`

### 2. 前向传播公式（代码中已实现）

1. 隐藏层：

 $Z_1 = W_1X + b_1,\quad A_1 = \sigma(Z_1)$ 

1. 输出层：

 $Z_2 = W_2A_1 + b_2,\quad A_2 = \sigma(Z_2)$ 

1. 损失函数（MSE）：

 $L = \frac{1}{m}\sum_{i=1}^m (Y - A_2)^2$ 

→ 对应代码：`loss = np.mean((Y - A2) ** 2)`

---

# 二、反向传播核心目标

反向传播的本质：**用链式法则，从后往前计算损失 ** $L$  ** 对所有参数 ** $W_1,b_1,W_2,b_2$  ** 的偏导数（梯度）**，再用梯度下降更新参数。

我们需要求6个梯度：

 $\frac{\partial L}{\partial Z_2},\ \frac{\partial L}{\partial W_2},\ \frac{\partial L}{\partial b_2},\ \frac{\partial L}{\partial Z_1},\ \frac{\partial L}{\partial W_1},\ \frac{\partial L}{\partial b_1}$ 

---

# 三、分步推导反向传播公式

## 第一步：推导输出层梯度（对应代码 `dZ2, dW2, db2`）

### 1. 求  $\boldsymbol{\frac{\partial L}{\partial Z_2}}$ （代码：`dZ2`）

用**链式法则**：

 $\frac{\partial L}{\partial Z_2} = \frac{\partial L}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2}$ 

- 第一步：求  $\frac{\partial L}{\partial A_2}$ 

对MSE损失求导：

 $\frac{\partial L}{\partial A_2} = \frac{2}{m}(A_2 - Y)$ 

- 第二步：求  $\frac{\partial A_2}{\partial Z_2}$ 

激活函数导数：

 $\frac{\partial A_2}{\partial Z_2} = \sigma'(Z_2)$ 

- 合并：

     $\frac{\partial L}{\partial Z_2} = \frac{2}{m}(A_2-Y) \cdot \sigma'(Z_2)$ 

✅ **代码对应**：

梯度下降中**常数因子不影响优化方向**，代码省略了  $2/m$ ，直接写：

```Python

dZ2 = (A2 - Y) * sigmoid_deriv(Z2)
```

### 2. 求  $\boldsymbol{\frac{\partial L}{\partial W_2}}$ （代码：`dW2`）

链式法则：

 $\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial W_2}$ 

由  $Z_2=W_2A_1+b_2$ ，得矩阵求导：

 $\frac{\partial Z_2}{\partial W_2} = A_1^T$ 

合并后**除以样本数 ** $m$ ：

 $\frac{\partial L}{\partial W_2} = \frac{1}{m} \cdot \frac{\partial L}{\partial Z_2} \cdot A_1^T$ 

✅ **代码对应**：

```Python

dW2 = np.dot(dZ2, A1.T) / 4
```

### 3. 求  $\boldsymbol{\frac{\partial L}{\partial b_2}}$ （代码：`db2`）

由  $Z_2=W_2A_1+b_2$ ，得：

 $\frac{\partial Z_2}{\partial b_2} = 1$ 

链式法则+求和+平均：

 $\frac{\partial L}{\partial b_2} = \frac{1}{m}\sum \frac{\partial L}{\partial Z_2}$ 

✅ **代码对应**：

```Python

db2 = np.sum(dZ2, axis=1, keepdims=True) / 4
```

---

## 第二步：推导隐藏层梯度（对应代码 `dZ1, dW1, db1`）

隐藏层梯度需要**继续向前链式传递**，把输出层的误差传到隐藏层。

### 1. 求  $\boldsymbol{\frac{\partial L}{\partial Z_1}}$ （代码：`dZ1`）

链式法则（三层嵌套）：

 $\frac{\partial L}{\partial Z_1} = \underbrace{\frac{\partial L}{\partial Z_2}}_{输出层误差} \cdot \underbrace{\frac{\partial Z_2}{\partial A_1}}_{权重传递} \cdot \underbrace{\frac{\partial A_1}{\partial Z_1}}_{激活函数导数}$ 

-  $\frac{\partial Z_2}{\partial A_1} = W_2^T$ （权重转置传递误差）

-  $\frac{\partial A_1}{\partial Z_1} = \sigma'(Z_1)$ 

最终公式：

 $\frac{\partial L}{\partial Z_1} = \left(W_2^T \cdot \frac{\partial L}{\partial Z_2}\right) \cdot \sigma'(Z_1)$ 

✅ **代码对应**：

```Python

dZ1 = np.dot(W2.T, dZ2) * sigmoid_deriv(Z1)
```

### 2. 求  $\boldsymbol{\frac{\partial L}{\partial W_1}}$ （代码：`dW1`）

和输出层权重推导完全一致：

 $\frac{\partial L}{\partial W_1} = \frac{1}{m} \cdot \frac{\partial L}{\partial Z_1} \cdot X^T$ 

✅ **代码对应**：

```Python

dW1 = np.dot(dZ1, X.T) / 4
```

### 3. 求  $\boldsymbol{\frac{\partial L}{\partial b_1}}$ （代码：`db1`）

和输出层偏置推导完全一致：

 $\frac{\partial L}{\partial b_1} = \frac{1}{m}\sum \frac{\partial L}{\partial Z_1}$ 

✅ **代码对应**：

```Python

db1 = np.sum(dZ1, axis=1, keepdims=True) / 4
```

---

## 第三步：梯度下降更新参数

得到所有梯度后，用**学习率**更新权重和偏置：

 $\begin{align*}
W_1 &= W_1 - \alpha \cdot \frac{\partial L}{\partial W_1} \\
b_1 &= b_1 - \alpha \cdot \frac{\partial L}{\partial b_1} \\
W_2 &= W_2 - \alpha \cdot \frac{\partial L}{\partial W_2} \\
b_2 &= b_2 - \alpha \cdot \frac{\partial L}{\partial b_2} \\
\end{align*}$ 

→ 对应代码中参数更新的4行代码。

---

# 四、公式与代码终极对照表

|数学公式|代码行|
|---|---|
| $\frac{\partial L}{\partial Z_2}$ |`dZ2 = (A2 - Y) * sigmoid_deriv(Z2)`|
| $\frac{\partial L}{\partial W_2}$ |`dW2 = np.dot(dZ2, A1.T) / 4`|
| $\frac{\partial L}{\partial b_2}$ |`db2 = np.sum(dZ2, axis=1, keepdims=True)/4`|
| $\frac{\partial L}{\partial Z_1}$ |`dZ1 = np.dot(W2.T, dZ2) * sigmoid_deriv(Z1)`|
| $\frac{\partial L}{\partial W_1}$ |`dW1 = np.dot(dZ1, X.T) / 4`|
| $\frac{\partial L}{\partial b_1}$ |`db1 = np.sum(dZ1, axis=1, keepdims=True)/4`|
---

### 总结

1. **反向传播核心**：**链式法则 + 从输出层向输入层反向求导**；

2. **关键公式**：Sigmoid导数  $\sigma'(z)=\sigma(z)(1-\sigma(z))$ ，是所有激活值梯度的核心；

3. **矩阵维度**：权重转置（ $W_2^T$ 、 $A_1^T$ ）是为了匹配矩阵乘法维度，保证误差正确传递；

4. **代码简化**：省略了MSE导数的常数因子  $2$ ，不影响模型训练效果。
> （注：文档部分内容可能由 AI 生成）