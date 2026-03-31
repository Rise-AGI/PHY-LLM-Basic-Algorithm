# MLP（多层滤波器系统）

## 零基础含义解释
MLP就像**物理实验的多层滤波器系统**，通过逐层变换找出实验数据中的特征。

## 物理场景类比解释
在粒子物理实验中：
- MLP相当于**多级滤波器系统**：
  - 输入层：探测器原始信号（如能谱数据）
  - 隐藏层：特征提取（如粒子类型区分）
  - 输出层：分类结果（如μ子/π子识别）

## 基本配置
```bash
# 安装物理实验的MLP工具包
pip install torch  # 获取张量计算能力
# 验证安装
python -c "import torch; print(torch.__version__)"  # 检查PyTorch版本
```

## 物理系学生常见应用场景
- 探测器信号分类（如粒子识别）
- 多层滤波器设计（如信号去噪）
- 实时数据特征提取（如粒子类型区分）

## 算法应用（物理实验场景）
```python
# 粒子能量分类器
import torch
from torch.nn import Linear, ReLU, LogSoftmax

class ParticleClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            Linear(1024, 512),  # 输入：1024点能谱数据
            ReLU(),
            Linear(512, 2),    # 输出：μ子/π子分类
            LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)
```

## VSCode操作指引
1. 安装Python插件(v2024.0.0)
2. 配置"Python: Select Interpreter"选择虚拟环境
3. 使用"Run and Debug"视图设置PyTorch断点
4. 在Jupyter插件中创建实时训练笔记本
