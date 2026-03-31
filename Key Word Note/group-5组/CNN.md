# CNN（粒子轨迹特征提取器）

## 零基础含义解释
CNN就像**物理实验的粒子轨迹特征提取器**，通过卷积层自动提取实验数据中的局部特征。

## 物理场景类比解释
在粒子物理实验中：
- CNN相当于**多层特征提取系统**：
  - 输入层：探测器原始信号（如能谱数据）
  - 卷积层：特征提取（如能量峰检测）
  - 输出层：特征分类（如粒子类型识别）

## 基本配置
```bash
# 安装物理实验的CNN工具包
pip install torch  # 获取PyTorch计算能力
# 验证安装
python -c "import torch; print(torch.__version__)"  # 检查PyTorch版本
```

## 物理系学生常见应用场景
- 能谱数据特征提取（如μ子/π子峰区分）
- 探测器信号模式识别（如噪声过滤）
- 实时轨迹特征分析（如粒子类型实时分类）

## 算法应用（物理实验场景）
```python
# 粒子轨迹特征提取
import torch
from torch.nn import Conv1d, ReLU, MaxPool1d

class TrajectoryFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            Conv1d(1, 16, 5),  # 输入：能谱数据
            ReLU(),
            MaxPool1d(2),      # 特征提取窗口
            Conv1d(16, 32, 3),
            ReLU(),
            MaxPool1d(2)
        )

    def forward(self, x):
        return self.net(x)
```

## VSCode操作指引
1. 安装Python插件(v2024.0.0)
2. 配置"Python: Select Interpreter"选择虚拟环境
3. 使用"Run and Debug"视图设置PyTorch断点
4. 在Jupyter插件中创建实时训练笔记本
