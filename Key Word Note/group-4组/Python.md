# Python（物理实验数据记录本）

## 零基础含义解释
Python就像**物理实验的快速数据记录本**，即时记录和分析实验数据，无需复杂配置。

## 物理场景类比解释
在粒子物理实验中：
- Python相当于**实验数据记录本**：
  - 快速记录探测器信号（如粒子计数）
  - 实时绘制能谱直方图
  - 自动化数据预处理（如滤波/校准）
- 类比实验记录本的"批处理模式"：可编写脚本自动完成重复性数据分析任务

## 物理系学生常见应用场景
- 光谱数据拟合（如高斯分布）
- 粒子轨迹可视化（如3D绘图）
- 实时数据采集与存储
- 批量处理实验日志文件（如自动化提取温度数据）
- 量子态概率计算（如叠加态模拟）

## 基本配置
```bash
# 运行环境：Windows 11 + Python 3.10 + VSCode Python插件(v2024.0.0)
# 安装NumPy物理数据处理库
pip install numpy  # pip = Python包管理器，安装科学计算库
# 验证安装
python -c "import numpy as np; print(np.__version__)"  # 检查NumPy版本：np=NumPy缩写
```

### Windows/macOS/Linux三平台安装教程
1. **Windows系统**
   - 下载安装包：https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
   - 双击运行安装向导 → 勾选"Add to PATH" → 点击"Install Now"
   - 验证安装：`python --version`（终端输入）

2. **macOS系统**
   - 安装Homebrew：`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
   - 安装Python：`brew install python@3.10`
   - 验证安装：`python3 --version`

3. **Linux系统（Ubuntu）**
   - 更新包列表：`sudo apt update`
   - 安装Python：`sudo apt install python3.10`
   - 验证安装：`python3 --version`

## 物理系学生常见应用场景
- 光谱数据拟合（如高斯分布）
- 粒子轨迹可视化（如3D绘图）
- 实时数据采集与存储

## 算法应用（物理实验场景）
```python
# 运行环境：Python 3.10 + PyTorch 2.1 + VSCode Python插件+Jupyter插件
# 粒子能谱分析与量子态模拟
import numpy as np
import matplotlib.pyplot as plt

# 加载探测器数据（CSV格式：每行记录一次粒子撞击能量）
data = np.loadtxt("detector_data.csv")  # np.loadtxt=NumPy加载文本数据函数

# 创建能谱直方图（bins=50表示将能量范围划分为50个区间）
hist, bins = np.histogram(data, bins=50)

# 绘制能谱图（bins[:-1]取区间左边界值）
plt.plot(bins[:-1], hist)
plt.title("粒子能谱分析（实验数据）")
plt.xlabel("能量 (MeV)")  # MeV=兆电子伏特，粒子物理常用单位
plt.ylabel("计数")  # Y轴表示每个能量区间内的粒子数量
plt.show()

# 量子态叠加模拟（|0> + e^{iθ}|1>）
theta = np.linspace(0, 2*np.pi, 100)
psi = np.cos(theta/2) + 1j*np.sin(theta/2)  # 1j=虚数单位，模拟量子叠加态
plt.plot(theta, np.abs(psi)**2)  # np.abs=计算模长，展示概率分布
plt.title("量子态概率分布模拟")
plt.xlabel("相位角 θ (rad)")
plt.ylabel("概率 |ψ|²")
plt.show()
```

## VSCode操作指引
1. 安装Python插件(v2024.0.0)和Jupyter插件
   - 打开VSCode → Ctrl+Shift+X → 搜索"Python" → 点击安装
   - 同样方式安装"Jupyter"插件

2. 创建虚拟环境
   - 终端执行：`python -m venv venv`（创建venv文件夹存放虚拟环境）

3. 激活环境
   - Linux/macOS：`source venv/bin/activate`（source=执行脚本文件）
   - Windows：`.\venv\Scripts\activate`（.\\=当前目录执行脚本）

4. 创建Jupyter Notebook
   - Ctrl+Shift+P → 输入"Jupyter: Create New Blank Notebook" → 回车
   - 在单元格输入代码：`import numpy as np` → Ctrl+Enter运行
