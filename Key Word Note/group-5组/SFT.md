# SFT（物理实验的微调）

## 零基础含义解释
SFT（Supervised Fine-Tuning）就像**物理实验的微调过程**，在预训练基础上使用特定实验数据进一步优化模型。

## 物理场景类比解释
在粒子物理实验中：
- SFT相当于**探测器的微调阶段**：
  - 使用标准粒子源预训练（Pre-train）
  - 用特定实验数据微调（如μ子/π子分类）
  - 类似实验设备的参数校准

## 基本配置
```bash
# 安装物理实验微调工具
pip install torch  # 获取PyTorch计算能力
pip install peft  # 参数高效微调库
# 验证安装
python -c "from peft import PeftModel"
```

## 物理系学生常见应用场景
- 优化探测器粒子分类
- 针对特定实验数据微调
- 提高粒子识别精度

## 算法应用（物理实验场景）
```python
# 标准粒子源预训练模型
from transformers import AutoModelForSequenceClassification
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 特定实验数据微调
from peft import PeftConfig, PeftModel
config = PeftConfig(pet="lora", r=8, alpha=16, dropout=0.1)
model = PeftModel(base_model, config)  # 创建微调模型
```

## VSCode操作指引
1. 安装Python插件(v2024.0.0)
2. 配置"Python: Select Interpreter"选择虚拟环境
3. 使用"Run and Debug"视图调试微调过程
4. 在Jupyter插件中可视化微调结果
