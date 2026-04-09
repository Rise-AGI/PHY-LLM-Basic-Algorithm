# CuPy 神经网络训练报告
## 一、模型基础信息
- 模型类型：全连接神经网络
- 网络结构：2 → 40 → 20 → 1
- 运行环境：CuPy GPU 加速
- 数据来源：手动导入(异或数据集)

## 二、训练超参数
- 总训练轮次：15000
- 学习率：0.5
- 激活函数：Sigmoid
- 损失函数：均方误差(MSE)
- 参数保存间隔：每3000轮保存一次

## 三、训练损失详情

Epoch 0      | Loss: 0.259285
Epoch 3000   | Loss: 0.000001
Epoch 6000   | Loss: 0.000000
Epoch 9000   | Loss: 0.000000
Epoch 12000  | Loss: 0.000000

- 最终损失值：0.000000

## 四、训练状态
✅ 模型收敛成功

## 五、参数文件说明
1. 训练中参数：`trained_params/W1_epoch_3000.csv` 等（每3000轮权重）
2. 最终模型参数：`trained_params/*_epoch_final.csv`
3. 文件格式：标准CSV，可直接用Excel/Python查看

## 六、输出文件
- `loss_curve.html`：训练损失可视化曲线
- `training_report.md`：本训练报告
- `trained_params/`：模型权重W、偏置B矩阵
