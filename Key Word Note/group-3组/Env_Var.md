# Env Var（实验环境参数）

## 零基础含义解释
Env Var就像**实验的环境参数配置**，记录着实验进行时需要的各种环境变量。

## 物理场景类比解释
在粒子物理实验中：
- Env Var相当于**实验参数配置表**：
  - 设置探测器电压（DETECTOR_VOLTAGE=3.3V）- 类比实验前配置探测器的供电参数
  - 定义数据存储路径（DATA_PATH=/experiment/day1）- 类比实验记录本上的数据存放位置标记
  - 控制实验阶段（EXPERIMENT_PHASE=calibration）- 类比实验的不同阶段（准备、采集、分析）

## 基本配置
```bash
# 设置实验环境变量
export DETECTOR_MODE=high_gain  # 设定探测器高增益模式
# 查看实验参数
echo $DETECTOR_MODE  # 检查探测器模式设置
```

## 物理系学生常见应用场景
- 配置实验仪器参数
- 设置数据存储路径
- 定义模拟计算的精度要求

## 算法应用（物理实验场景）
```bash
# 运行环境：Windows 11 + Python 3.10 + VSCode Python插件(v2024.0.0)
# 实验参数动态配置 - 粒子加速器控制示例
export PARTICLE_TYPE=muon  # 设置粒子类型(muon=缪子, electron=电子)
export ENERGY_LEVEL=high   # 设置能量级别(high=高能, low=低能)
export MAGNET_STRENGTH=5.0T  # 设置磁铁强度

# 启动实验模拟 - 传递环境变量到物理模拟程序
python run_simulation.py  # 执行粒子轨迹模拟程序
# 物理参数说明：
# - PARTICLE_TYPE: 粒子种类影响相互作用截面
# - ENERGY_LEVEL: 控制粒子束流能量分布
# - MAGNET_STRENGTH: 影响粒子轨迹偏转半径
```

## 物理系学生扩展代码
```bash
# 批量处理实验数据 - 环境变量驱动的自动化示例
export EXPERIMENT_DATE=20260331  # 设置实验日期
export DETECTOR_ID=ATLAS-04      # 设置探测器编号
export DATA_PATH=/experiment/data/$EXPERIMENT_DATE/$DETECTOR_ID  # 构建数据路径

# 创建实验数据目录结构
mkdir -p $DATA_PATH/raw $DATA_PATH/processed

# 使用环境变量配置数据处理流程
python preprocess.py --input $DATA_PATH/raw --output $DATA_PATH/processed
```

## VSCode操作指引
1. 在终端设置环境变量：
   - Windows: `set PARTICLE_TYPE=muon`
   - Linux/macOS: `export PARTICLE_TYPE=muon`
2. 使用".env"文件保存实验配置：
   ```env
   # .env 文件示例 - 粒子实验配置
   PARTICLE_TYPE=muon
   ENERGY_LEVEL=high
   MAGNET_STRENGTH=5.0T
   ```
3. 安装"DotENV"插件管理变量：
   - 自动加载.env文件
   - 变量高亮显示
   - 语法错误检查
4. 在调试配置中设置实验参数：
   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Python: 调试粒子模拟",
         "type": "python",
         "request": "launch",
         "program": "run_simulation.py",
         "console": "integratedTerminal",
         "env": {
           "PARTICLE_TYPE": "muon",
           "ENERGY_LEVEL": "high"
         }
       }
     ]
   }
   ```
5. 使用"Python Environments"管理器切换解释器版本
6. 安装"Python Environment Manager"插件可视化配置环境变量
