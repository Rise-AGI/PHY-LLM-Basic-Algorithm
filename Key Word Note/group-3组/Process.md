# Process（独立实验组）

## 零基础含义解释
Process就像**物理实验的独立实验组**，每个实验组都在进行不同的研究任务，互不干扰。

## 物理场景类比解释
在粒子物理实验中：
- Process相当于**独立的探测器数据采集组**：
  - 每个进程管理一组探测器的数据记录
  - 多个进程同时处理不同粒子源的数据
  - 进程间通信类似实验小组间的数据交换

## 基本配置
```bash
# 运行环境：Windows 11 + VSCode 1.88.0 + Process Explorer插件
# 查看进程树（实验组结构）
ps -ef | head -n 20  # 查看前20个进程（Linux）

# Windows查看命令
tasklist | findstr "python"  # 查找Python实验进程
```

## 物理系学生常见应用场景
- 管理多粒子源数据采集
- 并行处理光谱数据
- 实验模拟的独立运行环境

## 算法应用（物理实验场景）
```bash
# 运行环境：Windows 11 + Python 3.10 + VSCode Python插件(v2024.0.0)
# 多进程处理实验数据 - 探测器信号分析示例
python -c "from multiprocessing import Pool; Pool(4).map(process_data, files)"  # 使用4个实验组并行处理

# 物理参数说明：
# - Pool(4)：创建4个进程，对应4个探测器通道
# - process_data：信号处理函数，执行滤波/放大等操作
# - files：实验数据文件列表，每个文件对应不同时间窗口的数据
```

## 物理系学生扩展代码
```bash
# 实验进程优先级管理示例
# 设置不同实验进程的优先级
nice -n 10 python high_priority_analysis.py &  # 高优先级分析进程
nice -n 19 python background_simulation.py &   # 低优先级模拟进程

# 查看进程优先级
ps -eo pid,ni,pri,cmd | grep "python"  # 显示Python实验进程的优先级信息

# 动态调整进程优先级
renice 5 -p $(pgrep background_simulation)  # 降低模拟进程的优先级
```

## VSCode操作指引
1. 安装Process Explorer插件：
   - 提供进程树状图可视化
   - 支持资源占用实时监控
   - 显示进程详细属性
2. 打开"Processes"视图：
   - 菜单栏选择"View" > "Process Explorer"
   - 查看实验进程的层级关系
   - 颜色编码显示资源占用情况
3. 右键进程选择"Kill"终止异常实验进程：
   - 强制停止无响应的模拟程序
   - 释放被占用的实验设备资源
4. 使用"Search"查找特定实验进程：
   - 按进程名过滤（如"detector"）
   - 按CPU/GPU使用率排序
   - 按实验日期筛选进程
5. 安装"Terminal Tabs"插件管理多个监控会话
6. 使用"Performance Monitor"跟踪进程性能指标
