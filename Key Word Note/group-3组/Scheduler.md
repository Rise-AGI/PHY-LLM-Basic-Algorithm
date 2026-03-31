# Scheduler（实验任务调度员）

## 零基础含义解释
Scheduler就像**实验任务分配员**，决定哪个实验进程/线程何时使用探测器资源。

## 物理场景类比解释
在粒子物理实验中：
- Scheduler相当于**实验任务调度员**：
  - 分配探测器使用时间片（如每组10分钟）
  - 优先处理高能粒子探测任务
  - 管理实验数据传输队列

## 基本配置
```bash
# 查看实时优先级（实验紧急程度）
ps -eo pri,ni,cmd --sort -pri | head  # 按优先级排序

# 设置实验进程优先级
nice -n 5 python high_priority_analysis.py  # 低优先级运行
renice 0 -p 1234  # 提高进程ID=1234的优先级
```

## 物理系学生常见应用场景
- 优先处理高能粒子数据
- 管理低温实验实时监控
- 调度多探测器并行采集

## 算法应用（物理实验场景）
```python
# 运行环境：Windows 11 + Python 3.10 + VSCode Python插件(v2024.0.0)
# 多线程实验调度 - 粒子探测器优先级调度示例
import os
import time

# 函数说明：探测器调度核心函数
# 参数：detector_id - 探测器编号（0-3）
# 物理应用：按优先级调度不同探测器的数据采集
def schedule_detector(detector_id):
    os.nice(detector_id)  # 按探测器ID设置优先级（0-3）
    print(f"Starting detector {detector_id} at {time.time()}")  # 记录启动时间

# 创建并行实验任务
for d in range(4):  # 管理4个探测器通道
    if os.fork() == 0:  # 创建子进程
        schedule_detector(d)  # 启动探测器调度
        break
```

## 物理系学生扩展代码
```bash
# 实验任务优先级动态调整
# 设置不同实验进程的优先级
renice 5 -p $(pgrep "detector")  # 提高所有探测器进程优先级
renice 15 -p $(pgrep "simulation")  # 降低模拟进程优先级

# 查看实时优先级变化
ps -eo pri,ni,cmd --sort -pri | grep -E "detector|simulation"
```

## VSCode操作指引
1. 安装Process Explorer插件：
   - 提供进程优先级可视化
   - 支持资源调度监控
   - 显示进程亲和性设置
2. 在"Processes"视图调整实验进程优先级：
   - 右键进程选择"Set Priority"调整优先级
   - 颜色编码显示进程状态（绿色=正常，黄色=高优先级，红色=异常）
3. 使用"Timeline"面板查看资源调度：
   - 显示进程调度时间线
   - 查看CPU核心分配情况
   - 分析进程等待时间
4. 在"Settings"中配置进程亲和性：
   ```json
   {
     "processExplorer": {
       "affinity": {
         "enabled": true,
         "cores": [0, 1, 2, 3]  # 指定使用核心0-3处理实验进程
       }
     }
   }
   ```
5. 安装"Process Monitor"插件跟踪进程活动
6. 使用"Performance"视图分析调度效率
