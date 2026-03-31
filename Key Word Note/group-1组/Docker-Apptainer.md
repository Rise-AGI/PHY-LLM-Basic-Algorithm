# Docker/Apptainer（容器技术）

## 零基础解释
Docker/Apptainer是软件打包工具，像**实验室的标准化实验箱**，能将实验所需的所有设备和材料打包，确保在不同实验室复现实验环境。

## 物理场景类比解释
在粒子物理实验中：
- Docker相当于**标准化实验运输箱**：
  - 打包探测器数据处理软件栈（类比集装箱装载实验设备）
  - 确保CERN与本地实验室环境一致（跨平台复现实验条件）
  - 支持多版本实验软件共存（如GEANT4仿真不同版本并行）

## 基本配置
```dockerfile
# 运行环境：Windows 11 + Docker 24.0 + VSCode Docker插件(v1.18.0)
# 粒子模拟Dockerfile示例
FROM nvidia/cuda:12.1-base
# 安装实验软件栈
RUN apt-get update && apt-get install -y python3-pip geant4
# 挂载实验数据目录
VOLUME /experiment_data
# 设置工作路径
WORKDIR /app
# 启动模拟脚本
CMD ["bash", "run_simulation.sh"]
```

## 算法应用
### 物理实验场景：GEANT4仿真环境容器化
```dockerfile
# 运行环境：Windows 11 + Docker 24.0 + VSCode Docker插件(v1.18.0)
# GEANT4物理仿真容器化示例
FROM geant4/geant4-base:10.7-py38
# 安装实验依赖（模拟实验设备清单）
RUN apt-get update && \
    apt-get install -y cmake g++ libgl1 libx11-dev && \
    pip install numpy pandas

# 挂载实验数据目录（类似实验材料存储区）
VOLUME /experiment_data

# 设置工作路径（实验操作台区域）
WORKDIR /app

# 拷贝实验代码（模拟设备安装）
COPY simulation_code/ /app/

# 构建仿真程序（实验装置组装）
RUN mkdir build && \
    cd build && \
    cmake .. && \
    make

# 启动仿真（实验运行）
CMD ["./build/simulation", "--input", "/experiment_data/input.root"]
```
```
- **关键注释**：
  - `FROM geant4/geant4-base:10.7-py38`：基于官方GEANT4镜像构建
  - `VOLUME /experiment_data`：挂载实验数据目录（类似更换实验样品）
  - `CMD ["./build/simulation", ...]`：指定仿真执行命令（实验启动按钮）

## 物理系学生常见应用场景
- **跨平台实验复现**：在本地PC和超算中心运行完全一致的GEANT4仿真环境
- **多版本实验并行**：同时运行GEANT4 10.6和10.7版本对比实验结果
- **实验环境共享**：通过DockerHub共享完整的实验环境配置（类似共享实验装置）
- **批量数据处理**：使用Docker容器编排批量处理探测器原始数据

## VSCode操作指引
### Docker物理实验环境配置
1. 插件安装：
   - 打开扩展面板（`Ctrl+Shift+X`）
   - 搜索"Docker"插件（Microsoft官方出品）
   - 点击安装按钮（需管理员权限时输入密码）

2. 容器操作：
   - 查看运行状态：
     - 左侧活动栏点击Docker图标
     - 在"Containers"部分查看实验容器状态
   - 实时监控资源：
     - 右键容器 > "Stats" > 观察GPU/内存使用（类似实验监控仪表盘）

3. 物理实验调试：
   - 进入容器终端：
     - 右键运行中的容器 > "Attach Shell"
     - 输入`tail -f logs/simulation.log`实时查看实验日志
   - 可视化调试：
     - 安装"Remote - Containers"插件
     - 在容器内安装Jupyter扩展（实验数据可视化）

4. 镜像管理：
   - 构建镜像：
     - 右键Dockerfile > "Build Image"
     - 输入镜像名称：`geant4-experiment:latest`
   - 推送镜像：
     - 右键镜像 > "Push" > 登录DockerHub后共享实验环境
