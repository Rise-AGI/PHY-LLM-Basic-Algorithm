# GPU（图形处理器）

## 零基础解释

GPU是专为并行计算设计的处理器，像**粒子轨迹重建的并行处理阵列**，特别适合同时处理数千个粒子信号。

## 物理场景类比解释
在粒子物理实验中：
- GPU相当于**探测器信号并行处理系统**：
  - 同时处理数千个粒子探测器通道（类似CUDA核心处理并行计算）
  - 加速粒子轨迹重建算法（如Kalman滤波并行化）
  - 大规模蒙特卡洛模拟加速（如GEANT4仿真）

## 基本配置
```bash
# 验证CUDA安装（物理场景）
nvidia-smi # 查看GPU状态（类似监测探测器工作状态）
nvcc --version # 检查CUDA编译器版本
```

## 算法应用
### 物理实验场景：粒子轨迹重建加速
```cpp
// 运行环境：Windows 11 + CUDA 12.1 + VSCode C/C++插件(v1.18.0)
// CUDA核函数：并行计算粒子轨迹参数
__global__ void calculateTrajectory(float* positions, float* results, int numParticles) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < numParticles) {
        // 模拟粒子轨迹计算（类似探测器信号处理）
        results[i] = sqrt(positions[i*3]*positions[i*3] + 
                         positions[i*3+1]*positions[i*3+1] + 
                         positions[i*3+2]*positions[i*3+2]);
    }
}

int main() {
    // 初始化粒子数据（模拟探测器采集）
    int numParticles = 100000;
    float* positions = new float[numParticles*3];
    float* results = new float[numParticles];
    
    // GPU内存分配
    float *d_positions, *d_results;
    cudaMalloc(&d_positions, numParticles*3*sizeof(float));
    cudaMalloc(&d_results, numParticles*sizeof(float));
    
    // 数据传输到GPU（类似探测器数据上载）
    cudaMemcpy(d_positions, positions, numParticles*3*sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动CUDA核函数（实验开始）
    dim3 blockSize(256);
    dim3 gridSize((numParticles + blockSize.x - 1) / blockSize.x);
    calculateTrajectory<<<gridSize, blockSize>>>(d_positions, d_results, numParticles);
    
    // 传输结果回主机（实验数据保存）
    cudaMemcpy(results, d_results, numParticles*sizeof(float), cudaMemcpyDeviceToHost);
    
    // 清理资源
    cudaFree(d_positions);
    cudaFree(d_results);
    delete[] positions;
    delete[] results;
    
    return 0;
}
```
```
- **关键注释**：
  - `__global__ void calculateTrajectory`：CUDA核函数定义（实验处理算法）
  - `threadIdx.x + blockIdx.x * blockDim.x`：线程索引计算（类似探测器通道编号）
  - `cudaMemcpy`：设备与主机间数据传输（实验数据采集与存储）

## 物理系学生常见应用场景
- **粒子轨迹重建**：加速Kalman滤波算法处理数千探测器信号
- **蒙特卡洛模拟**：GPU并行生成粒子碰撞事件（如LHC实验模拟）
- **实时数据筛选**：利用CUDA流处理实验数据（在线触发系统）
- **量子场论计算**：格点QCD模拟中的矩阵运算加速

## VSCode操作指引
### GPU物理实验环境配置
1. CUDA环境准备：
   - 安装NVIDIA插件：
     - 打开扩展面板（`Ctrl+Shift+X`）
     - 搜索"NVIDIA"插件（NVIDIA官方出品）
     - 点击安装并重启VSCode
   - 验证安装：
     ```bash
     nvidia-smi # 查看GPU状态（Windows PowerShell）
     nvcc --version # 检查CUDA编译器版本
```

2. CUDA代码调试：
   - 创建编译任务：
     - `Ctrl+Shift+P` > "C/C++: nvcc 生成活动文件"
     - 选择"GPU"架构配置（对应实验设备型号）
   - 启动NSight调试：
     - 安装"NSight Visual Studio Code Edition"插件
     - 在CUDA核函数设置断点（实验过程暂停点）
     - 使用"Parallel Nsight"窗口观察线程状态

3. 容器化实验：
   - 配置GPU容器：
     ```powershell
     # 安装NVIDIA Container Toolkit（实验环境容器化）
     docker run --gpus all nvidia/cuda:12.1-base nvidia-smi
     ```
   - VSCode连接：
     - 安装"Docker"插件
     - 右键Dockerfile > "Build Image" > 添加构建参数：
       ```dockerfile
       ARG NVIDIA_DRIVER_CAPABILITIES=compute,utility
       ```

4. 实验监控：
   - 实时GPU监控：
     - 安装"GPU Info and Monitor"扩展
     - 点击状态栏GPU图标查看利用率（类似实验监控仪表）
   - 能耗分析：
     - 使用NSight系统分析器：
       - `Ctrl+Shift+P` > "NSight > Start Analysis"
       - 记录实验过程中的GPU能耗数据
