# OS（实验控制中心）

## 零基础含义解释
OS就像**物理实验的中央控制系统**，管理所有实验设备、数据流和操作流程。

## 物理场景类比解释
在粒子物理实验中：
- OS相当于**实验控制中心**：
  - 管理探测器、数据采集系统
  - 分配实验资源（如计算节点）
  - 协调实验流程（如启动/停止采集）

## 基本配置
```bash
# 查看系统版本（实验控制软件）
uname -a  # Linux系统信息
# Windows查看命令
systeminfo | findstr "OS"  # 查找操作系统信息
```

## 物理系学生常见应用场景
- 控制探测器数据采集系统
- 管理实验计算资源
- 协调多设备同步操作

## 算法应用（物理实验场景）
```bash
# 运行环境：Windows 11 + VSCode 1.88.0 + Process Explorer插件
# 实时资源监控 - 粒子加速器控制系统示例
top  # 监控实验设备负载（Linux）
# 物理参数说明：
# - PID：进程ID对应实验设备编号
# - %CPU：探测器数据采集负载
# - MEM：实验数据缓存使用情况

# Windows实时监控命令
tasklist | findstr "python"  # 查找实验进程
perfmon  # 系统性能监视器
```

## 物理系学生扩展代码
```bash
# 实验资源自动化监控脚本
while true; do
  echo "当前时间：$(date)"
  echo "GPU资源使用："
  nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu --format=csv
  echo "实验进程状态："
  ps -ef | grep "experiment" | grep -v "grep"
  sleep 5  # 每5秒更新一次
done
```

## VSCode操作指引
1. 安装"Remote - SSH"插件连接远程实验设备：
   - 配置SSH连接参数（IP、端口、认证方式）
   - 使用"Connect to Host"建立远程连接
   - 在远程服务器上直接编辑实验代码
2. 使用"Terminal"运行系统命令：
   - Windows快捷键：Ctrl+`
   - Linux/macOS快捷键：Ctrl+` 或 Cmd+`
   - 使用多个终端标签页同时监控不同实验参数
3. 安装"Process Explorer"插件可视化管理：
   - 实时查看实验进程树状图
   - 右键进程选择"Kill"终止异常实验进程
   - 查看进程资源占用（CPU/GPU/内存）
4. 通过"Settings"同步实验环境配置：
   ```json
   {
     "settings.sync": {
       "enabled": true,
       "exclude": [
         "**/node_modules", 
         "**/.git"
       ]
     }
   }
   ```
5. 安装"Live Share"插件实现实验协同
6. 使用"Timeline"面板跟踪实验关键事件
