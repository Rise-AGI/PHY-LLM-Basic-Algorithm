# GitHub（全球实验平台）

## 零基础解释
GitHub就像**全球物理实验平台**，让不同实验室的科学家共享实验数据和代码。就像粒子加速器的全球合作项目，GitHub能协调世界各地的代码修改。

## 物理场景类比解释
在高能物理实验中：
- GitHub相当于**全球探测器数据共享平台**：
  - `git clone`复制实验方案（类似复制探测器设置）
  - `Pull Request`提交实验结果（类似同行评审）
  - `Actions`自动化实验流程（如数据采集/分析）
  - `Issues`记录实验问题（类似实验室日志）
  - `Wiki`编写实验手册（如同设备操作指南）

## 物理场景类比
在粒子物理实验中：
- GitHub相当于**ATLAS/CMS实验数据平台**：
  - `git clone`复制实验方案（类似复制探测器设置）
  - `Pull Request`提交实验结果（类似同行评审）
  - `Actions`自动化实验流程（如数据采集/分析）
  - `Issues`记录实验问题（类似实验室日志）

## 基本配置
```bash
# 运行环境：Windows 11 + GitHub CLI 2.40.0 + VSCode GitHub插件(v3.0.0)
# 安装GitHub CLI（Windows系统）
# 方法1：通过PowerShell安装
winget install --id GitHub.cli -e

# 方法2：手动下载安装
# 1. 访问 https://github.com/cli/cli/releases
# 2. 下载 Windows x86_64 版本
# 3. 双击安装包 → 选择安装路径 → 确认安装
```

## 物理系学生常见应用场景
- 共享粒子模拟代码
- 协作调整探测器参数
- 自动化实验数据采集

## 算法应用（物理实验场景）
```bash
# 批量提交实验数据
# 运行环境：Windows 11 + Git 2.40.0 + VSCode GitHub插件(v3.0.0)
git clone https://github.com/physical-lab/accelerator.git  # 克隆实验仓库 = 获取探测器初始设置
# clone = 复制远程仓库，如同获取实验标准模板

cd accelerator  # 进入实验目录（change directory）

git checkout -b detector-3.3  # 创建并切换实验分支
# checkout = 切换分支，如同切换实验参数组
# -b = 新建分支，detector-3.3是新分支名

git add calibration_data.csv  # 添加校准数据文件
# add = 添加文件，如同整理实验原始记录

git commit -m "记录探测器校准数据"  # 提交带注释
# commit = 提交更改，如同归档实验日志

git push origin detector-3.3  # 推送新分支到远程仓库
# push = 推送，如同共享实验结果
```

# 物理系专属代码示例
```bash
# 批量处理光谱实验数据
for file in *.txt; do
  git add "$file"  # 添加每个数据文件
  git commit -m "提交光谱数据：$file"  # 提交注释包含文件名
done
git push origin main  # 批量推送所有数据
```

## VSCode操作指引
1. 安装GitHub插件（扩展商店搜索"GitHub"）
2. 使用"Git: Clone"复制全球实验方案（支持可视化选择分支）
3. 在"Pull Request"中提交实验结果（实时查看评审反馈）
4. 配置"Actions"自动化实验流程（如数据校准自动化）
5. 使用"GitHub Pull Requests"侧边栏查看评审意见
6. 通过"Git: Sign in to GitHub"登录账号（支持扫码认证）
