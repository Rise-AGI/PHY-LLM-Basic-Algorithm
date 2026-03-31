# Git（实验数据溯源记录本）

## 零基础解释
Git就像**物理实验的实验数据溯源记录本**，记录每次实验数据的修改历史。就像科学家记录实验步骤一样，Git能追踪代码的每个改动。

## 物理场景类比解释
在粒子物理实验中：
- Git相当于**实验数据的版本控制**：
  - 每次`commit`记录实验参数变更（如探测器电压调整）
  - `branch`管理不同实验方案（如粒子源A/B对比）
  - `merge`整合多个实验小组的观测数据
  - `diff`比对实验数据差异（如同对比实验结果变化）
  - `reset`回退错误数据（类似修正错误的实验记录）

## 物理场景类比
在粒子物理实验中：
- Git相当于**实验数据的版本控制**：
  - 每次`commit`记录实验参数变更（如探测器电压调整）
  - `branch`管理不同实验方案（如粒子源A/B对比）
  - `merge`整合多个实验小组的观测数据

## 基本配置
```bash
# 运行环境：Windows 11 + Git 2.40.0 + VSCode GitLens插件(v1.0.0)
# 初始化仓库（创建实验记录本）
git init  # init = initialize，创建空的实验数据追踪目录

# 配置实验人员信息
git config --global user.name "Alice"  # 实验人员姓名
git config --global user.email "alice@lab.edu"  # 实验室邮箱
```

## 物理系学生常见应用场景
- 管理粒子模拟代码版本
- 追踪实验参数调整记录
- 协作处理光谱数据

## 算法应用（物理实验场景）
```bash
# 批量提交实验数据
# 运行环境：Windows 11 + Git 2.40.0 + VSCode GitLens插件(v1.0.0)
git add data_20260331.txt  # 添加实验数据文件（类似整理实验原始记录）
git commit -m "记录3月31日探测器校准数据"  # 提交带注释（如同归档实验日志）
git push origin main  # 同步到远程仓库（类似共享实验记录）
# 注：可使用GitLens插件的"Commit"按钮可视化提交
```

## VSCode操作指引
1. 安装GitLens插件
2. 打开命令面板（Ctrl+Shift+P）→ 输入"Git: Initialize Repository"
3. 在"Git Changes"侧边栏查看修改差异
4. 使用"Commit"按钮提交带注释的更改
5. 在"Timeline"视图查看历史提交（如同实验记录本的页码）
6. 右键文件选择"Git: Show File History"查看版本差异（类似对比实验数据修正）
