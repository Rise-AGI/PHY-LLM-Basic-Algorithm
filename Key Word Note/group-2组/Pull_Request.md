# Pull Request（实验方案同行评审）

## 零基础解释
Pull Request（PR）就像**物理实验的同行评审流程**，当你完成实验方案设计后，通过PR邀请导师和同行对你的实验方法进行评审和建议。

## 物理场景类比解释
在量子物理研究中：
- PR相当于**实验方案的专家评审**：
  - 类似超导量子计算实验的方案审查流程
  - 评审者可提出参数调整建议（如量子比特频率设置）
  - 支持多人对同一实验方案讨论（如同步验证量子态测量方法）
  - 支持"Review"功能如同实验数据交叉验证

## 物理场景类比
在粒子物理实验中：
- PR相当于**实验方案的专家评审**：
  - 类比ATLAS实验的方案审查流程
  - 评审者可提出实验参数调整建议
  - 支持多人对同一实验方案讨论

## 基本操作
```bash
# 运行环境：Windows 11 + Git 2.40.0 + VSCode GitLens插件(v1.0.0)
# 创建特性分支（实验新方案）
git checkout -b new-experiment-plan  # 创建并切换分支
# checkout = 切换分支，如同切换实验参数组
# -b = 新建分支，detector-3.3是新分支名

# 推送分支到远程仓库
git push origin new-experiment-plan  # 同步实验分支
# push = 推送，如同共享实验方案
```

## 物理系学生应用场景
- 实验方案优化建议
- 多人协作修改探测器参数
- 实验数据处理方法评审

## 算法应用（物理实验场景）
```bash
# 创建PR后添加评审注释
# 运行环境：Windows 11 + Git 2.40.0 + VSCode GitLens插件(v1.0.0)
git diff main new-experiment-plan  # 查看实验方案差异
# diff = 比对差异，如同验证实验数据一致性

# 合并评审通过的方案
git checkout main  # 切换到主分支
# checkout = 切换分支，如同切换到标准实验方案

git merge new-experiment-plan  # 合并实验方案
# merge = 合并，如同整合多方实验验证结果
```

# 物理系专属代码示例
```bash
# 量子模拟代码评审流程
git checkout -b quantum-sim-3qubits  # 创建量子模拟分支
# ...进行量子算法开发...
git push origin quantum-sim-3qubits  # 推送分支

# 在GitHub创建PR后
git fetch origin  # 获取评审反馈
git diff origin/main quantum-sim-3qubits  # 查看差异
# 根据评审意见修改代码
git add corrected_code.py
git commit -m "修正量子门时序问题"
git push origin quantum-sim-3qubits  # 更新PR
```

## VSCode操作指引
1. 打开GitHub插件（扩展商店搜索"GitHub"）
2. 在"Branches"视图右键创建PR（支持可视化对比分支）
3. 使用"Diff Viewer"查看实验方案差异（如同对比实验数据变化）
4. 在"Files Changed"标签添加评审注释（类似实验日志批注）
5. 通过"Review"按钮提交整体评审意见（如同实验报告评审签字）
6. 使用"Timeline"查看评审讨论历史（如同查看实验方案修改记录）
