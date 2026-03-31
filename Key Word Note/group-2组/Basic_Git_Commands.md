# Basic Git Commands（实验记录操作）

## 零基础含义解释
Git基础命令就像**物理实验记录本的基本操作**：
- `add`：标记待记录的实验数据
- `commit`：保存实验步骤记录
- `push/pull`：同步实验记录到共享平台

## 物理场景类比解释
在量子光学实验中：
- `git add`相当于**标记待存档的干涉测量数据**
- `git commit`相当于**封存当前实验状态（如同保存关键观测时刻）**
- `git push`相当于**上传数据到实验室共享存储**
- `git pull`相当于**获取他人更新的实验方案**
- `git diff`相当于**对比不同版本的实验参数**

## 物理场景类比解释
在粒子物理实验中：
- `git add`相当于**标记待记录的实验数据**
- `git commit`相当于**归档当前实验状态**
- `git push`相当于**提交实验记录到共享平台**

## 基本操作
```bash
# 运行环境：Windows 11 + Git Bash 2.40.0 + VSCode GitLens插件(v1.0.0)
# 添加单个实验数据文件
git add experiment_data.csv  # 添加特定文件（类似整理待存档的实验数据）
# add = 添加文件，如同标记待归档的实验记录

# 添加所有修改
git add .  # 点号表示当前目录所有改动（批量添加实验数据）
# . = 当前目录，如同选择所有新生成的实验文件

# 提交带注释的实验记录
git commit -m "更新：优化探测器阈值设置"  # 提交说明（如同实验日志备注）
# commit = 提交，-m = 添加备注信息

# 查看实验记录历史
git log --oneline  # 查看简洁版提交历史（类似翻阅实验记录本）
# log = 日志，--oneline = 简洁显示
```

## 物理系学生常见应用场景
- 提交粒子模拟代码更新
- 记录实验参数调整过程
- 追踪数据处理脚本变更

## 算法应用（物理实验场景）
```bash
# 实验数据版本对比
# 运行环境：Windows 11 + Git Bash 2.40.0
git diff  # 查看当前工作区与暂存区的差异（如同对比实时数据与参考值）
# diff = 差异，显示未暂存的修改

git diff --cached  # 查看已暂存的修改（如同检查待归档数据）
# --cached = 查看暂存区内容

# 回退到上一实验版本
git checkout HEAD^ experiment_data.csv  # 回退特定文件
# checkout = 切换版本，HEAD^ = 上一提交
```

# 物理系专属代码示例
```bash
# 批量处理光谱实验数据
for file in *.txt; do
  git add "$file"  # 添加每个数据文件
  git commit -m "提交光谱数据：$file"  # 提交注释包含文件名
done
git push origin main  # 批量推送所有数据

# 恢复错误修改的实验数据
git reset --hard HEAD~1  # 回退到最后一次正确提交
# reset = 重置，--hard = 强制回退，HEAD~1 = 上一次提交
```

## VSCode操作指引
1. 在Git Changes侧边栏勾选待提交文件（如同选择待归档的实验数据）
2. 在输入框输入中文提交信息（记录实验操作说明）
3. 点击"Commit"按钮提交（完成实验记录归档）
4. 使用"..."菜单中的"Discard Changes"撤销修改（如同清除错误记录）
5. 通过"Compare"按钮可视化对比实验版本（如同对比参数变化）
6. 右键文件选择"Git: Revert File"回退特定文件修改（如同修正异常数据）
