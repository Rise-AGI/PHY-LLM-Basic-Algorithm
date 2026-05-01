"""
训练曲线可视化
用法：python plot_training.py [training_log.json 路径]
默认读取：qwen-sft-recovered/training_log.json
"""

import json
import sys
import os

# ── 读取日志文件 ──────────────────────────────────────────────
log_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "qwen-sft-recovered", "training_log.json"
)

if not os.path.exists(log_path):
    print(f"找不到日志文件：{log_path}")
    print()
    print("可能原因：训练被中断（OOM），save_final 未执行，日志未保存到本地。")
    print("解决办法：")
    print("  1. 重新提交训练，训练完成后日志会自动打包")
    print("  2. 若有其他 training_log.json，指定路径：")
    print("     python plot_training.py <training_log.json路径>")
    sys.exit(1)

with open(log_path, "r", encoding="utf-8") as f:
    logs = json.load(f)

# ── 分离 step 级别和 epoch 级别的记录 ────────────────────────
step_logs  = [r for r in logs if "lr" in r]           # 有 lr 的是 step 日志
epoch_logs = [r for r in logs if "elapsed_sec" in r]  # 有 elapsed_sec 的是 epoch 日志

steps       = [r["global_step"] for r in step_logs]
train_loss  = [r["train_loss"]  for r in step_logs]
lr_values   = [r["lr"]          for r in step_logs]

epoch_steps = [r["global_step"] for r in epoch_logs]
epoch_train = [r["train_loss"]  for r in epoch_logs]
epoch_eval  = [r["eval_loss"]   for r in epoch_logs if r.get("eval_loss") is not None]
epoch_eval_steps = [r["global_step"] for r in epoch_logs if r.get("eval_loss") is not None]

print(f"日志文件：{log_path}")
print(f"共 {len(step_logs)} 条 step 日志，{len(epoch_logs)} 条 epoch 日志")
if step_logs:
    print(f"训练步数：{steps[0]} → {steps[-1]}")
    print(f"Train Loss：{train_loss[0]:.4f} → {train_loss[-1]:.4f}")
if epoch_eval:
    print(f"Eval  Loss：{epoch_eval[0]:.4f} → {epoch_eval[-1]:.4f}")

# ── 绘图 ─────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("TkAgg")  # Windows 本地显示
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print("\n缺少 matplotlib，请先运行：pip install matplotlib")
    sys.exit(1)

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig = plt.figure(figsize=(13, 8))
fig.suptitle("SFT 训练曲线", fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── 图1：Train Loss（逐步） ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
if steps:
    ax1.plot(steps, train_loss, color="#4C72B0", linewidth=1.2, alpha=0.7, label="Train Loss（逐步）")
    # 滑动平均（窗口=10）
    if len(train_loss) >= 10:
        window = 10
        smoothed = [
            sum(train_loss[max(0, i-window):i+1]) / min(i+1, window)
            for i in range(len(train_loss))
        ]
        ax1.plot(steps, smoothed, color="#C44E52", linewidth=2.0, label=f"平滑（窗口={window}）")
if epoch_steps:
    ax1.scatter(epoch_steps, epoch_train, color="#DD8452", s=60, zorder=5, label="Epoch 结束点")
ax1.set_xlabel("Global Step")
ax1.set_ylabel("Loss")
ax1.set_title("Train Loss 曲线")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── 图2：Train vs Eval Loss（epoch 级别） ────────────────────
ax2 = fig.add_subplot(gs[1, 0])
if epoch_logs:
    epochs_x = [r["epoch"] for r in epoch_logs]
    ax2.plot(epochs_x, epoch_train, "o-", color="#4C72B0", linewidth=1.8, markersize=6, label="Train Loss")
if epoch_eval:
    epochs_eval_x = [r["epoch"] for r in epoch_logs if r.get("eval_loss") is not None]
    ax2.plot(epochs_eval_x, epoch_eval, "s--", color="#C44E52", linewidth=1.8, markersize=6, label="Eval Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.set_title("Train vs Eval Loss（每 Epoch）")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
if not epoch_logs:
    ax2.text(0.5, 0.5, "暂无 Epoch 数据", ha="center", va="center", transform=ax2.transAxes, color="gray")

# ── 图3：学习率曲线 ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
if steps:
    ax3.plot(steps, lr_values, color="#55A868", linewidth=1.8)
    ax3.set_xlabel("Global Step")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("学习率调度曲线")
    ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, "暂无学习率数据", ha="center", va="center", transform=ax3.transAxes, color="gray")

# ── 保存 + 显示 ──────────────────────────────────────────────
save_path = os.path.join(os.path.dirname(log_path), "training_curves.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\n图片已保存：{save_path}")
plt.show()
