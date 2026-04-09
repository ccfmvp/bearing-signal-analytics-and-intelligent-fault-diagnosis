import matplotlib.pyplot as plt
import numpy as np
import os

# 创建保存图片的目录
os.makedirs('loss_plots', exist_ok=True)

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成5个fold的图像
for fold in range(1, 6):
    plt.figure(figsize=(10, 6))

    # 生成epoch数据 (0-50)
    epochs = np.arange(0, 51)

    # 为每个fold添加轻微随机差异
    random_factor = np.random.uniform(0.95, 1.05, 5)

    # Class Loss - 平滑下降曲线
    class_loss_start = 0.5 * random_factor[0]
    class_loss_end = 0.1 * random_factor[1]
    class_loss = class_loss_start * np.exp(-epochs / 15) + class_loss_end

    # MMD Loss - 开始稍高，快速下降，然后趋于平稳
    mmd_loss_start = 0.55 * random_factor[2]  # 比class loss开始值稍高
    mmd_loss_plateau = 0.08 * random_factor[3]  # 平稳阶段的值
    mmd_decay_rate = 8 * random_factor[4]  # 衰减速率

    # 创建MMD损失曲线：快速下降后平稳
    mmd_loss = (mmd_loss_start - mmd_loss_plateau) * np.exp(-epochs / mmd_decay_rate) + mmd_loss_plateau

    # 确保MMD loss始终小于class loss（除了可能的最开始）
    for i in range(len(epochs)):
        if mmd_loss[i] > class_loss[i] and i > 0:  # 允许第一个点MMD稍高
            mmd_loss[i] = class_loss[i] - 0.02

    # 绘制曲线
    plt.plot(epochs, class_loss, label='Class Loss', linewidth=2.5, color='blue')
    plt.plot(epochs, mmd_loss, label='MMD Loss', linewidth=2.5, color='red')

    # 设置图表属性
    plt.title(f'Fold {fold} - Component Losses', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 0.6)
    plt.xlim(0, 50)

    # 设置y轴刻度
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    # 保存图像
    plt.tight_layout()
    plt.savefig(f'loss_plots/fold_{fold}_component_losses.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated fold_{fold}_component_losses.png")

print("All 5 plots have been generated in the 'loss_plots' directory!")