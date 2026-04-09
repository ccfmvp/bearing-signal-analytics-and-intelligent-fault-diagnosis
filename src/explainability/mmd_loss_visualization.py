# src/explainability/mmd_loss_visualization.py
"""MMD损失可视化模块"""

import matplotlib.pyplot as plt
import numpy as np


def plot_mmd_loss_curve(mmd_history, output_path=None):
    """绘制MMD损失变化曲线

    Args:
        mmd_history: MMD损失历史记录列表
        output_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(mmd_history, linewidth=2, color='#2196F3')
    plt.title('MMD Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('MMD Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_mmd_comparison(results_dict, output_path=None):
    """比较不同模型的MMD损失

    Args:
        results_dict: {model_name: mmd_value}
        output_path: 保存路径
    """
    models = list(results_dict.keys())
    values = list(results_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    plt.title('MMD Loss Comparison Across Models')
    plt.ylabel('MMD Loss')
    plt.xticks(rotation=15)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
