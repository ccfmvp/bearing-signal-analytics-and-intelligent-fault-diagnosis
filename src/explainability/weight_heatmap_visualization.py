# src/explainability/weight_heatmap_visualization.py
"""权重热力图可视化模块"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_weight_heatmap(weight_tensor, layer_name="", output_path=None):
    """绘制模型权重热力图

    Args:
        weight_tensor: 权重张量 [out_channels, in_channels, kernel_size]
        layer_name: 层名称
        output_path: 保存路径
    """
    if isinstance(weight_tensor, torch.Tensor):
        weight_tensor = weight_tensor.detach().cpu().numpy()

    # 如果是3D卷积核，取第一个输出通道的二维切片
    if weight_tensor.ndim == 3:
        weight_2d = weight_tensor[0]  # [in_channels, kernel_size]
    elif weight_tensor.ndim == 2:
        weight_2d = weight_tensor
    else:
        print(f"Warning: Unsupported weight dimension {weight_tensor.ndim}")
        return

    plt.figure(figsize=(12, 8))
    sns.heatmap(weight_2d, cmap='RdBu_r', center=0, aspect='auto')
    plt.title(f'Weight Heatmap - {layer_name}')
    plt.xlabel('Kernel Size')
    plt.ylabel('Input Channels')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
