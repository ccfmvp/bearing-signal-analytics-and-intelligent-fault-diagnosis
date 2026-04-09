import scipy.io
import numpy as np
from tqdm import tqdm
import pandas as pd
import os


def sliding_window_sampling(data, window_size, stride=1):
    """
    对时间序列数据进行滑动窗口采样

    参数:
        data: 输入数据，形状为(n_samples, n_features)
        window_size: 窗口大小
        stride: 滑动步长，默认为1

    返回:
        采样后的数据，形状为(n_windows, window_size, n_features)
    """
    n_samples = data.shape[0]
    n_features = data.shape[1] if len(data.shape) > 1 else 1

    # 确保数据是二维的
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    # 计算窗口数量
    n_windows = (n_samples - window_size) // stride + 1

    # 初始化结果数组
    sampled_data = np.zeros((n_windows, window_size, n_features))

    # 滑动窗口采样
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        sampled_data[i] = data[start:end]

    return sampled_data

for ID in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']:
    mat_data = scipy.io.loadmat(f'../目标域数据集/{ID}.mat')[ID]
    N = 800
    sampled_data = sliding_window_sampling(mat_data, N, stride=int(N/2))
    sampled_data = sampled_data[:, :, 0]
    df = pd.DataFrame(sampled_data)
    df.to_csv(f'目标域数据集滑动窗口采样结果/{ID}.csv', index=None)