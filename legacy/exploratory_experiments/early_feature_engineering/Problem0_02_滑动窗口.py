# %%
import scipy.io
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

# %%
expanded_df = pd.read_excel('周期计算.xlsx', index_col=None)

# %%
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

# %%
# for index, row in tqdm(expanded_df.iterrows(), total=expanded_df.shape[0]):
#     file_path = os.path.join(*[comp for comp in [row['Level_0'], row['Level_1'], row['Level_2'], row['Level_3'], row['Level_4'], row['Level_5'], row['Level_6']] if comp is not np.nan])
#     mat_data = scipy.io.loadmat(file_path)[row['Var_Name']]
#     if row['Level_2'].startswith('48'):
#         N = 800
#     else:
#         N = 200
#     sampled_data = sliding_window_sampling(mat_data, N, stride=int(N/2))
#     sampled_data = sampled_data[:, :, 0]
#     df = pd.DataFrame(sampled_data)
#     df.to_csv(f'源域数据集滑动窗口采样结果/{index+1:03d}.csv', index=None)

# %%
for index, row in tqdm(expanded_df.iterrows(), total=expanded_df.shape[0]):
    file_path = os.path.join(*[comp for comp in [row['Level_0'], row['Level_1'], row['Level_2'], row['Level_3'], row['Level_4'], row['Level_5'], row['Level_6']] if comp is not np.nan])
    mat_data = scipy.io.loadmat(file_path)[row['Var_Name']]
    if row['Level_2'] == '48kHz_Normal_data':
        N = 800
        sampled_data = sliding_window_sampling(mat_data, N, stride=int(N/2/10))
        sampled_data = sampled_data[:, :, 0]
        df = pd.DataFrame(sampled_data)
        df.to_csv(f'源域数据集滑动窗口采样结果/{index+1:03d}.csv', index=None)


