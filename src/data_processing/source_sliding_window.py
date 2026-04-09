# %%
import scipy.io
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

# %%
expanded_df = pd.read_excel('cycle.xlsx', index_col=None)


# %%
def sliding_window_sampling(data, window_size, stride=1):
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


for index, row in tqdm(expanded_df.iterrows(), total=expanded_df.shape[0]):
    file_path = os.path.join(*[comp for comp in
                               [row['Level_0'], row['Level_1'], row['Level_2'], row['Level_3'], row['Level_4'],
                                row['Level_5'], row['Level_6']] if comp is not np.nan])
    mat_data = scipy.io.loadmat(file_path)[row['Var_Name']]

    # 确定窗口大小
    if row['Level_2'].startswith('48'):
        N = 800
    else:
        N = 200

    # 确定步长（重叠率）
    if row['Level_2'] == '48kHz_Normal_data':
        # 正常数据使用小步长（高重叠率）
        stride = int(N / 20)  # 95%重叠
    else:
        # 故障数据使用大步长（低重叠率）
        stride = int(N / 2)  # 50%重叠

    # 执行滑动窗口采样
    sampled_data = sliding_window_sampling(mat_data, N, stride=stride)
    sampled_data = sampled_data[:, :, 0]

    # 保存结果
    df = pd.DataFrame(sampled_data)
    df.to_csv(f'original_sliding_window_result/{index + 1:03d}.csv', index=None)

