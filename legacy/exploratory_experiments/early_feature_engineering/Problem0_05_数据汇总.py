# %%
import scipy.io
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

# %%
expanded_df = pd.read_excel('周期计算.xlsx', index_col=None)

# %%
for index, row in tqdm(expanded_df.iterrows(), total=expanded_df.shape[0]):
    all_features = pd.read_csv(f'特征提取/特征提取_{index+1:03d}.csv', index_col=None)
    all_features.columns = [f'特征{i+1}' for i in range(53)]
    all_features['故障类型'] = row['故障类型']

# %%
all_dataframes = []

for index, row in tqdm(expanded_df.iterrows(), total=expanded_df.shape[0]):
    all_features = pd.read_csv(f'特征提取/特征提取_{index+1:03d}.csv', index_col=None)
    all_features.columns = [f'特征{i+1}' for i in range(53)]
    all_features['故障类型'] = row['故障类型']

    level_2 = row['Level_2']
    if '12kHz_DE_data' in level_2:
        all_features['轴承类别'] = 'DE'
    elif '12kHz_FE_data' in level_2:
        all_features['轴承类别'] = 'FE'
    elif '48kHz_DE_data' in level_2:
        all_features['轴承类别'] = 'DE'
    elif '48kHz_Normal_data' in level_2:
        all_features['轴承类别'] = 'N'

    all_dataframes.append(all_features)

if all_dataframes:
    final_dataframe = pd.concat(all_dataframes, ignore_index=True)
    print(f"最终DataFrame形状: {final_dataframe.shape}")

# %%
final_dataframe['标签'] = final_dataframe['故障类型'] + '--' + final_dataframe['轴承类别']

# %%
final_dataframe.to_csv('问题1_特征提取及标签.csv', index=None)


