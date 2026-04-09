import scipy.io
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

for ID in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']:
    all_features = pd.read_csv(f'目标域特征提取/目标域特征提取_{ID}.csv', index_col=None)
    all_features.columns = [f'特征{i+1}' for i in range(53)]
    all_features['目标代码'] = ID

all_dataframes = []

for ID in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']:
    all_features = pd.read_csv(f'目标域特征提取/目标域特征提取_{ID}.csv', index_col=None)
    all_features.columns = [f'特征{i+1}' for i in range(53)]
    all_features['目标代码'] = ID

    all_dataframes.append(all_features)

if all_dataframes:
    final_dataframe = pd.concat(all_dataframes, ignore_index=True)
    print(f"最终DataFrame形状: {final_dataframe.shape}")

final_dataframe.to_csv('问题3_目标域特征提取.csv', index=None)