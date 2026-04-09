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
    all_features_mean = all_features.mean()

# %%
result_df = expanded_df.copy()

# 为每个特征创建列名
feature_columns = [f'特征{i+1}' for i in range(53)]
for col in feature_columns:
    result_df[col] = np.nan

# 遍历每一行，读取对应的特征文件并计算均值
for index, row in tqdm(result_df.iterrows(), total=result_df.shape[0]):
    try:
        # 读取对应的特征文件
        feature_file = f'特征提取/特征提取_{index+1:03d}.csv'
        if os.path.exists(feature_file):
            all_features = pd.read_csv(feature_file, index_col=None)
            # 计算每列的均值
            all_features_mean = all_features.mean()
            
            # 将均值填入对应的列
            for i, (col_name, mean_value) in enumerate(all_features_mean.items()):
                if i < len(feature_columns):
                    result_df.loc[index, feature_columns[i]] = mean_value
        else:
            print(f"警告: 文件 {feature_file} 不存在")
    except Exception as e:
        print(f"处理第 {index+1} 行时出错: {e}")

# %%
result_df.to_excel('加入特征之后.xlsx', index=None)


