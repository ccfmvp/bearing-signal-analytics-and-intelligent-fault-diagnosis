# %%
import scipy.io
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import re

# %%
expanded_df = pd.read_excel('cycle.xlsx', index_col=None)

# %%
def extract_sensor_position(row):
    """
    从Level_2列中提取传感器位置
    格式如：12kHz_DE_data, 48kHz_Normal_data等
    """
    if pd.notna(row['Level_2']):
        level_2_str = str(row['Level_2'])
        # 使用正则表达式提取传感器位置
        match = re.search(r'(\d+kHz_)([^_]+)(_data)', level_2_str)
        if match:
            sensor_pos = match.group(2)
            # 如果是Normal，返回'N'
            if sensor_pos == 'Normal':
                return 'N'
            return sensor_pos
    return None

# %%
all_dataframes = []

for index, row in tqdm(expanded_df.iterrows(), total=expanded_df.shape[0]):
    # 读取特征文件
    all_features = pd.read_csv(f'featureExtraction/特征提取_{index + 1:03d}.csv', index_col=None)

    # 动态获取特征数量并生成列名
    num_features = all_features.shape[1]
    all_features.columns = [f'特征{i + 1}' for i in range(num_features)]

    # 添加故障类型
    all_features['故障类型'] = row['故障类型']

    # 从Level_2提取传感器位置
    sensor_position = extract_sensor_position(row)
    all_features['轴承类别'] = sensor_position

    all_dataframes.append(all_features)

# 合并所有DataFrame
if all_dataframes:
    final_dataframe = pd.concat(all_dataframes, ignore_index=True)
    print(f"最终DataFrame形状: {final_dataframe.shape}")

# %%
# 创建标签列
def create_label(row):
    """
    创建标签，格式为：故障类型--轴承类别
    """
    return f"{row['故障类型']}"

final_dataframe['标签'] = final_dataframe.apply(create_label, axis=1)

# %%
# 统计生成的标签项
label_counts = final_dataframe['标签'].value_counts()
print("\n生成的标签项统计:")
for label, count in label_counts.items():
    print(f"{label}: {count}个样本")

# 显示所有唯一的标签
unique_labels = final_dataframe['标签'].unique()
print(f"\n总共生成 {len(unique_labels)} 种不同的标签:")
for i, label in enumerate(sorted(unique_labels), 1):
    print(f"{i}. {label}")

# %%
# 保存结果
final_dataframe.to_csv('labeledData4.csv', index=None)
print(f"\n数据已保存到 labeledData4.csv")