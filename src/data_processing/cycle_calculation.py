# %% 可视化部分 - 添加多种图表
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import scipy.io
import numpy as np
from tqdm import tqdm
import os
import pandas as pd


def traverse_directory_to_dataframe(root_path):
    all_data = []

    for root, dirs, files in os.walk(root_path):
        path_parts = os.path.normpath(root).split(os.sep)

        for file in files:
            row = path_parts + [file]
            all_data.append(row)

    max_depth = max(len(row) for row in all_data) if all_data else 0

    columns = [f'Level_{i}' for i in range(max_depth)]

    df = pd.DataFrame(all_data, columns=columns)

    return df


directory_path1 = r"../源域数据集/12kHz_DE_data"
df_directory1 = traverse_directory_to_dataframe(directory_path1)
directory_path2 = r"../源域数据集/12kHz_FE_data"
df_directory2 = traverse_directory_to_dataframe(directory_path2)
directory_path3 = r"../源域数据集/48kHz_DE_data"
df_directory3 = traverse_directory_to_dataframe(directory_path3)
directory_path4 = r"../源域数据集/48kHz_Normal_data"
df_directory4 = traverse_directory_to_dataframe(directory_path4)

df_directorys = pd.concat([df_directory1, df_directory2, df_directory3, df_directory4]).replace({np.nan: None})

# %%
df_directorys

# %%
# 创建新列用于存储扩展后的信息
expanded_rows = []

# 遍历DataFrame的每一行
for index, row in tqdm(df_directorys.iterrows(), total=df_directorys.shape[0]):
    # 构建文件路径
    file_path = os.path.join(*[comp for comp in
                               [row['Level_0'], row['Level_1'], row['Level_2'], row['Level_3'], row['Level_4'],
                                row['Level_5'], row['Level_6']] if comp is not None])

    try:
        # 读取.mat文件[1,2](@ref)
        mat_data = scipy.io.loadmat(file_path)

        # 获取所有变量名
        variable_names = list(mat_data.keys())

        # 过滤出以特定后缀结尾的变量
        de_vars = [var for var in variable_names if var.endswith('_DE_time')]
        fe_vars = [var for var in variable_names if var.endswith('_FE_time')]
        ba_vars = [var for var in variable_names if var.endswith('_BA_time')]

        # 查找RPM值
        rpm_value = None
        for var in variable_names:
            if var.endswith('RPM') or var.endswith('_RPM'):
                rpm_value = mat_data[var][0, 0] if isinstance(mat_data[var], np.ndarray) else mat_data[var]
                break

        # 处理找到的变量
        found_vars = []
        if de_vars:
            found_vars.extend([(var, 'DE') for var in de_vars])
        if fe_vars:
            found_vars.extend([(var, 'FE') for var in fe_vars])
        if ba_vars:
            found_vars.extend([(var, 'BA') for var in ba_vars])

        # 如果没有找到任何相关变量，至少保留原始行
        if not found_vars:
            new_row = row.to_dict()
            new_row['传感器位置'] = None
            new_row['RPM'] = rpm_value
            expanded_rows.append(new_row)
        else:
            # 为每个找到的变量创建新行
            for var_name, 传感器位置 in found_vars:
                new_row = row.to_dict()
                new_row['传感器位置'] = 传感器位置
                new_row['RPM'] = rpm_value
                new_row['Var_Name'] = var_name  # 可选：保存变量名
                expanded_rows.append(new_row)

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        # 即使出错也保留原始行
        new_row = row.to_dict()
        new_row['传感器位置'] = 'Error'
        new_row['RPM'] = None
        expanded_rows.append(new_row)

# 创建新的DataFrame
expanded_df = pd.DataFrame(expanded_rows)

# 重新排列列的顺序，将新列放在前面
cols = ['传感器位置', 'RPM'] + [col for col in expanded_df.columns if col not in ['传感器位置', 'RPM', 'Var_Name']]
if 'Var_Name' in expanded_df.columns:
    cols.append('Var_Name')
expanded_df = expanded_df[cols]

import re

# 遍历DataFrame的每一行，根据Level_3的文件名设置RPM值
for index, row in expanded_df.iterrows():
    level_3_value = row['Level_3']

    # 检查Level_3是否包含.rpm信息
    if isinstance(level_3_value, str) and '.mat' in level_3_value:
        # 使用正则表达式提取rpm值
        rpm_match = re.search(r'\((\d+)rpm\)', level_3_value)
        if rpm_match:
            rpm_value = int(rpm_match.group(1))
            expanded_df.at[index, 'RPM'] = rpm_value

# 遍历DataFrame的每一行，根据Level_3的文件名设置RPM值
for index, row in expanded_df.iterrows():
    level_3_value = row['Level_5']

    # 检查Level_5是否包含.rpm信息
    if isinstance(level_3_value, str) and '.mat' in level_3_value:
        # 使用正则表达式提取rpm值
        rpm_match = re.search(r'\((\d+)rpm\)', level_3_value)
        if rpm_match:
            rpm_value = int(rpm_match.group(1))
            expanded_df.at[index, 'RPM'] = rpm_value


# %%
def determine_fault_type(row):
    # 检查所有列中是否包含特定的故障类型标识
    row_values = row.astype(str).values

    # 检查是否包含OR
    if any('OR' in str(val) for val in row_values):
        return 'OR'
    # 检查是否包含IR
    elif any('IR' in str(val) for val in row_values):
        return 'IR'
    # 检查是否包含B但不包含OR或IR
    elif any('B' in str(val) and 'OR' not in str(val) and 'IR' not in str(val) for val in row_values):
        return 'B'
    # 如果都不是，则为正常(N)
    else:
        return 'N'


# 应用函数创建新的故障类型列
expanded_df['故障类型'] = expanded_df.apply(determine_fault_type, axis=1)

# %%
expanded_df['Nd'] = np.where(expanded_df['Level_2'] != '48kHz_Normal_data', 9, None)

expanded_df.loc[expanded_df['Level_2'].isin(['12kHz_DE_data', '48kHz_DE_data']), 'd'] = 0.3126 * 0.0254
expanded_df.loc[expanded_df['Level_2'].isin(['12kHz_DE_data', '48kHz_DE_data']), 'D'] = 1.537 * 0.0254

expanded_df.loc[expanded_df['Level_2'] == '12kHz_FE_data', 'd'] = 0.2656 * 0.0254
expanded_df.loc[expanded_df['Level_2'] == '12kHz_FE_data', 'D'] = 1.122 * 0.0254


# %%
def calculate_bearing_fault_frequency(fault_type, n, d, D, Nd):
    """
    计算轴承故障特征频率

    参数:
    fault_type (str): 故障类型，可选 'BPFO'（外圈）、'BPFI'（内圈）或 'BSF'（滚动体）
    n (float): 轴承内圈转速，单位 rpm
    d (float): 滚动体直径，单位 mm
    D (float): 轴承节径，单位 mm
    Nd (int): 滚动体个数

    返回:
    float: 特征频率，单位 Hz
    """
    # 计算轴承转频 (Hz)
    fr = n / 60.0

    if fault_type == 'BPFO':
        # 外圈故障特征频率
        return fr * (Nd / 2.0) * (1 - d / D)
    elif fault_type == 'BPFI':
        # 内圈故障特征频率
        return fr * (Nd / 2.0) * (1 + d / D)
    elif fault_type == 'BSF':
        # 滚动体故障特征频率
        return fr * (D / d) * (1 - (d / D) ** 2)
    else:
        raise ValueError("不支持的故障类型。请选择 'BPFO', 'BPFI' 或 'BSF'")


# %%
expanded_df['BPFO'] = np.nan
expanded_df['BPFI'] = np.nan
expanded_df['BSF'] = np.nan

for idx, row in expanded_df.iterrows():
    try:
        n = row['RPM']
        d = row['d']
        D = row['D']
        Nd = row['Nd']

        if pd.isna(n) or pd.isna(d) or pd.isna(D) or pd.isna(Nd):
            continue

        # 计算各特征频率
        expanded_df.at[idx, 'BPFO'] = calculate_bearing_fault_frequency('BPFO', n, d, D, Nd)
        expanded_df.at[idx, 'BPFI'] = calculate_bearing_fault_frequency('BPFI', n, d, D, Nd)
        expanded_df.at[idx, 'BSF'] = calculate_bearing_fault_frequency('BSF', n, d, D, Nd)

    except (ValueError, TypeError):
        continue

# %%
expanded_df['T_BPFO'] = 1 / expanded_df['BPFO']
expanded_df['T_BPFI'] = 1 / expanded_df['BPFI']
expanded_df['T_BSF'] = 1 / expanded_df['BSF']

# %%
expanded_df

# %%
expanded_df['N_BPFO'] = np.nan

expanded_df.loc[
    expanded_df['Level_2'].isin(['12kHz_DE_data', '12kHz_FE_data']),
    'N_BPFO'
] = expanded_df['T_BPFO'] * 12 * 1000

expanded_df.loc[
    expanded_df['Level_2'] == '48kHz_DE_data',
    'N_BPFO'
] = expanded_df['T_BPFO'] * 48 * 1000

# %%
expanded_df['N_BPFI'] = np.nan

expanded_df.loc[
    expanded_df['Level_2'].isin(['12kHz_DE_data', '12kHz_FE_data']),
    'N_BPFI'
] = expanded_df['T_BPFI'] * 12 * 1000

expanded_df.loc[
    expanded_df['Level_2'] == '48kHz_DE_data',
    'N_BPFI'
] = expanded_df['T_BPFI'] * 48 * 1000

# %%
expanded_df['N_BSF'] = np.nan

expanded_df.loc[
    expanded_df['Level_2'].isin(['12kHz_DE_data', '12kHz_FE_data']),
    'N_BSF'
] = expanded_df['T_BSF'] * 12 * 1000

expanded_df.loc[
    expanded_df['Level_2'] == '48kHz_DE_data',
    'N_BSF'
] = expanded_df['T_BSF'] * 48 * 1000

# %%
expanded_df['ID'] = [f"{i + 1:03d}" for i in range(len(expanded_df))]

# %%
expanded_df

# %%
expanded_df.to_excel('cycle.xlsx', index=None)
