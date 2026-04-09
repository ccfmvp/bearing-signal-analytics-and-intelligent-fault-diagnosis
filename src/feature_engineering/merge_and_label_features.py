import scipy.io
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

# 读取主数据文件
expanded_df = pd.read_excel('cycle.xlsx', index_col=None)


# 动态确定特征数量
def get_feature_count():
    """动态获取特征文件中的列数"""
    # 尝试找到第一个存在的特征文件
    for index in range(expanded_df.shape[0]):
        feature_file = f'featureExtraction/特征提取_{index + 1:03d}.csv'
        if os.path.exists(feature_file):
            try:
                # 读取文件但不加载全部数据，只获取列数
                all_features = pd.read_csv(feature_file, nrows=0)  # 只读取列名
                return len(all_features.columns)
            except Exception as e:
                print(f"读取文件 {feature_file} 列数时出错: {e}")
                continue


# 动态获取特征数量
feature_count = get_feature_count()
print(f"检测到特征数量: {feature_count}")

# 准备结果数据框
result_df = expanded_df.copy()

# 动态创建特征列
feature_columns = [f'特征{i + 1}' for i in range(feature_count)]
for col in feature_columns:
    result_df[col] = np.nan  # 初始化为空值

# 处理每个特征文件
for index, row in tqdm(result_df.iterrows(), total=result_df.shape[0], desc="处理特征文件"):
    try:
        # 构建特征文件名
        feature_file = f'featureExtraction/特征提取_{index + 1:03d}.csv'

        if os.path.exists(feature_file):
            # 读取特征文件
            all_features = pd.read_csv(feature_file, index_col=None)

            # 检查列数是否匹配
            if len(all_features.columns) != feature_count:
                print(f"警告: 文件 {feature_file} 的特征数({len(all_features.columns)})与预期({feature_count})不匹配")
                # 可以选择跳过或特殊处理
                continue

            # 计算每列的均值
            all_features_mean = all_features.mean()

            # 动态填充特征列（处理列数可能变化的情况）
            for i, (col_name, mean_value) in enumerate(all_features_mean.items()):
                if i < len(feature_columns):
                    result_df.loc[index, feature_columns[i]] = mean_value
                else:
                    # 如果特征文件列数多于预期，动态创建新列
                    new_col_name = f'特征{i + 1}'
                    if new_col_name not in result_df.columns:
                        result_df[new_col_name] = np.nan
                    result_df.loc[index, new_col_name] = mean_value
        else:
            print(f"警告: 文件 {feature_file} 不存在")
    except Exception as e:
        print(f"处理第 {index + 1} 行时出错: {e}")

# 保存结果
result_df.to_excel('addFeature.xlsx', index=None)
print(
    f"处理完成！共处理 {result_df.shape[0]} 行数据，包含 {len([col for col in result_df.columns if col.startswith('特征')])} 个特征")

# 可选：显示结果统计信息
print("\n结果统计:")
print(f"总列数: {result_df.shape[1]}")
print(f"特征列: {[col for col in result_df.columns if col.startswith('特征')]}")
print(f"缺失值统计:")
missing_stats = result_df.isnull().sum()
print(missing_stats[missing_stats > 0])