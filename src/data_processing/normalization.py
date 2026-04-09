import pandas as pd
import numpy as np
import os


def normalize_csv_file(input_file, output_file):
    """
    简洁的CSV文件归一化处理

    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    """
    # 读取数据
    df = pd.read_csv(input_file)
    print(f"成功读取数据: {df.shape[0]} 行, {df.shape[1]} 列")

    # 识别特征列（假设最后4列是非特征列）
    feature_columns = df.columns[:-4]
    print(f"处理 {len(feature_columns)} 个特征列")

    # 创建归一化后的数据副本
    df_normalized = df.copy()
    normalization_params = {}

    # 对每个特征列进行Z-score归一化
    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            mean_val = df[column].mean()
            std_val = df[column].std()

            # 处理标准差为0的情况
            if std_val < 1e-8:
                std_val = 1.0

            # 应用归一化
            df_normalized[column] = (df[column] - mean_val) / std_val

            # 保存参数
            normalization_params[column] = {'mean': mean_val, 'std': std_val}

            print(f"✓ {column}: 归一化完成")

    # 保存归一化后的数据
    df_normalized.to_csv(output_file, index=False)
    print(f"归一化数据已保存至: {output_file}")

    # 保存归一化参数
    params_file = output_file.replace('.csv', '_params.csv')
    params_df = pd.DataFrame(normalization_params).T
    params_df.to_csv(params_file)
    print(f"归一化参数已保存至: {params_file}")

    return df_normalized


# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径
    input_csv = "processed_dataset.csv"  # 输入文件
    output_csv = "normalized_data.csv"  # 输出文件

    # 执行归一化
    normalized_data = normalize_csv_file(input_csv, output_csv)

    print("\n处理完成!")