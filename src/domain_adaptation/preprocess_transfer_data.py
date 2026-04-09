import pandas as pd


def extract_top_30_features(file_name, output_file):
    """提取前30个重要特征到新文件"""
    # 读取特征重要性文件
    df_importance = pd.read_csv("all_features_importance.csv")

    # 获取前30个最重要的特征
    top_30_features = df_importance.nlargest(30, 'importance')['feature'].tolist()
    print(f"前30个最重要的特征: {top_30_features}")

    # 读取原始数据文件（请将文件名替换为你的实际文件名）
    df_data = pd.read_csv(file_name)  # 替换为你的CSV文件名

    # 确保保留故障类型、轴承类别和标签列
    additional_columns = ['标签', '目标代码']
    columns_to_keep = top_30_features + [col for col in additional_columns if col in df_data.columns]

    # 提取指定列
    df_selected = df_data[columns_to_keep]

    # 保存到新文件
    df_selected.to_csv(output_file, index=False)
    print(f"已提取 {len(columns_to_keep)} 列到 {output_file}")

    return df_selected


# 执行函数
if __name__ == "__main__":
    extract_top_30_features("original_labeledData.csv", "solving_original_data.csv")
    extract_top_30_features("target_labeledData.csv", "solving_target_data.csv")