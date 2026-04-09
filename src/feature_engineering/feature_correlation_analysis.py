import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取映射文件
mapping_df = pd.read_csv('mappingFeature.csv')
# 创建特征索引到英文名称的映射字典
feature_mapping = dict(zip(mapping_df['索引'], mapping_df['特征名称']))

# 读取数据文件
data_df = pd.read_csv('normalized_data.csv')  # 替换为您的文件名

# 提取特征列（排除最后的非特征列：故障类型,轴承类别,尺寸,标签）
feature_columns = [col for col in data_df.columns if col.startswith('特征')]
features_data = data_df[feature_columns]

# 将特征名称映射为英文
def map_feature_name(feature_chinese):
    # 从"特征52"中提取数字52
    feature_num = int(feature_chinese.replace('特征', ''))
    # 使用模运算将特征编号映射到0-14的范围
    mapped_index = feature_num % 15  # 因为mapping文件有15个特征(0-14)
    return feature_mapping.get(mapped_index, feature_chinese)

# 重命名列
new_column_names = [map_feature_name(col) for col in feature_columns]
features_data.columns = new_column_names

# 计算相关性矩阵
print("计算特征相关性矩阵...")
corr_matrix = features_data.corr()

# 创建热力图
plt.figure(figsize=(16, 12))

# 使用马卡龙色系配色
cmap = sns.diverging_palette(250, 15, s=75, l=50, center="light", as_cmap=True)

# 只显示下三角（避免重复）
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8},
            annot=True, fmt=".2f", annot_kws={"size": 8})

plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
plt.tight_layout()

# 确保可视化文件夹存在
visualization_folder = 'graph'
os.makedirs(visualization_folder, exist_ok=True)

# 保存图像
plt.savefig(os.path.join(visualization_folder, 'feature_correlation_heatmap.png'),
            dpi=300, bbox_inches='tight')
plt.close()

print("特征相关性热力图已保存到 feature_correlation_heatmap.png")

# 显示相关性矩阵的前几行
print("\n相关性矩阵前5x5:")
print(corr_matrix.iloc[:5, :5].round(3))