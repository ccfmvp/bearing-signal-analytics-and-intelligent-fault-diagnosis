import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib
import seaborn as sns
from scipy.stats import zscore

matplotlib.use('Agg')  # 使用非交互式后端，提高速度

# 设置马卡龙色系配色方案
MACARON_COLORS = [
    '#FFB6C1', '#FFD700', '#98FB98', '#87CEFA', '#DDA0DD',
    '#FFA07A', '#20B2AA', '#F0E68C', '#E6E6FA', '#B0E0E6',
    '#FFC0CB', '#FFE4E1', '#F5DEB3', '#F0FFF0', '#F5F5DC',
    '#E0FFFF', '#F0F8FF', '#FFF0F5', '#FFEFD5', '#FFE4B5'
]

# 设置冷色调马卡龙色系（更适合数据可视化）
COLD_MACARON_COLORS = [
    '#87CEEB', '#AFEEEE', '#B0E0E6', '#ADD8E6', '#87CEFA',
    '#00BFFF', '#1E90FF', '#6495ED', '#7B68EE', '#6A5ACD',
    '#9370DB', '#BA55D3', '#DA70D6', '#EE82EE', '#DDA0DD',
    '#FFB6C1', '#FFA07A', '#FFD700', '#98FB98', '#90EE90'
]

# 使用冷色调马卡龙色系作为主配色
COLORS = COLD_MACARON_COLORS

# 设置输出目录
visualization_folder = "graph/1_03"
if not os.path.exists(visualization_folder):
    os.makedirs(visualization_folder)

# 定义时域和频域特征名称
time_domain_feature_names = [
    'mean', 'median', 'std', 'variance', 'rms', 'peak', 'peak_to_peak',
    'skewness', 'kurtosis', 'crest_factor', 'form_factor', 'impulse_factor',
    'energy', 'mean_abs', 'abs_energy', 'q25', 'q75', 'iqr',
    'zero_crossing_rate', 'peak_count', 'peak_rate', 'autocorr_max',
    'autocorr_first_zero', 'shannon_entropy', 'envelope_mean', 'envelope_std',
    'diff_mean', 'diff_std', 'fractal_dimension', 'lyapunov_exponent'
]

frequency_domain_feature_names = [
    'spectral_centroid', 'spectral_spread', 'spectral_skewness', 'spectral_kurtosis',
    'spectral_energy', 'spectral_entropy', 'dominant_frequency', 'dominant_magnitude',
    'dominant_frequency_ratio', 'frequency_std', 'frequency_variance',
    'spectral_rolloff', 'spectral_flatness', 'spectrogram_peak_var', 'spectrogram_entropy'
]

# 添加频带能量特征
for i in range(1, 6):
    frequency_domain_feature_names.extend([f'band_{i}_energy', f'band_{i}_energy_ratio'])

# 创建列名列表
column_names = time_domain_feature_names + frequency_domain_feature_names

# 读取所有CSV文件
csv_files = glob.glob("featureExtraction/特征提取_*.csv")
all_dfs = {}

print("正在读取CSV文件...")
for file in tqdm(csv_files):
    sample_id = os.path.basename(file).split('_')[1].split('.')[0]
    df = pd.read_csv(file, header=None, names=column_names)
    df = df.replace([np.inf, -np.inf], np.nan)
    all_dfs[sample_id] = df

# 分类特征
time_features = time_domain_feature_names
freq_features = frequency_domain_feature_names

print(f"时域特征: {len(time_features)} 个")
print(f"频域特征: {len(freq_features)} 个")

# 组织特征数据
print("正在组织特征数据...")
feature_data_dict = {}

for column in tqdm(column_names):
    feature_data_dict[column] = []
    for sample_id, df in all_dfs.items():
        if column in df.columns:
            data = df[column].dropna()
            data = data[np.isfinite(data)]
            if len(data) > 0:
                feature_data_dict[column].append(data)

# 设置图形参数
DPI = 150
LINE_WIDTH = 0.8
ALPHA = 0.7
SCATTER_ALPHA = 0.6
SCATTER_SIZE = 15
SAMPLING_INTERVAL = 200  # 每隔200个点取一个点

# 定义常用颜色（从冷色调马卡龙色系中选择）
LINE_COLOR = '#87CEEB'        # 天蓝 - 用于折线图
HIST_COLOR = '#DDA0DD'        # 淡梅紫 - 用于直方图
VIOLIN_COLOR = '#B0E0E6'      # 粉雾蓝 - 用于小提琴图
SCATTER_COLOR = '#FFB6C1'     # 樱花粉 - 用于散点图
EDGE_COLOR = '#B0E0E6'        # 粉雾蓝 - 用于边缘


def create_subplot_grid(n_features, n_cols=5):
    """创建子图网格"""
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    return fig, axes, n_rows, n_cols


def hide_empty_subplots(axes, n_features, n_rows, n_cols):
    """隐藏多余的空子图"""
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)


def sample_data(data, interval=SAMPLING_INTERVAL):
    """对数据进行采样，每隔interval个点取一个点"""
    if len(data) <= interval:
        return data
    return data[::interval]


def get_color(index, color_list=COLORS):
    """获取循环颜色"""
    return color_list[index % len(color_list)]


# 1. 时域特征折线图大图（优化版 - 使用单一颜色）
print("生成时域特征折线图大图...")
fig, axes, n_rows, n_cols = create_subplot_grid(len(time_features))

for i, feature in enumerate(time_features):
    row, col = i // n_cols, i % n_cols
    feature_data = feature_data_dict.get(feature, [])

    if not feature_data:
        axes[row, col].set_visible(False)
        continue

    for j, data in enumerate(feature_data):
        sampled_data = sample_data(data.values)
        axes[row, col].plot(sampled_data,
                            color=LINE_COLOR,
                            alpha=ALPHA, linewidth=LINE_WIDTH)

    axes[row, col].set_xlabel('Window Index', fontsize=8)
    axes[row, col].set_ylabel('Value', fontsize=8)
    axes[row, col].tick_params(axis='both', which='major', labelsize=6)

hide_empty_subplots(axes, len(time_features), n_rows, n_cols)
plt.tight_layout()
plt.savefig(os.path.join(visualization_folder, 'time_domain_line_plots.png'),
            dpi=DPI, bbox_inches='tight')
plt.close()
print("时域特征折线图大图已保存")

# 2. 时域特征直方图大图（优化版 - 调整数据范围，使用单一颜色）
print("生成时域特征直方图大图...")
fig, axes, n_rows, n_cols = create_subplot_grid(len(time_features))

for i, feature in enumerate(time_features):
    row, col = i // n_cols, i % n_cols
    feature_data = feature_data_dict.get(feature, [])

    if not feature_data:
        axes[row, col].set_visible(False)
        continue

    combined_data = np.concatenate(feature_data)

    # 调整直方图范围，使数据分布更广
    data_range = np.ptp(combined_data)  # 数据范围
    if data_range == 0:  # 避免除零
        data_range = 1

    # 扩展直方图范围，使数据更分散
    hist_range = (
        np.min(combined_data) - 0.2 * data_range,
        np.max(combined_data) + 0.2 * data_range
    )

    # 动态调整bins数量
    bins = min(80, max(30, len(combined_data) // 50))

    axes[row, col].hist(combined_data, bins=bins, range=hist_range,
                        alpha=0.7, color=HIST_COLOR, edgecolor=EDGE_COLOR)
    axes[row, col].set_xlabel('Value', fontsize=8)
    axes[row, col].set_ylabel('Frequency', fontsize=8)
    axes[row, col].tick_params(axis='both', which='major', labelsize=6)

hide_empty_subplots(axes, len(time_features), n_rows, n_cols)
plt.tight_layout()
plt.savefig(os.path.join(visualization_folder, 'time_domain_histogram_plots.png'),
            dpi=DPI, bbox_inches='tight')
plt.close()
print("时域特征直方图大图已保存")

# 3. 时域特征散点图大图（新增）
print("生成时域特征散点图大图...")
fig, axes, n_rows, n_cols = create_subplot_grid(len(time_features))

for i, feature in enumerate(time_features):
    row, col = i // n_cols, i % n_cols
    feature_data = feature_data_dict.get(feature, [])

    if not feature_data:
        axes[row, col].set_visible(False)
        continue

    for j, data in enumerate(feature_data):
        sampled_data = sample_data(data.values)
        x_values = np.arange(len(sampled_data))
        axes[row, col].scatter(x_values, sampled_data,
                              color=SCATTER_COLOR,
                              alpha=SCATTER_ALPHA, s=SCATTER_SIZE)

    axes[row, col].set_xlabel('Window Index', fontsize=8)
    axes[row, col].set_ylabel('Value', fontsize=8)
    axes[row, col].tick_params(axis='both', which='major', labelsize=6)

hide_empty_subplots(axes, len(time_features), n_rows, n_cols)
plt.tight_layout()
plt.savefig(os.path.join(visualization_folder, 'time_domain_scatter_plots.png'),
            dpi=DPI, bbox_inches='tight')
plt.close()
print("时域特征散点图大图已保存")

# 4. 频域特征折线图大图（优化版 - 使用单一颜色）
print("生成频域特征折线图大图...")
fig, axes, n_rows, n_cols = create_subplot_grid(len(freq_features))

for i, feature in enumerate(freq_features):
    row, col = i // n_cols, i % n_cols
    feature_data = feature_data_dict.get(feature, [])

    if not feature_data:
        axes[row, col].set_visible(False)
        continue

    for j, data in enumerate(feature_data):
        sampled_data = sample_data(data.values)
        axes[row, col].plot(sampled_data,
                            color=LINE_COLOR,
                            alpha=ALPHA, linewidth=LINE_WIDTH)

    axes[row, col].set_xlabel('Window Index', fontsize=8)
    axes[row, col].set_ylabel('Value', fontsize=8)
    axes[row, col].tick_params(axis='both', which='major', labelsize=6)

hide_empty_subplots(axes, len(freq_features), n_rows, n_cols)
plt.tight_layout()
plt.savefig(os.path.join(visualization_folder, 'frequency_domain_line_plots.png'),
            dpi=DPI, bbox_inches='tight')
plt.close()
print("频域特征折线图大图已保存")

# 5. 频域特征直方图大图（优化版 - 调整数据范围，使用单一颜色）
print("生成频域特征直方图大图...")
fig, axes, n_rows, n_cols = create_subplot_grid(len(freq_features))

for i, feature in enumerate(freq_features):
    row, col = i // n_cols, i % n_cols
    feature_data = feature_data_dict.get(feature, [])

    if not feature_data:
        axes[row, col].set_visible(False)
        continue

    combined_data = np.concatenate(feature_data)

    # 调整直方图范围，使数据分布更广
    data_range = np.ptp(combined_data)
    if data_range == 0:
        data_range = 1

    hist_range = (
        np.min(combined_data) - 0.2 * data_range,
        np.max(combined_data) + 0.2 * data_range
    )

    bins = min(80, max(30, len(combined_data) // 50))

    axes[row, col].hist(combined_data, bins=bins, range=hist_range,
                        alpha=0.7, color=HIST_COLOR, edgecolor=EDGE_COLOR)
    axes[row, col].set_xlabel('Value', fontsize=8)
    axes[row, col].set_ylabel('Frequency', fontsize=8)
    axes[row, col].tick_params(axis='both', which='major', labelsize=6)

hide_empty_subplots(axes, len(freq_features), n_rows, n_cols)
plt.tight_layout()
plt.savefig(os.path.join(visualization_folder, 'frequency_domain_histogram_plots.png'),
            dpi=DPI, bbox_inches='tight')
plt.close()
print("频域特征直方图大图已保存")

# 6. 频域特征散点图大图（新增）
print("生成频域特征散点图大图...")
fig, axes, n_rows, n_cols = create_subplot_grid(len(freq_features))

for i, feature in enumerate(freq_features):
    row, col = i // n_cols, i % n_cols
    feature_data = feature_data_dict.get(feature, [])

    if not feature_data:
        axes[row, col].set_visible(False)
        continue

    for j, data in enumerate(feature_data):
        sampled_data = sample_data(data.values)
        x_values = np.arange(len(sampled_data))
        axes[row, col].scatter(x_values, sampled_data,
                              color=SCATTER_COLOR,
                              alpha=SCATTER_ALPHA, s=SCATTER_SIZE)

    axes[row, col].set_xlabel('Window Index', fontsize=8)
    axes[row, col].set_ylabel('Value', fontsize=8)
    axes[row, col].tick_params(axis='both', which='major', labelsize=6)

hide_empty_subplots(axes, len(freq_features), n_rows, n_cols)
plt.tight_layout()
plt.savefig(os.path.join(visualization_folder, 'frequency_domain_scatter_plots.png'),
            dpi=DPI, bbox_inches='tight')
plt.close()
print("频域特征散点图大图已保存")

# 7. 时域特征小提琴图大图（优化颜色 - 使用单一颜色）
print("生成时域特征小提琴图大图...")
fig, axes, n_rows, n_cols = create_subplot_grid(len(time_features))

for i, feature in enumerate(time_features):
    row, col = i // n_cols, i % n_cols
    feature_data = feature_data_dict.get(feature, [])

    if not feature_data:
        axes[row, col].set_visible(False)
        continue

    data_to_plot = [data.values for data in feature_data]
    violin_parts = axes[row, col].violinplot(data_to_plot, showmeans=True, showmedians=True)

    # 优化颜色设置 - 使用单一颜色
    for j, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(VIOLIN_COLOR)
        pc.set_alpha(0.7)
        pc.set_edgecolor(EDGE_COLOR)
        pc.set_linewidth(0.8)

    # 设置统计线颜色
    violin_parts['cmeans'].set_color('#FF6B6B')  # 柔和的红色
    violin_parts['cmeans'].set_linewidth(1.5)

    violin_parts['cmedians'].set_color('#4ECDC4')  # 青绿色
    violin_parts['cmedians'].set_linewidth(1.5)

    violin_parts['cbars'].set_color('#45B7D1')  # 天蓝色
    violin_parts['cbars'].set_linewidth(1.2)

    violin_parts['cmins'].set_color('#96CEB4')  # 淡绿色
    violin_parts['cmaxes'].set_color('#96CEB4')

    axes[row, col].set_xlabel('Sample', fontsize=8)
    axes[row, col].set_ylabel('Value', fontsize=8)
    axes[row, col].set_xticks(range(1, len(feature_data) + 1))
    axes[row, col].set_xticklabels([f'S{i + 1}' for i in range(len(feature_data))], fontsize=6)
    axes[row, col].tick_params(axis='both', which='major', labelsize=6)

hide_empty_subplots(axes, len(time_features), n_rows, n_cols)
plt.tight_layout()
plt.savefig(os.path.join(visualization_folder, 'time_domain_violin_plots.png'),
            dpi=DPI, bbox_inches='tight')
plt.close()
print("时域特征小提琴图大图已保存")

# 8. 频域特征小提琴图大图（优化颜色 - 使用单一颜色）
print("生成频域特征小提琴图大图...")
fig, axes, n_rows, n_cols = create_subplot_grid(len(freq_features))

for i, feature in enumerate(freq_features):
    row, col = i // n_cols, i % n_cols
    feature_data = feature_data_dict.get(feature, [])

    if not feature_data:
        axes[row, col].set_visible(False)
        continue

    data_to_plot = [data.values for data in feature_data]
    violin_parts = axes[row, col].violinplot(data_to_plot, showmeans=True, showmedians=True)

    # 优化颜色设置 - 使用单一颜色
    for j, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(VIOLIN_COLOR)
        pc.set_alpha(0.7)
        pc.set_edgecolor(EDGE_COLOR)
        pc.set_linewidth(0.8)

    violin_parts['cmeans'].set_color('#FF6B6B')
    violin_parts['cmeans'].set_linewidth(1.5)

    violin_parts['cmedians'].set_color('#4ECDC4')
    violin_parts['cmedians'].set_linewidth(1.5)

    violin_parts['cbars'].set_color('#45B7D1')
    violin_parts['cbars'].set_linewidth(1.2)

    violin_parts['cmins'].set_color('#96CEB4')
    violin_parts['cmaxes'].set_color('#96CEB4')

    axes[row, col].set_xlabel('Sample', fontsize=8)
    axes[row, col].set_ylabel('Value', fontsize=8)
    axes[row, col].set_xticks(range(1, len(feature_data) + 1))
    axes[row, col].set_xticklabels([f'S{i + 1}' for i in range(len(feature_data))], fontsize=6)
    axes[row, col].tick_params(axis='both', which='major', labelsize=6)

hide_empty_subplots(axes, len(freq_features), n_rows, n_cols)
plt.tight_layout()
plt.savefig(os.path.join(visualization_folder, 'frequency_domain_violin_plots.png'),
            dpi=DPI, bbox_inches='tight')
plt.close()
print("频域特征小提琴图大图已保存")

# 9. 热力图 - 特征相关性矩阵（优化颜色）
print("生成特征相关性热力图...")
# 准备数据：计算每个特征的平均值
feature_means = {}
for feature in column_names:
    feature_data = feature_data_dict.get(feature, [])
    if feature_data:
        # 计算每个样本的特征平均值
        sample_means = [np.mean(data) for data in feature_data]
        feature_means[feature] = sample_means

# 创建DataFrame
if feature_means:
    df_means = pd.DataFrame(feature_means)

    # 计算相关性矩阵
    corr_matrix = df_means.corr()

    # 创建热力图
    plt.figure(figsize=(20, 16))

    # 使用马卡龙色系配色
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角

    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})

    plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_folder, 'feature_correlation_heatmap.png'),
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print("特征相关性热力图已保存")

# 为每个特征生成单独的小图
print("为每个特征生成单独的小图...")
for column in tqdm(column_names):
    feature_folder = os.path.join(visualization_folder, "individual_features", column)
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)

    all_data = feature_data_dict.get(column, [])
    if not all_data:
        continue

    # 折线图小图
    plt.figure(figsize=(8, 5))
    for i, data in enumerate(all_data):
        sampled_data = sample_data(data.values)
        plt.plot(sampled_data, color=LINE_COLOR, alpha=0.7, linewidth=1.2)
    plt.xlabel('Window Index')
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(os.path.join(feature_folder, f'{column}_line_plot.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

    # 直方图小图（优化范围）
    plt.figure(figsize=(8, 5))
    combined_data = np.concatenate(all_data)

    # 调整直方图范围
    data_range = np.ptp(combined_data)
    if data_range == 0:
        data_range = 1

    hist_range = (
        np.min(combined_data) - 0.2 * data_range,
        np.max(combined_data) + 0.2 * data_range
    )

    bins = min(60, max(25, len(combined_data) // 40))

    plt.hist(combined_data, bins=bins, range=hist_range,
             alpha=0.7, color=HIST_COLOR, edgecolor=EDGE_COLOR)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(feature_folder, f'{column}_histogram.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

    # 散点图小图（新增）
    plt.figure(figsize=(8, 5))
    for i, data in enumerate(all_data):
        sampled_data = sample_data(data.values)
        x_values = np.arange(len(sampled_data))
        plt.scatter(x_values, sampled_data, color=SCATTER_COLOR, alpha=0.6, s=20)
    plt.xlabel('Window Index')
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(os.path.join(feature_folder, f'{column}_scatter_plot.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

    # 小提琴图小图
    plt.figure(figsize=(8, 5))
    data_to_plot = [data.values for data in all_data]
    violin_parts = plt.violinplot(data_to_plot, showmeans=True, showmedians=True)

    for j, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(VIOLIN_COLOR)
        pc.set_alpha(0.7)
        pc.set_edgecolor(EDGE_COLOR)
        pc.set_linewidth(0.8)

    violin_parts['cmeans'].set_color('#FF6B6B')
    violin_parts['cmedians'].set_color('#4ECDC4')
    violin_parts['cbars'].set_color('#45B7D1')

    plt.xlabel('Sample', fontsize=10)
    plt.ylabel(column, fontsize=10)
    plt.xticks(range(1, len(all_data) + 1), [f'S{i + 1}' for i in range(len(all_data))], fontsize=9)
    plt.title(f'{column} - Violin Plot', fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(feature_folder, f'{column}_violin_plot.png'), dpi=DPI, bbox_inches='tight')
    plt.close()


print("所有特征的小图已保存")

print(f"所有图表已保存到 {visualization_folder} 文件夹")
print("十一个大图：")
print("1. 时域特征折线图: time_domain_line_plots.png")
print("2. 时域特征直方图: time_domain_histogram_plots.png")
print("3. 时域特征散点图: time_domain_scatter_plots.png")
print("4. 频域特征折线图: frequency_domain_line_plots.png")
print("5. 频域特征直方图: frequency_domain_histogram_plots.png")
print("6. 频域特征散点图: frequency_domain_scatter_plots.png")
print("7. 时域特征小提琴图: time_domain_violin_plots.png")
print("8. 频域特征小提琴图: frequency_domain_violin_plots.png")
print("9. 特征相关性热力图: feature_correlation_heatmap.png")
print(f"小图保存在: {visualization_folder}/individual_features/ 文件夹中")