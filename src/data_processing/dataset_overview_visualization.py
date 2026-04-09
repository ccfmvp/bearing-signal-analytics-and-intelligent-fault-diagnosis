import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# 设置马卡龙色系配色方案
MACARON_COLORS = [
    '#FFB6C1', '#FFD700', '#98FB98', '#87CEFA', '#DDA0DD',
    '#FFA07A', '#20B2AA', '#F0E68C', '#E6E6FA', '#B0E0E6',
    '#FFC0CB', '#FFE4E1', '#F5DEB3', '#F0FFF0', '#F5F5DC',
    '#E0FFFF', '#F0F8FF', '#FFF0F5', '#FFEFD5', '#FFE4B5'
]

COLD_MACARON_COLORS = [
    '#87CEEB', '#AFEEEE', '#B0E0E6', '#ADD8E6', '#87CEFA',
    '#00BFFF', '#1E90FF', '#6495ED', '#7B68EE', '#6A5ACD',
    '#9370DB', '#BA55D3', '#DA70D6', '#EE82EE', '#DDA0DD',
    '#FFB6C1', '#FFA07A', '#FFD700', '#98FB98', '#90EE90'
]

# 使用冷色调作为主配色
COLORS = COLD_MACARON_COLORS

# 设置输出目录
output_folder = "graph/1_01"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取数据 - 修改为读取Excel文件
print("正在读取cycle.xlsx文件...")
try:
    df = pd.read_excel('cycle.xlsx')
    print("Excel文件读取成功！")
except FileNotFoundError:
    print("错误：找不到cycle.xlsx文件，请检查文件路径和名称")
    exit()
except Exception as e:
    print(f"读取文件时出错：{e}")
    exit()

# 显示数据基本信息
print(f"数据形状：{df.shape}")
print(f"列名：{df.columns.tolist()}")
print("\n数据前5行：")
print(df.head())

# 数据预处理
# 检查是否存在必要的列，如果不存在则创建或处理
if '故障类型' in df.columns:
    df['故障类型'] = df['故障类型'].fillna('Unknown')
else:
    print("警告：数据中未找到'故障类型'列")

if '传感器位置' in df.columns:
    df['传感器位置'] = df['传感器位置'].fillna('Unknown')
else:
    print("警告：数据中未找到'传感器位置'列")

# 设置图形参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False

# 1. 传感器位置分布饼图（如果存在该列）
if '传感器位置' in df.columns:
    plt.figure(figsize=(10, 8))
    sensor_counts = df['传感器位置'].value_counts()
    colors = [COLORS[i % len(COLORS)] for i in range(len(sensor_counts))]
    plt.pie(sensor_counts.values, labels=sensor_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title('传感器位置分布', fontsize=14, pad=20)
    plt.savefig(os.path.join(output_folder, 'sensor_location_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("已生成传感器位置分布饼图")

# 2. 故障类型分布条形图（如果存在该列）
if '故障类型' in df.columns:
    plt.figure(figsize=(12, 6))
    fault_counts = df['故障类型'].value_counts()
    colors = [COLORS[i % len(COLORS)] for i in range(len(fault_counts))]
    bars = plt.bar(fault_counts.index, fault_counts.values, color=colors, alpha=0.8)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}', ha='center', va='bottom')

    plt.xlabel('故障类型', fontsize=12)
    plt.ylabel('数量', fontsize=12)
    plt.title('故障类型分布', fontsize=14, pad=20)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'fault_type_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("已生成故障类型分布条形图")

# 3. RPM分布直方图和小提琴图（如果存在RPM列）
if 'RPM' in df.columns:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 直方图
    ax1.hist(df['RPM'], bins=20, color=COLORS[0], alpha=0.7, edgecolor=COLORS[5])
    ax1.set_xlabel('RPM', fontsize=12)
    ax1.set_ylabel('频率', fontsize=12)
    ax1.set_title('RPM分布直方图', fontsize=14)

    # 小提琴图
    sns.violinplot(y=df['RPM'], ax=ax2, color=COLORS[0], alpha=0.7)
    ax2.set_ylabel('RPM', fontsize=12)
    ax2.set_title('RPM分布小提琴图', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'rpm_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("已生成RPM分布图")

# 4. 故障特征频率关系散点图（如果存在相关列）
if all(col in df.columns for col in ['BPFO', 'BPFI', 'RPM']):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['BPFO'], df['BPFI'], c=df['RPM'],
                          cmap='Blues', alpha=0.7, s=50)
    plt.colorbar(scatter, label='RPM')
    plt.xlabel('BPFO (外圈故障频率)', fontsize=12)
    plt.ylabel('BPFI (内圈故障频率)', fontsize=12)
    plt.title('故障特征频率关系图', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_folder, 'fault_frequency_relationship.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("已生成故障特征频率关系图")

# 5. 不同传感器位置的RPM箱线图（如果存在相关列）
if all(col in df.columns for col in ['传感器位置', 'RPM']):
    plt.figure(figsize=(10, 6))
    box_data = [df[df['传感器位置'] == loc]['RPM'] for loc in df['传感器位置'].unique()]
    box_plot = plt.boxplot(box_data, labels=df['传感器位置'].unique(),
                           patch_artist=True)

    # 设置箱线图颜色
    for patch, color in zip(box_plot['boxes'], COLORS[:len(box_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.xlabel('传感器位置', fontsize=12)
    plt.ylabel('RPM', fontsize=12)
    plt.title('不同传感器位置的RPM分布', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'rpm_by_sensor_location.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("已生成传感器位置RPM箱线图")

# 6. 故障特征频率热力图（相关性矩阵）
freq_columns = [col for col in ['BPFO', 'BPFI', 'BSF', 'RPM'] if col in df.columns]
if len(freq_columns) >= 2:  # 至少需要两列才能计算相关性
    corr_matrix = df[freq_columns].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('故障特征频率相关性热力图', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'frequency_correlation_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("已生成相关性热力图")

# 7. 时间周期关系图（如果存在相关列）
time_columns = [col for col in ['T_BPFO', 'T_BPFI', 'T_BSF'] if col in df.columns]
if time_columns:
    fig, axes = plt.subplots(1, len(time_columns), figsize=(5 * len(time_columns), 5))
    if len(time_columns) == 1:
        axes = [axes]  # 确保axes是列表形式

    titles = {'T_BPFO': '外圈故障周期', 'T_BPFI': '内圈故障周期', 'T_BSF': '滚动体故障周期'}

    for i, col in enumerate(time_columns):
        axes[i].hist(df[col], bins=15, color=COLORS[i * 2], alpha=0.7, edgecolor=COLORS[i * 2 + 1])
        axes[i].set_xlabel(titles.get(col, col), fontsize=10)
        axes[i].set_ylabel('频率', fontsize=10)
        axes[i].set_title(titles.get(col, col) + '分布', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'fault_period_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("已生成时间周期关系图")

# 8. 多变量关系散点图矩阵（选择存在的关键特征）
key_columns = [col for col in ['RPM', 'BPFO', 'BPFI', 'BSF'] if col in df.columns]
if len(key_columns) >= 2:  # 至少需要两列才能生成散点图矩阵
    sns.pairplot(df[key_columns], diag_kind='hist', plot_kws={'alpha': 0.6, 's': 30})
    plt.suptitle('多变量关系散点图矩阵', y=1.02, fontsize=14)
    plt.savefig(os.path.join(output_folder, 'multivariate_scatter_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("已生成多变量关系散点图矩阵")

print(f"\n所有图表已保存到 {output_folder} 文件夹")
print("生成完成！")