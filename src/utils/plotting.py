import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_individual_f1_barchart(csv_file="detailed_cv_results.csv"):
    """
    绘制每个模型五次F1值的柱状图
    """
    # 读取数据
    df = pd.read_csv(csv_file)

    # 马卡龙配色方案
    macaron_colors = [
        '#FFB6C1',  # 浅粉红
        '#87CEEB',  # 天蓝色
        '#98FB98',  # 浅绿色
        '#DDA0DD',  # 梅红色
        '#FFD700',  # 金黄色
    ]

    # 设置图形样式
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    # 获取模型列表
    models = df['Model'].unique()

    # 设置x轴位置
    x_pos = np.arange(len(models))
    bar_width = 0.15  # 每个折的柱子宽度
    fold_positions = [x_pos + i * bar_width for i in range(5)]

    # 绘制每个模型的五折F1值
    for fold in range(1, 6):
        fold_data = []
        for model in models:
            fold_value = df[(df['Model'] == model) & (df['Fold'] == fold)]['Weighted_Avg_F1']
            if not fold_value.empty:
                fold_data.append(fold_value.values[0])
            else:
                fold_data.append(0)

        plt.bar(fold_positions[fold - 1], fold_data, width=bar_width - 0.02,
                color=macaron_colors[fold - 1], alpha=0.8, label=f'Fold {fold}',
                edgecolor='white', linewidth=1.2)

    # 美化图表
    plt.xlabel('Models', fontsize=13, fontweight='bold', labelpad=10)
    plt.ylabel('Weighted Average F1-Score', fontsize=13, fontweight='bold', labelpad=10)
    plt.title('Weighted F1-Score for Each Model Across 5-Fold Cross Validation',
              fontsize=16, fontweight='bold', pad=20)

    # 设置x轴标签
    plt.xticks(x_pos + bar_width * 2, models, rotation=45, ha='right')

    # 添加图例
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True,
               fancybox=True, shadow=True)

    # 添加网格线
    plt.grid(True, alpha=0.3, axis='y')

    # 调整y轴范围，使图表更美观
    y_min = df['Weighted_Avg_F1'].min() - 0.02
    y_max = df['Weighted_Avg_F1'].max() + 0.02
    plt.ylim(y_min, y_max)

    # 添加数值标签（可选，如果数据点不太密集）
    for fold in range(5):
        for i, model in enumerate(models):
            fold_value = df[(df['Model'] == model) & (df['Fold'] == fold + 1)]['Weighted_Avg_F1']
            if not fold_value.empty:
                value = fold_value.values[0]
                plt.text(fold_positions[fold][i], value + 0.005, f'{value:.3f}',
                         ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('../graph/individual_f1_scores_macaron.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("单个F1值柱状图已保存为 'individual_f1_scores_macaron.png'")


def plot_average_f1_linechart(csv_file="detailed_cv_results.csv"):
    """
    绘制每个模型五次F1平均值的折线图
    """
    # 读取数据
    df = pd.read_csv(csv_file)

    # 马卡龙配色方案
    macaron_colors = [
        '#FF6B9D',  # 更鲜艳的粉红色
        '#4ECDC4',  # 青绿色
        '#45B7D1',  # 蓝色
        '#96CEB4',  # 薄荷绿
        '#FFA07A',  # 浅橙红色
        '#BA68C8',  # 紫色
    ]

    # 设置图形样式
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    # 计算每个模型的平均值和标准差
    model_means = df.groupby('Model')['Weighted_Avg_F1'].mean().sort_values(ascending=False)
    model_stds = df.groupby('Model')['Weighted_Avg_F1'].std()

    # 创建模型列表（按平均值排序）
    models_sorted = model_means.index.tolist()
    means_sorted = model_means.values
    stds_sorted = [model_stds[model] for model in models_sorted]

    # 绘制折线图
    x_pos = np.arange(len(models_sorted))

    # 绘制主折线
    line = plt.plot(x_pos, means_sorted, marker='o', markersize=10,
                    linewidth=3, color='#FF6B9D', alpha=0.9,
                    markerfacecolor='white', markeredgewidth=2,
                    markeredgecolor='#FF6B9D', label='Average F1-Score')

    # 添加误差线（标准差）
    plt.errorbar(x_pos, means_sorted, yerr=stds_sorted, fmt='none',
                 color='#FF6B9D', alpha=0.7, capsize=5, capthick=2,
                 linewidth=2, label='Standard Deviation')

    # 添加填充区域（表示标准差范围）
    plt.fill_between(x_pos, means_sorted - stds_sorted, means_sorted + stds_sorted,
                     color='#FF6B9D', alpha=0.2, label='Std Dev Range')

    # 美化图表
    plt.xlabel('Models', fontsize=13, fontweight='bold', labelpad=10)
    plt.ylabel('Weighted Average F1-Score', fontsize=13, fontweight='bold', labelpad=10)
    plt.title('Average Weighted F1-Score with Standard Deviation by Model',
              fontsize=16, fontweight='bold', pad=20)

    # 设置x轴标签
    plt.xticks(x_pos, models_sorted, rotation=45, ha='right')

    # 添加图例
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)

    # 添加网格线
    plt.grid(True, alpha=0.3, axis='y')

    # 调整y轴范围
    y_min = min(means_sorted - stds_sorted) - 0.01
    y_max = max(means_sorted + stds_sorted) + 0.01
    plt.ylim(y_min, y_max)

    # 添加数值标签
    for i, (mean, std) in enumerate(zip(means_sorted, stds_sorted)):
        plt.text(i, mean + 0.005, f'{mean:.4f}\n(±{std:.4f})',
                 ha='center', va='bottom', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # 添加水平参考线
    plt.axhline(y=np.mean(means_sorted), color='gray', linestyle='--', alpha=0.7)
    plt.text(len(models_sorted) - 0.5, np.mean(means_sorted) + 0.002,
             f'Overall Mean: {np.mean(means_sorted):.4f}',
             ha='right', va='bottom', fontsize=10, color='gray')

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('../graph/average_f1_linechart_macaron.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("平均值折线图已保存为 'average_f1_linechart_macaron.png'")

    # 打印统计摘要
    print("\n" + "=" * 70)
    print("模型性能统计摘要 (按平均F1排序)")
    print("=" * 70)
    for i, model in enumerate(models_sorted):
        print(f"{i + 1:2d}. {model:15} | 平均F1: {means_sorted[i]:.4f} | "
              f"标准差: {stds_sorted[i]:.4f} | 范围: {means_sorted[i] - stds_sorted[i]:.4f} - {means_sorted[i] + stds_sorted[i]:.4f}")

    best_model = models_sorted[0]
    best_score = means_sorted[0]
    print(f"\n最佳模型: {best_model} (平均F1: {best_score:.4f})")


def create_enhanced_individual_chart(csv_file="detailed_cv_results.csv"):
    """
    创建增强版的单个F1值图表，使用更细致的分组
    """
    df = pd.read_csv(csv_file)

    # 更丰富的马卡龙配色
    macaron_colors = [
        '#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD', '#FFD700',
        '#FFA07A', '#20B2AA', '#DEB887', '#F0E68C', '#D8BFD8'
    ]

    # 设置图形
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.set_style("whitegrid")

    # 获取模型和折数
    models = df['Model'].unique()
    folds = sorted(df['Fold'].unique())

    # 设置位置
    x = np.arange(len(models))
    width = 0.75 / len(folds)  # 动态宽度

    # 绘制每个折的柱状图
    bars = []
    for i, fold in enumerate(folds):
        fold_data = [df[(df['Model'] == model) & (df['Fold'] == fold)]['Weighted_Avg_F1'].values[0]
                     for model in models]
        bar = ax.bar(x + i * width - width * (len(folds) - 1) / 2, fold_data, width,
                     label=f'Fold {fold}', color=macaron_colors[i], alpha=0.85,
                     edgecolor='white', linewidth=1.2)
        bars.append(bar)

        # 添加数值标签
        for j, value in enumerate(fold_data):
            ax.text(x[j] + i * width - width * (len(folds) - 1) / 2, value + 0.003,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 美化图表
    ax.set_xlabel('Models', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('Weighted Average F1-Score', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_title('Detailed Weighted F1-Score for Each Fold by Model',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Fold Number')

    # 调整y轴范围
    y_min = df['Weighted_Avg_F1'].min() - 0.02
    y_max = df['Weighted_Avg_F1'].max() + 0.03
    ax.set_ylim(y_min, y_max)

    # 添加平均参考线
    overall_mean = df['Weighted_Avg_F1'].mean()
    ax.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(len(models) - 0.5, overall_mean + 0.005, f'Overall Mean: {overall_mean:.4f}',
            ha='right', va='bottom', fontsize=11, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig('../graph/enhanced_individual_f1_scores.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("增强版单个F1值图表已保存为 'enhanced_individual_f1_scores.png'")


if __name__ == "__main__":
    # 确保CSV文件存在
    try:
        # 绘制单个F1值的柱状图
        plot_individual_f1_barchart("detailed_cv_results.csv")

        # 绘制平均值的折线图
        plot_average_f1_linechart("detailed_cv_results.csv")

        # 绘制增强版图表
        create_enhanced_individual_chart("detailed_cv_results.csv")

        print("\n所有图表已生成完成！")
        print("生成的文件:")
        print("1. individual_f1_scores_macaron.png - 五折F1值柱状图")
        print("2. average_f1_linechart_macaron.png - 平均值折线图")
        print("3. enhanced_individual_f1_scores.png - 增强版柱状图")

    except FileNotFoundError:
        print("错误: 找不到文件 'detailed_cv_results.csv'")
        print("请先运行汇总脚本生成数据文件")
    except Exception as e:
        print(f"发生错误: {e}")