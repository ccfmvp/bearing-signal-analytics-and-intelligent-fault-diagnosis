# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# 特征名称映射字典（索引从0开始）
FEATURE_NAME_MAPPING = {
    0: 'mean', 1: 'median', 2: 'std', 3: 'variance', 4: 'rms', 5: 'peak',
    6: 'peak_to_peak', 7: 'skewness', 8: 'kurtosis', 9: 'crest_factor',
    10: 'form_factor', 11: 'impulse_factor', 12: 'energy', 13: 'mean_abs',
    14: 'abs_energy', 15: 'q25', 16: 'q75', 17: 'iqr', 18: 'zero_crossing_rate',
    19: 'peak_count', 20: 'peak_rate', 21: 'autocorr_max', 22: 'autocorr_first_zero',
    23: 'shannon_entropy', 24: 'envelope_mean', 25: 'envelope_std', 26: 'diff_mean',
    27: 'diff_std', 28: 'fractal_dimension', 29: 'lyapunov_exponent',
    30: 'spectral_centroid', 31: 'spectral_spread', 32: 'spectral_skewness',
    33: 'spectral_kurtosis', 34: 'spectral_energy', 35: 'spectral_entropy',
    36: 'band_1_energy', 37: 'band_2_energy', 38: 'band_3_energy',
    39: 'band_4_energy', 40: 'band_5_energy', 41: 'band_1_energy_ratio',
    42: 'band_2_energy_ratio', 43: 'band_3_energy_ratio', 44: 'band_4_energy_ratio',
    45: 'band_5_energy_ratio', 46: 'dominant_frequency', 47: 'dominant_magnitude',
    48: 'dominant_frequency_ratio', 49: 'frequency_std', 50: 'frequency_variance',
    51: 'spectral_rolloff', 52: 'spectral_flatness', 53: 'spectrogram_peak_var',
    54: 'spectrogram_entropy'
}

# Set cyan-green color scheme
GREEN_COLORS = ['#2E8B57', '#3CB371', '#20B2AA', '#48D1CC', '#40E0D0']
sns.set_palette(GREEN_COLORS)
sns.set_style("whitegrid")


def load_and_preprocess_data(file_path):
    """Load and preprocess data"""
    print("Loading data...")
    data = pd.read_csv(file_path)

    print(f"Data shape: {data.shape}")
    print(f"Number of features: {data.shape[1] - 4}")
    print(f"Number of samples: {data.shape[0]}")

    # Separate features and labels
    feature_columns = data.columns[:55]
    non_feature_columns = data.columns[55:]

    X = data[feature_columns]
    y = data[data.columns[-1]]

    print(f"Label categories: {len(np.unique(y))}")
    print("Label distribution:")
    print(y.value_counts())

    return X, y, data, feature_columns, non_feature_columns


def plot_feature_importance(feature_importance, top_n=30, filename='feature_importance_ranking.png', fixed_xlim=None, title='Top Feature Importance Ranking'):
    """绘制特征重要性排名图，使用英文特征名"""
    plt.figure(figsize=(10, 8))  # 调整图表的大小

    top_features = feature_importance.head(top_n)

    # 将特征索引转换为英文名称
    feature_names = []
    for feature_idx in top_features['feature']:
        try:
            # 提取数字部分并减去1，映射到英文名称
            idx = int(feature_idx.replace('特征', '')) - 1
            feature_name = FEATURE_NAME_MAPPING.get(idx, f"Unknown_{idx}")
            feature_names.append(feature_name)
        except (ValueError, TypeError):
            # 如果转换失败，直接使用原索引作为特征名
            feature_names.append(str(feature_idx))

    colors = plt.cm.GnBu(np.linspace(0.6, 0.9, len(top_features)))

    bars = plt.barh(range(len(top_features)), top_features['importance'],
                    color=colors, edgecolor='#2E8B57', linewidth=0.7, alpha=0.8)

    # 使用支持中文的字体，例如 SimHei 或 Arial
    plt.yticks(range(len(top_features)), feature_names, fontsize=10, family='SimHei', rotation=45)  # 设置字体和大小，并旋转标签
    plt.xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Feature Name', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)  # 根据标题参数更新
    plt.gca().invert_yaxis()
    plt.gca().set_facecolor('#F8FFFC')
    plt.grid(True, alpha=0.3)

    # 添加数值标签
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{importance:.4f}', va='center', fontsize=9)

    # 设置横坐标的范围，确保后20个图与前30个图一致
    if fixed_xlim:
        plt.xlim(fixed_xlim)

    plt.tight_layout()  # 自动调整布局，确保标签不会被裁剪
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征重要性排名图已保存：{filename}")


def random_forest_feature_selection(X, y, n_estimators=100, random_state=42):
    """Use random forest for feature selection"""
    print("=" * 50)
    print("Random Forest Feature Importance Analysis")
    print("=" * 50)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=random_state, stratify=y_encoded
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )

    print("Training random forest model...")
    rf.fit(X_train, y_train)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    return rf, feature_importance, le


def select_top_features(feature_importance, top_n=30):
    """Select top N features"""
    selected_features = feature_importance.head(top_n)['feature'].values
    print(f"\nSelected top {top_n} features")
    return selected_features, top_n


def process_original_dataset(original_data, selected_features, feature_columns, non_feature_columns):
    """Keep selected feature columns in original dataset"""
    print("\nProcessing original dataset...")

    columns_to_keep = list(selected_features) + list(non_feature_columns)
    processed_data = original_data[columns_to_keep]

    print(f"Original dataset shape: {original_data.shape}")
    print(f"Processed dataset shape: {processed_data.shape}")
    print(f"Deleted {original_data.shape[1] - processed_data.shape[1]} feature columns")
    print(f"Number of features kept: {len(selected_features)}")

    return processed_data


def main():
    """Main function"""
    file_path = "labeledData23.csv"
    try:
        X, y, original_data, feature_columns, non_feature_columns = load_and_preprocess_data(file_path)
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # Random forest feature selection
    rf, feature_importance, le = random_forest_feature_selection(X, y)

    # Select top 30 features
    selected_features, n_selected = select_top_features(feature_importance, top_n=30)

    # Plot feature importance with English names
    # 前30个特征图
    plot_feature_importance(feature_importance, top_n=30, filename='graph/feature_importance_top30.png',
                            title='Top 30 Feature Importance Ranking')

    # 获取前30个图的横坐标最大值
    top30_max_x = feature_importance.head(30)['importance'].max()

    # 后25个特征图，设置相同的横坐标范围
    plot_feature_importance(feature_importance.tail(25), top_n=25, filename='graph/feature_importance_bottom25.png',
                            fixed_xlim=(0, top30_max_x), title='Bottom 25 Feature Importance Ranking')

    # Process original dataset
    processed_data = process_original_dataset(original_data, selected_features, feature_columns, non_feature_columns)

    print("\nSelected feature list (with English names):")
    for i, feature in enumerate(selected_features, 1):
        importance = feature_importance[feature_importance['feature'] == feature]['importance'].values[0]
        # 转换为英文名称显示
        try:
            idx = int(feature)
            feature_name = FEATURE_NAME_MAPPING.get(idx, f"Unknown_{idx}")
        except (ValueError, TypeError):
            feature_name = str(feature)
        print(f"{i:2d}. {feature_name} (Index {feature}): {importance:.6f}")

    # Save results
    feature_importance.to_csv('all_features_importance.csv', index=False, encoding='utf-8-sig')
    processed_data.to_csv('processed_dataset.csv', index=False, encoding='utf-8-sig')

    reduction = (X.shape[1] - n_selected) / X.shape[1] * 100
    print("\n" + "=" * 50)
    print("Feature selection completed!")
    print(f"- Original features: {X.shape[1]}")
    print(f"- Selected features: {n_selected}")
    print(f"- Reduction: {reduction:.1f}%")
    print(f"- Files saved: all_features_importance.csv, processed_dataset.csv, feature_importance_ranking.png")
    print("=" * 50)


if __name__ == "__main__":
    main()