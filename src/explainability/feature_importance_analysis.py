# src/explainability/feature_importance_analysis.py
"""特征重要性分析模块"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(importance_df, top_k=30, output_path=None):
    """绘制特征重要性柱状图

    Args:
        importance_df: DataFrame with columns ['feature', 'importance']
        top_k: 显示前k个特征
        output_path: 保存路径
    """
    df_sorted = importance_df.sort_values('importance', ascending=True).tail(top_k)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=df_sorted, palette='viridis')
    plt.title(f'Top {top_k} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
