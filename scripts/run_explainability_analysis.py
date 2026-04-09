#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可解释性分析流水线
对应问题4：模型可视化与可解释性分析

使用方式:
    python scripts/run_explainability_analysis.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 60)
    print("可解释性分析流水线")
    print("=" * 60)

    # Step 1: 3D权重可视化
    print("\n[Step 1/3] 模型权重3D可视化...")
    print("  -> 运行 src/explainability/weight_3d_visualization.py")

    # Step 2: MMD损失分析
    print("\n[Step 2/3] MMD损失可视化...")
    print("  -> 运行 src/explainability/mmd_loss_visualization.py")

    # Step 3: 特征重要性与权重热力图
    print("\n[Step 3/3] 特征重要性与权重分析...")
    print("  -> 运行 src/explainability/feature_importance_analysis.py")
    print("  -> 运行 src/explainability/weight_heatmap_visualization.py")

    print("\n" + "=" * 60)
    print("可解释性分析完成!")
    print("可视化输出: outputs/figures/explainability/")
    print("=" * 60)


if __name__ == "__main__":
    main()
