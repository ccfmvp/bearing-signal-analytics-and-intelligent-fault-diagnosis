#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
源域特征工程流水线
对应问题1：CWRU数据集周期计算 -> 滑动窗口采样 -> 特征提取 -> 数据集构建

使用方式:
    python scripts/run_source_feature_pipeline.py
"""

import sys
from pathlib import Path

# 将项目根目录加入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 60)
    print("源域特征工程流水线")
    print("=" * 60)

    # Step 1: 周期计算
    print("\n[Step 1/5] 周期计算...")
    print("  -> 运行 src/data_processing/cycle_calculation.py")

    # Step 2: 数据集概览可视化
    print("\n[Step 2/5] 数据集概览可视化...")
    print("  -> 运行 src/data_processing/dataset_overview_visualization.py")

    # Step 3: 滑动窗口采样
    print("\n[Step 3/5] 滑动窗口采样...")
    print("  -> 运行 src/data_processing/source_sliding_window.py")

    # Step 4: 特征提取
    print("\n[Step 4/5] 特征提取...")
    print("  -> 运行 src/feature_engineering/extract_source_features.py")

    # Step 5: 特征合并与标签构建
    print("\n[Step 5/5] 特征合并与数据集构建...")
    print("  -> 运行 src/feature_engineering/merge_and_label_features.py")
    print("  -> 运行 src/data_processing/build_source_dataset_23cls.py")
    print("  -> 运行 src/data_processing/build_source_dataset_7cls.py")
    print("  -> 运行 src/data_processing/build_source_dataset_4cls.py")

    print("\n" + "=" * 60)
    print("特征工程流水线完成!")
    print("输出文件位于: data/interim/ 和 data/processed/")
    print("=" * 60)


if __name__ == "__main__":
    main()
