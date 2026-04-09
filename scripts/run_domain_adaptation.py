#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
域适应训练流水线
对应问题3：跨域迁移学习（MMD / DANN）

使用方式:
    python scripts/run_domain_adaptation.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 60)
    print("域适应训练流水线")
    print("=" * 60)

    # Step 1: 目标域数据准备
    print("\n[Step 1/5] 目标域数据处理...")
    print("  -> 运行 src/data_processing/target_sliding_window.py")
    print("  -> 运行 src/feature_engineering/extract_target_features.py")
    print("  -> 运行 src/data_processing/target_dataset_builder.py")

    # Step 2: 迁移数据预处理
    print("\n[Step 2/5] 迁移数据预处理...")
    print("  -> 运行 src/domain_adaptation/preprocess_transfer_data.py")
    print("  -> 运行 src/domain_adaptation/transfer_normalization.py")

    # Step 3: MMD迁移方法
    print("\n[Step 3/5] MMD域适应训练...")
    mmd_models = [
        ("CNN-BiLSTM + MMD", "src/domain_adaptation/train_cnn_bilstm_mmd.py"),
        ("CNN-BiLSTM + MMD v1", "src/domain_adaptation/train_cnn_bilstm_mmd_v1.py"),
        ("ResNet-Transformer + MMD", "src/domain_adaptation/train_resnet_transformer_mmd.py"),
        ("XGBoost + MMD", "src/domain_adaptation/train_xgboost_mmd.py"),
    ]
    for name, path in mmd_models:
        print(f"  -> 训练 {name}: {path}")

    # Step 4: DANN迁移方法
    print("\n[Step 4/5] DANN域适应训练...")
    print("  -> 运行 src/domain_adaptation/train_cnn_bilstm_dann.py")

    # Step 5: 结果比较
    print("\n[Step 5/5] 迁移模型比较...")
    print("  -> 对比各迁移方法在目标域上的预测结果")

    print("\n" + "=" * 60)
    print("域适应训练流水线完成!")
    print("迁移模型: outputs/models/transfer_models/")
    print("预测结果: outputs/evaluation/target_predictions/")
    print("=" * 60)


if __name__ == "__main__":
    main()
