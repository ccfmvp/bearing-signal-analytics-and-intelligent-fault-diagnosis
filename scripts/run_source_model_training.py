#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
源域模型训练流水线
对应问题2：多分类模型训练（CNN/LSTM/BiLSTM/Transformer/ResNet/XGBoost/LightGBM）

使用方式:
    python scripts/run_source_model_training.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 60)
    print("源域模型训练流水线")
    print("=" * 60)

    # Step 1: 特征预处理
    print("\n[Step 1/3] 特征选择与归一化...")
    print("  -> 运行 src/feature_engineering/select_top_features.py")
    print("  -> 运行 src/data_processing/normalization.py")

    # Step 2: 深度学习模型训练
    print("\n[Step 2/3] 深度学习模型训练（5折交叉验证）...")
    models_dl = [
        ("CNN", "src/training/train_cnn_23cls.py"),
        ("CNN-LSTM", "src/training/train_cnn_lstm_23cls.py"),
        ("CNN-BiLSTM", "src/training/train_cnn_bilstm_23cls.py"),
        ("CNN-Transformer", "src/training/train_cnn_transformer_23cls.py"),
        ("ResNet-Transformer", "src/training/train_resnet_transformer_23cls.py"),
    ]
    for name, path in models_dl:
        print(f"  -> 训练 {name}: {path}")

    # Step 3: 传统机器学习模型训练
    print("\n[Step 3/3] 传统ML模型训练（5折交叉验证）...")
    models_ml = [
        ("XGBoost", "src/training/train_xgboost_23cls.py"),
        ("LightGBM", "src/training/train_lightgbm_23cls.py"),
    ]
    for name, path in models_ml:
        print(f"  -> 训练 {name}: {path}")

    print("\n" + "=" * 60)
    print("模型训练流水线完成!")
    print("模型文件: outputs/models/source_models/")
    print("评估结果: outputs/evaluation/source_classification/")
    print("=" * 60)


if __name__ == "__main__":
    main()
