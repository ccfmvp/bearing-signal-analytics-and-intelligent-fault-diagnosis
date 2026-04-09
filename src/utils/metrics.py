import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)


def compute_metrics(y_true, y_pred, average='weighted'):
    """计算分类指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        average: 多分类平均方式

    Returns:
        dict: 包含accuracy, f1, precision, recall的字典
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average=average),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
    }


def print_classification_report(y_true, y_pred, target_names=None):
    """打印并返回分类报告"""
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=target_names))
    return report
