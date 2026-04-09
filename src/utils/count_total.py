import os
import pandas as pd
import numpy as np
from pathlib import Path


def process_fold_reports(directory_path, output_file="cross_validation_summary.csv"):
    """
    处理包含五折交叉验证结果的目录，汇总所有模型的评估指标

    Parameters:
    directory_path: 包含模型目录的路径
    output_file: 输出汇总文件的名称
    """

    # 定义要处理的模型目录
    model_dirs = [
        "CNN23",
        "CNNBILSTM23",
        "CNNLSTM23",
        "CNNTransformer23",
        "LightGBM23",
        "XGBoost23",
        "ResNetTransformer23"
    ]

    # 存储所有模型结果的列表
    all_results = []

    # 遍历每个模型目录
    for model_dir in model_dirs:
        model_path = Path(directory_path) / model_dir

        if not model_path.exists():
            print(f"警告: 目录 {model_path} 不存在，跳过")
            continue

        print(f"处理模型: {model_dir}")

        # 存储当前模型五折结果的列表
        fold_results = []

        # 查找并读取五折结果文件
        for fold_num in range(1, 6):
            fold_file = model_path / f"fold_{fold_num}_classification_report.csv"

            if not fold_file.exists():
                print(f"警告: 文件 {fold_file} 不存在，跳过")
                continue

            try:
                # 读取CSV文件
                df = pd.read_csv(fold_file, index_col=0)

                # 提取需要的三行数据
                accuracy_row = df.loc['accuracy']
                macro_avg_row = df.loc['macro avg']
                weighted_avg_row = df.loc['weighted avg']

                # 存储当前折的结果
                fold_result = {
                    'fold': fold_num,
                    'accuracy': accuracy_row.iloc[0],  # 取precision列的值作为accuracy
                    'macro_avg_precision': macro_avg_row['precision'],
                    'macro_avg_recall': macro_avg_row['recall'],
                    'macro_avg_f1': macro_avg_row['f1-score'],
                    'weighted_avg_precision': weighted_avg_row['precision'],
                    'weighted_avg_recall': weighted_avg_row['recall'],
                    'weighted_avg_f1': weighted_avg_row['f1-score']
                }

                fold_results.append(fold_result)
                print(f"  成功读取第 {fold_num} 折数据")

            except Exception as e:
                print(f"错误: 读取文件 {fold_file} 时出错: {e}")
                continue

        # 计算五折平均值
        if fold_results:
            # 转换为DataFrame便于计算
            folds_df = pd.DataFrame(fold_results)

            # 计算平均值
            avg_results = {
                'Model': model_dir,
                'accuracy_mean': folds_df['accuracy'].mean(),
                'accuracy_std': folds_df['accuracy'].std(),
                'macro_avg_precision_mean': folds_df['macro_avg_precision'].mean(),
                'macro_avg_precision_std': folds_df['macro_avg_precision'].std(),
                'macro_avg_recall_mean': folds_df['macro_avg_recall'].mean(),
                'macro_avg_recall_std': folds_df['macro_avg_recall'].std(),
                'macro_avg_f1_mean': folds_df['macro_avg_f1'].mean(),
                'macro_avg_f1_std': folds_df['macro_avg_f1'].std(),
                'weighted_avg_precision_mean': folds_df['weighted_avg_precision'].mean(),
                'weighted_avg_precision_std': folds_df['weighted_avg_precision'].std(),
                'weighted_avg_recall_mean': folds_df['weighted_avg_recall'].mean(),
                'weighted_avg_recall_std': folds_df['weighted_avg_recall'].std(),
                'weighted_avg_f1_mean': folds_df['weighted_avg_f1'].mean(),
                'weighted_avg_f1_std': folds_df['weighted_avg_f1'].std(),
                'folds_processed': len(fold_results)
            }

            all_results.append(avg_results)
            print(f"  完成处理，成功处理 {len(fold_results)}/5 折数据")
        else:
            print(f"  警告: 模型 {model_dir} 没有找到有效的折数据")

    # 创建汇总DataFrame
    if all_results:
        summary_df = pd.DataFrame(all_results)

        # 设置Model为索引
        summary_df.set_index('Model', inplace=True)

        # 保存到CSV文件
        summary_df.to_csv(output_file)
        print(f"\n汇总完成！结果已保存到: {output_file}")

        # 打印简要结果
        print("\n各模型五折交叉验证平均结果:")
        print("=" * 80)
        for model in summary_df.index:
            acc_mean = summary_df.loc[model, 'accuracy_mean']
            macro_f1_mean = summary_df.loc[model, 'macro_avg_f1_mean']
            weighted_f1_mean = summary_df.loc[model, 'weighted_avg_f1_mean']
            folds = summary_df.loc[model, 'folds_processed']
            print(f"{model:15} | Accuracy: {acc_mean:.4f} | Macro F1: {macro_f1_mean:.4f} | "
                  f"Weighted F1: {weighted_f1_mean:.4f} | 折数: {folds}/5")

        return summary_df
    else:
        print("错误: 没有找到任何有效数据")
        return None


def create_detailed_summary(directory_path, output_file="detailed_cv_results.csv"):
    """
    创建详细的汇总表，包含每一折的详细结果
    """
    model_dirs = [
        "CNN23", "ResNetTransformer23","CNNBILSTM23", "CNNLSTM23",
        "CNNTransformer23", "LightGBM23", "XGBoost23"
    ]

    detailed_results = []

    for model_dir in model_dirs:
        model_path = Path(directory_path) / model_dir

        if not model_path.exists():
            continue

        for fold_num in range(1, 6):
            fold_file = model_path / f"fold_{fold_num}_classification_report.csv"

            if not fold_file.exists():
                continue

            try:
                df = pd.read_csv(fold_file, index_col=0)

                detailed_results.append({
                    'Model': model_dir,
                    'Fold': fold_num,
                    'Accuracy': df.loc['accuracy'].iloc[0],
                    'Macro_Avg_Precision': df.loc['macro avg']['precision'],
                    'Macro_Avg_Recall': df.loc['macro avg']['recall'],
                    'Macro_Avg_F1': df.loc['macro avg']['f1-score'],
                    'Weighted_Avg_Precision': df.loc['weighted avg']['precision'],
                    'Weighted_Avg_Recall': df.loc['weighted avg']['recall'],
                    'Weighted_Avg_F1': df.loc['weighted avg']['f1-score']
                })

            except Exception as e:
                continue

    if detailed_results:
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(output_file, index=False)
        print(f"详细结果已保存到: {output_file}")
        return detailed_df
    return None


if __name__ == "__main__":

    current_directory = "../result"  # 当前目录，可以根据需要修改

    # 处理并汇总结果
    summary = process_fold_reports(current_directory)

    # 创建详细结果表
    detailed = create_detailed_summary(current_directory)

    if summary is not None:
        print(f"\n汇总统计:")
        print(f"成功处理的模型数量: {len(summary)}")
        print(f"平均处理的折数: {summary['folds_processed'].mean():.1f}")

        # 找出最佳模型（按accuracy）
        best_model = summary['accuracy_mean'].idxmax()
        best_accuracy = summary['accuracy_mean'].max()
        print(f"最佳模型: {best_model} (Accuracy: {best_accuracy:.4f})")