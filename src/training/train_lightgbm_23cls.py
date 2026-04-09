# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, \
    recall_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 创建保存结果的目录
os.makedirs('result/LightGBM23', exist_ok=True)

# 加载数据（注意：指定 dtype 或设置 low_memory=False 以避免警告）
data = pd.read_csv('normalized_data.csv', low_memory=False)

# 分离特征和标签
X = data.iloc[:, :-4].values  # X 现在是 NumPy 数组
y = data.iloc[:, -1].values

# 对文本标签进行编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 创建5折交叉验证
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 存储每折的结果
fold_results = []
fold = 1

for train_index, val_index in kf.split(X, y_encoded):
    print(f"\n=================== Fold {fold} ===================")

    # 划分训练集和验证集（直接使用整数索引）
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]

    # 创建LightGBM分类器
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=23,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        random_state=42,
        verbose=-1  # 不输出训练过程信息
    )

    # 训练模型
    print("开始训练模型...")
    model.fit(X_train, y_train)
    print("训练完成")

    # 最终预测
    y_pred = model.predict(X_val)

    # 计算准确率
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')

    print(f"Fold {fold} 准确率: {accuracy:.4f}")
    print(f"Fold {fold} F1分数: {f1:.4f}")
    print(f"Fold {fold} 精确率: {precision:.4f}")
    print(f"Fold {fold} 召回率: {recall:.4f}")

    # 打印分类报告
    print(f"\nFold {fold} 分类报告:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    # 保存分类报告
    report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'result/LightGBM23/fold_{fold}_classification_report.csv')

    # 创建混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    fold_results.append((cm, y_val, y_pred))

    # 将混淆矩阵转为 DataFrame 并保存为 CSV
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_df.to_csv(f'result/LightGBM23/fold_{fold}_confusion_matrix.csv')

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'result/LightGBM23/fold_{fold}_confusion_matrix.png', dpi=150)
    plt.show()

    fold += 1

# 生成汇总图：五个折的混淆矩阵（上面三个，下面两个）
plt.figure(figsize=(20, 16))
for i, (cm, _, _) in enumerate(fold_results, 1):
    plt.subplot(3, 2, i)  # 3行2列，共5个子图

    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                cbar_kws={'shrink': 0.8})
    plt.title(f'Fold {i} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig('result/LightGBM23/all_folds_confusion_matrix_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# 打印总体统计信息
print("\n=================== 总体统计信息 ===================")
all_accuracies = [accuracy_score(true_labels, preds) for _, true_labels, preds in fold_results]
all_f1_scores = [f1_score(true_labels, preds, average='weighted') for _, true_labels, preds in fold_results]
all_precisions = [precision_score(true_labels, preds, average='weighted') for _, true_labels, preds in fold_results]
all_recalls = [recall_score(true_labels, preds, average='weighted') for _, true_labels, preds in fold_results]

print(f"平均准确率: {np.mean(all_accuracies):.4f} (±{np.std(all_accuracies):.4f})")
print(f"平均F1分数: {np.mean(all_f1_scores):.4f} (±{np.std(all_f1_scores):.4f})")
print(f"平均精确率: {np.mean(all_precisions):.4f} (±{np.std(all_precisions):.4f})")
print(f"平均召回率: {np.mean(all_recalls):.4f} (±{np.std(all_recalls):.4f})")
print(f"各折准确率: {[f'{acc:.4f}' for acc in all_accuracies]}")
print(f"各折F1分数: {[f'{f1:.4f}' for f1 in all_f1_scores]}")

# 保存总体统计结果
overall_stats = pd.DataFrame({
    'Fold': list(range(1, 6)),
    'Accuracy': all_accuracies,
    'F1_Score': all_f1_scores,
    'Precision': all_precisions,
    'Recall': all_recalls
})

overall_stats.loc['Mean'] = overall_stats.mean()
overall_stats.loc['Std'] = overall_stats.std()
overall_stats.to_csv('result/LightGBM23/overall_statistics.csv')