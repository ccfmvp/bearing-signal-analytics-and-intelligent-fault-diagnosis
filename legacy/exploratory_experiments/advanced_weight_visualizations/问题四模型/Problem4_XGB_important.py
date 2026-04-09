import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存目录
os.makedirs('XGBoost', exist_ok=True)

# 加载数据
data = pd.read_csv('问题1_特征提取及标签.csv')

# 分离特征和标签
X = data.iloc[:, :53]
y = data.iloc[:, -1]

# 对文本标签进行编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 创建5折交叉验证
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 用来存储所有 fold 的特征重要性
all_feature_importances = []

fold = 1
for train_index, val_index in kf.split(X, y_encoded):
    print(f"\n=================== Fold {fold} ===================")

    # 划分训练集和验证集
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]

    # 创建XGBoost分类器
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        random_state=42,
        eval_metric='mlogloss'
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_val)

    # 计算准确率
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Fold {fold} 准确率: {accuracy:.4f}")

    # 打印分类报告
    print(f"\nFold {fold} 分类报告:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    # 保存分类报告
    report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'XGBoost/fold_{fold}_classification_report.csv')

    # 创建混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_df.to_csv(f'XGBoost/fold_{fold}_confusion_matrix.csv')

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'XGBoost/fold_{fold}_confusion_matrix.png', dpi=150)
    plt.close()

    # ============ 新增：特征重要性 ============
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    feature_importance_df.to_csv(f'XGBoost/feature_importances_fold_{fold}.csv', index=False)

    # 画图
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.title(f'Feature Importance - Fold {fold}')
    plt.tight_layout()
    plt.savefig(f'XGBoost/feature_importances_fold_{fold}.png', dpi=150)
    plt.close()

    # 保存进列表，用于之后平均
    all_feature_importances.append(importance)

    fold += 1

# ============ 最终平均特征重要性 ============
all_feature_importances = np.array(all_feature_importances)
mean_importance = np.mean(all_feature_importances, axis=0)

final_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean_Importance': mean_importance
}).sort_values(by='Mean_Importance', ascending=False)

final_importance_df.to_csv('XGBoost/feature_importances_mean.csv', index=False)

plt.figure(figsize=(12, 6))
sns.barplot(x="Mean_Importance", y="Feature", data=final_importance_df)
plt.title('Feature Importance - Mean of 5 Folds')
plt.tight_layout()
plt.savefig('XGBoost/feature_importances_mean.png', dpi=150)
plt.close()

print("\n✅ 所有 fold 的特征重要性已保存，并生成平均特征重要性。")

# ============ 新增：提取前80%累计重要性的特征 ============
os.makedirs('XGBoost/TopFeatures_80', exist_ok=True)

final_importance_df["Cumulative"] = final_importance_df["Mean_Importance"].cumsum() / final_importance_df["Mean_Importance"].sum()

top80_df = final_importance_df[final_importance_df["Cumulative"] <= 0.8]

# 保存结果
top80_df.to_csv("XGBoost/TopFeatures_80/top80_features.csv", index=False)

plt.figure(figsize=(12, 6))
sns.barplot(x="Mean_Importance", y="Feature", data=top80_df)
plt.title('Top Features covering 80% importance')
plt.tight_layout()
plt.savefig("XGBoost/TopFeatures_80/top80_features.png", dpi=150)
plt.close()

print(f"\n✅ 已输出前80%累计贡献的重要特征，共 {len(top80_df)} 个，保存在 XGBoost/TopFeatures_80/")
