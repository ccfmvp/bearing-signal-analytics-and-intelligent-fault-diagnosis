# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# 创建保存目录
os.makedirs('XGB_Transfer', exist_ok=True)

# ======================
# 1. 加载数据
# ======================
print("加载数据...")
source_data = pd.read_csv('问题1_特征提取及标签.csv')
target_data = pd.read_csv('问题3_目标域特征提取.csv')

# 只取前53列特征
X_source = source_data.iloc[:, :53].values
y_source = source_data.iloc[:, -1].values
X_target = target_data.iloc[:, :53].values

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y_source)

# 标准化
scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_source)
X_target_scaled = scaler.transform(X_target)

# ======================
# 2. 定义 MMD 损失函数
# ======================
def compute_mmd_loss(X_src, X_tgt):
    mean_src = np.mean(X_src, axis=0)
    mean_tgt = np.mean(X_tgt, axis=0)
    return np.sum((mean_src - mean_tgt) ** 2)

def custom_obj(preds, dtrain):
    """自定义目标函数：交叉熵 + MMD 正则"""
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))  # sigmoid
    grad = preds - labels
    hess = preds * (1.0 - preds)

    # MMD 正则项
    batch_size = min(256, len(preds))
    idx_src = np.random.choice(len(X_source_scaled), batch_size, replace=False)
    idx_tgt = np.random.choice(len(X_target_scaled), batch_size, replace=True)
    mmd_loss = compute_mmd_loss(X_source_scaled[idx_src], X_target_scaled[idx_tgt])
    grad += 0.01 * mmd_loss  # 权重可调
    hess += 0.01
    return grad, hess

# ======================
# 3. 五折交叉验证训练
# ======================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
target_predictions = pd.DataFrame()

for train_idx, val_idx in kf.split(X_source_scaled, y_encoded):
    print(f"\n================ Fold {fold} ================")
    X_train, X_val = X_source_scaled[train_idx], X_source_scaled[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        obj=custom_obj,
        evals=[(dval, 'eval')],
        verbose_eval=10
    )

    # 评估验证集
    preds_val = bst.predict(dval)
    preds_val_label = (preds_val > 0.5).astype(int)
    acc = accuracy_score(y_val, preds_val_label)
    print(f"Fold {fold} 准确率: {acc:.4f}")
    print(classification_report(y_val, preds_val_label, target_names=le.classes_))

    # 预测目标域
    dtarget = xgb.DMatrix(X_target_scaled)
    preds_target = bst.predict(dtarget)
    preds_target_label = (preds_target > 0.5).astype(int)
    preds_target_names = le.inverse_transform(preds_target_label)

    target_predictions[f'fold_{fold}_pred'] = preds_target_names
    fold += 1

# ======================
# 4. 保存 Fold 1-5 预测结果
# ======================
target_data_with_preds = target_data.copy()
for i in range(1, 6):
    target_data_with_preds[f'fold_{i}_pred'] = target_predictions[f'fold_{i}_pred']

target_data_with_preds.to_csv('XGB_Transfer/target_data_with_predictions.csv', index=False)
print("保存 target_data_with_predictions.csv")

# 统计 A-P ID 的多数投票结果
raw_ids = target_data['目标代码'].unique()
final_results = []

for ID in raw_ids:
    rows = target_data_with_preds[target_data['目标代码'] == ID]
    votes = []
    for i in range(1, 6):
        votes.extend(rows[f'fold_{i}_pred'].tolist())
    final_pred = pd.Series(votes).mode()[0]  # 多数投票
    final_results.append([ID, final_pred])

final_df = pd.DataFrame(final_results, columns=['ID', 'Final_Prediction'])
final_df.to_csv('XGB_Transfer 问题三 7 分类 A-P 结果.csv', index=False)

# 保存 Fold 1-5 结果统计
id_list, fold_list, pred_list, count_list = [], [], [], []
for ID in raw_ids:
    for i in range(1, 6):
        values = target_data_with_preds[target_data['目标代码'] == ID][f'fold_{i}_pred']
        if len(values) > 0:
            pred = values.mode()[0]
            cnt = values.value_counts().iloc[0]
        else:
            pred, cnt = None, None
        id_list.append(ID)
        fold_list.append(i)
        pred_list.append(pred)
        count_list.append(cnt)

temp_df = pd.DataFrame({
    'ID': id_list,
    'Fold': fold_list,
    'Prediction': pred_list,
    'Count': count_list
})

result_pred_df = temp_df.pivot(index='ID', columns='Fold', values='Prediction')
result_count_df = temp_df.pivot(index='ID', columns='Fold', values='Count')
result_pred_df.columns = [f'Fold {c}' for c in result_pred_df.columns]
result_count_df.columns = [f'Fold {c} Count' for c in result_count_df.columns]

result_df = pd.concat([result_pred_df, result_count_df], axis=1).reset_index()
result_df.to_csv('XGB_Transfer 问题三 7 分类 Fold 1-5 结果.csv', index=False)

print("✅ 所有结果已生成：")
print(" - XGB_Transfer/target_data_with_predictions.csv")
print(" - XGB_Transfer 问题三 7 分类 Fold 1-5 结果.csv")
print(" - XGB_Transfer 问题三 7 分类 A-P 结果.csv")
