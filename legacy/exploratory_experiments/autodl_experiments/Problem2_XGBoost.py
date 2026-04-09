# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# 加载数据（请替换为实际数据路径）
data = pd.read_csv('问题1_特征提取及标签.csv')  # 假设数据是CSV格式

numeric_cols = data.select_dtypes(include=['int', 'float']).columns
means = data[numeric_cols].mean()
data.fillna(means, inplace=True)

# 分离特征和标签
X = data.iloc[:, :53]  # 前53列是特征
y = data.iloc[:, -1]   # 最后一列是标签

# 对文本标签进行编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 创建XGBoost分类器
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=7,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    random_state=42,
    eval_metric='mlogloss'
)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)

import os

os.makedirs('XGBoost', exist_ok=True)

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

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

# 存储每折的结果
fold = 1
for train_index, val_index in kf.split(X, y_encoded):
    print(f"\n=================== Fold {fold} ===================")
    
    # 划分训练集和验证集
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]
    
    # 创建XGBoost分类器
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=7,
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

    # 在打印分类报告后添加：
    report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'XGBoost/fold_{fold}_classification_report.csv')
    
    # 创建混淆矩阵
    cm = confusion_matrix(y_val, y_pred)

    # 将混淆矩阵转为 DataFrame 并保存为 CSV
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
    plt.savefig(f'XGBoost/fold_{fold}.png', dpi=150)
    plt.show()
    
    fold += 1


