# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV
df = pd.read_csv("MMD_Transfer 问题三 7 分类 Fold 1-5 结果.csv")

# 构造一个长表形式：ID - Fold - Prediction - Count
folds = [3, 4, 5]
records = []

for _, row in df.iterrows():
    ID = row['ID']
    for f in folds:
        pred = row[f'Fold {f}']
        count = row[f'Fold {f} Count']
        records.append([ID, f'Fold {f}', pred, count])

long_df = pd.DataFrame(records, columns=['ID', 'Fold', 'Prediction', 'Count'])

# 统计每个 ID 在不同类别的概率分布
prob_df = long_df.pivot_table(index='ID', columns='Prediction', values='Count', aggfunc='sum').fillna(0)

# 行归一化，得到概率
prob_df = prob_df.div(prob_df.sum(axis=1), axis=0)

# 画热力图
plt.figure(figsize=(12, 8))
sns.heatmap(prob_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Probability'})

plt.title("Prediction Probability Heatmap (ID vs Classes)")
plt.ylabel("ID")
plt.xlabel("Predicted Class")
plt.tight_layout()
plt.savefig("概率矩阵热力图.png", dpi=150)
plt.show()
