import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import torch.nn.functional as F

# 创建保存结果的目录
os.makedirs('MMD_Transfer', exist_ok=True)

# 设置设备（自动检测GPU或使用CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ========== 读取前80%特征 ==========
top80_features = pd.read_csv("top80_features.csv")["Feature"].tolist()
print(f"选择的特征数量: {len(top80_features)}")

# 加载源域数据
source_data = pd.read_csv('问题1_特征提取及标签.csv')
# 加载目标域数据
target_data = pd.read_csv('问题3_目标域特征提取.csv')

# 只保留前80%重要性的特征
source_data = source_data[top80_features + [source_data.columns[-1]]]  # 特征 + 标签
target_data = target_data[top80_features]

# 处理源域数据
numeric_cols = source_data.select_dtypes(include=['int', 'float']).columns
means = source_data[numeric_cols].mean()
source_data.fillna(means, inplace=True)

# 处理目标域数据
target_numeric_cols = target_data.select_dtypes(include=['int', 'float']).columns
target_data.fillna(means[target_numeric_cols], inplace=True)

# 分离源域特征和标签
X_source = source_data[top80_features].values
y_source = source_data.iloc[:, -1].values

# 目标域只有特征，没有标签
X_target = target_data.values

# 对文本标签进行编码
le = LabelEncoder()
y_encoded = le.fit_transform(y_source)

# 数据标准化
scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_source)
X_target_scaled = scaler.transform(X_target)

# 转换为PyTorch张量
X_target_tensor = torch.FloatTensor(X_target_scaled).to(device)

# 创建5折交叉验证
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# 定义线性映射层
class LinearMapping(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearMapping, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


# 定义一维CNN模型
class CNN1D(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN1D, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 组合模型
class CombinedModel(nn.Module):
    def __init__(self, linear_input_dim, linear_output_dim, cnn_input_channels, num_classes):
        super(CombinedModel, self).__init__()
        self.linear_mapping = LinearMapping(linear_input_dim, linear_output_dim)
        self.cnn = CNN1D(cnn_input_channels, num_classes)

    def forward(self, x):
        x = self.linear_mapping(x)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        return x

    def get_linear_features(self, x):
        return self.linear_mapping(x)


# MMD损失函数
def mmd_loss(source_features, target_features, kernel_mul=2.0, kernel_num=5):
    batch_size = source_features.size(0)
    features = torch.cat([source_features, target_features], dim=0)
    kernel_val = 0
    for sigma in range(-kernel_num, kernel_num + 1):
        sigma = kernel_mul ** sigma
        pairwise_dist = torch.cdist(features, features, p=2)
        kernel_val += torch.exp(-pairwise_dist ** 2 / (2 * sigma ** 2))
    XX = kernel_val[:batch_size, :batch_size]
    YY = kernel_val[batch_size:, batch_size:]
    XY = kernel_val[:batch_size, batch_size:]
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd


# KLD损失函数
def kld_loss(source_features, target_features, epsilon=1e-8):
    batch_size_src = source_features.size(0)
    batch_size_tgt = target_features.size(0)
    min_batch_size = min(batch_size_src, batch_size_tgt)
    src_indices = torch.randperm(batch_size_src)[:min_batch_size]
    tgt_indices = torch.randperm(batch_size_tgt)[:min_batch_size]
    source_features = source_features[src_indices]
    target_features = target_features[tgt_indices]
    source_features = F.normalize(source_features, p=2, dim=1)
    target_features = F.normalize(target_features, p=2, dim=1)
    source_probs = F.softmax(source_features, dim=1)
    target_probs = F.softmax(target_features, dim=1)
    source_probs = torch.clamp(source_probs, min=epsilon)
    target_probs = torch.clamp(target_probs, min=epsilon)
    source_probs = source_probs / source_probs.sum(dim=1, keepdim=True)
    target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)
    kld = F.kl_div(
        input=source_probs.log(),
        target=target_probs,
        reduction='batchmean'
    )
    if torch.isnan(kld):
        return torch.tensor(0.0, device=source_features.device)
    return kld


# 训练函数
def train_model_with_mmd(model, train_loader, target_data, criterion, optimizer, epochs=100, mmd_weight=0.25):
    model.train()
    n_target = len(target_data)
    subset_size = int(0.1 * n_target)
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            source_features = model.get_linear_features(inputs)
            indices = torch.randperm(n_target)[:subset_size]
            target_subset = target_data[indices].to(device)
            target_features = model.get_linear_features(target_subset)
            mmd = mmd_loss(source_features, target_features)
            outputs = model(inputs)
            class_loss = criterion(outputs, labels)
            total_loss = class_loss + mmd_weight * mmd
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * inputs.size(0)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader.dataset):.4f}')


# 评估函数
def evaluate_model(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds


# 预测函数
def predict_target(model, target_tensor):
    model.eval()
    all_preds = []
    with torch.no_grad():
        batch_size = 512
        for i in range(0, len(target_tensor), batch_size):
            batch = target_tensor[i:i + batch_size]
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    return all_preds


# 主训练流程
fold = 1
target_predictions = pd.DataFrame()

for train_index, val_index in kf.split(X_source_scaled, y_encoded):
    print(f"\n=================== Fold {fold} ===================")
    X_train, X_val = X_source_scaled[train_index], X_source_scaled[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=512, shuffle=False)

    linear_output_dim = 100
    model = CombinedModel(
        linear_input_dim=X_source.shape[1],
        linear_output_dim=linear_output_dim,
        cnn_input_channels=1,
        num_classes=len(le.classes_)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    print("开始训练模型...")
    train_model_with_mmd(model, train_loader, X_target_tensor, criterion, optimizer, epochs=50, mmd_weight=0.02)

    print("评估模型...")
    true_labels, predictions = evaluate_model(model, val_loader)
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Fold {fold} 准确率: {accuracy:.4f}")
    print(f"\nFold {fold} 分类报告:")
    print(classification_report(true_labels, predictions, target_names=le.classes_))

    report = classification_report(true_labels, predictions, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'MMD_Transfer/fold_{fold}_classification_report.csv')

    cm = confusion_matrix(true_labels, predictions)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_df.to_csv(f'MMD_Transfer/fold_{fold}_confusion_matrix.csv')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'MMD_Transfer/fold_{fold}_confusion_matrix.png', dpi=150)
    plt.close()

    print(f"对目标域数据进行预测 (Fold {fold})...")
    target_preds = predict_target(model, X_target_tensor)
    target_pred_labels = le.inverse_transform(target_preds)
    target_predictions[f'fold_{fold}_pred'] = target_pred_labels

    fold += 1

target_data_with_predictions = target_data.copy()
for i in range(1, 6):
    target_data_with_predictions[f'fold_{i}_pred'] = target_predictions[f'fold_{i}_pred']

target_data_with_predictions.to_csv('MMD_Transfer/target_data_with_predictions.csv', index=False)
print("目标域数据预测完成并已保存到 'MMD_Transfer/target_data_with_predictions.csv'")

result_data = pd.read_csv('MMD_Transfer/target_data_with_predictions.csv', index_col=None)
rawdata = pd.read_csv('问题3_目标域特征提取.csv', index_col=None)

result_final = pd.concat([
    result_data[['fold_1_pred', 'fold_2_pred', 'fold_3_pred', 'fold_4_pred', 'fold_5_pred']],
    rawdata[['目标代码']]
], axis=1)

id_list, fold_list, prediction_list, count_list = [], [], [], []
for ID in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']:
    for i in range(1, 6):
        try:
            value_counts = result_final[result_final['目标代码'] == ID][f'fold_{i}_pred'].value_counts()
            pred_value = value_counts.index[0]
            pred_count = value_counts.iloc[0]
            id_list.append(ID)
            fold_list.append(i)
            prediction_list.append(pred_value)
            count_list.append(pred_count)
        except IndexError:
            id_list.append(ID)
            fold_list.append(i)
            prediction_list.append(None)
            count_list.append(None)

temp_df = pd.DataFrame({
    'ID': id_list,
    'Fold': fold_list,
    'Prediction': prediction_list,
    'Count': count_list
})

result_pred_df = temp_df.pivot(index='ID', columns='Fold', values='Prediction')
result_count_df = temp_df.pivot(index='ID', columns='Fold', values='Count')

result_pred_df.columns = [f'Fold {col}' for col in result_pred_df.columns]
result_count_df.columns = [f'Fold {col} Count' for col in result_count_df.columns]

result_df = pd.concat([result_pred_df, result_count_df], axis=1)
result_df.reset_index(inplace=True)

new_columns = []
for i in range(1, 6):
    new_columns.append(f'Fold {i}')
    new_columns.append(f'Fold {i} Count')

result_df = result_df[['ID'] + new_columns]
result_df.to_csv('MMD_Transfer 问题四 7 分类 Fold 1-5 结果.csv', index=None)

final_results = []
for idx, row in result_df.iterrows():
    id_name = row['ID']
    fold3_pred, fold4_pred, fold5_pred = row['Fold 3'], row['Fold 4'], row['Fold 5']
    fold3_count, fold4_count, fold5_count = row['Fold 3 Count'], row['Fold 4 Count'], row['Fold 5 Count']
    predictions = [(fold3_pred, fold3_count), (fold4_pred, fold4_count), (fold5_pred, fold5_count)]
    pred_counts, pred_max_values = {}, {}
    for pred, count in predictions:
        if pred in pred_counts:
            pred_counts[pred] += 1
        else:
            pred_counts[pred] = 1
            pred_max_values[pred] = count
        if count > pred_max_values[pred]:
            pred_max_values[pred] = count
    max_frequency = max(pred_counts.values())
    candidates = [pred for pred, freq in pred_counts.items() if freq == max_frequency]
    if len(candidates) == 1:
        final_prediction = candidates[0]
    else:
        max_count = -1
        final_prediction = None
        for candidate in candidates:
            if pred_max_values[candidate] > max_count:
                max_count = pred_max_values[candidate]
                final_prediction = candidate
    final_results.append([id_name, final_prediction])

final_df = pd.DataFrame(final_results, columns=['ID', 'Final_Prediction'])
final_df.to_csv('MMD_Transfer 问题四 7 分类 A-P 结果.csv', index=None)
