import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, \
    recall_score
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
os.makedirs('result/CNNBiLSTM_MMD', exist_ok=True)

# 设置设备（自动检测GPU或使用CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载源域数据
source_data = pd.read_csv('final_original_data.csv')
# 加载目标域数据
target_data = pd.read_csv('final_target_data.csv')

# 只取目标域的前30列
target_data = target_data.iloc[:, :-1]

# 处理源域数据
numeric_cols = source_data.select_dtypes(include=['int', 'float']).columns
means = source_data[numeric_cols].mean()
source_data.fillna(means, inplace=True)

# 处理目标域数据
target_numeric_cols = target_data.select_dtypes(include=['int', 'float']).columns
target_data.fillna(means[target_numeric_cols], inplace=True)

# 分离源域特征和标签
X_source = source_data.iloc[:, :-1].values
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


# ==================== 模型定义部分 ====================
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


class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNBiLSTM, self).__init__()

        # CNN特征提取部分
        self.cnn_features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # 计算输出长度（只经过一次池化）
        self.cnn_output_length = input_dim // 2

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        cnn_out = self.cnn_features(x)
        lstm_input = cnn_out.transpose(1, 2)

        lstm_out, (hidden, cell) = self.lstm(lstm_input)

        # 使用所有时间步的平均值
        averaged_output = lstm_out.mean(dim=1)

        output = self.classifier(averaged_output)
        return output


class CombinedModel(nn.Module):
    def __init__(self, linear_input_dim, linear_output_dim, cnnbilstm_input_dim, num_classes):
        super(CombinedModel, self).__init__()
        self.linear_mapping = LinearMapping(linear_input_dim, linear_output_dim)
        self.cnn_bilstm = CNNBiLSTM(cnnbilstm_input_dim, num_classes)

    def forward(self, x):
        # 首先通过线性映射
        x = self.linear_mapping(x)

        # 通过CNN-BiLSTM
        x = self.cnn_bilstm(x)
        return x

    def get_linear_features(self, x):
        """获取线性映射后的特征，不经过CNN-BiLSTM"""
        return self.linear_mapping(x)


# ==================== 损失函数部分 ====================
def mmd_loss(source_features, target_features, kernel_mul=2.0, kernel_num=5):
    """计算最大均值差异(MMD)损失"""
    batch_size = source_features.size(0)
    total_size = batch_size * 2

    # 合并源域和目标域特征
    features = torch.cat([source_features, target_features], dim=0)

    # 计算核矩阵
    kernel_val = 0
    for sigma in range(-kernel_num, kernel_num + 1):
        sigma = kernel_mul ** sigma  # 带宽参数
        # 计算高斯核
        pairwise_dist = torch.cdist(features, features, p=2)
        kernel_val += torch.exp(-pairwise_dist ** 2 / (2 * sigma ** 2))

    # 计算MMD
    XX = kernel_val[:batch_size, :batch_size]
    YY = kernel_val[batch_size:, batch_size:]
    XY = kernel_val[:batch_size, batch_size:]

    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd


# ==================== 训练和评估函数 ====================
def train_model_with_mmd(model, train_loader, val_loader, target_data, criterion, optimizer, epochs=100,
                         mmd_weight=0.25):
    model.train()
    n_target = len(target_data)
    subset_size = int(0.1 * n_target)  # 计算目标数据的10%

    # 初始化记录列表
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    val_precisions = []
    val_recalls = []
    class_losses = []
    mmd_losses = []

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        running_class_loss = 0.0
        running_mmd_loss = 0.0

        # 每个epoch随机选择10%的目标数据
        indices = torch.randperm(n_target)[:subset_size]
        target_subset = target_data[indices].to(device)

        # 训练阶段
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            optimizer.zero_grad()

            # 获取源域线性特征
            source_features = model.get_linear_features(inputs)

            # 获取目标域线性特征（使用子集）
            target_features = model.get_linear_features(target_subset)

            # 计算MMD损失
            mmd = mmd_loss(source_features, target_features)

            # 计算分类输出和损失
            outputs = model(inputs)
            class_loss = criterion(outputs, labels)

            # 总损失
            total_loss = class_loss + mmd_weight * mmd

            # 反向传播
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * inputs.size(0)
            running_class_loss += class_loss.item() * inputs.size(0)
            running_mmd_loss += mmd.item() * inputs.size(0)

        # 计算平均损失
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_class_loss = running_class_loss / len(train_loader.dataset)
        epoch_mmd_loss = running_mmd_loss / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        class_losses.append(epoch_class_loss)
        mmd_losses.append(epoch_mmd_loss)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')

        val_accuracies.append(accuracy)
        val_f1_scores.append(f1)
        val_precisions.append(precision)
        val_recalls.append(recall)

        # 打印每10个epoch的指标
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Total Loss: {epoch_loss:.4f}, '
                  f'Class Loss: {epoch_class_loss:.4f}, MMD Loss: {epoch_mmd_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, '
                  f'Precision: {precision:.4f}, Recall: {recall:.4f}')

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores,
        'val_precisions': val_precisions,
        'val_recalls': val_recalls,
        'class_losses': class_losses,
        'mmd_losses': mmd_losses
    }


def evaluate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


# 预测目标域数据
def predict_target(model, target_tensor):
    model.eval()
    all_preds = []

    with torch.no_grad():
        # 分批处理目标域数据，避免内存不足
        batch_size = 512
        for i in range(0, len(target_tensor), batch_size):
            batch = target_tensor[i:i + batch_size]
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds


# ==================== 主程序 ====================
# 存储每折的结果和指标历史
fold_results = []
fold_histories = []
fold = 1

# 初始化一个DataFrame来存储所有fold的预测结果
target_predictions = pd.DataFrame()

for train_index, val_index in kf.split(X_source_scaled, y_encoded):
    print(f"\n=================== Fold {fold} ===================")

    # 划分训练集和验证集
    X_train, X_val = X_source_scaled[train_index], X_source_scaled[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # 初始化模型、损失函数和优化器
    linear_output_dim = 100  # 线性映射的输出维度
    model = CombinedModel(
        linear_input_dim=X_source.shape[1],
        linear_output_dim=linear_output_dim,
        cnnbilstm_input_dim=linear_output_dim,  # CNN-BiLSTM的输入维度
        num_classes=len(le.classes_)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 训练模型（带MMD损失）
    print("开始训练模型...")
    history = train_model_with_mmd(model, train_loader, val_loader, X_target_tensor,
                                   criterion, optimizer, epochs=50, mmd_weight=0.02)
    fold_histories.append(history)

    # 评估模型
    print("评估模型...")
    true_labels, predictions = evaluate_model(model, val_loader)

    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Fold {fold} 准确率: {accuracy:.4f}")

    # 打印分类报告
    print(f"\nFold {fold} 分类报告:")
    print(classification_report(true_labels, predictions, target_names=le.classes_))

    # 保存分类报告
    report = classification_report(true_labels, predictions, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'result/CNNBiLSTM_MMD/fold_{fold}_classification_report.csv')

    # 创建混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    fold_results.append((cm, true_labels, predictions))

    # 将混淆矩阵转为 DataFrame 并保存为 CSV
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_df.to_csv(f'result/CNNBiLSTM_MMD/fold_{fold}_confusion_matrix.csv')

    # 绘制每折的训练指标变化图 - 分成单独的图片

    # 1. 绘制损失曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Train Loss', linewidth=2)
    plt.plot(history['val_losses'], label='Val Loss', linewidth=2)
    plt.title(f'Fold {fold} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'result/CNNBiLSTM_MMD/fold_{fold}_loss_curves.png', dpi=150)
    plt.show()

    # 2. 绘制分类损失和MMD损失图
    plt.figure(figsize=(10, 6))
    plt.plot(history['class_losses'], label='Class Loss', linewidth=2)
    plt.plot(history['mmd_losses'], label='MMD Loss', linewidth=2)
    plt.title(f'Fold {fold} - Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'result/CNNBiLSTM_MMD/fold_{fold}_component_losses.png', dpi=150)
    plt.show()

    # 3. 绘制指标曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_accuracies'], label='Accuracy', color='green', linewidth=2)
    plt.plot(history['val_f1_scores'], label='F1 Score', color='red', linewidth=2)
    plt.plot(history['val_precisions'], label='Precision', color='blue', linewidth=2)
    plt.plot(history['val_recalls'], label='Recall', color='orange', linewidth=2)
    plt.title(f'Fold {fold} - Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'result/CNNBiLSTM_MMD/fold_{fold}_validation_metrics.png', dpi=150)
    plt.show()

    # 4. 绘制混淆矩阵图
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'result/CNNBiLSTM_MMD/fold_{fold}_confusion_matrix.png', dpi=150)
    plt.show()

    # 对目标域数据进行预测
    print(f"对目标域数据进行预测 (Fold {fold})...")
    target_preds = predict_target(model, X_target_tensor)

    # 将预测结果转换为原始标签
    target_pred_labels = le.inverse_transform(target_preds)

    # 将预测结果添加到DataFrame中
    target_predictions[f'fold_{fold}_pred'] = target_pred_labels

    fold += 1

# ==================== 生成汇总图表 ====================
# 生成汇总图1：五个折的指标变化图
plt.figure(figsize=(12, 8))        # 2 行 3 列，宽 12 高 8 足够
for i, history in enumerate(fold_histories, 1):
    plt.subplot(2, 3, i)           # 2 行 3 列

    # 绘制指标曲线
    plt.plot(history['val_accuracies'], label='Accuracy', color='green', linewidth=2.5)
    plt.plot(history['val_f1_scores'], label='F1 Score', color='red', linewidth=2.5)
    plt.plot(history['val_precisions'], label='Precision', color='blue', linewidth=2.5)
    plt.plot(history['val_recalls'], label='Recall', color='orange', linewidth=2.5)

    plt.title(f'Fold {i} - Validation Metrics', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)  # 确保y轴范围一致便于比较

plt.tight_layout()
plt.savefig('result/CNNBiLSTM_MMD/all_folds_metrics_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# 生成汇总图2：五个折的混淆矩阵
plt.figure(figsize=(20, 16))
for i, (cm, _, _) in enumerate(fold_results, 1):
    plt.subplot(3, 2, i)  # 3行2列，共5个子图

    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                cbar_kws={'shrink': 0.8})
    plt.title(f'Fold {i} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig('result/CNNBiLSTM_MMD/all_folds_confusion_matrix_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 保存模型和结果 ====================
# 保存模型示例（保存最后一个fold的模型）
torch.save(model.state_dict(), 'model/cnn_bilstm_mmd_model.pth')
print("模型已保存为 cnn_bilstm_mmd_model.pth")

# 打印总体统计信息
print("\n=================== 总体统计信息 ===================")
all_accuracies = [accuracy_score(true_labels, preds) for _, true_labels, preds in fold_results]
all_f1_scores = [f1_score(true_labels, preds, average='weighted') for _, true_labels, preds in fold_results]

print(f"平均准确率: {np.mean(all_accuracies):.4f} (±{np.std(all_accuracies):.4f})")
print(f"平均F1分数: {np.mean(all_f1_scores):.4f} (±{np.std(all_f1_scores):.4f})")
print(f"各折准确率: {[f'{acc:.4f}' for acc in all_accuracies]}")

# ==================== 处理目标域预测结果 ====================
# 将目标域预测结果合并到原始目标域数据
target_data_with_predictions = target_data.copy()
for i in range(1, 6):
    target_data_with_predictions[f'fold_{i}_pred'] = target_predictions[f'fold_{i}_pred']

# 保存带有预测结果的目标域数据
target_data_with_predictions.to_csv('result/CNNBiLSTM_MMD/target_data_with_predictions.csv', index=False)

# 读取结果数据
result_data = pd.read_csv('result/CNNBiLSTM_MMD/target_data_with_predictions.csv', index_col=None)
rawdata = pd.read_csv('target_labeledData.csv', index_col=None)

# 合并预测结果和原始数据
result_final = pd.concat([
    result_data[['fold_1_pred', 'fold_2_pred', 'fold_3_pred', 'fold_4_pred', 'fold_5_pred']],
    rawdata[['目标代码']]
], axis=1)

# 创建空列表存储结果
id_list = []
fold_list = []
prediction_list = []
count_list = []

# 遍历每个ID和fold
for ID in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']:
    for i in range(1, 6):
        # 获取预测结果
        try:
            value_counts = result_final[result_final['目标代码'] == ID][f'fold_{i}_pred'].value_counts()
            pred_value = value_counts.index[0]
            pred_count = value_counts.iloc[0]

            id_list.append(ID)
            fold_list.append(i)
            prediction_list.append(pred_value)
            count_list.append(pred_count)
        except IndexError:
            # 如果找不到值，添加NaN
            id_list.append(ID)
            fold_list.append(i)
            prediction_list.append(None)
            count_list.append(None)

# 创建临时DataFrame
temp_df = pd.DataFrame({
    'ID': id_list,
    'Fold': fold_list,
    'Prediction': prediction_list,
    'Count': count_list
})

# 转换为所需的格式（ID为行，Fold为列）
result_pred_df = temp_df.pivot(index='ID', columns='Fold', values='Prediction')
result_count_df = temp_df.pivot(index='ID', columns='Fold', values='Count')

# 重命名列
result_pred_df.columns = [f'Fold {col}' for col in result_pred_df.columns]
result_count_df.columns = [f'Fold {col} Count' for col in result_count_df.columns]

# 合并两个DataFrame
result_df = pd.concat([result_pred_df, result_count_df], axis=1)

# 重置索引使ID成为常规列
result_df.reset_index(inplace=True)

# 重新排列列的顺序，使预测值和计数相邻
new_columns = []
for i in range(1, 6):
    new_columns.append(f'Fold {i}')
    new_columns.append(f'Fold {i} Count')

result_df = result_df[['ID'] + new_columns]

# 保存详细结果
result_df.to_csv('result/CNNBiLSTM_MMD/问题三_7分类_Fold_1-5_结果.csv', index=None)

# 基于Fold 3, 4, 5进行最终判断
final_results = []

# 遍历每一行（每个ID）
for idx, row in result_df.iterrows():
    id_name = row['ID']

    # 获取Fold 3, 4, 5的预测结果
    fold3_pred = row['Fold 3']
    fold4_pred = row['Fold 4']
    fold5_pred = row['Fold 5']

    # 获取Fold 3, 4, 5的计数值
    fold3_count = row['Fold 3 Count']
    fold4_count = row['Fold 4 Count']
    fold5_count = row['Fold 5 Count']

    # 创建预测结果和计数的配对列表
    predictions = [
        (fold3_pred, fold3_count),
        (fold4_pred, fold4_count),
        (fold5_pred, fold5_count)
    ]

    # 统计每个预测结果出现的次数
    pred_counts = {}
    pred_max_values = {}

    for pred, count in predictions:
        if pred in pred_counts:
            pred_counts[pred] += 1
        else:
            pred_counts[pred] = 1
            pred_max_values[pred] = count
        # 更新最大计数值
        if count > pred_max_values[pred]:
            pred_max_values[pred] = count

    # 找到出现次数最多的预测结果
    max_frequency = max(pred_counts.values())
    candidates = [pred for pred, freq in pred_counts.items() if freq == max_frequency]

    # 如果只有一个候选者，直接选择它
    if len(candidates) == 1:
        final_prediction = candidates[0]
    else:
        # 如果有多个候选者（出现次数相同），选择计数值最大的
        max_count = -1
        final_prediction = None
        for candidate in candidates:
            if pred_max_values[candidate] > max_count:
                max_count = pred_max_values[candidate]
                final_prediction = candidate

    final_results.append([id_name, final_prediction])

# 创建最终的DataFrame
final_df = pd.DataFrame(final_results, columns=['ID', 'Final_Prediction'])
final_df.to_csv('result/CNNBiLSTM_MMD/问题三_7分类_A-P_结果.csv', index=None)

print("所有处理完成！")