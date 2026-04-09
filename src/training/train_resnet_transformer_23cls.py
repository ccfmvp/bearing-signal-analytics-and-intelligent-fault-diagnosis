# %%
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
import math
from scipy.ndimage import gaussian_filter1d

# 设置设备（自动检测GPU或使用CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载数据
data = pd.read_csv('normalized_data.csv')

numeric_cols = data.select_dtypes(include=['int', 'float']).columns
means = data[numeric_cols].mean()
data.fillna(means, inplace=True)

# 分离特征和标签
X = data.iloc[:, :-4].values
y = data.iloc[:, -1].values

# 对文本标签进行编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建5折交叉验证
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        x = x + self.pe[:x.size(0), :]
        return x

# 定义ResNetTransformer模型
class ResNetTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.2):
        super(ResNetTransformer, self).__init__()

        # 初始投影到 d_model
        self.input_proj = nn.Conv1d(1, d_model, kernel_size=1)

        # 构建 ResNet blocks
        self.resnet = nn.Sequential(
            ResNetBlock(d_model, d_model),
            nn.MaxPool1d(2),
            ResNetBlock(d_model, d_model),
            nn.MaxPool1d(2) if input_dim // 4 >= 4 else nn.Identity()
        )

        self.d_model = d_model
        self.seq_len = input_dim // 4  # 经过两次 MaxPool1d(2)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: [B, input_dim]
        x = x.unsqueeze(1)  # [B, 1, input_dim]

        # 初始投影
        x = self.input_proj(x)  # [B, d_model, L]

        # ResNet blocks
        x = self.resnet(x)  # [B, d_model, seq_len]

        # 转置为 [seq_len, B, d_model]
        x = x.permute(2, 0, 1)  # [seq_len, B, d_model]

        # 添加位置编码
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer_encoder(x)  # [seq_len, B, d_model]

        # 转回 [B, d_model, seq_len] 用于分类
        x = x.permute(1, 2, 0)  # [B, d_model, seq_len]

        # 分类
        out = self.classifier(x)  # [B, num_classes]
        return out


# 训练和评估函数 - 修改为记录每个epoch的指标
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model.train()

    # 初始化记录列表
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    val_precisions = []
    val_recalls = []

    for epoch in tqdm(range(epochs)):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

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
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores,
        'val_precisions': val_precisions,
        'val_recalls': val_recalls
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


# 存储每折的结果和指标历史
fold_results = []
fold_histories = []
fold = 1

for train_index, val_index in kf.split(X_scaled, y_encoded):
    print(f"\n=================== Fold {fold} ===================")

    # 划分训练集和验证集
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
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
    model = ResNetTransformer(input_dim=X.shape[1], num_classes=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 训练模型并记录历史指标
    print("开始训练模型...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50)
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
    report_df.to_csv(f'result/ResNetTransformer23/fold_{fold}_classification_report.csv')

    # 创建混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    fold_results.append((cm, true_labels, predictions))

    # 将混淆矩阵转为 DataFrame 并保存为 CSV
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_df.to_csv(f'result/ResNetTransformer23/fold_{fold}_confusion_matrix.csv')

    # 绘制每折的训练指标变化图
    plt.figure(figsize=(10, 8))

    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    # 直接使用原始数据绘制，不使用平滑
    plt.plot(history['train_losses'], label='Train Loss', linewidth=2)
    plt.plot(history['val_losses'], label='Val Loss', linewidth=2)
    plt.title(f'Fold {fold} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制指标曲线
    plt.subplot(2, 1, 2)
    # 直接使用原始数据绘制，不使用平滑
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
    plt.savefig(f'result/ResNetTransformer23/fold_{fold}_metrics.png', dpi=150)
    plt.show()

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'result/ResNetTransformer23/fold_{fold}_confusion_matrix.png', dpi=150)
    plt.show()

    fold += 1

# 生成汇总图1：五个折的指标变化图（上面三个，下面两个）
plt.figure(figsize=(20, 16))

for i, history in enumerate(fold_histories, 1):
    plt.subplot(3, 2, i)  # 3行2列，共5个子图

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
plt.savefig('result/ResNetTransformer23/all_folds_metrics_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# 生成汇总图2：五个折的混淆矩阵（上面三个，下面两个）
plt.figure(figsize=(20, 16))
for i, (cm, _, _) in enumerate(fold_results, 1):
    plt.subplot(3, 2, i)  # 3行2列，共5个子图

    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                cbar_kws={'shrink': 0.8})
    plt.title(f'Fold {i} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig('result/ResNetTransformer23/all_folds_confusion_matrix_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# 保存模型示例（保存最后一个fold的模型）
torch.save(model.state_dict(), 'model/resnet_transformer_model.pth')
print("模型已保存为 cnn_model.pth")

# 打印总体统计信息
print("\n=================== 总体统计信息 ===================")
all_accuracies = [accuracy_score(true_labels, preds) for _, true_labels, preds in fold_results]
all_f1_scores = [f1_score(true_labels, preds, average='weighted') for _, true_labels, preds in fold_results]

print(f"平均准确率: {np.mean(all_accuracies):.4f} (±{np.std(all_accuracies):.4f})")
print(f"平均F1分数: {np.mean(all_f1_scores):.4f} (±{np.std(all_f1_scores):.4f})")
print(f"各折准确率: {[f'{acc:.4f}' for acc in all_accuracies]}")