# %%
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

# 设置设备（自动检测GPU或使用CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载数据
data = pd.read_csv('问题1_特征提取及标签.csv')

numeric_cols = data.select_dtypes(include=['int', 'float']).columns
means = data[numeric_cols].mean()
data.fillna(means, inplace=True)

# 分离特征和标签
X = data.iloc[:, :53].values
y = data.iloc[:, -1].values

# 对文本标签进行编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建5折交叉验证
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 定义一维CNN模型
class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN1D, self).__init__()
        
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # 第二个卷积块
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # 第三个卷积块
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # 增加通道维度 [batch_size, 1, feature_dim]
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

# 训练和评估函数
def train_model(model, train_loader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        # 打印每10个epoch的损失
        if (epoch + 1) % 1 == 0:
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

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

# 存储每折的结果
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
    model = CNN1D(input_dim=X.shape[1], num_classes=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 训练模型
    print("开始训练模型...")
    train_model(model, train_loader, criterion, optimizer, epochs=25)
    
    # 评估模型
    print("评估模型...")
    true_labels, predictions = evaluate_model(model, val_loader)
    
    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Fold {fold} 准确率: {accuracy:.4f}")
    
    # 打印分类报告
    print(f"\nFold {fold} 分类报告:")
    print(classification_report(true_labels, predictions, target_names=le.classes_))

    # 在打印分类报告后添加：
    report = classification_report(true_labels, predictions, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'CNN/fold_{fold}_classification_report.csv')
    
    # 创建混淆矩阵
    cm = confusion_matrix(true_labels, predictions)

    # 将混淆矩阵转为 DataFrame 并保存为 CSV
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_df.to_csv(f'CNN/fold_{fold}_confusion_matrix.csv')
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, 
                yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'CNN/fold_{fold}.png', dpi=150)
    plt.show()
    
    fold += 1

# 保存模型示例（保存最后一个fold的模型）
torch.save(model.state_dict(), 'cnn_model.pth')
print("模型已保存为 cnn_model.pth")


