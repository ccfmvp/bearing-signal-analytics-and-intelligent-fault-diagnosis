import math
import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """ResNet残差块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
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
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class ResNetTransformer(nn.Module):
    """ResNet+Transformer模型，用于轴承故障多分类"""

    def __init__(self, input_dim, num_classes, d_model=128, nhead=8, num_layers=2,
                 dim_feedforward=512, dropout=0.2):
        super(ResNetTransformer, self).__init__()

        self.input_proj = nn.Conv1d(1, d_model, kernel_size=1)

        self.resnet = nn.Sequential(
            ResNetBlock(d_model, d_model),
            nn.MaxPool1d(2),
            ResNetBlock(d_model, d_model),
            nn.MaxPool1d(2) if input_dim // 4 >= 4 else nn.Identity()
        )

        self.d_model = d_model
        self.seq_len = input_dim // 4

        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.resnet(x)
        x = x.permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        out = self.classifier(x)
        return out
