import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """CNN+BiLSTM模型，用于轴承故障多分类"""

    def __init__(self, input_dim, num_classes):
        super(CNNBiLSTM, self).__init__()

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

        self.cnn_output_length = input_dim // 2

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
        averaged_output = lstm_out.mean(dim=1)
        output = self.classifier(averaged_output)
        return output
