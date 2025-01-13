import torch
import torch.nn as nn
from .SEBlock import SEBlock
from .ECABlock import ECABlock
from .CBAMBlock import CBAMBlock
from .SABlock import SABlock

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, cnn_channels=[64, 128, 256], lstm_hidden_dim=128, lstm_num_layers=2, dropout=0.2, attention_type=None):
        super(CNN_LSTM, self).__init__()
        
        # 1D卷积层
        self.conv1 = nn.Conv1d(1, cnn_channels[0], kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_channels[0])
        self.conv2 = nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_channels[1])
        self.conv3 = nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(cnn_channels[2])
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.leaky_relu = nn.LeakyReLU(inplace=False)
        self.dropout = nn.Dropout(dropout)
        
        # 注意力模块
        self.attention_type = attention_type
        if attention_type == 'SE':
            self.attention = SEBlock(cnn_channels[2])
        elif attention_type == 'ECA':
            self.attention = ECABlock(cnn_channels[2])
        elif attention_type == 'CBAM':
            self.attention = CBAMBlock(cnn_channels[2])
        elif attention_type == 'SA':
            self.attention = SABlock(cnn_channels[2])
        else:
            self.attention = None
        
        # LSTM层
        self.lstm = nn.LSTM(cnn_channels[2], lstm_hidden_dim, lstm_num_layers, batch_first=True, dropout=dropout)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        # CNN部分
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        
        if self.attention:
            x = self.attention(x)
        
        x = self.pool(x)
        x = self.dropout(x)
        
        # 调整维度为LSTM输入
        x = x.permute(0, 2, 1)  # (batch, seq, feature)
        
        # LSTM部分
        lstm_out, _ = self.lstm(x)
        
        # 使用最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        
        return out
