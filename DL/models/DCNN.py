import torch.nn as nn
from .SEBlock import SEBlock
from .ECABlock import ECABlock
from .CBAMBlock import CBAMBlock
from .SABlock import SABlock  # 添加 SABlock 的导入

class DCNN(nn.Module):
    def __init__(self, input_dim, attention_type=None):
        super(DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=4, dilation=2)  # Added dilation=2 and updated padding
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=4, dilation=2)  # Added dilation=2 and updated padding
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2)  # Added dilation=2 and updated padding
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout(0.7)  # 进一步增加 dropout
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        attention_blocks = {
            'SE': SEBlock,
            'ECA': ECABlock,
            'CBAM': CBAMBlock,
            'SA': SABlock  # 添加 'SA' 对应的 SABlock
        }
        self.attentions = nn.ModuleList([
            attention_blocks[attention_type](32) if attention_type else None,    # 对应32通道
            attention_blocks[attention_type](64) if attention_type else None,    # 对应64通道
            attention_blocks[attention_type](128) if attention_type else None    # 对应128通道
        ])

        self.fc = nn.Sequential(
            nn.Linear(128, 64),  # 增加全连接层的尺寸
            nn.ReLU(inplace=False),
            nn.Dropout(0.7),
            nn.Linear(64, 1)
        )
        self.leaky_relu = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        for i in range(3):
            x = self.leaky_relu(getattr(self, f'bn{i+1}')(getattr(self, f'conv{i+1}')(x)))
            if self.attentions[i]:
                x = self.attentions[i](x)
            x = self.pool(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x