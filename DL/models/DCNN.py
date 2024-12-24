import torch.nn as nn
from .SEBlock import SEBlock
from .ECABlock import ECABlock
from .CBAMBlock import CBAMBlock
from .SABlock import SABlock  # 新增SABlock导入

class DCNN(nn.Module):
    def __init__(self, input_dim, attention_type=None):
        super(DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        attention_blocks = {
            'SE': SEBlock,
            'ECA': ECABlock,
            'CBAM': CBAMBlock,
            'SA': SABlock  # 添加SA模块
        }
        self.attentions = nn.ModuleList([
            attention_blocks[attention_type](16) if attention_type else None,
            attention_blocks[attention_type](32) if attention_type else None,
            attention_blocks[attention_type](64) if attention_type else None,
            attention_blocks[attention_type](128) if attention_type else None
        ])

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        self.relu = nn.ReLU(inplace=False)
        self.leaky_relu = nn.LeakyReLU(inplace=False)
        self.elu = nn.ELU(inplace=False)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        if self.attentions[0]:
            x = self.attentions[0](x)
        x = self.pool1(x)

        x = self.leaky_relu(self.bn2(self.conv2(x)))
        if self.attentions[1]:
            x = self.attentions[1](x)
        x = self.pool2(x)

        x = self.leaky_relu(self.bn3(self.conv3(x)))
        if self.attentions[2]:
            x = self.attentions[2](x)

        x = self.leaky_relu(self.bn4(self.conv4(x)))
        if self.attentions[3]:
            x = self.attentions[3](x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x