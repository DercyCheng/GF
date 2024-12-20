import torch.nn as nn
from .SEBlock import SEBlock
from .ECABlock import ECABlock
from .CBAMBlock import CBAMBlock

class DCNN(nn.Module):
    def __init__(self, input_dim, attention_type=None):
        super(DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(512)
        self.conv6 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(1024)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        attention_blocks = {
            'SE': SEBlock,
            'ECA': ECABlock,
            'CBAM': CBAMBlock
        }
        self.attentions = nn.ModuleList([
            attention_blocks[attention_type](32) if attention_type else None,
            attention_blocks[attention_type](64) if attention_type else None,
            attention_blocks[attention_type](128) if attention_type else None,
            attention_blocks[attention_type](256) if attention_type else None,
            attention_blocks[attention_type](512) if attention_type else None,
            attention_blocks[attention_type](1024) if attention_type else None
        ])

        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        self.relu = nn.ReLU(inplace=False)
        self.leaky_relu = nn.LeakyReLU(inplace=False)
        self.elu = nn.ELU(inplace=False)

    def forward(self, x):
        for i in range(6):
            x = self.leaky_relu(getattr(self, f'bn{i+1}')(getattr(self, f'conv{i+1}')(x)))
            if self.attentions[i]:
                x = self.attentions[i](x)
            x = self.pool(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x