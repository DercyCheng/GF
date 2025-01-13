import torch.nn as nn
from .SEBlock import SEBlock
from .ECABlock import ECABlock
from .CBAMBlock import CBAMBlock
from .SABlock import SABlock

class DCNN(nn.Module):
    def __init__(self, input_dim, attention_type=None):
        super(DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(1024)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.attention_type = attention_type
        if attention_type == 'SE':
            self.attention1 = SEBlock(64)
            self.attention2 = SEBlock(128)
            self.attention3 = SEBlock(256)
            self.attention4 = SEBlock(512)
            self.attention5 = SEBlock(1024)
        elif attention_type == 'ECA':
            self.attention1 = ECABlock(64)
            self.attention2 = ECABlock(128)
            self.attention3 = ECABlock(256)
            self.attention4 = ECABlock(512)
            self.attention5 = ECABlock(1024)
        elif attention_type == 'CBAM':
            self.attention1 = CBAMBlock(64)
            self.attention2 = CBAMBlock(128)
            self.attention3 = CBAMBlock(256)
            self.attention4 = CBAMBlock(512)
            self.attention5 = CBAMBlock(1024)
        elif attention_type == 'SA':
            self.attention1 = SABlock(64)
            self.attention2 = SABlock(128)
            self.attention3 = SABlock(256)
            self.attention4 = SABlock(512)
            self.attention5 = SABlock(1024)
        else:
            self.attention1 = self.attention2 = self.attention3 = self.attention4 = self.attention5 = None

        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 1)
        self.relu = nn.ReLU(inplace=False)
        self.leaky_relu = nn.LeakyReLU(inplace=False)
        self.elu = nn.ELU(inplace=False)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        if self.attention1:
            x = self.attention1(x)
        x = self.pool(x)
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        if self.attention2:
            x = self.attention2(x)
        x = self.pool(x)
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        if self.attention3:
            x = self.attention3(x)
        x = self.pool(x)
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        if self.attention4:
            x = self.attention4(x)
        x = self.pool(x)
        x = self.leaky_relu(self.bn5(self.conv5(x)))
        if self.attention5:
            x = self.attention5(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.elu(self.fc4(x))
        x = self.fc5(x)
        return x