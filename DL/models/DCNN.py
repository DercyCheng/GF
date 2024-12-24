import torch.nn as nn
from .SEBlock import SEBlock
from .ECABlock import ECABlock
from .CBAMBlock import CBAMBlock
from .SABlock import SABlock  # 新增导入 SABlock

class DCNN(nn.Module):
    def __init__(self, input_dim, attention_type=None):
        super(DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)  # 将32减少到16
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)  # 将64减少到32
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # 将128减少到64
        self.bn3 = nn.BatchNorm1d(64)
        # 移除多余的卷积层和批归一化层
        # self.conv4 = ...
        # self.bn4 = ...
        # self.conv5 = ...
        # self.bn5 = ...
        # self.conv6 = ...
        # self.bn6 = ...
        
        # 修改池化层，减少池化次数或调整参数
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # 取消后续池化层
        # self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.7)  # 增加Dropout比率
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        attention_blocks = {
            'SE': SEBlock,
            'ECA': ECABlock,
            'CBAM': CBAMBlock,
            'SA': SABlock  # 增加 SA 选项
        }
        self.attentions = nn.ModuleList([
            attention_blocks[attention_type](16) if attention_type else None,
            attention_blocks[attention_type](32) if attention_type else None,
            attention_blocks[attention_type](64) if attention_type else None,
            # attention_blocks[attention_type](256) if attention_type else None,
            # attention_blocks[attention_type](512) if attention_type else None,
            # attention_blocks[attention_type](1024) if attention_type else None
        ])

        self.fc = nn.Sequential(
            nn.Linear(64, 128),  # 调整全连接层大小
            nn.ReLU(inplace=False),
            nn.Dropout(0.7),
            nn.Linear(128, 1)
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
        # 取消池化
        # x = self.pool3(x)

        # x = self.leaky_relu(self.bn4(self.conv4(x)))
        # if self.attentions[3]:
        #     x = self.attentions[3](x)
        # 取消池化
        # x = self.pool4(x)

        # x = self.leaky_relu(self.bn5(self.conv5(x)))
        # if self.attentions[4]:
        #     x = self.attentions[4](x)
        # 取消池化
        # x = self.pool5(x)

        # x = self.leaky_relu(self.bn6(self.conv6(x)))
        # if self.attentions[5]:
        #     x = self.attentions[5](x)
        # 取消池化
        # x = self.pool6(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x