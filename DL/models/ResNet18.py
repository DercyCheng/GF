import torch.nn as nn
from torchvision.models import resnet18
from .SEBlock import SEBlock
from .ECABlock import ECABlock
from .CBAMBlock import CBAMBlock
from .SABlock import SABlock

class ResNet18(nn.Module):
    def __init__(self, attention_type=None):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.attention_type = attention_type
        if attention_type == 'SE':
            self.attention = SEBlock(512)
        elif attention_type == 'ECA':
            self.attention = ECABlock(512)
        elif attention_type == 'CBAM':
            self.attention = CBAMBlock(512)
        elif attention_type == 'SA':
            self.attention = SABlock(512)
        else:
            self.attention = None

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.resnet(x)
        if self.attention:
            x = self.attention(x)
        return x