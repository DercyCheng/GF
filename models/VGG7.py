import torch.nn as nn
from .SEBlock import SEBlock
from .ECABlock import ECABlock
from .CBAMBlock import CBAMBlock
from .SABlock import SABlock

class VGG7(nn.Module):
    def __init__(self, input_dim, attention_type=None):
        super(VGG7, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.attention_type = attention_type
        if attention_type == 'SE':
            self.attention = SEBlock(256)
        elif attention_type == 'ECA':
            self.attention = ECABlock(256)
        elif attention_type == 'CBAM':
            self.attention = CBAMBlock(256)
        elif attention_type == 'SA':
            self.attention = SABlock(256)
        else:
            self.attention = None
        self.classifier = nn.Sequential(
            nn.Linear(256 * (input_dim // 8), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1)
        )

    def forward(self, x):
        x = self.features(x)
        if self.attention:
            x = self.attention(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x