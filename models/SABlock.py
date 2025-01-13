
import torch
import torch.nn as nn

class SABlock(nn.Module):
    def __init__(self, channels):
        super(SABlock, self).__init__()
        # Define layers for scaled attention
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights