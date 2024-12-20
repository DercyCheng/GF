from torch import nn
import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from .ECABlock import ECABlock
from .CBAMBlock import CBAMBlock
from .SEBlock import SEBlock
from .SABlock import SABlock

class SSLT(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, dropout=0.1, attention_type=None):
        super(SSLT, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, 512)
        encoder_layers = TransformerEncoderLayer(d_model=512, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        attention_blocks = {
            'ECA': ECABlock(512),
            'CBAM': CBAMBlock(512),
            'SE': SEBlock(512),
            'SA': SABlock(512)
        }
        self.attention_block = attention_blocks.get(attention_type, None)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Default device

    def forward(self, x):
        x = self.embedding(x)  # Shape: [batch_size, input_dim] -> [batch_size, 512]
        x = x.unsqueeze(1)  # Add sequence dimension: [batch_size, 1, 512]
        x = self.transformer_encoder(x)  # Shape: [batch_size, 1, 512]
        if self.attention_block:
            x = self.attention_block(x.transpose(1, 2)).transpose(1, 2)  # Apply selected attention
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x.squeeze()  # 确保输出形状正确