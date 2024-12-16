
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from models.ECABlock import ECABlock
from models.CBAMBlock import CBAMBlock
from models.SEBlock import SEBlock
from models.SABlock import SABlock

class SSLT(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, dropout=0.1, attention_type=None):
        super(SSLT, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, 512)
        encoder_layers = TransformerEncoderLayer(d_model=512, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        if attention_type == 'ECA':
            self.attention_block = ECABlock(512)
        elif attention_type == 'CBAM':
            self.attention_block = CBAMBlock(512)
        elif attention_type == 'SE':
            self.attention_block = SEBlock(512)
        elif attention_type == 'SA':
            self.attention_block = SABlock(512)
        else:
            self.attention_block = None
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.embedding(x)  # Shape: [batch_size, input_dim] -> [batch_size, 512]
        x = x.unsqueeze(1)  # Add sequence dimension: [batch_size, 1, 512]
        x = self.transformer_encoder(x)  # Shape: [batch_size, 1, 512]
        if self.attention_block:
            x = self.attention_block(x.transpose(1, 2)).transpose(1, 2)  # Apply selected attention
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x