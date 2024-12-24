import torch.nn as nn
from .SEBlock import SEBlock
from .ECABlock import ECABlock
from .CBAMBlock import CBAMBlock
from .SABlock import SABlock

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, attention_type=None):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.attention_type = attention_type
        if attention_type == 'SE':
            self.attention = SEBlock(hidden_dim)
        elif attention_type == 'ECA':
            self.attention = ECABlock(hidden_dim)
        elif attention_type == 'CBAM':
            self.attention = CBAMBlock(hidden_dim)
        elif attention_type == 'SA':
            self.attention = SABlock(hidden_dim)
        else:
            self.attention = None

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if self.attention:
            lstm_out = self.attention(lstm_out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(lstm_out[:, -1, :])
        return out
