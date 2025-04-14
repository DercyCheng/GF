import torch
import torch.nn as nn
import torch.nn.functional as F
from .SEBlock import SEBlock
from .ECABlock import ECABlock
from .CBAMBlock import CBAMBlock  # 确保导入CBAM

class DynamicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4):
        super(DynamicConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.K = K
        
        # Initialize weights with correct dimensions
        self.weight = nn.Parameter(
            torch.randn(K, out_channels, in_channels//groups, kernel_size),
            requires_grad=True
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_channels), requires_grad=True)
        else:
            self.bias = None
            
        # Modified attention layer
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, K, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Generate attention scores (B, K, 1)
        scores = self.attention(x)
        
        # Compute dynamic weights
        dynamic_weights = torch.sum(self.weight.unsqueeze(0) * scores.view(batch_size, self.K, 1, 1, 1), dim=1)
        
        # Compute dynamic bias if needed
        if self.bias is not None:
            dynamic_bias = torch.sum(self.bias.unsqueeze(0) * scores.view(batch_size, self.K, 1), dim=1)
        else:
            dynamic_bias = None
        
        # Apply convolution with dynamic weights and bias
        return F.conv1d(
            x,
            weight=dynamic_weights,
            bias=dynamic_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

class AdaptiveFeatureFusion(nn.Module):
    """创新性的自适应特征融合模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        weights = self.attention(x1 + x2)
        return weights * x1 + (1 - weights) * x2

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class DCNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5, attention_type=None):
        super().__init__()
        
        # Wider channels
        channels = [64, 128, 256, 512]
        
        # Initial conv layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual layers
        self.backbone = nn.ModuleList([
            ResidualBlock(in_ch, out_ch, stride=2)
            for in_ch, out_ch in zip(channels[:-1], channels[1:])
        ])
        
        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            self._get_attention_block(attention_type, ch)
            for ch in channels[1:]
        ]) if attention_type else None
        
        # Output layers with better regularization
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(channels[-1], 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def _get_attention_block(self, attention_type, channels):
        """创建指定类型的注意力模块"""
        if attention_type is None:
            return None
        elif attention_type.lower() == 'SE':
            return SEBlock(channels)
        elif attention_type.lower() == 'ECA':
            return ECABlock(channels)
        elif attention_type.lower() == 'CBAM':
            return CBAMBlock(channels)
        else:
            raise ValueError(f"不支持的注意力机制类型: {attention_type}")
    
    def forward(self, x):
        x = self.conv1(x)
        
        for i, (res_block, attn) in enumerate(zip(self.backbone, 
                                                 self.attention_blocks if self.attention_blocks else [None]*len(self.backbone))):
            x = res_block(x)
            if attn is not None:
                x = attn(x)
        
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.fc2(x)
        return x

    def get_feature_importance(self):
        """获取特征重要性"""
        with torch.no_grad():
            # 使用第一层卷积权重作为特征重要性指标
            weights = self.backbone[0][0].weight
            importance = torch.abs(weights).mean(dim=(0, 1))
            return importance.cpu().numpy()
            
    def get_attention_weights(self, x):
        """获取注意力权重"""
        if not self.attention_blocks:
            return None
        with torch.no_grad():
            # 传播到每个注意力块
            x = self.conv1(x)
            attention_weights = []
            
            for i, (res_block, attn) in enumerate(zip(self.backbone, self.attention_blocks)):
                x = res_block(x)
                if hasattr(attn, 'get_attention_weights'):
                    weights = attn.get_attention_weights(x)
                    attention_weights.append(weights)
                else:
                    # 对于没有实现get_attention_weights方法的注意力块
                    attention_weights.append(None)
                    
            return attention_weights