
import torch
import torch.nn as nn

class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=16, k_size=7):
        super(CBAMBlock, self).__init__()
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv1d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        avg_out = self.channel_attention(x)
        channel_out = x * avg_out
        # 空间注意力
        avg_out = torch.mean(channel_out, dim=1, keepdim=True)
        max_out, _ = torch.max(channel_out, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.spatial_attention(spatial_out)
        return channel_out * spatial_out