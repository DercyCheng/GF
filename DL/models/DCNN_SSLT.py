import torch
import torch.nn as nn
from .DCNN import DCNN
from .SSLT import SSLT

class DCNN_SSLT(nn.Module):
    def __init__(self, input_dim_dcnn, input_dim_sslt, attention_type=None):
        super(DCNN_SSLT, self).__init__()
        self.dcnn = DCNN(input_dim=input_dim_dcnn, attention_type=attention_type)
        self.sslt = SSLT(input_dim=input_dim_sslt, attention_type=attention_type)
        self.fc = nn.Sequential(
            nn.Linear(1024 + 512, 256),  # 假设 DCNN 输出 1024，SSLT 输出 512
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x_dcnn, x_sslt):
        out_dcnn = self.dcnn(x_dcnn)
        out_sslt = self.sslt(x_sslt)
        combined = torch.cat((out_dcnn, out_sslt), dim=1)  # 直接拼接特征
        out = self.fc(combined)
        return out.squeeze()