import torch
import torch.nn as nn

# ResidualBlock 클래스
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation,
                               dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation,
                               dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)\
            if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # 🔧 padding으로 생긴 길이 차이 제거
        if out.size(-1) != x.size(-1):
            out = out[:, :, :x.size(-1)]

        res = x if self.downsample is None else self.downsample(x)
        return out + res

# TCN 전체 클래스
class TCN(nn.Module): 
    def __init__(self, input_size, output_size, num_channels, kernel_size=7, dropout=0.2):
        super(TCN, self).__init__()

        layers = []
        for i in range(len(num_channels)):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation_size = 2 ** i
            layers += [ResidualBlock(in_ch, out_ch, kernel_size, dilation_size, dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):  # x: [B, L, C] → [B, C, L]
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]  # 마지막 시점만 예측
        out = self.fc(out)
        return out

