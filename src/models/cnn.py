import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# Residual Block
# -------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3,
            padding=dilation, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3,
            padding=dilation, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


# -------------------------------
# Main Model
# -------------------------------
class SpectraCNN(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_blocks=8):
        super().__init__()

        # Input projection
        self.in_conv = nn.Conv1d(in_channels, base_channels, kernel_size=5, padding=2)

        # Residual stack with increasing dilation
        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** (i % 4)  # cycles: 1,2,4,8
            blocks.append(ResBlock(base_channels, dilation=dilation))
        self.blocks = nn.Sequential(*blocks)

        # Bottleneck
        self.mid = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
        )

        # Output projection
        self.out_conv = nn.Conv1d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        x: (B, L, 1)
        """
        # Convert to (B, C, L)
        x_in = x.transpose(1, 2)

        x_feat = self.in_conv(x_in)
        x_feat = self.blocks(x_feat)
        x_feat = self.mid(x_feat)

        correction = self.out_conv(x_feat)

        # Residual learning (CRITICAL)
        y = x_in + correction

        return y.transpose(1, 2)