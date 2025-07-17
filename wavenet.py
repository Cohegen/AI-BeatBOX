import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv_filter = nn.Conv1d(channels, channels, kernel_size, padding=0, dilation=dilation)
        self.conv_gate = nn.Conv1d(channels, channels, kernel_size, padding=0, dilation=dilation)
        self.conv_residual = nn.Conv1d(channels, channels, 1)
        self.conv_skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        # Manually pad for causality
        pad = (self.dilation * (self.kernel_size - 1), 0)
        x_padded = F.pad(x, pad)
        filter_out = torch.tanh(self.conv_filter(x_padded))
        gate_out = torch.sigmoid(self.conv_gate(x_padded))
        z = filter_out * gate_out
        residual = self.conv_residual(z) + x
        skip = self.conv_skip(z)
        return residual, skip

class WaveNet(nn.Module):
    def __init__(self, in_channels=1, channels=32, kernel_size=2, num_blocks=3, num_layers=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.causal = nn.Conv1d(in_channels, channels, kernel_size, padding=0)
        self.residual_blocks = nn.ModuleList()
        for b in range(num_blocks):
            for n in range(num_layers):
                dilation = 2 ** n
                self.residual_blocks.append(ResidualBlock(channels, kernel_size, dilation))
        self.output1 = nn.Conv1d(channels, channels, 1)
        self.output2 = nn.Conv1d(channels, 1, 1)

    def forward(self, x):
        # Manually pad for causality in the first layer
        pad = (self.kernel_size - 1, 0)
        x = F.pad(x, pad)
        x = self.causal(x)
        skip_connections = 0
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections = skip_connections + skip if isinstance(skip_connections, torch.Tensor) else skip
        out = F.relu(skip_connections)
        out = F.relu(self.output1(out))
        out = self.output2(out)
        return out
