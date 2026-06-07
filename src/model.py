from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class TemporalAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, max(1, channels // 2), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(1, channels // 2), 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.attn(x)
        return x * w


class ConvBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.drop(out)
        out = self.conv2(out)
        out = self.drop(out)
        return self.act(out + x)


class MSATCN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        seq_len: int,
        num_classes: int,
        hidden_channels: int = 32,
        num_blocks: int = 3,
        dilations: List[int] | None = None,
        kernel_sizes: List[int] | None = None,
        dropout: float = 0.2,
        use_channel_attention: bool = True,
        use_temporal_attention: bool = True,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4]
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        branch_out = hidden_channels // len(kernel_sizes)
        remainder = hidden_channels - branch_out * len(kernel_sizes)
        branches = []
        for i, k in enumerate(kernel_sizes):
            out_ch = branch_out + (1 if i < remainder else 0)
            branches.append(ConvBranch(input_channels, out_ch, k, dilation=1, dropout=dropout))
        self.stem_branches = nn.ModuleList(branches)
        self.stem_merge = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        blocks = []
        for i in range(num_blocks):
            d = dilations[i] if i < len(dilations) else dilations[-1]
            blocks.append(ResidualTemporalBlock(hidden_channels, kernel_size=3, dilation=d, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        self.channel_attention = ChannelAttention(hidden_channels) if use_channel_attention else nn.Identity()
        self.temporal_attention = TemporalAttention(hidden_channels) if use_temporal_attention else nn.Identity()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([branch(x) for branch in self.stem_branches], dim=1)
        x = self.stem_merge(x)
        x = self.blocks(x)
        x = self.channel_attention(x)
        x = self.temporal_attention(x)
        x = self.pool(x)
        return self.classifier(x)
