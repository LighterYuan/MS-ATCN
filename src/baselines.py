from __future__ import annotations

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNBaseline(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class LSTMBaseline(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, num_classes: int, hidden_dim: int = 128, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out, _ = self.rnn(x)
        return self.classifier(out[:, -1, :])


class GRUBaseline(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, num_classes: int, hidden_dim: int = 128, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out, _ = self.rnn(x)
        return self.classifier(out[:, -1, :])


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if y.size(-1) > x.size(-1):
            y = y[..., : x.size(-1)]
        return y + self.downsample(x)


class TCNBaseline(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            TemporalBlock(in_channels, hidden_dim, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.net(x))


def build_baseline_model(model_name: str, in_channels: int, seq_len: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.3) -> nn.Module:
    name = model_name.lower()
    if name == "mlp":
        return MLPBaseline(in_channels, seq_len, num_classes, hidden_dim, dropout)
    if name == "cnn":
        return CNNBaseline(in_channels, seq_len, num_classes, hidden_dim, dropout)
    if name == "lstm":
        return LSTMBaseline(in_channels, seq_len, num_classes, hidden_dim, 1, dropout)
    if name == "gru":
        return GRUBaseline(in_channels, seq_len, num_classes, hidden_dim, 1, dropout)
    if name == "tcn":
        return TCNBaseline(in_channels, seq_len, num_classes, hidden_dim, dropout)
    raise ValueError(f"Unsupported deep baseline: {model_name}")
