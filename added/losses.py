from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FormulaConsistentFocalLoss(nn.Module):
    """Implements alpha_y * (1-p_y)^gamma * CE exactly as stated in the manuscript."""

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha.detach().clone().float())
        else:
            self.alpha = None
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction={self.reduction}")
