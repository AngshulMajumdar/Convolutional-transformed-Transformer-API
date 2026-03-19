from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


TensorStats = Dict[str, torch.Tensor]


class BaseAttention(nn.Module):
    """Base interface for attention backends used by this package."""

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, TensorStats]:
        raise NotImplementedError

    def regularization_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device if any(True for _ in self.parameters()) else "cpu"
        return torch.tensor(0.0, device=device)
