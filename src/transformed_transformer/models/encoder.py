from __future__ import annotations

import torch
from torch import nn

from transformed_transformer.attention.base import BaseAttention


class TinyEncoderBlock(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, attention: BaseAttention) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attention = attention
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attn_out, stats = self.attention(self.norm(hidden), attention_mask=attention_mask)
        hidden = hidden + attn_out
        hidden = hidden + self.ff(hidden)
        return hidden, stats
