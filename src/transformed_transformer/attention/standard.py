from __future__ import annotations

import math

import torch
from torch import nn

from .base import BaseAttention, TensorStats


def _apply_attention_mask(attn_logits: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return attn_logits
    mask = attention_mask.to(dtype=torch.bool, device=attn_logits.device)
    while mask.dim() < attn_logits.dim():
        mask = mask.unsqueeze(0)
    return attn_logits.masked_fill(~mask, torch.finfo(attn_logits.dtype).min)


class StandardAttention(BaseAttention):
    """Single-head dense self-attention with learned Q/K/V projections."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TensorStats]:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        attn_logits = _apply_attention_mask(attn_logits, attention_mask)
        attn = torch.softmax(attn_logits, dim=-1)
        y = self.o(torch.matmul(attn, v))

        stats: TensorStats = {
            "attn_mean": attn.mean().detach(),
            "attn_std": attn.std().detach(),
        }
        return y, stats
