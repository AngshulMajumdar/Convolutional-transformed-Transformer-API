from __future__ import annotations

import math

import torch
from torch import nn

from .base import BaseAttention, TensorStats
from .transformed_dense import transform_regularizer


def _apply_attention_mask(attn_logits: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return attn_logits
    mask = attention_mask.to(dtype=torch.bool, device=attn_logits.device)
    while mask.dim() < attn_logits.dim():
        mask = mask.unsqueeze(0)
    return attn_logits.masked_fill(~mask, torch.finfo(attn_logits.dtype).min)


class TransformSparseAttention(BaseAttention):
    """
    Surrogate sparse transformed attention.

    This implementation uses top-k sparsification of transformed values.
    It is useful for API demonstration and smoke testing, but should not be
    advertised as a finalized sparse-transformer algorithm.
    """

    def __init__(self, d_model: int, sparsity: int = 4, reg_weight: float = 1e-3) -> None:
        super().__init__()
        self.d_model = d_model
        self.sparsity = sparsity
        self.reg_weight = reg_weight
        self.Wq = nn.Parameter(torch.eye(d_model) + 0.01 * torch.randn(d_model, d_model))
        self.Wk = nn.Parameter(torch.eye(d_model) + 0.01 * torch.randn(d_model, d_model))
        self.Wv = nn.Parameter(torch.eye(d_model) + 0.01 * torch.randn(d_model, d_model))
        self.o = nn.Linear(d_model, d_model)

    def regularization_loss(self) -> torch.Tensor:
        reg = transform_regularizer(self.Wq) + transform_regularizer(self.Wk) + transform_regularizer(self.Wv)
        return self.reg_weight * reg

    def sparsify_values(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = min(self.sparsity, values.shape[-1])
        idx = torch.topk(values.abs(), k=k, dim=-1).indices
        mask = torch.zeros_like(values).scatter_(-1, idx, 1.0)
        sparse_values = values * mask
        density = (mask != 0).float().mean()
        tail_mse = torch.mean((values - sparse_values) ** 2)
        return sparse_values, density, tail_mse

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TensorStats]:
        q = x @ self.Wq.T
        k = x @ self.Wk.T
        v = x @ self.Wv.T
        v_sparse, density, tail_mse = self.sparsify_values(v)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        attn_logits = _apply_attention_mask(attn_logits, attention_mask)
        attn = torch.softmax(attn_logits, dim=-1)
        y = self.o(torch.matmul(attn, v_sparse))

        with torch.no_grad():
            sq = torch.linalg.svdvals(self.Wq)
            sk = torch.linalg.svdvals(self.Wk)
            sv = torch.linalg.svdvals(self.Wv)
            stats: TensorStats = {
                "reg_loss": self.regularization_loss().detach(),
                "cond_q": torch.linalg.cond(self.Wq).detach(),
                "cond_k": torch.linalg.cond(self.Wk).detach(),
                "cond_v": torch.linalg.cond(self.Wv).detach(),
                "sigma_min_q": sq.min().detach(),
                "sigma_min_k": sk.min().detach(),
                "sigma_min_v": sv.min().detach(),
                "density": density.detach(),
                "tail_mse": tail_mse.detach(),
                "attn_mean": attn.mean().detach(),
                "attn_std": attn.std().detach(),
            }
        return y, stats
