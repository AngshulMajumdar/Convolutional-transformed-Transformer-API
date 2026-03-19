from __future__ import annotations

import torch
from torch import nn

from transformed_transformer.attention.base import BaseAttention
from transformed_transformer.models.encoder import TinyEncoderBlock


class BertStyleEmbeddings(nn.Module):
    """Minimal BERT-style embeddings with token, position, and segment terms."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, type_vocab_size: int = 2) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        hidden = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        hidden = self.norm(hidden)
        hidden = self.dropout(hidden)
        return hidden


class MiniBertEncoder(nn.Module):
    """Small encoder-only stack intended as a clean extension path toward miniBERT demos."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        ff_dim: int,
        num_layers: int,
        attention_factory: callable,
    ) -> None:
        super().__init__()
        self.embeddings = BertStyleEmbeddings(vocab_size=vocab_size, max_seq_len=max_seq_len, d_model=d_model)
        self.layers = nn.ModuleList(
            [
                TinyEncoderBlock(
                    d_model=d_model,
                    ff_dim=ff_dim,
                    attention=attention_factory(),
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.pooler = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())

    def extra_loss(self) -> torch.Tensor:
        losses = [layer.attention.regularization_loss() for layer in self.layers]
        if not losses:
            return torch.tensor(0.0, device=self.final_norm.weight.device)
        return torch.stack(losses).sum()

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        stats: dict[str, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.layers):
            hidden, layer_stats = layer(hidden, attention_mask=None)
            for key, value in layer_stats.items():
                stats[f"layer_{layer_idx}_{key}"] = value
        hidden = self.final_norm(hidden)
        pooled = self.pooler(hidden[:, 0])
        return hidden, pooled, stats


class MiniBertForSequenceClassification(nn.Module):
    """Mini BERT-style encoder plus CLS classifier head."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        ff_dim: int,
        num_layers: int,
        num_classes: int,
        attention_factory: callable,
    ) -> None:
        super().__init__()
        self.encoder = MiniBertEncoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            ff_dim=ff_dim,
            num_layers=num_layers,
            attention_factory=attention_factory,
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(d_model, num_classes)

    def extra_loss(self) -> torch.Tensor:
        return self.encoder.extra_loss()

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, pooled, stats = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids)
        logits = self.classifier(self.dropout(pooled))
        return logits, stats


class MiniBertForMaskedLM(nn.Module):
    """Mini BERT-style masked language modeling head for later pretraining demos."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        ff_dim: int,
        num_layers: int,
        attention_factory: callable,
    ) -> None:
        super().__init__()
        self.encoder = MiniBertEncoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            ff_dim=ff_dim,
            num_layers=num_layers,
            attention_factory=attention_factory,
        )
        self.lm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

    def extra_loss(self) -> torch.Tensor:
        return self.encoder.extra_loss()

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden, _, stats = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids)
        logits = self.lm_head(hidden)
        return logits, stats
