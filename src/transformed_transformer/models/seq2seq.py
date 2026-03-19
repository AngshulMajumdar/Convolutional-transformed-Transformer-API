from __future__ import annotations

import math

import torch
from torch import nn

from transformed_transformer.attention.transformed_conv import ConvTransform1D
from transformed_transformer.models.bert import BertStyleEmbeddings
from transformed_transformer.models.encoder import TinyEncoderBlock
from transformed_transformer.models.slm import CausalTokenEmbeddings


class StandardCrossAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.weight.device)

    def forward(
        self,
        query_hidden: torch.Tensor,
        memory_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q = self.q_proj(query_hidden)
        k = self.k_proj(memory_hidden)
        v = self.v_proj(memory_hidden)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool, device=attn_logits.device)
            while mask.dim() < attn_logits.dim():
                mask = mask.unsqueeze(0)
            attn_logits = attn_logits.masked_fill(~mask, torch.finfo(attn_logits.dtype).min)
        attn = torch.softmax(attn_logits, dim=-1)
        out = self.out_proj(torch.matmul(attn, v))
        stats = {"attn_mean": attn.mean().detach(), "attn_std": attn.std().detach()}
        return out, stats


class TransformConvCrossAttention(nn.Module):
    """Cross-attention with DCTL-inspired local transforms on queries, keys, and values."""

    def __init__(self, d_model: int, kernel_size: int = 3, reg_weight: float = 1e-3) -> None:
        super().__init__()
        self.d_model = d_model
        self.reg_weight = reg_weight
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_transform = ConvTransform1D(d_model=d_model, kernel_size=kernel_size)
        self.k_transform = ConvTransform1D(d_model=d_model, kernel_size=kernel_size)
        self.v_transform = ConvTransform1D(d_model=d_model, kernel_size=kernel_size)
        self.out_proj = nn.Linear(d_model, d_model)

    def regularization_loss(self) -> torch.Tensor:
        return (
            self.q_transform.regularization_loss(self.reg_weight)
            + self.k_transform.regularization_loss(self.reg_weight)
            + self.v_transform.regularization_loss(self.reg_weight)
        )

    def forward(
        self,
        query_hidden: torch.Tensor,
        memory_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q = self.q_transform(self.q_proj(query_hidden))
        k = self.k_transform(self.k_proj(memory_hidden))
        v = self.v_transform(self.v_proj(memory_hidden))
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool, device=attn_logits.device)
            while mask.dim() < attn_logits.dim():
                mask = mask.unsqueeze(0)
            attn_logits = attn_logits.masked_fill(~mask, torch.finfo(attn_logits.dtype).min)
        attn = torch.softmax(attn_logits, dim=-1)
        out = self.out_proj(torch.matmul(attn, v))
        stats = {
            "attn_mean": attn.mean().detach(),
            "attn_std": attn.std().detach(),
            "q_gate": torch.tanh(self.q_transform.gate).detach(),
            "k_gate": torch.tanh(self.k_transform.gate).detach(),
            "v_gate": torch.tanh(self.v_transform.gate).detach(),
            "reg_loss": self.regularization_loss().detach(),
        }
        return out, stats


class ConvDictionarySynthesis(nn.Module):
    """Dictionary-learning inspired synthesis block for the decoder side.

    This is the synthesis counterpart to the analysis-side convolutional transform.
    It is trained end-to-end with backprop inside a transformer-like seq2seq model,
    rather than via standalone alternating convolutional dictionary learning.
    """

    def __init__(self, d_model: int, kernel_size: int = 3, reg_weight: float = 1e-3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = kernel_size // 2
        self.reg_weight = reg_weight
        self.synthesis = nn.ConvTranspose1d(d_model, d_model, kernel_size=kernel_size, padding=padding)
        self.mix = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def regularization_loss(self) -> torch.Tensor:
        reg = torch.sum(self.synthesis.weight.square()) + torch.sum(self.mix.weight.square())
        return self.reg_weight * reg

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # hidden: (batch, seq, d_model)
        y = self.synthesis(hidden.transpose(1, 2)).transpose(1, 2)
        y = self.mix(y)
        out = self.norm(hidden + torch.tanh(self.gate) * y)
        stats = {"dict_gate": torch.tanh(self.gate).detach(), "dict_reg_loss": self.regularization_loss().detach()}
        return out, stats


class Seq2SeqDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        ff_dim: int,
        self_attention: nn.Module,
        cross_attention: nn.Module,
        dictionary_synthesis: ConvDictionarySynthesis | None = None,
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.dictionary_synthesis = dictionary_synthesis
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )

    def regularization_loss(self) -> torch.Tensor:
        device = self.self_norm.weight.device
        total = torch.tensor(0.0, device=device)
        if hasattr(self.self_attention, "regularization_loss"):
            total = total + self.self_attention.regularization_loss()
        if hasattr(self.cross_attention, "regularization_loss"):
            total = total + self.cross_attention.regularization_loss()
        if self.dictionary_synthesis is not None:
            total = total + self.dictionary_synthesis.regularization_loss()
        return total

    def forward(
        self,
        hidden: torch.Tensor,
        memory_hidden: torch.Tensor,
        self_attention_mask: torch.Tensor | None = None,
        memory_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stats: dict[str, torch.Tensor] = {}
        self_attn_out, self_stats = self.self_attention(self.self_norm(hidden), attention_mask=self_attention_mask)
        hidden = hidden + self_attn_out
        cross_attn_out, cross_stats = self.cross_attention(
            query_hidden=self.cross_norm(hidden),
            memory_hidden=memory_hidden,
            attention_mask=memory_attention_mask,
        )
        hidden = hidden + cross_attn_out
        if self.dictionary_synthesis is not None:
            synth_out, synth_stats = self.dictionary_synthesis(hidden)
            hidden = synth_out
            stats.update({f"synth_{key}": value for key, value in synth_stats.items()})
        hidden = hidden + self.ff(hidden)
        stats.update({f"self_{key}": value for key, value in self_stats.items()})
        stats.update({f"cross_{key}": value for key, value in cross_stats.items()})
        return hidden, stats


class MiniSeq2SeqBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        ff_dim: int,
        num_layers: int,
        encoder_attention_factory: callable,
        decoder_self_attention_factory: callable,
        cross_attention_factory: callable,
        use_dictionary_decoder: bool,
        conv_kernel_size: int,
        reg_weight: float,
    ) -> None:
        super().__init__()
        self.encoder_embeddings = BertStyleEmbeddings(vocab_size=vocab_size, max_seq_len=max_seq_len, d_model=d_model)
        self.encoder_layers = nn.ModuleList(
            [
                TinyEncoderBlock(d_model=d_model, ff_dim=ff_dim, attention=encoder_attention_factory())
                for _ in range(num_layers)
            ]
        )
        self.encoder_final_norm = nn.LayerNorm(d_model)

        self.decoder_embeddings = CausalTokenEmbeddings(vocab_size=vocab_size, max_seq_len=max_seq_len, d_model=d_model)
        self.decoder_layers = nn.ModuleList(
            [
                Seq2SeqDecoderBlock(
                    d_model=d_model,
                    ff_dim=ff_dim,
                    self_attention=decoder_self_attention_factory(),
                    cross_attention=cross_attention_factory(),
                    dictionary_synthesis=(
                        ConvDictionarySynthesis(d_model=d_model, kernel_size=conv_kernel_size, reg_weight=reg_weight)
                        if use_dictionary_decoder
                        else None
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder_final_norm = nn.LayerNorm(d_model)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    def extra_loss(self) -> torch.Tensor:
        losses = [layer.attention.regularization_loss() for layer in self.encoder_layers]
        losses.extend(layer.regularization_loss() for layer in self.decoder_layers)
        if not losses:
            return torch.tensor(0.0, device=self.decoder_final_norm.weight.device)
        return torch.stack(losses).sum()

    def encode(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.encoder_embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        stats: dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.encoder_layers):
            hidden, layer_stats = layer(hidden, attention_mask=None)
            for key, value in layer_stats.items():
                stats[f"encoder_layer_{idx}_{key}"] = value
        return self.encoder_final_norm(hidden), stats

    def decode(self, decoder_input_ids: torch.Tensor, memory_hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.decoder_embeddings(decoder_input_ids)
        self_mask = self._causal_mask(seq_len=decoder_input_ids.shape[1], device=decoder_input_ids.device)
        memory_mask = torch.ones(
            decoder_input_ids.shape[0], decoder_input_ids.shape[1], memory_hidden.shape[1],
            device=decoder_input_ids.device, dtype=torch.bool
        )
        stats: dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.decoder_layers):
            hidden, layer_stats = layer(
                hidden=hidden,
                memory_hidden=memory_hidden,
                self_attention_mask=self_mask,
                memory_attention_mask=memory_mask,
            )
            for key, value in layer_stats.items():
                stats[f"decoder_layer_{idx}_{key}"] = value
        return self.decoder_final_norm(hidden), stats

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        memory_hidden, enc_stats = self.encode(input_ids=input_ids, token_type_ids=token_type_ids)
        decoder_hidden, dec_stats = self.decode(decoder_input_ids=decoder_input_ids, memory_hidden=memory_hidden)
        stats = enc_stats | dec_stats
        return decoder_hidden, stats


class MiniSeq2SeqForConditionalGeneration(nn.Module):
    """Mini encoder-decoder transformer.

    With ``use_dictionary_decoder=True``, the encoder uses convolutional transform-style
    attention and the decoder adds a convolutional dictionary-style synthesis block.
    This keeps the model recognizably transformer-based while exposing an
    analysis-synthesis variant for demos.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        ff_dim: int,
        num_layers: int,
        encoder_attention_factory: callable,
        decoder_self_attention_factory: callable,
        cross_attention_factory: callable,
        use_dictionary_decoder: bool,
        conv_kernel_size: int,
        reg_weight: float,
    ) -> None:
        super().__init__()
        self.backbone = MiniSeq2SeqBackbone(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            ff_dim=ff_dim,
            num_layers=num_layers,
            encoder_attention_factory=encoder_attention_factory,
            decoder_self_attention_factory=decoder_self_attention_factory,
            cross_attention_factory=cross_attention_factory,
            use_dictionary_decoder=use_dictionary_decoder,
            conv_kernel_size=conv_kernel_size,
            reg_weight=reg_weight,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.backbone.decoder_embeddings.token_embeddings.weight

    def extra_loss(self) -> torch.Tensor:
        return self.backbone.extra_loss()

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden, stats = self.backbone(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            token_type_ids=token_type_ids,
        )
        logits = self.lm_head(hidden)
        return logits, stats
