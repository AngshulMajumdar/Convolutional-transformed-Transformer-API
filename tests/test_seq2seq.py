from __future__ import annotations

import torch
import torch.nn.functional as F

from transformed_transformer import (
    ModelConfig,
    build_analysis_synthesis_transformer,
    build_mini_seq2seq,
)


ATTENTION_TYPES = ["standard", "transformed_conv"]


def _make_seq2seq_batch(vocab_size: int, seq_len: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src = torch.randint(3, vocab_size, (batch_size, seq_len))
    tgt = torch.flip(src, dims=[1])
    decoder_input_ids = torch.cat([torch.full((batch_size, 1), 2, dtype=torch.long), tgt[:, :-1]], dim=1)
    return src, decoder_input_ids, tgt


def test_seq2seq_forward_variants() -> None:
    for attention_type in ATTENTION_TYPES:
        config = ModelConfig(vocab_size=37, seq_len=10, d_model=16, ff_dim=32, num_layers=2, attention_type=attention_type)
        standard_model = build_mini_seq2seq(config)
        proposed_model = build_analysis_synthesis_transformer(config)
        src, decoder_input_ids, tgt = _make_seq2seq_batch(config.vocab_size, config.seq_len, batch_size=3)

        logits_std, stats_std = standard_model(src, decoder_input_ids)
        logits_prop, stats_prop = proposed_model(src, decoder_input_ids)
        assert logits_std.shape == (3, config.seq_len, config.vocab_size)
        assert logits_prop.shape == (3, config.seq_len, config.vocab_size)
        assert torch.isfinite(logits_std).all()
        assert torch.isfinite(logits_prop).all()
        assert isinstance(stats_std, dict) and isinstance(stats_prop, dict)
        assert len(stats_std) > 0 and len(stats_prop) > 0


def test_analysis_synthesis_transformer_one_step() -> None:
    config = ModelConfig(vocab_size=41, seq_len=9, d_model=16, ff_dim=32, num_layers=2, attention_type="transformed_conv")
    model = build_analysis_synthesis_transformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    src, decoder_input_ids, tgt = _make_seq2seq_batch(config.vocab_size, config.seq_len, batch_size=4)

    logits, _ = model(src, decoder_input_ids)
    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), tgt.reshape(-1)) + model.extra_loss()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).item()
