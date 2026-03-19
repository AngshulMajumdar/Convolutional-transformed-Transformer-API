from __future__ import annotations

import torch
import torch.nn.functional as F

from transformed_transformer import ModelConfig, build_mini_slm


def _shift_targets(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return input_ids[:, :-1], input_ids[:, 1:]


def test_minislm_forward_all_attention_backends() -> None:
    for attention_type in ["standard", "transformed_dense", "transformed_sparse", "transformed_conv"]:
        config = ModelConfig(
            vocab_size=31,
            seq_len=12,
            d_model=24,
            ff_dim=48,
            num_layers=2,
            attention_type=attention_type,
            sparse_k=4,
        )
        model = build_mini_slm(config)
        input_ids = torch.randint(0, config.vocab_size, (3, config.seq_len))
        logits, stats = model(input_ids)
        assert logits.shape == (3, config.seq_len, config.vocab_size)
        assert isinstance(stats, dict)


def test_minislm_one_training_step_all_attention_backends() -> None:
    for attention_type in ["standard", "transformed_dense", "transformed_sparse", "transformed_conv"]:
        config = ModelConfig(
            vocab_size=29,
            seq_len=10,
            d_model=16,
            ff_dim=32,
            num_layers=2,
            attention_type=attention_type,
            sparse_k=4,
        )
        model = build_mini_slm(config)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, config.vocab_size, (4, config.seq_len))
        inputs, targets = _shift_targets(input_ids)

        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1)) + model.extra_loss()
        optim.zero_grad()
        loss.backward()
        optim.step()

        assert torch.isfinite(loss).item()
