import torch
from transformed_transformer import ModelConfig, build_attention, build_mini_slm

ATTENTION_TYPES = ["standard", "transformed_dense", "transformed_sparse", "transformed_conv"]


def test_attention_backends_accept_boolean_mask() -> None:
    x = torch.randn(2, 5, 12)
    mask = torch.tensor([[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]], dtype=torch.bool)
    for attention_type in ATTENTION_TYPES:
        module = build_attention(ModelConfig(vocab_size=32, seq_len=5, d_model=12, attention_type=attention_type))
        y, stats = module(x, attention_mask=mask)
        assert y.shape == x.shape
        assert isinstance(stats, dict)


def test_minislm_causal_forward_is_finite() -> None:
    for attention_type in ATTENTION_TYPES:
        model = build_mini_slm(ModelConfig(vocab_size=23, seq_len=9, d_model=16, ff_dim=32, num_layers=2, attention_type=attention_type))
        ids = torch.randint(0, 23, (2, 9))
        logits, _ = model(ids)
        assert torch.isfinite(logits).all()
