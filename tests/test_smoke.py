import torch

from transformed_transformer import ModelConfig, build_attention, build_tiny_classifier
from transformed_transformer.utils import make_synthetic_classification_data, set_seed, train_one_step


ATTENTION_TYPES = ["standard", "transformed_dense", "transformed_sparse", "transformed_conv"]


def test_imports_and_attention_builders() -> None:
    for attention_type in ATTENTION_TYPES:
        config = ModelConfig(vocab_size=32, seq_len=12, attention_type=attention_type)
        module = build_attention(config)
        assert module is not None


def test_model_instantiation_and_forward() -> None:
    set_seed(0)
    x, _ = make_synthetic_classification_data(n_samples=4, seq_len=12, vocab_size=32)

    for attention_type in ATTENTION_TYPES:
        config = ModelConfig(vocab_size=32, seq_len=12, attention_type=attention_type)
        model = build_tiny_classifier(config)
        logits, stats = model(x)
        assert logits.shape == (4, 2)
        assert isinstance(stats, dict)
        assert len(stats) > 0
        assert torch.isfinite(logits).all()


def test_tiny_training_step() -> None:
    set_seed(0)
    x, y = make_synthetic_classification_data(n_samples=8, seq_len=12, vocab_size=32)

    for attention_type in ATTENTION_TYPES:
        config = ModelConfig(vocab_size=32, seq_len=12, attention_type=attention_type)
        model = build_tiny_classifier(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        info = train_one_step(model, x, y, optimizer)
        assert info["total_loss"] >= 0.0
