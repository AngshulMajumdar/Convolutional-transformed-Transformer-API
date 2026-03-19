import torch
import torch.nn.functional as F

from transformed_transformer import ModelConfig, build_mini_bert_classifier, build_mini_bert_mlm
from transformed_transformer.utils import make_synthetic_classification_data, set_seed


ATTENTION_TYPES = ["standard", "transformed_dense", "transformed_sparse", "transformed_conv"]


def _make_masked_batch(vocab_size: int, seq_len: int, batch_size: int, mask_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(2, vocab_size, (batch_size, seq_len))
    input_ids[:, 0] = 2  # stand-in CLS token for demo stability
    labels = input_ids.clone()
    mask_positions = torch.full((batch_size,), 2, dtype=torch.long)
    masked_input = input_ids.clone()
    masked_input[torch.arange(batch_size), mask_positions] = mask_token_id
    return masked_input, labels


def test_minibert_classifier_forward() -> None:
    set_seed(0)
    x, _ = make_synthetic_classification_data(n_samples=4, seq_len=12, vocab_size=32)
    token_type_ids = torch.zeros_like(x)

    for attention_type in ATTENTION_TYPES:
        config = ModelConfig(vocab_size=32, seq_len=12, attention_type=attention_type, num_layers=2)
        model = build_mini_bert_classifier(config)
        logits, stats = model(x, token_type_ids=token_type_ids)
        assert logits.shape == (4, 2)
        assert torch.isfinite(logits).all()
        assert isinstance(stats, dict)
        assert len(stats) > 0


def test_minibert_classifier_tiny_training_step() -> None:
    set_seed(0)
    x, y = make_synthetic_classification_data(n_samples=8, seq_len=12, vocab_size=32)

    for attention_type in ATTENTION_TYPES:
        config = ModelConfig(vocab_size=32, seq_len=12, attention_type=attention_type, num_layers=2)
        model = build_mini_bert_classifier(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y) + model.extra_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert float(loss.detach()) >= 0.0


def test_minibert_mlm_forward_and_step() -> None:
    set_seed(0)

    for attention_type in ATTENTION_TYPES:
        config = ModelConfig(vocab_size=41, seq_len=10, attention_type=attention_type, num_layers=2, mask_token_id=1)
        model = build_mini_bert_mlm(config)
        masked_input, labels = _make_masked_batch(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            batch_size=4,
            mask_token_id=config.mask_token_id,
        )
        logits, stats = model(masked_input)
        assert logits.shape == (4, config.seq_len, config.vocab_size)
        assert torch.isfinite(logits).all()
        assert len(stats) > 0

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1)) + model.extra_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert float(loss.detach()) >= 0.0
